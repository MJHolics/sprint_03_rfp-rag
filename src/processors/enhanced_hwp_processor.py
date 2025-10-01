"""
통합 HWP 문서 처리기
다양한 방법으로 텍스트 추출, 메타데이터 자동 추출, 멀티모달 이미지 분석
"""
import os
import re
import subprocess
import sys
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shutil

from .base import DocumentProcessor, DocumentChunk

# 멀티모달 처리를 위한 선택적 import
try:
    import olefile
    import openai
    from PIL import Image
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# 설정 import
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL
except ImportError:
    OPENAI_API_KEY = None
    OPENAI_CHAT_MODEL = None

class EnhancedHWPProcessor(DocumentProcessor):
    """통합 HWP 문서 처리기 - 텍스트 추출, 메타데이터 분석, 이미지 분석 통합"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, enable_multimodal: bool = True):
        super().__init__(chunk_size, overlap)
        self.supported_extensions = ['.hwp']
        self.enable_multimodal = enable_multimodal and MULTIMODAL_AVAILABLE

        # OpenAI 클라이언트 초기화 (멀티모달 분석용)
        if self.enable_multimodal and OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.openai_client = None

        # 메타데이터 추출 패턴
        self.metadata_patterns = {
            'agency': [
                r'발주기관\s*[:\：]\s*([^\n\r]+)',
                r'발주청\s*[:\：]\s*([^\n\r]+)',
                r'발주처\s*[:\：]\s*([^\n\r]+)',
                r'주관기관\s*[:\：]\s*([^\n\r]+)',
                r'계약기관\s*[:\：]\s*([^\n\r]+)',
                r'(한국\w+(?:공단|공사|연구원|재단|진흥원|개발원))',
                r'(\w+(?:대학교|대학))',
                r'(\w+(?:특별시|광역시|도))',
                r'(\w+(?:시청|구청|군청|도청))',
            ],
            'business_type_patterns': {
                '정보시스템구축': [r'시스템\s*구축', r'시스템\s*개발', r'플랫폼\s*구축'],
                '정보시스템고도화': [r'시스템\s*고도화', r'시스템\s*개선', r'기능개선'],
                'ISP/ISMP': [r'ISP\s*수립', r'ISMP\s*수립', r'정보화전략계획'],
                '홈페이지구축': [r'홈페이지\s*구축', r'웹사이트\s*구축', r'포털\s*구축'],
                'PMC용역': [r'PMC\s*용역', r'사업관리', r'프로젝트\s*관리'],
                '운영용역': [r'운영\s*용역', r'시스템\s*운영'],
                'ERP구축': [r'ERP\s*구축', r'전사적\s*자원관리'],
                '보안시스템': [r'보안\s*시스템', r'사이버\s*보안'],
                '조사연구': [r'실태조사', r'현황조사', r'연구\s*용역']
            },
            'budget': [
                r'예산\s*[:\：]?\s*([0-9,]+)\s*원',
                r'사업비\s*[:\：]?\s*([0-9,]+)\s*원',
                r'총\s*사업비\s*[:\：]?\s*([0-9,]+)\s*원',
                r'계약금액\s*[:\：]?\s*([0-9,]+)\s*원',
                r'([0-9,]+)\s*억\s*원',
                r'([0-9,]+)\s*천만\s*원',
                r'금\s*([0-9,]+)\s*원',
                r'(\d{1,3}(?:,\d{3})*)\s*원',
                r'예산\s*(\d{1,3}(?:,\d{3})*)',
                r'([0-9]+억[0-9,]*만?원)',
                r'총액\s*[:\：]?\s*([0-9,]+)\s*원',
                r'사업규모\s*[:\：]?\s*([0-9,]+)\s*원'
            ],
            'deadline': [
                r'제출마감\s*[:\：]\s*(\d{4}[년.-]\d{1,2}[월.-]\d{1,2}일?)',
                r'접수마감\s*[:\：]\s*(\d{4}[년.-]\d{1,2}[월.-]\d{1,2}일?)',
                r'제안서\s*제출\s*[:\：]\s*(\d{4}[년.-]\d{1,2}[월.-]\d{1,2}일?)',
                r'마감일시\s*[:\：]\s*(\d{4}[년.-]\d{1,2}[월.-]\d{1,2}일?)',
                r'(\d{4})[년.-](\d{1,2})[월.-](\d{1,2})일?\s*까지'
            ]
        }

    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일에서 내용 추출 - 다중 방법 사용"""
        print(f"향상된 HWP 파일 처리: {Path(file_path).name}")

        text_content = ""
        extraction_method = "none"

        # 방법 1: LibreOffice를 통한 변환 시도
        text_content = self._extract_with_libreoffice(file_path)
        if self._is_meaningful_content(text_content):
            extraction_method = "libreoffice"
        else:
            # 방법 2: olefile 방법 시도
            text_content = self._extract_with_olefile(file_path)
            if self._is_meaningful_content(text_content):
                extraction_method = "olefile"
            else:
                # 방법 3: 바이너리 패턴 방법 시도
                text_content = self._extract_with_binary_pattern(file_path)
                if self._is_meaningful_content(text_content):
                    extraction_method = "binary_pattern"
                else:
                    # 방법 4: hwp5 라이브러리 시도 (설치되어 있다면)
                    text_content = self._extract_with_hwp5(file_path)
                    if self._is_meaningful_content(text_content):
                        extraction_method = "hwp5"

        # 여전히 실패 시 파일명에서 정보 추출
        if not self._is_meaningful_content(text_content):
            text_content = f"파일명: {Path(file_path).name}"
            extraction_method = "filename_only"

        # 기본 메타데이터 생성
        metadata = self._create_base_metadata(file_path)

        # 추출된 텍스트에서 메타데이터 자동 추출
        extracted_metadata = self._extract_metadata_from_text(text_content, Path(file_path).name)
        metadata.update(extracted_metadata)

        metadata.update({
            "source_type": "hwp",
            "extraction_method": extraction_method,
            "content_length": len(text_content)
        })

        content = {
            "text": text_content,
            "images": [],
            "tables": [],
            "metadata": metadata
        }

        # 멀티모달 이미지 분석 (활성화된 경우)
        if self.enable_multimodal:
            try:
                content = self._enhance_with_image_analysis(content, file_path)
            except Exception as e:
                print(f"HWP 이미지 분석 실패: {e}")

        return content

    def _extract_with_libreoffice(self, file_path: str) -> str:
        """LibreOffice를 통한 HWP → TXT 변환"""
        try:
            # LibreOffice가 설치되어 있는지 확인
            libreoffice_paths = [
                "libreoffice",
                "soffice",
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
            ]

            soffice_path = None
            for path in libreoffice_paths:
                if shutil.which(path) or Path(path).exists():
                    soffice_path = path
                    break

            if not soffice_path:
                return ""

            with tempfile.TemporaryDirectory() as temp_dir:
                # HWP를 TXT로 변환
                cmd = [
                    soffice_path,
                    "--headless",
                    "--convert-to", "txt",
                    "--outdir", temp_dir,
                    file_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    # 변환된 TXT 파일 찾기
                    txt_file = Path(temp_dir) / f"{Path(file_path).stem}.txt"
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        return self._clean_extracted_text(content)

        except Exception as e:
            print(f"LibreOffice 변환 실패: {e}")

        return ""

    def _extract_with_hwp5(self, file_path: str) -> str:
        """hwp5 라이브러리를 사용한 추출 (있다면)"""
        try:
            import hwp5
            from hwp5.xmlmodel import Hwp5File
            from hwp5.treeop import STARTTAG, ENDTAG, TEXT

            with Hwp5File(file_path) as hwp:
                text_parts = []

                for event in hwp.bodytext.events():
                    if event[0] == TEXT:
                        text_parts.append(event[1])

                if text_parts:
                    content = " ".join(text_parts)
                    return self._clean_extracted_text(content)

        except ImportError:
            # hwp5 라이브러리가 설치되지 않음
            pass
        except Exception as e:
            print(f"hwp5 추출 실패: {e}")

        return ""

    def _extract_with_olefile(self, file_path: str) -> str:
        """기존 olefile 방법 (개선됨)"""
        try:
            import olefile

            if not olefile.isOleFile(file_path):
                return ""

            ole = olefile.OleFileIO(file_path)
            text_parts = []

            # 모든 스트림에서 텍스트 찾기
            for stream_path in ole.listdir():
                if isinstance(stream_path, list):
                    stream_name = "/".join(stream_path)

                    # 텍스트 관련 스트림들
                    if any(keyword in stream_name.lower() for keyword in ['text', 'body', 'content', 'prvtext']):
                        try:
                            with ole.openfilepath(stream_path) as f:
                                raw_data = f.read()

                            # 여러 인코딩으로 시도
                            for encoding in ['utf-16le', 'utf-16be', 'cp949', 'utf-8']:
                                try:
                                    text = raw_data.decode(encoding)
                                    cleaned = self._clean_extracted_text(text)
                                    if len(cleaned.strip()) > 50:
                                        text_parts.append(cleaned)
                                        break
                                except:
                                    continue

                        except Exception:
                            continue

            ole.close()

            if text_parts:
                return " ".join(text_parts)

        except Exception as e:
            print(f"olefile 추출 실패: {e}")

        return ""

    def _extract_with_binary_pattern(self, file_path: str) -> str:
        """향상된 바이너리 패턴 추출"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            text_parts = []

            # UTF-16LE 패턴으로 긴 텍스트 조각 찾기
            i = 0
            while i < len(data) - 1:
                if data[i] != 0 and i + 1 < len(data) and data[i + 1] == 0:
                    start = i
                    text_length = 0

                    # 연속된 UTF-16LE 문자열 길이 측정
                    while i < len(data) - 1:
                        if data[i] != 0 and data[i + 1] == 0:
                            # 한글 또는 영문 범위 확인
                            char_code = data[i] + (data[i + 1] << 8)
                            if (0x20 <= char_code <= 0x7E) or (0xAC00 <= char_code <= 0xD7A3):
                                text_length += 2
                                i += 2
                            else:
                                break
                        else:
                            break

                    # 충분히 긴 텍스트 조각 추출
                    if text_length > 100:
                        try:
                            text_chunk = data[start:start + text_length].decode('utf-16le')
                            # 의미있는 텍스트인지 확인
                            if self._is_meaningful_text_chunk(text_chunk):
                                text_parts.append(text_chunk)
                        except:
                            pass

                i += 1

            if text_parts:
                combined = " ".join(text_parts)
                return self._clean_extracted_text(combined)

        except Exception as e:
            print(f"바이너리 패턴 추출 실패: {e}")

        return ""

    def _is_meaningful_content(self, text: str) -> bool:
        """추출된 내용이 의미있는지 확인"""
        if not text or len(text.strip()) < 100:
            return False

        # 한글 또는 영문이 충분히 포함되어 있는지 확인
        korean_chars = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)

        return (korean_chars > 20) or (english_chars > 50)

    def _is_meaningful_text_chunk(self, text: str) -> bool:
        """텍스트 조각이 의미있는지 확인"""
        if len(text.strip()) < 10:
            return False

        # 한글, 영문, 숫자가 포함되어 있는지 확인
        has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in text)
        has_alpha = any(c.isalpha() for c in text)
        has_common_words = any(word in text for word in ['사업', '시스템', '구축', '개발', '용역', '제안'])

        return has_korean or has_alpha or has_common_words

    def _clean_extracted_text(self, text: str) -> str:
        """추출된 텍스트 정리"""
        if not text:
            return ""

        # 널 문자 및 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # 연속된 공백 정리
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 의미없는 반복 문자 제거
        text = re.sub(r'(.)\1{10,}', r'\1', text)

        return text.strip()

    def _extract_metadata_from_text(self, text: str, filename: str) -> Dict[str, Any]:
        """텍스트에서 메타데이터 추출"""
        metadata = {}

        # 1. 파일명에서 발주기관 추출
        if '_' in filename:
            potential_agency = filename.split('_')[0]
            if len(potential_agency) > 1 and not potential_agency.isdigit():
                metadata['agency'] = potential_agency

        # 2. 텍스트에서 발주기관 추출
        if 'agency' not in metadata:
            agency = self._extract_agency_from_text(text)
            if agency:
                metadata['agency'] = agency

        # 3. 사업유형 추출
        business_type = self._extract_business_type_from_text(text, filename)
        if business_type:
            metadata['business_type'] = business_type

        # 4. 예산 추출
        budget = self._extract_budget_from_text(text)
        if budget:
            metadata['budget'] = budget

        # 5. 마감일 추출
        deadline = self._extract_deadline_from_text(text)
        if deadline:
            metadata['deadline'] = deadline

        return metadata

    def _extract_agency_from_text(self, text: str) -> Optional[str]:
        """텍스트에서 발주기관 추출"""
        # 앞부분에서 우선 검색
        search_text = text[:2000]

        for pattern in self.metadata_patterns['agency']:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                agency = matches[0].strip()
                if len(agency) > 1 and not agency.isdigit():
                    return re.sub(r'[^\w\s()（）]', '', agency).strip()

        return None

    def _extract_business_type_from_text(self, text: str, filename: str) -> Optional[str]:
        """텍스트와 파일명에서 사업유형 추출"""
        search_text = f"{text[:3000]} {filename}".lower()

        type_scores = {}
        for business_type, patterns in self.metadata_patterns['business_type_patterns'].items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, search_text, re.IGNORECASE))

            if score > 0:
                type_scores[business_type] = score

        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        return '기타'

    def _extract_budget_from_text(self, text: str) -> Optional[str]:
        """텍스트에서 예산 추출"""
        search_text = text[:5000]

        for pattern in self.metadata_patterns['budget']:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                budget = matches[0].replace(',', '').strip()
                if budget.isdigit() and int(budget) > 1000:
                    return f"{budget}원"

        return None

    def _extract_deadline_from_text(self, text: str) -> Optional[str]:
        """텍스트에서 마감일 추출"""
        search_text = text[:3000]

        for pattern in self.metadata_patterns['deadline']:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple) and len(matches[0]) == 3:
                    year, month, day = matches[0]
                    if len(year) == 2:
                        year = f"20{year}"
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    date_str = str(matches[0])
                    date_str = re.sub(r'[년월일]', '-', date_str)
                    date_str = re.sub(r'[-]+', '-', date_str).strip('-')
                    return date_str

        return None

    def chunk_content(self, content: Dict[str, Any]) -> List[DocumentChunk]:
        """HWP 내용을 의미있는 청크로 분할"""
        text = content["text"]
        metadata = content["metadata"]

        # 텍스트가 충분히 길면 스마트 청킹 적용
        if len(text) > 500:
            chunks = self._smart_chunk_text(text, metadata)
        else:
            # 짧은 텍스트는 전체를 하나의 청크로
            chunks = [DocumentChunk(
                content=text,
                metadata=metadata.copy(),
                chunk_id="",
                document_id=metadata.get('document_id', ''),
                chunk_index=0
            )]

        # HWP 특화 메타데이터 추가
        for chunk in chunks:
            chunk.metadata.update({
                "source_type": "hwp",
                "extraction_method": metadata.get("extraction_method", "unknown"),
                "content_length": len(chunk.content)
            })

        return chunks

    # 멀티모달 이미지 분석 기능 (hwp_multimodal_processor.py에서 통합)
    def _enhance_with_image_analysis(self, content: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """HWP 콘텐츠에 이미지 분석 결과 통합"""
        if not self.openai_client:
            return content

        try:
            print(f"HWP 멀티모달 분석 시작: {os.path.basename(file_path)}")

            # 이미지 분석 수행
            image_analyses = self._analyze_hwp_images(file_path)

            # 이미지에서 추출한 텍스트를 메인 텍스트에 추가
            enhanced_text = content.get("text", "")

            for img_analysis in image_analyses:
                stream_name = img_analysis.get("stream_name", "알 수 없음")
                extracted_text = img_analysis["extracted_text"]
                description = img_analysis["description"]
                content_type = img_analysis["content_type"]

                if extracted_text:
                    enhanced_text += f"\n\n[HWP 이미지 분석: {stream_name} - {content_type}]\n"
                    enhanced_text += f"추출된 텍스트: {extracted_text}\n"
                    if description != extracted_text:
                        enhanced_text += f"상세 설명: {description}\n"

            # 기존 content 업데이트
            enhanced_content = content.copy()
            enhanced_content["text"] = enhanced_text
            enhanced_content["hwp_image_analyses"] = image_analyses
            enhanced_content["total_analyzed_hwp_images"] = len(image_analyses)

            print(f"HWP 멀티모달 분석 완료: {len(image_analyses)}개 이미지 분석")
            return enhanced_content

        except Exception as e:
            print(f"HWP 멀티모달 콘텐츠 강화 실패: {e}")
            return content

    def _analyze_hwp_images(self, file_path: str, max_images: int = 5) -> List[Dict[str, Any]]:
        """HWP 이미지 분석 및 텍스트 추출"""
        if not self.openai_client:
            return []

        try:
            # HWP에서 이미지 추출
            images = self._extract_images_from_hwp(file_path, max_images)
            analyzed_images = []

            for img_data in images:
                try:
                    # GPT-4V로 이미지 분석
                    analysis = self._analyze_image_with_gpt4v(img_data['base64'])

                    analyzed_images.append({
                        **img_data,
                        "analysis": analysis,
                        "extracted_text": analysis.get("extracted_text", ""),
                        "description": analysis.get("description", ""),
                        "content_type": analysis.get("content_type", "unknown")
                    })

                    print(f"HWP 이미지 분석 완료: {img_data['stream_name']}")

                except Exception as e:
                    print(f"이미지 분석 실패 {img_data['stream_name']}: {e}")
                    continue

            return analyzed_images

        except Exception as e:
            print(f"HWP 이미지 분석 실패: {e}")
            return []

    def _extract_images_from_hwp(self, file_path: str, max_images: int = 5) -> List[Dict[str, Any]]:
        """HWP 파일에서 이미지 직접 추출"""
        images = []

        try:
            if not olefile.isOleFile(file_path):
                return []

            ole = olefile.OleFileIO(file_path)
            listdir = ole.listdir()

            image_count = 0
            for entry in listdir:
                if image_count >= max_images:
                    break

                stream_name = '/'.join(entry)

                if any(keyword in stream_name.lower() for keyword in ['bindata', 'picture', 'image', 'img']):
                    try:
                        with ole.openstream(entry) as stream:
                            stream_data = stream.read()

                            if self._is_image_data(stream_data):
                                try:
                                    img = Image.open(io.BytesIO(stream_data))

                                    img_buffer = io.BytesIO()
                                    if img.mode not in ('RGB', 'RGBA'):
                                        img = img.convert('RGB')
                                    img.save(img_buffer, format='PNG')
                                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                                    images.append({
                                        'index': image_count,
                                        'stream_name': stream_name,
                                        'width': img.size[0],
                                        'height': img.size[1],
                                        'format': img.format or 'Unknown',
                                        'mode': img.mode,
                                        'size': len(stream_data),
                                        'base64': img_base64
                                    })

                                    image_count += 1
                                    print(f"HWP 이미지 추출: {stream_name} ({img.size[0]}x{img.size[1]})")

                                except Exception as img_error:
                                    print(f"이미지 처리 실패 {stream_name}: {img_error}")
                                    continue

                    except Exception as stream_error:
                        print(f"스트림 읽기 실패 {stream_name}: {stream_error}")
                        continue

            ole.close()
            return images

        except Exception as e:
            print(f"HWP 이미지 추출 실패: {e}")
            return []

    def _is_image_data(self, data: bytes) -> bool:
        """데이터가 이미지인지 매직 바이트로 확인"""
        if len(data) < 8:
            return False

        image_signatures = [
            b'\xFF\xD8\xFF',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'BM',  # BMP
            b'RIFF',  # WebP (RIFF container)
            b'\x00\x00\x01\x00',  # ICO
        ]

        for signature in image_signatures:
            if data.startswith(signature):
                return True

        return False

    def _analyze_image_with_gpt4v(self, img_base64: str) -> Dict[str, str]:
        """GPT-4V로 이미지 분석"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """이 HWP 문서의 이미지를 분석해주세요. 다음 정보를 JSON 형태로 제공해주세요:
1. extracted_text: 이미지에서 추출한 모든 텍스트 (표, 차트의 숫자와 라벨, 한글 텍스트 포함)
2. description: 이미지 내용에 대한 상세한 설명 (차트 유형, 데이터 트렌드, 표 구조, 다이어그램 설명 등)
3. content_type: 이미지 타입 (table, chart, diagram, flowchart, text, screenshot, other)

RFP(제안요청서) 문서의 이미지이므로 다음에 특히 주의해주세요:
- 기술 요구사항, 시스템 구조도
- 예산 관련 표와 수치
- 일정표, 프로젝트 계획
- 업무 흐름도, 프로세스 다이어그램
- 조직도, 역할 분담표

한글 텍스트 추출에 특별히 신경써주세요."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # JSON 파싱 시도
            try:
                import json
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content

                parsed = json.loads(json_content)
                return {
                    "extracted_text": parsed.get("extracted_text", ""),
                    "description": parsed.get("description", ""),
                    "content_type": parsed.get("content_type", "other")
                }
            except:
                return {
                    "extracted_text": content,
                    "description": content,
                    "content_type": "unknown"
                }

        except Exception as e:
            print(f"GPT-4V 분석 실패: {e}")
            return {
                "extracted_text": "",
                "description": f"분석 실패: {str(e)}",
                "content_type": "error"
            }