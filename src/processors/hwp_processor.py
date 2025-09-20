"""
HWP 문서 처리기
pyhwpx를 사용한 직접 텍스트 추출 방식 구현
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Union

from .base import DocumentProcessor, DocumentChunk, TableImageChunk

class HWPProcessor(DocumentProcessor):
    """HWP 문서 처리기 (pyhwpx 사용)"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, 
                 extract_table_images: bool = False, xhtml_dir: Optional[str] = None):
        super().__init__(chunk_size, overlap)
        self.supported_extensions = ['.hwp']
        self.extract_table_images = extract_table_images
        self.xhtml_dir = xhtml_dir
        
        # 표 이미지 추출이 활성화된 경우 필요한 라이브러리 임포트
        if self.extract_table_images:
            self._import_table_processing_libraries()

    def _import_table_processing_libraries(self):
        """표 처리에 필요한 라이브러리들을 동적으로 임포트"""
        try:
            global BeautifulSoup, webdriver, ChromeDriverManager, By, WebDriverWait, expected_conditions
            global Image, Service
            
            from bs4 import BeautifulSoup
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions
            from PIL import Image
            
            print("표 이미지 처리 라이브러리 로드 완료")
        except ImportError as e:
            print(f"표 이미지 처리 라이브러리 로드 실패: {e}")
            print("pip install selenium webdriver-manager beautifulsoup4 pillow openai 명령으로 설치해주세요.")
            self.extract_table_images = False

    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일에서 텍스트 및 표 이미지 추출"""
        print(f"HWP 파일 처리 시도: {Path(file_path).name}")

        # 1. 기존 텍스트 추출
        text_content = self._extract_hwp_text(file_path)

        # 2. 표 이미지 추출 (옵션)
        table_images = []
        if self.extract_table_images and self.xhtml_dir:
            try:
                xhtml_path = self._find_corresponding_xhtml(file_path)
                if xhtml_path:
                    table_images = self._extract_table_images_from_xhtml(xhtml_path)
                    print(f"표 이미지 {len(table_images)}개 추출 완료")
                else:
                    print(f"대응하는 XHTML 파일을 찾을 수 없습니다: {file_path}")
            except Exception as e:
                print(f"표 이미지 추출 실패: {e}")

        # 기본 메타데이터 생성
        metadata = self._create_base_metadata(file_path)
        metadata.update({
            "source_type": "hwp",
            "extraction_method": "hybrid_with_tables" if table_images else "text_only",
            "table_count": len(table_images)
        })

        return {
            "text": text_content,
            "table_images": table_images,
            "metadata": metadata
        }

    def _extract_hwp_text(self, file_path: str) -> str:
        """HWP 파일에서 텍스트 추출 - 기존 로직"""
        # 1. olefile 방법 시도
        text_content = self._extract_with_olefile(file_path)

        # 2. 실패 시 바이너리 패턴 방법 시도
        if not text_content or len(text_content.strip()) < 10:
            text_content = self._extract_with_binary_pattern(file_path)

        # 3. 여전히 실패 시 기본 메시지
        if not text_content or len(text_content.strip()) < 10:
            text_content = f"HWP 파일 '{Path(file_path).name}'에서 텍스트를 추출할 수 없습니다. 파일이 암호화되어 있거나 지원되지 않는 형식일 수 있습니다."

        return text_content

    def _extract_with_olefile(self, file_path: str) -> str:
        """olefile을 사용한 텍스트 추출"""
        try:
            import olefile

            if not olefile.isOleFile(file_path):
                return ""

            ole = olefile.OleFileIO(file_path)

            # PrvText 스트림 찾기
            for stream_path in ole.listdir():
                if isinstance(stream_path, list) and 'PrvText' in str(stream_path):
                    try:
                        # 간단한 방법으로 데이터 읽기
                        with ole.openfilepath(stream_path) as f:
                            raw_data = f.read()

                        # UTF-16LE로 디코딩 시도 (HWP의 일반적인 인코딩)
                        if len(raw_data) > 2:
                            try:
                                text = raw_data.decode('utf-16le')
                                cleaned = self._clean_hwp_text(text)
                                if len(cleaned.strip()) > 10:
                                    ole.close()
                                    return cleaned
                            except:
                                pass

                    except Exception:
                        continue

            ole.close()
            return ""

        except Exception:
            return ""

    def _extract_with_binary_pattern(self, file_path: str) -> str:
        """바이너리 패턴을 사용한 텍스트 추출"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            # 한글 텍스트가 포함된 UTF-16 패턴 찾기
            text_parts = []

            # UTF-16LE 패턴으로 텍스트 조각 찾기
            i = 0
            while i < len(data) - 1:
                if data[i] != 0:  # 첫 번째 바이트가 0이 아니고
                    if i + 1 < len(data) and data[i + 1] == 0:  # 두 번째 바이트가 0이면
                        # 연속된 UTF-16LE 문자열 찾기
                        start = i
                        while i < len(data) - 1:
                            if data[i] != 0 and data[i + 1] == 0:
                                i += 2
                            else:
                                break

                        if i - start > 20:  # 충분히 긴 텍스트 조각
                            try:
                                text_chunk = data[start:i].decode('utf-16le')
                                # 한글이나 영문이 포함된 의미있는 텍스트인지 확인
                                if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in text_chunk) or \
                                   any(c.isalpha() for c in text_chunk):
                                    text_parts.append(text_chunk)
                            except:
                                pass
                i += 1

            if text_parts:
                combined_text = ' '.join(text_parts)
                return self._clean_hwp_text(combined_text)

            return ""

        except Exception:
            return ""

    def chunk_content(self, content: Dict[str, Any]) -> List[Union[DocumentChunk, TableImageChunk]]:
        """HWP 내용을 텍스트 청크와 표 이미지 청크로 분할"""
        text = content["text"]
        table_images = content.get("table_images", [])
        metadata = content["metadata"]

        # 1. 텍스트 청킹
        text_chunks = self._smart_chunk_text(text, metadata)

        # HWP 특화 메타데이터 추가
        for chunk in text_chunks:
            chunk.metadata.update({
                "source_type": "hwp",
                "chunk_type": "text",
                "extraction_method": metadata.get("extraction_method", "text_only")
            })

        # 2. 표 이미지 청킹 (옵션)
        table_chunks = []
        if table_images:
            table_chunks = self._create_table_image_chunks(table_images, metadata)

        # 3. 청크 결합 (텍스트 청크 + 표 이미지 청크)
        all_chunks = text_chunks + table_chunks

        # 청크 인덱스 재정렬
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i

        print(f"총 청크 생성: {len(text_chunks)}개 텍스트 + {len(table_chunks)}개 표 이미지")
        return all_chunks

    def _clean_hwp_text(self, raw_text: str) -> str:
        """HWP에서 추출된 원시 텍스트 정리"""
        import re

        if not raw_text:
            return ""

        # 널 문자 제거
        text = raw_text.replace('\x00', '')

        # 제어 문자 제거 (단, 개행과 탭은 유지)
        text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # 연속된 공백 정리
        text = re.sub(r' +', ' ', text)

        # 연속된 개행 정리 (최대 2개까지만)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def _extract_hwp_metadata_patterns(self, text: str) -> Dict[str, Any]:
        """HWP 문서에서 특화 메타데이터 추출"""
        import re

        extracted = {}

        # HWP 문서 특유의 패턴들
        patterns = {
            'hwp_version': r'HWP\s*(\d+\.\d+)',
            'creation_date': r'작성일\s*[:\s]*(\d{4}[.-]\d{1,2}[.-]\d{1,2})',
            'department': r'부서\s*[:\s]*([^\n]+)',
            'document_number': r'문서번호\s*[:\s]*([^\n]+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()

        return extracted

    def _find_corresponding_xhtml(self, hwp_path: str) -> Optional[str]:
        """HWP 파일에 대응하는 XHTML 파일 찾기"""
        hwp_filename = Path(hwp_path).stem  # 확장자 제거
        xhtml_dir = Path(self.xhtml_dir)
        
        # .xhtml 파일 먼저 시도
        xhtml_path = xhtml_dir / f"{hwp_filename}.xhtml"
        if xhtml_path.exists():
            return str(xhtml_path)
        
        return None

    def _extract_table_images_from_xhtml(self, xhtml_path: str) -> List[Dict]:
        """XHTML에서 표 이미지 추출 (컨텍스트 포함)"""
        if not self.extract_table_images:
            return []
        
        try:
            with open(xhtml_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        except Exception as e:
            print(f"XHTML 파일 읽기 실패: {e}")
            return []
        
        tables = soup.find_all('table')
        if not tables:
            print("XHTML에서 표를 찾을 수 없습니다.")
            return []
        
        print(f"XHTML에서 {len(tables)}개 표 발견")
        table_data_list = []
        
        for i, table in enumerate(tables):
            try:
                # 1. 표 앞뒤 컨텍스트 추출
                preceding_text, following_text = self._extract_table_context(soup, table)
                
                # 2. Standalone HTML 생성
                standalone_html = self._create_standalone_table_html(table, i)
                
                # 3. Selenium으로 스크린샷
                image_data = self._screenshot_table_html(standalone_html)
                
                # 4. GPT Vision으로 설명 생성 (컨텍스트 포함)
                gpt_description = self._generate_table_description_with_context(
                    image_data, preceding_text, following_text
                )
                
                table_data_list.append({
                    'image_data': image_data,
                    'gpt_description': gpt_description,
                    'table_html': str(table),
                    'table_index': i,
                    'preceding_context': preceding_text,
                    'following_context': following_text
                })
                
                print(f"표 {i+1}/{len(tables)} 처리 완료")
                
            except Exception as e:
                print(f"표 {i+1} 처리 실패: {e}")
                continue
        
        return table_data_list

    def _extract_table_context(self, soup, table_element):
        """표 앞뒤 텍스트 컨텍스트 추출"""
        preceding_text = ""
        following_text = ""
        
        try:
            # 표 앞 텍스트 추출
            current = table_element
            while current and len(preceding_text) < 500:
                prev_sibling = current.previous_sibling
                if prev_sibling:
                    if hasattr(prev_sibling, 'get_text'):
                        text = prev_sibling.get_text().strip()
                        if text:
                            preceding_text = text + " " + preceding_text
                    current = prev_sibling
                else:
                    current = current.parent
                    if current == soup:  # 루트에 도달하면 중단
                        break
            
            # 표 뒤 텍스트 추출  
            current = table_element
            while current and len(following_text) < 500:
                next_sibling = current.next_sibling
                if next_sibling:
                    if hasattr(next_sibling, 'get_text'):
                        text = next_sibling.get_text().strip()
                        if text:
                            following_text = following_text + " " + text
                    current = next_sibling
                else:
                    current = current.parent
                    if current == soup:  # 루트에 도달하면 중단
                        break
        except Exception as e:
            print(f"컨텍스트 추출 중 오류: {e}")
        
        return preceding_text.strip(), following_text.strip()

    def _create_standalone_table_html(self, table_soup, table_index: int) -> str:
        """깔끔한 표 전용 HTML 생성"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: 'Malgun Gothic', 'Microsoft YaHei', Arial, sans-serif; 
                    margin: 20px; 
                    background-color: white;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    max-width: 800px;
                    margin: 0 auto;
                }}
                th, td {{ 
                    border: 1px solid #333; 
                    padding: 8px 12px; 
                    text-align: left;
                    vertical-align: top;
                    word-wrap: break-word;
                }}
                th {{ 
                    background-color: #f5f5f5; 
                    font-weight: bold; 
                }}
                tr:nth-child(even) {{
                    background-color: #fafafa;
                }}
            </style>
        </head>
        <body>
            {table_html}
        </body>
        </html>
        """
        return html_template.format(table_html=str(table_soup))

    def _screenshot_table_html(self, html_content: str) -> bytes:
        """Selenium으로 표 HTML 스크린샷"""
        try:
            import tempfile
            import os
            
            # Chrome 옵션 설정
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1200,800')
            
            # 드라이버 초기화
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            try:
                # HTML을 임시 파일로 저장
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                    f.write(html_content)
                    temp_html_path = f.name
                
                # 페이지 로드
                driver.get(f"file://{temp_html_path}")
                
                # 표 요소 찾기 및 스크린샷
                table_element = WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
                # 표 영역 스크린샷
                screenshot = table_element.screenshot_as_png
                
                return screenshot
                
            finally:
                driver.quit()
                # 임시 파일 정리
                try:
                    os.unlink(temp_html_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"스크린샷 생성 실패: {e}")
            # 기본 이미지 반환 (빈 PNG)
            try:
                from PIL import Image
                import io
                img = Image.new('RGB', (400, 200), color='white')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                return img_bytes.getvalue()
            except:
                return b''  # 최후의 fallback

    def _generate_table_description_with_context(self, image_data: bytes, 
                                               preceding_text: str, 
                                               following_text: str) -> str:
        """GPT Vision API로 표 이미지 분석 (앞뒤 컨텍스트 포함)"""
        try:
            import base64
            from openai import OpenAI
            
            # OpenAI 클라이언트 초기화
            client = OpenAI()  # API 키는 환경변수 OPENAI_API_KEY에서 자동으로 가져옴
            
            # 컨텍스트 텍스트 정리 (너무 길면 자르기)
            preceding = preceding_text[-300:] if len(preceding_text) > 300 else preceding_text
            following = following_text[:300] if len(following_text) > 300 else following_text
            
            context_prompt = f"""
이 표 이미지를 분석하여 상세한 설명을 작성해주세요.

**문서 컨텍스트:**
- 표 앞 내용: "{preceding.strip()}"
- 표 뒤 내용: "{following.strip()}"

**분석 요청사항:**
1. 표의 주제와 목적
2. 주요 컬럼과 데이터 유형
3. 핵심 내용과 중요한 수치
4. 앞뒤 텍스트와의 연관성
5. 이 표가 문서에서 담당하는 역할

검색과 이해에 도움이 되도록 구체적이고 상세하게 작성해주세요.
"""
            
            # 새 버전 API 호출 방식
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": context_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"GPT Vision API 호출 실패: {e}")
            # Fallback: 기본 설명 반환
            return f"표 이미지 (크기: {len(image_data)} bytes). " + \
                   f"컨텍스트: {preceding[:100]}... → 표 → {following[:100]}..."

    def _create_table_image_chunks(self, table_images: List[Dict], base_metadata: Dict) -> List[TableImageChunk]:
        """표 이미지들을 TableImageChunk로 변환"""
        chunks = []
        
        for i, table_data in enumerate(table_images):
            try:
                # TableImageChunk 생성
                chunk_metadata = {
                    **base_metadata,
                    "chunk_type": "table_image", 
                    "table_index": table_data.get("table_index", i),
                    "has_image": True,
                    "extraction_method": "gpt_vision_with_context",
                    "preceding_context": table_data.get("preceding_context", ""),
                    "following_context": table_data.get("following_context", "")
                }
                
                chunk = TableImageChunk(
                    content=table_data.get("gpt_description", ""),
                    metadata=chunk_metadata,
                    chunk_id="",  # __post_init__에서 자동 생성
                    document_id=base_metadata.get('document_id', 'unknown'),
                    chunk_index=i,
                    
                    # 표 특화 필드들
                    table_html=table_data.get('table_html', ''),
                    image_data=table_data.get('image_data', b''),
                    gpt_description=table_data.get('gpt_description', '')
                )
                
                chunks.append(chunk)
                
            except Exception as e:
                print(f"표 {i} 청크 생성 실패: {e}")
                continue
        
        return chunks