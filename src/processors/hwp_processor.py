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
    
    # 표 분할 상수
    CROP_HEIGHT_PX = 1000   # 표 분할 높이 기준 (픽셀)
    OVERLAP_HEIGHT = 100  # 분할 시 겹침 높이 (픽셀)

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
                    # 텍스트 컨텍스트와 함께 표 이미지 추출
                    table_images = self._extract_table_images_from_xhtml(xhtml_path, text_content)
                    print(f"표 이미지 {len(table_images)}개 추출 완료 (대화형 GPT 분석 포함)")
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

    def _extract_table_images_from_xhtml(self, xhtml_path: str, document_text: str = "") -> List[Dict]:
        """XHTML에서 표 이미지 추출 (문서 전체 맥락 기반)"""
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
                
                # 3. 표 높이 체크 후 분할/단일 처리 결정
                image_parts = self._split_table_by_pixels(standalone_html)
                
                # 분할 여부 확인
                is_split = len(image_parts) > 1
                
                # 하나의 표 → 하나의 table_data로 처리 (GPT 설명은 나중에 일괄 생성)
                table_data_list.append({
                    'image_data': image_parts[0],  # 첫 번째 이미지 (호환성)
                    'image_parts': image_parts,    # 모든 분할된 이미지들
                    'gpt_description': "",         # 나중에 대화형 방식으로 생성
                    'table_html': str(table),
                    'table_index': i,
                    'preceding_context': preceding_text,
                    'following_context': following_text,
                    # 분할 관련 정보
                    'is_split_table': is_split,
                    'total_parts': len(image_parts),
                    'overlap_height': self.OVERLAP_HEIGHT if is_split else 0
                })
                
                print(f"표 {i+1}/{len(tables)} 처리 완료")
                
            except Exception as e:
                print(f"표 {i+1} 처리 실패: {e}")
                continue
        
        # 모든 표 이미지 생성 완료 후 대화형 GPT 분석 수행
        if table_data_list and document_text:
            print(f"📊 {len(table_data_list)}개 표에 대한 대화형 GPT 분석 시작...")
            try:
                descriptions = self._generate_all_table_descriptions_conversation_style(
                    document_text, table_data_list
                )
                
                # 생성된 설명을 각 표에 할당
                for i, description in enumerate(descriptions):
                    if i < len(table_data_list):
                        table_data_list[i]['gpt_description'] = description
                        
                print(f"✅ 대화형 GPT 분석 완료!")
                
            except Exception as e:
                print(f"❌ 대화형 GPT 분석 실패: {e}")
                # 실패 시 기본 설명으로 폴백
                for i, table_data in enumerate(table_data_list):
                    if not table_data.get('gpt_description'):
                        table_data['gpt_description'] = f"표 {i+1}: 분석 실패"
        
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
                    margin: 10px; 
                    background-color: white;
                    font-size: 16px; /* 글자 크기 증가 */
                    line-height: 1.4; /* 줄 간격 추가 */
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: auto;   
                    margin: 0 auto;
                }}
                th, td {{ 
                    border: 1px solid #333; 
                    padding: 6px 10px;   /* 패딩 증가 */
                    text-align: left;
                    vertical-align: middle;
                    word-wrap: break-word;
                    font-size: 15px; /* 셀 내 글자 크기 명시적 설정 */
                    line-height: 1.3;
                }}
                th {{ 
                    background-color: #f5f5f5; 
                    font-weight: bold; 
                }}
                tr:nth-child(even) {{
                    background-color: #fafafa;
                }}
                /* 셀 내부 p 태그 여백 제거 */
                td p, th p {{
                    margin: 0;
                }}
            </style>
        </head>
        <body>
            {table_html}
        </body>
        </html>
        """
        return html_template.format(table_html=str(table_soup))

    def _split_table_by_pixels(self, html_content: str) -> List[bytes]:
        """표를 픽셀 단위로 세로 분할하여 여러 이미지로 생성"""
        try:
            import tempfile
            import os
            import io
            from PIL import Image

            # Chrome 옵션 (기존과 동일)
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--force-device-scale-factor=1.5")
            options.add_argument("--disable-web-security")

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            try:
                # HTML을 임시 파일로 저장
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
                    f.write(html_content)
                    temp_html_path = f.name

                driver.get(f"file://{temp_html_path}")

                # 표 요소 대기
                table_element = WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located((By.TAG_NAME, "table"))
                )

                # 표 크기 계산 (기존 로직 보존)
                table_rect = table_element.rect
                table_width = int(table_rect["width"])
                table_height = int(table_rect["height"])

                # 기존 너비 설정 보존
                min_width = 1000
                min_height = 700
                max_width = 2400
                
                final_width = max(min_width, min(table_width + 100, max_width))
                final_height = max(min_height, table_height + 300)
                
                driver.set_window_size(final_width, final_height)
                driver.execute_script("arguments[0].scrollIntoView();", table_element)

                # 전체 표 스크린샷 먼저 촬영
                full_screenshot = table_element.screenshot_as_png
                full_img = Image.open(io.BytesIO(full_screenshot))

                # 기존 너비 조정 로직 보존
                if full_img.width > 1800:
                    scale = 1800 / full_img.width
                    new_size = (1800, int(full_img.height * scale))
                    full_img = full_img.resize(new_size, Image.LANCZOS)

                # 분할 로직
                image_parts = []
                img_height = full_img.height
                
                # CROP_HEIGHT_PX 초과 시에만 분할
                if img_height <= self.CROP_HEIGHT_PX:
                    # 분할 불필요
                    img_bytes = io.BytesIO()
                    full_img.save(img_bytes, format="PNG")
                    return [img_bytes.getvalue()]

                # 분할 실행
                current_y = 0
                part_number = 0
                
                while current_y < img_height:
                    part_number += 1
                    
                    # 마지막 부분 처리
                    if current_y + self.CROP_HEIGHT_PX >= img_height:
                        # 마지막 조각은 끝까지
                        end_y = img_height
                        start_y = max(0, end_y - self.CROP_HEIGHT_PX)
                    else:
                        # 일반적인 분할
                        start_y = current_y
                        end_y = current_y + self.CROP_HEIGHT_PX
                    
                    # 이미지 자르기
                    cropped_img = full_img.crop((0, start_y, full_img.width, end_y))
                    
                    # PNG로 변환
                    img_bytes = io.BytesIO()
                    cropped_img.save(img_bytes, format="PNG")
                    image_parts.append(img_bytes.getvalue())
                    
                    # 다음 시작점 (겹침 고려)
                    current_y = end_y - self.OVERLAP_HEIGHT
                    
                    # 무한루프 방지
                    if current_y >= img_height - self.OVERLAP_HEIGHT:
                        break

                return image_parts

            finally:
                driver.quit()
                try:
                    os.unlink(temp_html_path)
                except:
                    pass

        except Exception as e:
            print(f"표 분할 실패: {e}")
            # 실패 시 기존 방식으로 폴백
            return [self._screenshot_table_html(html_content)]

    def _screenshot_table_html(self, html_content: str) -> bytes:
        """Selenium으로 표 HTML 전체 스크린샷"""
        try:
            import tempfile
            import os
            import io
            from PIL import Image

            # Chrome 옵션
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--force-device-scale-factor=1.5")  # 고해상도 렌더링
            options.add_argument("--disable-web-security")  # 로컬 파일 접근 개선

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            try:
                # HTML을 임시 파일로 저장
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
                    f.write(html_content)
                    temp_html_path = f.name

                driver.get(f"file://{temp_html_path}")

                # 표 요소 대기
                table_element = WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located((By.TAG_NAME, "table"))
                )

                # 표 크기 계산
                table_rect = table_element.rect
                table_width = int(table_rect["width"])
                table_height = int(table_rect["height"])

                # 최소 크기 보장 및 윈도우 크기 설정
                min_width = 1000   # 최소 너비 보장
                min_height = 700   # 최소 높이 보장
                max_width = 2400   # 최대 너비 증가 (1500 → 2400)
                
                final_width = max(min_width, min(table_width + 100, max_width))
                final_height = max(min_height, table_height + 300)
                
                driver.set_window_size(final_width, final_height)

                # 표 스크롤 맞추기
                driver.execute_script("arguments[0].scrollIntoView();", table_element)

                # 표 영역 스크린샷
                screenshot = table_element.screenshot_as_png

                # 필요시 크기 조정 (가로폭 1800px 맞춤으로 증가)
                img = Image.open(io.BytesIO(screenshot))
                if img.width > 1800:  # 1200 → 1800으로 증가
                    scale = 1800 / img.width
                    new_size = (1800, int(img.height * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                return img_bytes.getvalue()

            finally:
                driver.quit()
                try:
                    os.unlink(temp_html_path)
                except:
                    pass

        except Exception as e:
            print(f"스크린샷 생성 실패: {e}")
            try:
                img = Image.new("RGB", (800, 400), color="white")  # 기본 이미지 크기 증가
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                return img_bytes.getvalue()
            except:
                return b""


    def _summarize_document_if_needed(self, document_text: str) -> str:
        """문서가 너무 길면 핵심 내용만 요약"""
        if len(document_text) > 30000:  # 약 20K 토큰
            print("  📄 문서가 길어서 핵심 내용만 요약 중...")
            lines = document_text.split('\n')
            important_lines = []
            
            for line in lines:
                line = line.strip()
                if len(line) < 10:  # 너무 짧은 라인 제외
                    continue
                    
                # 제목이나 중요한 키워드가 포함된 라인만 선택
                if any(keyword in line for keyword in [
                    '제안', '목적', '개요', '요약', '결론', '사업', '프로젝트', 
                    '배경', '필요성', '목표', '범위', '내용', '방법', '계획',
                    '예산', '일정', '팀', '조직', '기대효과', '성과'
                ]):
                    important_lines.append(line)
                    
                # 최대 150줄로 제한
                if len(important_lines) >= 150:
                    break
            
            summary = '\n'.join(important_lines)
            print(f"    📄 요약 완료: {len(document_text):,}자 → {len(summary):,}자")
            return summary
        
        return document_text

    def _generate_all_table_descriptions_conversation_style(self, 
                                                           document_text: str, 
                                                           table_data_list: List[Dict]) -> List[str]:
        """대화형 방식으로 모든 표 설명 생성"""
        try:
            import base64
            from openai import OpenAI
            
            client = OpenAI()
            
            # 1. 시스템 메시지로 문서 전체 맥락 설정
            messages = [
                {
                    "role": "system",
                    "content": f"""당신은 제안서 분석 전문가입니다. 다음 제안서를 분석해주세요.

**제안서 전체 내용:**
{self._summarize_document_if_needed(document_text)}

이제 이 문서의 표들을 하나씩 보여드릴 테니, 각 표에 대해 상세한 설명을 작성해주세요.
문서 전체 맥락에서 각 표의 역할과 의미를 파악하여 설명해주세요.

각 표마다 다음 형식으로 간결하게 답변해주세요:
• 표 제목/주제: 
• 문서에서의 역할: 
• 주요 컬럼과 데이터: 
• 핵심 내용: 
• 검색 키워드: 
• 비즈니스 의미: """
                }
            ]
            
            descriptions = []
            
            # 2. 각 표를 하나씩 대화로 처리
            for i, table_data in enumerate(table_data_list):
                try:
                    print(f"  📊 표 {i+1}/{len(table_data_list)} GPT 분석 중...")
                    
                    # 현재 표에 대한 질문 추가
                    user_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"표 {i+1}/{len(table_data_list)}를 분석해주세요."
                            }
                        ]
                    }
                    
                    # 분할된 표인 경우 모든 이미지 추가, 아니면 단일 이미지
                    image_parts = table_data.get('image_parts', [])
                    for img_data in image_parts:
                        user_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                            }
                        })
                    
                    messages.append(user_message)
                    
                    # 3. GPT 응답 받기
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=800,
                        temperature=0.3
                    )
                    
                    description = response.choices[0].message.content.strip()
                    descriptions.append(description)
                    
                    # 4. GPT 응답을 대화 기록에 추가 (맥락 누적)
                    messages.append({
                        "role": "assistant", 
                        "content": description
                    })
                    
                    print(f"    ✅ 표 {i+1} 분석 완료 ({len(description)}자)")
                    
                except Exception as e:
                    print(f"    ❌ 표 {i+1} 분석 실패: {e}")
                    descriptions.append(f"표 {i+1}: 분석 실패 - {str(e)}")
            
            return descriptions
            
        except Exception as e:
            print(f"대화형 GPT 분석 전체 실패: {e}")
            return [f"표 {i+1}: 전체 분석 실패" for i in range(len(table_data_list))]

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
                    "following_context": table_data.get("following_context", ""),
                    # 분할 관련 메타데이터
                    "is_split_table": table_data.get("is_split_table", False),
                    "total_parts": table_data.get("total_parts", 1),
                    "overlap_height": table_data.get("overlap_height", 0)
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
                    gpt_description=table_data.get('gpt_description', ''),
                    
                    # 분할 관련 필드들
                    image_parts=table_data.get('image_parts'),
                    is_split_table=table_data.get('is_split_table', False),
                    total_parts=table_data.get('total_parts', 1),
                    overlap_height=table_data.get('overlap_height', 0)
                )
                
                chunks.append(chunk)
                
            except Exception as e:
                print(f"표 {i} 청크 생성 실패: {e}")
                continue
        
        return chunks