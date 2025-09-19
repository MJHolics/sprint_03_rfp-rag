"""
HWP 문서 처리기
pyhwpx를 사용한 직접 텍스트 추출 방식 구현
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

from .base import DocumentProcessor, DocumentChunk

class HWPProcessor(DocumentProcessor):
    """HWP 문서 처리기 (pyhwpx 사용)"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        self.supported_extensions = ['.hwp']

    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """HWP 파일에서 텍스트 추출 - 간단한 방법 사용"""
        print(f"HWP 파일 처리 시도: {Path(file_path).name}")

        # 1. olefile 방법 시도
        text_content = self._extract_with_olefile(file_path)

        # 2. 실패 시 바이너리 패턴 방법 시도
        if not text_content or len(text_content.strip()) < 10:
            text_content = self._extract_with_binary_pattern(file_path)

        # 3. 여전히 실패 시 기본 메시지
        if not text_content or len(text_content.strip()) < 10:
            text_content = f"HWP 파일 '{Path(file_path).name}'에서 텍스트를 추출할 수 없습니다. 파일이 암호화되어 있거나 지원되지 않는 형식일 수 있습니다."

        # 기본 메타데이터 생성
        metadata = self._create_base_metadata(file_path)
        metadata.update({
            "source_type": "hwp",
            "extraction_method": "hybrid"
        })

        return {
            "text": text_content,
            "images": [],
            "tables": [],
            "metadata": metadata
        }

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

    def chunk_content(self, content: Dict[str, Any]) -> List[DocumentChunk]:
        """HWP 내용을 의미있는 청크로 분할"""
        text = content["text"]
        metadata = content["metadata"]

        # 스마트 청킹 적용
        chunks = self._smart_chunk_text(text, metadata)

        # HWP 특화 메타데이터 추가
        for chunk in chunks:
            chunk.metadata.update({
                "source_type": "hwp",
                "extraction_method": "olefile_direct"
            })

        return chunks

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