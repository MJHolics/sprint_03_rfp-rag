"""
PDF 문서 처리기
"""
import os
from typing import Dict, List, Any
import fitz  # PyMuPDF
from .base import DocumentProcessor, DocumentChunk, ProcessingResult
from .multimodal_processor import MultimodalProcessor

class PDFProcessor(DocumentProcessor):
    """PDF 문서 처리기"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, enable_multimodal: bool = True, streaming: bool = True):
        super().__init__(chunk_size, overlap)
        self.supported_extensions = ['.pdf']
        self.multimodal_processor = MultimodalProcessor() if enable_multimodal else None
        self.streaming = streaming  # 스트리밍 모드 활성화

    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """PDF에서 텍스트, 이미지, 메타데이터 추출"""
        if self.streaming:
            return self._extract_content_streaming(file_path)
        else:
            return self._extract_content_legacy(file_path)

    def _extract_content_streaming(self, file_path: str) -> Dict[str, Any]:
        """메모리 효율적인 스트리밍 방식으로 PDF 내용 추출"""
        try:
            # 기본 컨텐츠 구조 초기화
            content = {
                "text": "",
                "images": [],
                "tables": [],
                "metadata": self._create_base_metadata(file_path)
            }

            # PDF 열기 (메모리 최적화를 위해 컨텍스트 매니저 사용)
            with self._open_pdf_safely(file_path) as doc:
                if not doc:
                    return self._create_error_content(file_path, "PDF 파일을 열 수 없습니다")

                # PDF 메타데이터 추가
                try:
                    pdf_metadata = doc.metadata or {}
                    content["metadata"].update({
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "subject": pdf_metadata.get("subject", ""),
                        "creator": pdf_metadata.get("creator", ""),
                        "total_pages": len(doc)
                    })
                except Exception:
                    content["metadata"]["total_pages"] = len(doc)

                # 페이지별 스트리밍 처리
                for page_num in range(len(doc)):
                    try:
                        # 메모리 사용량 최적화를 위해 페이지별로 처리
                        page_content = self._process_page_streaming(doc, page_num)

                        if page_content["text"].strip():
                            content["text"] += f"\n[페이지 {page_num + 1}]\n{page_content['text']}"

                        # 이미지 정보 추가
                        content["images"].extend(page_content["images"])

                        # 메모리 정리 (가비지 컬렉션 유도)
                        if page_num % 10 == 0:  # 10페이지마다 메모리 정리
                            import gc
                            gc.collect()

                    except Exception as e:
                        content["text"] += f"\n[페이지 {page_num + 1}: 처리 오류 - {str(e)}]\n"
                        continue

            # 텍스트가 없는 경우 기본 메시지
            if not content["text"].strip():
                content["text"] = f"PDF 파일 '{os.path.basename(file_path)}'에서 텍스트를 추출할 수 없습니다."

            # 멀티모달 처리 (필요시)
            if self.multimodal_processor:
                try:
                    print(f"멀티모달 분석 시작: {os.path.basename(file_path)}")
                    content = self.multimodal_processor.enhance_content_with_images(content, file_path)
                    print(f"멀티모달 분석 완료: {content.get('total_analyzed_images', 0)}개 이미지 분석")
                except Exception as e:
                    print(f"멀티모달 분석 실패: {e}")

            return content

        except Exception as e:
            return self._create_error_content(file_path, str(e))

    def _process_page_streaming(self, doc, page_num: int) -> Dict[str, Any]:
        """단일 페이지를 메모리 효율적으로 처리"""
        page_content = {
            "text": "",
            "images": []
        }

        try:
            # 페이지 객체 가져오기 (with 구문으로 메모리 관리)
            page = doc[page_num]

            # 텍스트 추출 (stderr 억제)
            with self._suppress_stderr():
                page_text = page.get_text()
                if page_text.strip():
                    page_content["text"] = page_text

            # 이미지 정보 추출 (메모리 효율적)
            try:
                with self._suppress_stderr():
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        if len(img) >= 4:
                            page_content["images"].append({
                                "page": page_num + 1,
                                "index": img_index,
                                "xref": img[0],
                                "width": img[2] if len(img) > 2 else 0,
                                "height": img[3] if len(img) > 3 else 0
                            })
            except Exception:
                # 이미지 추출 오류는 무시
                pass

        except Exception:
            # 페이지 처리 오류
            page_content["text"] = f"페이지 {page_num + 1} 처리 오류"

        return page_content

    def _open_pdf_safely(self, file_path: str):
        """PDF를 안전하게 열기 위한 컨텍스트 매니저"""
        class PDFContextManager:
            def __init__(self, file_path):
                self.file_path = file_path
                self.doc = None

            def __enter__(self):
                try:
                    # stderr 억제하면서 PDF 열기
                    with self._suppress_stderr():
                        self.doc = fitz.open(self.file_path)
                    return self.doc
                except Exception:
                    return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.doc:
                    try:
                        self.doc.close()
                    except Exception:
                        pass

            def _suppress_stderr(self):
                """stderr 억제를 위한 컨텍스트 매니저"""
                class StderrSuppressor:
                    def __enter__(self):
                        import sys
                        import os
                        self.stderr_backup = sys.stderr
                        try:
                            if hasattr(os, 'devnull'):
                                self.devnull = open(os.devnull, 'w')
                                sys.stderr = self.devnull
                            else:
                                from io import StringIO
                                sys.stderr = StringIO()
                        except:
                            pass
                        return self

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        import sys
                        sys.stderr = self.stderr_backup
                        if hasattr(self, 'devnull'):
                            try:
                                self.devnull.close()
                            except:
                                pass

                return StderrSuppressor()

        return PDFContextManager(file_path)

    def _suppress_stderr(self):
        """stderr 억제를 위한 컨텍스트 매니저"""
        class StderrSuppressor:
            def __enter__(self):
                import sys
                import os
                self.stderr_backup = sys.stderr
                try:
                    if hasattr(os, 'devnull'):
                        self.devnull = open(os.devnull, 'w')
                        sys.stderr = self.devnull
                    else:
                        from io import StringIO
                        sys.stderr = StringIO()
                except:
                    pass
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                import sys
                sys.stderr = self.stderr_backup
                if hasattr(self, 'devnull'):
                    try:
                        self.devnull.close()
                    except:
                        pass

        return StderrSuppressor()

    def _create_error_content(self, file_path: str, error_message: str) -> Dict[str, Any]:
        """오류 발생 시 기본 컨텐츠 생성"""
        return {
            "text": f"PDF 파일 처리 실패: {error_message}",
            "images": [],
            "tables": [],
            "metadata": {
                **self._create_base_metadata(file_path),
                "processing_error": error_message
            }
        }

    def _extract_content_legacy(self, file_path: str) -> Dict[str, Any]:
        """기존 방식의 PDF 내용 추출 (호환성 유지)"""
        try:
            # MuPDF 오류 억제를 위한 설정
            import warnings
            import sys
            import os
            from contextlib import redirect_stderr
            from io import StringIO

            warnings.filterwarnings("ignore")

            # stderr을 완전히 억제
            stderr_backup = sys.stderr
            devnull = open(os.devnull, 'w') if hasattr(os, 'devnull') else StringIO()

            try:
                # stderr 완전 리다이렉트
                sys.stderr = devnull

                # MuPDF 로깅 레벨 설정 (가능한 경우)
                try:
                    fitz.TOOLS.mupdf_warnings(False)
                except:
                    pass

                doc = fitz.open(file_path)
            finally:
                sys.stderr = stderr_backup
                if hasattr(devnull, 'close'):
                    try:
                        devnull.close()
                    except:
                        pass

            content = {
                "text": "",
                "images": [],
                "tables": [],
                "metadata": self._create_base_metadata(file_path)
            }

            # PDF 메타데이터 추가 (안전하게)
            try:
                pdf_metadata = doc.metadata or {}
                content["metadata"].update({
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "creator": pdf_metadata.get("creator", ""),
                    "total_pages": len(doc)
                })
            except Exception:
                content["metadata"]["total_pages"] = len(doc)

            # 페이지별 텍스트 추출 (안전하게)
            for page_num, page in enumerate(doc):
                try:
                    # stderr을 완전히 억제하면서 처리
                    page_devnull = open(os.devnull, 'w') if hasattr(os, 'devnull') else StringIO()

                    try:
                        sys.stderr = page_devnull
                        page_text = page.get_text()
                        if page_text.strip():
                            content["text"] += f"\n[페이지 {page_num + 1}]\n{page_text}"
                    finally:
                        sys.stderr = stderr_backup
                        if hasattr(page_devnull, 'close'):
                            try:
                                page_devnull.close()
                            except:
                                pass

                    # 이미지 정보 추출 (필요시) - 오류 발생 시 무시
                    try:
                        img_devnull = open(os.devnull, 'w') if hasattr(os, 'devnull') else StringIO()
                        try:
                            sys.stderr = img_devnull
                            images = page.get_images(full=True)
                            for img_index, img in enumerate(images):
                                if len(img) >= 4:
                                    content["images"].append({
                                        "page": page_num + 1,
                                        "index": img_index,
                                        "xref": img[0],
                                        "width": img[2] if len(img) > 2 else 0,
                                        "height": img[3] if len(img) > 3 else 0
                                    })
                        finally:
                            sys.stderr = stderr_backup
                            if hasattr(img_devnull, 'close'):
                                try:
                                    img_devnull.close()
                                except:
                                    pass
                    except Exception:
                        # 이미지 추출 오류는 무시
                        pass

                except Exception as e:
                    # 페이지 처리 오류 시 해당 페이지만 건너뛰기
                    content["text"] += f"\n[페이지 {page_num + 1}: 처리 오류]\n"
                    continue

            doc.close()

            # 텍스트가 없는 경우 기본 메시지
            if not content["text"].strip():
                content["text"] = f"PDF 파일 '{os.path.basename(file_path)}'에서 텍스트를 추출할 수 없습니다."

            # 멀티모달 처리 (이미지 분석 및 텍스트 강화)
            if self.multimodal_processor:
                try:
                    print(f"멀티모달 분석 시작: {os.path.basename(file_path)}")
                    content = self.multimodal_processor.enhance_content_with_images(content, file_path)
                    print(f"멀티모달 분석 완료: {content.get('total_analyzed_images', 0)}개 이미지 분석")
                except Exception as e:
                    print(f"멀티모달 분석 실패: {e}")

            return content

        except Exception as e:
            # 전체 처리 실패 시
            return {
                "text": f"PDF 파일 처리 실패: {str(e)}",
                "images": [],
                "tables": [],
                "metadata": {
                    **self._create_base_metadata(file_path),
                    "processing_error": str(e)
                }
            }

    def chunk_content(self, content: Dict[str, Any]) -> List[DocumentChunk]:
        """PDF 내용을 의미있는 청크로 분할"""
        text = content["text"]
        metadata = content["metadata"]

        # 스마트 청킹 적용
        chunks = self._smart_chunk_text(text, metadata)

        # PDF 특화 메타데이터 추가
        for chunk in chunks:
            chunk.metadata.update({
                "total_images": len(content["images"]),
                "source_type": "pdf"
            })

        return chunks

    def _extract_rfp_metadata(self, text: str) -> Dict[str, Any]:
        """RFP 문서에서 특화 메타데이터 추출"""
        import re

        extracted = {}

        # 예산 정보 추출
        budget_patterns = [
            r'총\s*사업비\s*[:\s]*([0-9,]+)\s*(원|백만원|억원)',
            r'예산\s*[:\s]*([0-9,]+)\s*(원|백만원|억원)',
            r'금액\s*[:\s]*([0-9,]+)\s*(원|백만원|억원)'
        ]

        for pattern in budget_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['budget'] = match.group(0)
                break

        # 사업명 추출
        title_patterns = [
            r'사업명\s*[:\s]*([^\n]+)',
            r'과업명\s*[:\s]*([^\n]+)',
            r'프로젝트명\s*[:\s]*([^\n]+)'
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['business_name'] = match.group(1).strip()
                break

        # 발주기관 추출
        agency_patterns = [
            r'발주기관\s*[:\s]*([^\n]+)',
            r'주관기관\s*[:\s]*([^\n]+)',
            r'발주처\s*[:\s]*([^\n]+)'
        ]

        for pattern in agency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['agency'] = match.group(1).strip()
                break

        # 마감일 추출
        deadline_patterns = [
            r'마감일\s*[:\s]*(\d{4}[.-]\d{1,2}[.-]\d{1,2})',
            r'제출기한\s*[:\s]*(\d{4}[.-]\d{1,2}[.-]\d{1,2})',
            r'접수마감\s*[:\s]*(\d{4}[.-]\d{1,2}[.-]\d{1,2})'
        ]

        for pattern in deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['deadline'] = match.group(1)
                break

        return extracted