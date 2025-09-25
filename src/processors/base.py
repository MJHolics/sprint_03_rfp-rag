"""
문서 처리를 위한 추상 기본 클래스 및 데이터 구조
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid
import hashlib
from pathlib import Path

@dataclass
class DocumentChunk:
    """문서 청크 데이터 구조"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    chunk_index: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())

@dataclass
class ProcessingResult:
    """문서 처리 결과"""
    success: bool
    chunks: List[DocumentChunk]
    total_chunks: int
    error_message: Optional[str] = None
    processing_time: float = 0.0
    extracted_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.extracted_metadata is None:
            self.extracted_metadata = {}

class DocumentProcessor(ABC):
    """문서 처리기 추상 기본 클래스"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.supported_extensions = []

    @abstractmethod
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """
        문서에서 텍스트, 이미지, 테이블 등을 추출

        Args:
            file_path: 처리할 파일 경로

        Returns:
            Dict containing:
            - text: 추출된 텍스트
            - images: 이미지 정보 리스트
            - tables: 테이블 정보 리스트
            - metadata: 문서 메타데이터
        """
        pass

    @abstractmethod
    def chunk_content(self, content: Dict[str, Any]) -> List[DocumentChunk]:
        """
        내용을 의미있는 청크로 분할

        Args:
            content: extract_content에서 반환된 내용

        Returns:
            DocumentChunk 리스트
        """
        pass

    def process_document(self, file_path: str, additional_metadata: Dict[str, Any] = None) -> ProcessingResult:
        """
        문서 처리 메인 메서드

        Args:
            file_path: 처리할 파일 경로
            additional_metadata: 추가 메타데이터

        Returns:
            ProcessingResult 객체
        """
        import time
        start_time = time.time()

        try:
            # 1. 파일 확장자 검증
            if not self._is_supported_file(file_path):
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    error_message=f"지원하지 않는 파일 형식: {Path(file_path).suffix}"
                )

            # 2. 내용 추출
            content = self.extract_content(file_path)

            # 3. 추가 메타데이터 병합
            if additional_metadata:
                content.setdefault('metadata', {}).update(additional_metadata)

            # 4. 청킹
            chunks = self.chunk_content(content)

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                chunks=chunks,
                total_chunks=len(chunks),
                processing_time=processing_time,
                extracted_metadata=content.get('metadata', {})
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                chunks=[],
                total_chunks=0,
                error_message=str(e),
                processing_time=processing_time
            )

    def _is_supported_file(self, file_path: str) -> bool:
        """파일 확장자 지원 여부 확인"""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions

    def _create_base_metadata(self, file_path: str) -> Dict[str, Any]:
        """기본 메타데이터 생성"""
        file_path = Path(file_path)
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_extension': file_path.suffix.lower(),
            'document_id': self._generate_document_id(file_path)
        }

    def _generate_document_id(self, file_path: Path) -> str:
        """파일 경로 기반으로 고유한 문서 ID 생성"""
        # 파일의 절대 경로와 이름을 조합하여 해시 생성
        file_key = f"{file_path.resolve()}_{file_path.name}_{file_path.suffix}"
        return hashlib.md5(file_key.encode('utf-8')).hexdigest()

    def _smart_chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """스마트 텍스트 청킹 (구조 인식)"""
        chunks = []

        # 섹션 기반 분할 시도
        sections = self._split_by_sections(text)

        if len(sections) > 1:
            # 섹션별로 청킹
            for i, section in enumerate(sections):
                section_chunks = self._chunk_by_size(section, metadata)
                for chunk in section_chunks:
                    chunk.metadata['section_index'] = i
                chunks.extend(section_chunks)
        else:
            # 일반 크기 기반 청킹
            chunks = self._chunk_by_size(text, metadata)

        return chunks

    def _split_by_sections(self, text: str) -> List[str]:
        """RFP 문서의 섹션별 분리"""
        import re

        # RFP 문서의 일반적인 섹션 패턴
        section_patterns = [
            r'\d+\.\s*[가-힣]+',  # "1. 사업개요"
            r'[가-힣]+\s*:\s*',   # "사업개요:"
            r'\[[가-힣]+\]',      # "[사업개요]"
            r'제\s*\d+\s*장',     # "제1장"
        ]

        sections = []
        current_section = ""

        for line in text.split('\n'):
            is_header = any(re.match(pattern, line.strip()) for pattern in section_patterns)

            if is_header and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'

        if current_section.strip():
            sections.append(current_section.strip())

        return sections if len(sections) > 1 else [text]

    def _chunk_by_size(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """크기 기반 청킹"""
        chunks = []
        text_length = len(text)

        for i in range(0, text_length, self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]

            if chunk_text.strip():  # 빈 청크 제외
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    metadata=metadata.copy(),
                    chunk_id=str(uuid.uuid4()),
                    document_id=metadata.get('document_id', str(uuid.uuid4())),
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)

        return chunks