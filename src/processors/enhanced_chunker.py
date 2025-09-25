"""
향상된 청킹 시스템
폰트 크기, 제목 구조, 의미적 경계를 고려한 지능형 청킹
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .base import DocumentChunk
from .cnn_metadata_analyzer import StructuralElement, DocumentStructure

@dataclass
class EnhancedChunk:
    """향상된 청크 클래스"""
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any]

    # 추가 구조 정보
    chunk_type: str  # 'title', 'content', 'table', 'list', 'mixed'
    semantic_level: int  # 의미적 깊이 (1=최상위, 2=하위 등)
    parent_section: str  # 상위 섹션 제목
    keywords: List[str]  # 주요 키워드
    importance_score: float  # 중요도 점수

class EnhancedChunker:
    """향상된 청킹 시스템"""

    def __init__(self, base_chunk_size: int = 1000, overlap: int = 200):
        self.base_chunk_size = base_chunk_size
        self.overlap = overlap

        # 청킹 규칙 정의
        self.chunk_boundaries = {
            'strong': [  # 강한 경계 (여기서 반드시 분할)
                r'^\d+\.\s+[가-힣A-Za-z]+',  # 1. 제목
                r'^제\s*\d+\s*장',  # 제1장
                r'^[가-힣]+\s*사업\s*개요',  # 사업 개요
                r'^[가-힣]+\s*기술\s*요구사항',  # 기술 요구사항
            ],
            'medium': [  # 중간 경계 (적절한 위치에서 분할)
                r'^\s*\([가-힣]\)',  # (가)
                r'^\s*[가-힣]\.',  # 가.
                r'^\s*\d+\)',  # 1)
            ],
            'weak': [  # 약한 경계 (필요시에만 분할)
                r'^\s*-\s+',  # - 항목
                r'^\s*•\s+',  # • 항목
                r'\n\n',  # 빈 줄
            ]
        }

    def create_enhanced_chunks(self, document_structure: DocumentStructure,
                             document_id: str) -> List[EnhancedChunk]:
        """문서 구조를 기반으로 향상된 청킹 수행"""

        chunks = []
        current_section = ""
        semantic_level = 1

        # 1. 구조적 요소 그룹화
        grouped_elements = self._group_elements_by_structure(document_structure.elements)

        # 2. 각 그룹별로 청킹
        for group_idx, element_group in enumerate(grouped_elements):
            group_chunks = self._chunk_element_group(
                element_group, document_id, group_idx, current_section, semantic_level
            )
            chunks.extend(group_chunks)

            # 섹션 업데이트
            if element_group and element_group[0].element_type == 'title':
                current_section = element_group[0].text_content
                semantic_level = self._calculate_semantic_level(element_group[0])

        # 3. 청크 품질 검증 및 조정
        chunks = self._validate_and_adjust_chunks(chunks)

        return chunks

    def _group_elements_by_structure(self, elements: List[StructuralElement]) -> List[List[StructuralElement]]:
        """구조적 요소들을 그룹화"""
        groups = []
        current_group = []

        for element in elements:
            if element.element_type == 'title' and current_group:
                # 새로운 제목이 나오면 이전 그룹 완료
                groups.append(current_group)
                current_group = [element]
            else:
                current_group.append(element)

        if current_group:
            groups.append(current_group)

        return groups

    def _chunk_element_group(self, elements: List[StructuralElement], document_id: str,
                           group_idx: int, parent_section: str, semantic_level: int) -> List[EnhancedChunk]:
        """요소 그룹을 청킹"""

        chunks = []

        # 그룹 내 텍스트 결합
        combined_text = '\n'.join([elem.text_content for elem in elements])

        # 청크 타입 결정
        chunk_type = self._determine_chunk_type(elements)

        # 텍스트 길이에 따른 분할 전략
        if len(combined_text) <= self.base_chunk_size:
            # 단일 청크로 처리
            chunk = self._create_single_chunk(
                combined_text, elements, document_id, group_idx,
                parent_section, semantic_level, chunk_type
            )
            chunks.append(chunk)
        else:
            # 다중 청킹 필요
            sub_chunks = self._create_multiple_chunks(
                combined_text, elements, document_id, group_idx,
                parent_section, semantic_level, chunk_type
            )
            chunks.extend(sub_chunks)

        return chunks

    def _determine_chunk_type(self, elements: List[StructuralElement]) -> str:
        """청크 타입 결정"""
        if not elements:
            return 'content'

        types = [elem.element_type for elem in elements]

        if 'title' in types and len(types) == 1:
            return 'title'
        elif 'table' in types:
            return 'table'
        elif all(t == 'list' for t in types):
            return 'list'
        elif len(set(types)) > 2:
            return 'mixed'
        else:
            return 'content'

    def _create_single_chunk(self, text: str, elements: List[StructuralElement],
                           document_id: str, chunk_idx: int, parent_section: str,
                           semantic_level: int, chunk_type: str) -> EnhancedChunk:
        """단일 청크 생성"""

        chunk_id = f"{document_id}_{chunk_idx:04d}"
        keywords = self._extract_keywords(text, elements)
        importance_score = self._calculate_importance_score(elements, text)

        metadata = {
            'document_id': document_id,
            'chunk_type': chunk_type,
            'semantic_level': semantic_level,
            'parent_section': parent_section,
            'element_count': len(elements),
            'avg_font_size': np.mean([elem.font_size for elem in elements]),
            'has_bold': any(elem.font_weight == 'bold' for elem in elements),
            'page_numbers': list(set([elem.page_number for elem in elements])),
            'keywords': keywords,
            'importance_score': importance_score
        }

        return EnhancedChunk(
            content=text,
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_idx,
            metadata=metadata,
            chunk_type=chunk_type,
            semantic_level=semantic_level,
            parent_section=parent_section,
            keywords=keywords,
            importance_score=importance_score
        )

    def _create_multiple_chunks(self, text: str, elements: List[StructuralElement],
                              document_id: str, base_idx: int, parent_section: str,
                              semantic_level: int, chunk_type: str) -> List[EnhancedChunk]:
        """긴 텍스트를 여러 청크로 분할"""

        chunks = []

        # 지능형 분할점 찾기
        split_points = self._find_optimal_split_points(text, elements)

        if not split_points:
            # 분할점이 없으면 강제 분할
            split_points = self._force_split(text)

        # 분할점 기준으로 청킹
        start = 0
        for i, split_point in enumerate(split_points + [len(text)]):
            chunk_text = text[start:split_point].strip()

            if len(chunk_text) < 50:  # 너무 짧은 청크는 이전과 병합
                if chunks:
                    chunks[-1].content += '\n' + chunk_text
                continue

            # 오버랩 추가 (이전 청크가 있는 경우)
            if chunks and self.overlap > 0:
                overlap_text = text[max(0, start-self.overlap):start]
                chunk_text = overlap_text + chunk_text

            chunk_id = f"{document_id}_{base_idx:04d}_{i:02d}"
            keywords = self._extract_keywords(chunk_text, elements)
            importance_score = self._calculate_importance_score_for_text(chunk_text)

            metadata = {
                'document_id': document_id,
                'chunk_type': chunk_type,
                'semantic_level': semantic_level,
                'parent_section': parent_section,
                'sub_chunk_index': i,
                'is_continuation': i > 0,
                'keywords': keywords,
                'importance_score': importance_score
            }

            chunk = EnhancedChunk(
                content=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=base_idx * 100 + i,  # 고유한 인덱스
                metadata=metadata,
                chunk_type=chunk_type,
                semantic_level=semantic_level,
                parent_section=parent_section,
                keywords=keywords,
                importance_score=importance_score
            )
            chunks.append(chunk)
            start = split_point

        return chunks

    def _find_optimal_split_points(self, text: str, elements: List[StructuralElement]) -> List[int]:
        """최적의 분할점 찾기"""
        split_points = []
        lines = text.split('\n')
        current_pos = 0
        current_chunk_size = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            # 청크 크기 한계 도달 확인
            if current_chunk_size + line_length > self.base_chunk_size:
                # 적절한 분할점인지 확인
                if self._is_good_split_point(line):
                    split_points.append(current_pos)
                    current_chunk_size = 0

            current_pos += line_length
            current_chunk_size += line_length

        return split_points

    def _is_good_split_point(self, line: str) -> bool:
        """좋은 분할점인지 판단"""
        line = line.strip()

        # 강한 경계 확인
        for pattern in self.chunk_boundaries['strong']:
            if re.match(pattern, line):
                return True

        # 중간 경계 확인
        for pattern in self.chunk_boundaries['medium']:
            if re.match(pattern, line):
                return True

        return False

    def _force_split(self, text: str) -> List[int]:
        """강제 분할 (문장 단위 우선)"""
        sentences = re.split(r'[.!?]\s+', text)
        split_points = []
        current_pos = 0
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.base_chunk_size:
                split_points.append(current_pos)
                current_size = 0
            current_pos += sentence_size
            current_size += sentence_size

        return split_points

    def _calculate_semantic_level(self, element: StructuralElement) -> int:
        """의미적 깊이 계산"""
        text = element.text_content

        # 숫자 기반 레벨 감지 (1., 1.1., 1.1.1. 등)
        match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if match:
            return len(match.group(1).split('.'))

        # 한글 번호 기반 (가., 나., 다. 등)
        if re.match(r'^[가-힣]\. ', text):
            return 2

        # 기본값
        return 1

    def _extract_keywords(self, text: str, elements: List[StructuralElement] = None) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (향후 NLP 모델로 개선)
        import re

        # 한글 단어 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', text)

        # 영어 단어 추출 (3글자 이상)
        english_words = re.findall(r'[A-Za-z]{3,}', text)

        # 빈도수 기반 상위 키워드 선택
        from collections import Counter
        all_words = korean_words + english_words
        word_counts = Counter(all_words)

        # 상위 10개 키워드
        keywords = [word for word, count in word_counts.most_common(10)]
        return keywords

    def _calculate_importance_score(self, elements: List[StructuralElement], text: str) -> float:
        """중요도 점수 계산"""
        score = 0.5  # 기본 점수

        # 요소 타입 기반 점수
        for element in elements:
            if element.element_type == 'title':
                score += 0.3
            elif element.element_type == 'subtitle':
                score += 0.2
            elif element.font_weight == 'bold':
                score += 0.1

        # 텍스트 길이 기반 점수
        if len(text) > 500:
            score += 0.1

        # 키워드 밀도 기반 점수
        important_keywords = ['시스템', '구축', '개발', '요구사항', '기술', '예산']
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        score += keyword_count * 0.05

        return min(score, 1.0)  # 최대값 1.0 제한

    def _calculate_importance_score_for_text(self, text: str) -> float:
        """텍스트만으로 중요도 점수 계산"""
        score = 0.5  # 기본 점수

        # 텍스트 길이 기반
        if len(text) > 500:
            score += 0.1

        # 키워드 기반
        important_keywords = ['시스템', '구축', '개발', '요구사항', '기술', '예산']
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        score += keyword_count * 0.05

        return min(score, 1.0)

    def _validate_and_adjust_chunks(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """청크 품질 검증 및 조정"""
        validated_chunks = []

        for chunk in chunks:
            # 너무 짧은 청크 처리
            if len(chunk.content.strip()) < 50:
                if validated_chunks:
                    # 이전 청크와 병합
                    validated_chunks[-1].content += '\n' + chunk.content
                    continue

            # 너무 긴 청크 처리 (강제 분할)
            if len(chunk.content) > self.base_chunk_size * 1.5:
                sub_chunks = self._emergency_split_chunk(chunk)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def _emergency_split_chunk(self, chunk: EnhancedChunk) -> List[EnhancedChunk]:
        """응급 청크 분할"""
        text = chunk.content
        target_size = self.base_chunk_size

        # 문장 단위로 분할
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sub_chunks = []
        current_text = ""
        sub_idx = 0

        for sentence in sentences:
            if len(current_text) + len(sentence) > target_size and current_text:
                # 현재 텍스트를 청크로 생성
                sub_chunk = EnhancedChunk(
                    content=current_text.strip(),
                    chunk_id=f"{chunk.chunk_id}_emergency_{sub_idx}",
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index * 1000 + sub_idx,
                    metadata=chunk.metadata.copy(),
                    chunk_type=chunk.chunk_type,
                    semantic_level=chunk.semantic_level,
                    parent_section=chunk.parent_section,
                    keywords=self._extract_keywords(current_text),
                    importance_score=self._calculate_importance_score_for_text(current_text)
                )
                sub_chunks.append(sub_chunk)
                current_text = sentence
                sub_idx += 1
            else:
                current_text += " " + sentence

        # 마지막 청크 추가
        if current_text.strip():
            sub_chunk = EnhancedChunk(
                content=current_text.strip(),
                chunk_id=f"{chunk.chunk_id}_emergency_{sub_idx}",
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index * 1000 + sub_idx,
                metadata=chunk.metadata.copy(),
                chunk_type=chunk.chunk_type,
                semantic_level=chunk.semantic_level,
                parent_section=chunk.parent_section,
                keywords=self._extract_keywords(current_text),
                importance_score=self._calculate_importance_score_for_text(current_text)
            )
            sub_chunks.append(sub_chunk)

        return sub_chunks