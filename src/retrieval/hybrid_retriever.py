"""
하이브리드 검색 시스템 - Vector + BM25
"""
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import Counter

from ..storage.vector_store import VectorStore
from ..storage.metadata_store import MetadataStore
from ..processors.base import DocumentChunk

class BM25:
    """BM25 키워드 검색 구현"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = []
        self.idf_cache = {}

    def fit(self, documents: List[str]):
        """BM25 인덱스 구축"""
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # 각 문서의 단어 빈도 계산
        self.doc_freqs = []
        for doc in documents:
            words = doc.lower().split()
            word_freq = Counter(words)
            self.doc_freqs.append(word_freq)

        # IDF 계산을 위한 문서 빈도 계산
        self._calculate_idf()

    def _calculate_idf(self):
        """IDF (Inverse Document Frequency) 계산"""
        all_words = set()
        for doc_freq in self.doc_freqs:
            all_words.update(doc_freq.keys())

        N = len(self.documents)
        for word in all_words:
            df = sum(1 for doc_freq in self.doc_freqs if word in doc_freq)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[word] = idf

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 점수로 문서 검색"""
        if not self.documents:
            return []

        query_words = query.lower().split()
        scores = []

        for doc_idx, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]

            for word in query_words:
                if word in doc_freq:
                    tf = doc_freq[word]
                    idf = self.idf_cache.get(word, 0)

                    # BM25 공식
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                    score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # 점수 기준 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class HybridRetriever:
    """Vector + BM25 하이브리드 검색기"""

    def __init__(self, vector_store: VectorStore, metadata_store: MetadataStore):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.bm25 = BM25()
        self.document_chunks = []

    def build_bm25_index(self, chunks: List[DocumentChunk]):
        """BM25 인덱스 구축"""
        self.document_chunks = chunks
        documents = [chunk.content for chunk in chunks]
        self.bm25.fit(documents)
        print(f"BM25 인덱스 구축 완료: {len(documents)}개 문서")

    def hybrid_search(self, query: str, k: int = 10,
                     vector_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            vector_weight: 벡터 검색 가중치
            keyword_weight: 키워드 검색 가중치
            filters: 메타데이터 필터
        """
        try:
            # 1. Vector 검색
            vector_results = self.vector_store.similarity_search(
                query, k=k*2, filters=filters  # 더 많이 가져와서 다양성 확보
            )

            # 2. BM25 키워드 검색
            keyword_results = []
            if self.document_chunks:
                bm25_scores = self.bm25.search(query, top_k=k*2)
                for idx, score in bm25_scores:
                    if idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        keyword_results.append({
                            'content': chunk.content,
                            'metadata': chunk.metadata,
                            'id': chunk.chunk_id,
                            'bm25_score': score
                        })

            # 3. 결과 융합
            combined_results = self._combine_results(
                vector_results, keyword_results,
                vector_weight, keyword_weight
            )

            # 4. 상위 k개 선택
            final_results = combined_results[:k]

            # 5. 검색 로그 기록
            self.metadata_store.log_search(
                query=query,
                search_method="hybrid",
                results_count=len(final_results)
            )

            return final_results

        except Exception as e:
            print(f"하이브리드 검색 오류: {e}")
            return []

    def _combine_results(self, vector_results: List[Dict], keyword_results: List[Dict],
                        vector_weight: float, keyword_weight: float) -> List[Dict[str, Any]]:
        """Vector와 키워드 검색 결과 융합"""

        # 정규화를 위한 최대 점수 계산
        max_vector_score = max([1 - r.get('distance', 0) for r in vector_results], default=1.0)
        max_keyword_score = max([r.get('bm25_score', 0) for r in keyword_results], default=1.0)

        # 결과 통합 딕셔너리
        combined = {}

        # Vector 결과 처리
        for result in vector_results:
            doc_id = result['id']
            # 거리를 유사도로 변환 (1 - distance)
            vector_similarity = 1 - result.get('distance', 0)
            normalized_vector_score = vector_similarity / max_vector_score if max_vector_score > 0 else 0

            combined[doc_id] = {
                'content': result['content'],
                'metadata': result['metadata'],
                'id': doc_id,
                'vector_score': normalized_vector_score,
                'keyword_score': 0.0,
                'hybrid_score': vector_weight * normalized_vector_score
            }

        # 키워드 결과 처리
        for result in keyword_results:
            doc_id = result['id']
            normalized_keyword_score = result['bm25_score'] / max_keyword_score if max_keyword_score > 0 else 0

            if doc_id in combined:
                # 이미 있는 문서는 키워드 점수 추가
                combined[doc_id]['keyword_score'] = normalized_keyword_score
                combined[doc_id]['hybrid_score'] = (
                    vector_weight * combined[doc_id]['vector_score'] +
                    keyword_weight * normalized_keyword_score
                )
            else:
                # 새 문서는 키워드 점수만으로 추가
                combined[doc_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'id': doc_id,
                    'vector_score': 0.0,
                    'keyword_score': normalized_keyword_score,
                    'hybrid_score': keyword_weight * normalized_keyword_score
                }

        # 하이브리드 점수로 정렬
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        return sorted_results

    def vector_only_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Vector 검색만 수행"""
        results = self.vector_store.similarity_search(query, k=k, filters=filters)

        # 검색 로그 기록
        self.metadata_store.log_search(
            query=query,
            search_method="vector_only",
            results_count=len(results)
        )

        return results

    def keyword_only_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """키워드 검색만 수행"""
        if not self.document_chunks:
            return []

        bm25_scores = self.bm25.search(query, top_k=k)
        results = []

        for idx, score in bm25_scores:
            if idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                results.append({
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'id': chunk.chunk_id,
                    'bm25_score': score
                })

        # 검색 로그 기록
        self.metadata_store.log_search(
            query=query,
            search_method="keyword_only",
            results_count=len(results)
        )

        return results

    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        # 여기서는 기본 통계만 반환, 실제로는 metadata_store에서 조회
        return {
            'total_indexed_chunks': len(self.document_chunks),
            'vector_store_stats': self.vector_store.get_collection_stats(),
            'bm25_indexed': len(self.document_chunks) > 0
        }