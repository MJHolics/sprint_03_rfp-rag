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

            # 3. 하이브리드 점수 계산 (RRF - Reciprocal Rank Fusion 방식)
            final_results = []

            # 벡터 결과를 순위 기반으로 점수화
            vector_scores = {}
            for rank, result in enumerate(vector_results):
                vector_scores[result['id']] = {
                    'result': result,
                    'rank_score': 1.0 / (rank + 1),  # 순위 기반 점수
                    'distance_score': max(0.0, min(1.0, 1.0 / (1.0 + result.get('distance', 1.0))))
                }

            # 키워드 결과를 순위 기반으로 점수화
            keyword_scores = {}
            for rank, result in enumerate(keyword_results):
                keyword_scores[result['id']] = {
                    'result': result,
                    'rank_score': 1.0 / (rank + 1),  # 순위 기반 점수
                    'bm25_score': max(0.0, min(1.0, result.get('bm25_score', 0) / 15.0))
                }

            # 모든 ID 수집
            all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

            # 각 ID에 대해 하이브리드 점수 계산
            for doc_id in all_ids:
                v_data = vector_scores.get(doc_id)
                k_data = keyword_scores.get(doc_id)

                # 기본값 설정
                vector_score = 0.0
                keyword_score = 0.0
                content = ""
                metadata = {}

                if v_data:
                    vector_score = v_data['distance_score']
                    content = v_data['result']['content']
                    metadata = v_data['result']['metadata']

                if k_data:
                    keyword_score = k_data['bm25_score']
                    if not content:  # 벡터 결과가 없는 경우만
                        content = k_data['result']['content']
                        metadata = k_data['result']['metadata']

                # 하이브리드 점수 = 가중 평균 + 순위 보너스
                hybrid_score = (vector_weight * vector_score + keyword_weight * keyword_score)

                # 두 검색에서 모두 발견된 경우 보너스 점수
                if v_data and k_data:
                    hybrid_score *= 1.2  # 20% 보너스

                source_type = 'hybrid' if (v_data and k_data) else ('vector' if v_data else 'keyword')

                final_results.append({
                    'content': content,
                    'metadata': metadata,
                    'id': doc_id,
                    'vector_score': vector_score,
                    'keyword_score': keyword_score,
                    'hybrid_score': hybrid_score,
                    'source': source_type
                })

            # 하이브리드 점수로 정렬하고 상위 k개 선택
            final_results = sorted(final_results, key=lambda x: x['hybrid_score'], reverse=True)[:k]

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
        """Vector와 키워드 검색 결과 융합 - 독립 점수 기반"""

        all_results = []

        # Vector 결과 처리 - 거리 기반 점수
        for i, result in enumerate(vector_results):
            distance = result.get('distance', 1.0)
            # 거리를 0-1 점수로 변환: 낮은 거리 = 높은 점수
            vector_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
            # 순위 기반 보너스 (상위 결과일수록 높은 점수)
            rank_bonus = max(0.0, (10 - i) / 10.0) if i < 10 else 0.0
            final_vector_score = (vector_score + rank_bonus * 0.1) / 1.1  # 정규화

            all_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'id': result['id'],
                'vector_score': final_vector_score,
                'keyword_score': 0.0,
                'hybrid_score': vector_weight * final_vector_score,
                'source': 'vector',
                'original_rank': i + 1
            })

        # 키워드 결과 처리 - BM25 점수
        for i, result in enumerate(keyword_results):
            bm25_score = result.get('bm25_score', 0)
            # BM25 점수를 0-1로 정규화 (일반적으로 0-15 범위)
            keyword_score = max(0.0, min(1.0, bm25_score / 15.0))
            # 순위 기반 보너스
            rank_bonus = max(0.0, (10 - i) / 10.0) if i < 10 else 0.0
            final_keyword_score = (keyword_score + rank_bonus * 0.1) / 1.1  # 정규화

            all_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'id': result['id'],
                'vector_score': 0.0,
                'keyword_score': final_keyword_score,
                'hybrid_score': keyword_weight * final_keyword_score,
                'source': 'keyword',
                'original_rank': i + 1
            })

        # 하이브리드 점수로 정렬
        sorted_results = sorted(
            all_results,
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        return sorted_results

    def vector_only_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Vector 검색만 수행"""
        try:
            results = self.vector_store.similarity_search(query, k=k, filters=filters)

            # 점수 정규화 및 추가
            processed_results = []
            for result in results:
                # 거리를 유사도로 변환 (Chroma에서 반환되는 distance는 보통 0-2 사이)
                distance = result.get('distance', 1.0)
                # 더 정확한 유사도 변환: exp(-distance) 또는 1/(1+distance)
                vector_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))

                processed_result = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'id': result['id'],
                    'vector_score': vector_score,
                    'distance': distance
                }
                processed_results.append(processed_result)

            # 검색 로그 기록
            self.metadata_store.log_search(
                query=query,
                search_method="vector_only",
                results_count=len(processed_results)
            )

            return processed_results

        except Exception as e:
            print(f"Vector 검색 오류: {e}")
            return []

    def keyword_only_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """키워드 검색만 수행 - 벡터 스토어에서 전체 컨텐츠 가져오기"""
        try:
            # BM25 인덱스가 비어있으면 메타데이터에서 청크를 가져와서 구축
            if not self.document_chunks:
                self._rebuild_bm25_from_metadata()

            if not self.document_chunks:
                print("키워드 검색용 문서 청크가 없습니다.")
                return []

            bm25_scores = self.bm25.search(query, top_k=k)
            results = []

            for idx, score in bm25_scores:
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    # BM25 점수 정규화 (일반적으로 0-10 사이 값)
                    normalized_score = max(0.0, min(1.0, score / 10.0))

                    # 벡터 스토어에서 전체 컨텐츠와 메타데이터 가져오기
                    full_content = chunk.content
                    enhanced_metadata = chunk.metadata

                    try:
                        # 유사한 내용으로 벡터 검색해서 완전한 정보 가져오기
                        content_query = chunk.content[:50] if chunk.content else "시스템 구축"
                        vector_results = self.vector_store.similarity_search(
                            content_query, k=3  # 상위 3개 검색
                        )

                        # 가장 유사한 결과의 메타데이터와 컨텐츠 사용
                        if vector_results:
                            best_match = vector_results[0]  # 가장 유사한 결과
                            if len(best_match['content']) > len(chunk.content):
                                full_content = best_match['content']
                                enhanced_metadata = best_match['metadata']

                    except Exception as e:
                        pass  # 실패하면 기존 정보 사용

                    results.append({
                        'content': full_content,
                        'metadata': enhanced_metadata,
                        'id': chunk.chunk_id,
                        'bm25_score': score,
                        'keyword_score': normalized_score
                    })

            # 검색 로그 기록
            self.metadata_store.log_search(
                query=query,
                search_method="keyword_only",
                results_count=len(results)
            )

            return results

        except Exception as e:
            print(f"키워드 검색 오류: {e}")
            return []

    def _rebuild_bm25_from_metadata(self):
        """메타데이터 스토어에서 청크를 가져와서 BM25 인덱스 재구축"""
        try:
            print("BM25 인덱스 재구축 중...")

            # 메타데이터 스토어에서 모든 청크 가져오기
            all_chunks = self.metadata_store.get_all_chunks()

            if all_chunks:
                # DocumentChunk 객체로 변환
                from ..processors.base import DocumentChunk
                self.document_chunks = []

                for chunk_data in all_chunks:
                    chunk = DocumentChunk(
                        content=chunk_data.get('content', ''),
                        chunk_id=chunk_data.get('chunk_id', ''),
                        document_id=chunk_data.get('document_id', ''),
                        chunk_index=chunk_data.get('chunk_index', 0),
                        metadata=chunk_data.get('metadata', {})
                    )
                    self.document_chunks.append(chunk)

                # BM25 인덱스 구축
                documents = [chunk.content for chunk in self.document_chunks]
                self.bm25.fit(documents)
                print(f"BM25 인덱스 재구축 완료: {len(documents)}개 문서")
            else:
                print("재구축할 청크 데이터가 없습니다.")

        except Exception as e:
            print(f"BM25 인덱스 재구축 오류: {e}")

    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        # 여기서는 기본 통계만 반환, 실제로는 metadata_store에서 조회
        return {
            'total_indexed_chunks': len(self.document_chunks),
            'vector_store_stats': self.vector_store.get_collection_stats(),
            'bm25_indexed': len(self.document_chunks) > 0
        }