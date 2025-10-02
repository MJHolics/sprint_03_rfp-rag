"""
RAG 시스템 3단계 - 비동기 + 고급 캐싱
"""
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import time

from .rag_system import RAGSystem
from .storage.async_vector_store import AsyncVectorStore, run_async
from .caching.advanced_cache import AdvancedCacheManager


class RAGSystemStage3(RAGSystem):
    """3단계 최적화가 적용된 RAG 시스템"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 비동기 벡터 스토어
        self.async_vector_store = AsyncVectorStore(self.vector_store)

        # 고급 캐싱 시스템
        self.advanced_cache = AdvancedCacheManager(
            l1_max_size=100,    # L1: 메모리 100개
            l1_ttl=300,         # L1: 5분
            l2_max_size=1000,   # L2: 디스크 1000개
            l2_ttl=3600         # L2: 1시간
        )

        print("3단계 RAG 시스템 초기화 완료")
        print("   - 비동기 OpenAI API 처리")
        print("   - L1/L2 다단계 캐싱")

    async def search_and_answer_async(self,
                                     query: str,
                                     search_method: str = "hybrid",
                                     top_k: int = 5) -> Dict[str, Any]:
        """
        비동기 검색 및 답변 생성

        Args:
            query: 사용자 질문
            search_method: 검색 방법
            top_k: 반환할 문서 수

        Returns:
            답변 및 메타데이터
        """
        start_time = time.time()

        # 캐시 키 생성
        cache_key = f"{search_method}:{query}:{top_k}"

        # 고급 캐시 확인
        cached_result = self.advanced_cache.get(cache_key)
        if cached_result is not None:
            print(f"캐시 히트! (L1 또는 L2)")
            cached_result['response_time'] = time.time() - start_time
            cached_result['from_cache'] = True
            return cached_result

        try:
            # 비동기 검색 수행
            if search_method == "vector":
                search_results = await self.async_vector_store.similarity_search_async(query, k=top_k)
            elif search_method == "keyword":
                # 키워드 검색은 동기 (BM25)
                search_results = self.retriever.keyword_only_search(query, k=top_k)
            else:  # hybrid
                # Vector와 Keyword를 완전 병렬 실행
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                # 비동기로 Vector 시작
                vector_task = self.async_vector_store.similarity_search_async(query, k=top_k)

                # 동기 Keyword를 별도 스레드에서 실행 (비동기 변환)
                loop = asyncio.get_event_loop()
                keyword_task = loop.run_in_executor(
                    None,  # 기본 executor
                    self.retriever.keyword_only_search,
                    query,
                    top_k
                )

                # 두 작업을 완전 병렬로 실행
                vector_results, keyword_results = await asyncio.gather(
                    vector_task,
                    keyword_task
                )

                # 하이브리드 결합
                search_results = self._combine_hybrid_results(
                    vector_results, keyword_results, top_k
                )

                # 원본 결과 저장 (신뢰도 계산용)
                self._last_vector_results = vector_results
                self._last_keyword_results = keyword_results

            # 신뢰도 계산 (하이브리드 전용 로직)
            if search_method == "hybrid" and search_results:
                # 하이브리드는 원본 vector/keyword 결과를 기반으로 신뢰도 계산
                confidence = self._calculate_hybrid_confidence_stage3(
                    self._last_vector_results,
                    self._last_keyword_results
                )
            else:
                confidence = self._calculate_confidence(search_results, query)

            # 답변 생성 (비동기로 변경 가능)
            if confidence < self.confidence_threshold:
                answer = "문서에서 관련 정보를 찾을 수 없습니다."
                answer_confidence = "low"
            else:
                answer = self._generate_answer(query, search_results)
                answer_confidence = "high" if confidence > 0.7 else "medium"

            response_time = time.time() - start_time

            # 응답 구성
            result = {
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'answer_confidence': answer_confidence,
                'search_method': search_method,
                'sources': [
                    {
                        'content_preview': r.get('content', '')[:200] + "...",
                        'file_name': r.get('metadata', {}).get('file_name', 'Unknown'),
                        'agency': r.get('metadata', {}).get('agency', 'Unknown'),
                        'score': r.get('hybrid_score', r.get('vector_score', r.get('score', 0)))
                    }
                    for r in search_results[:3]
                ],
                'response_time': response_time,
                'total_results': len(search_results),
                'from_cache': False
            }

            # 고급 캐시에 저장
            self.advanced_cache.set(cache_key, result)

            return result

        except Exception as e:
            return {
                'query': query,
                'answer': f"검색 중 오류: {str(e)}",
                'confidence': 0.0,
                'answer_confidence': "error",
                'search_method': search_method,
                'sources': [],
                'response_time': time.time() - start_time,
                'total_results': 0,
                'from_cache': False
            }

    def _calculate_hybrid_confidence_stage3(self, vector_results: List[Dict], keyword_results: List[Dict]) -> float:
        """
        Stage 3 하이브리드 신뢰도 계산
        원본 vector와 keyword 검색 결과를 기반으로 계산 (Stage 2 로직 완전 동일)
        """
        # Vector 신뢰도 계산
        vector_confidence = 0.0
        if vector_results:
            v_score = vector_results[0].get('vector_score', 0)
            if v_score == 0:  # vector_score가 없으면 distance로 계산
                distance = vector_results[0].get('distance', 1.0)
                v_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
            vector_confidence = v_score

        # Keyword 신뢰도 계산
        keyword_confidence = 0.0
        if keyword_results:
            k_score = keyword_results[0].get('keyword_score', keyword_results[0].get('bm25_score', 0))
            if k_score > 1.0:  # BM25 원점수인 경우 정규화
                k_score = min(1.0, k_score / 15.0)
            keyword_confidence = k_score

        # Stage 2와 동일한 로직 사용
        VECTOR_WEIGHT = 0.7
        KEYWORD_WEIGHT = 0.3

        # 기본 신뢰도는 두 검색 중 더 높은 값 사용
        base_confidence = max(vector_confidence, keyword_confidence)

        # 가중 평균으로 추가 신뢰도 계산
        weighted_avg = (VECTOR_WEIGHT * vector_confidence + KEYWORD_WEIGHT * keyword_confidence)

        # 두 검색 모두 좋은 결과를 내면 가중 평균 보너스 추가
        if vector_confidence > 0.3 and keyword_confidence > 0.3:
            hybrid_confidence = max(base_confidence, weighted_avg * 1.1)
        else:
            hybrid_confidence = base_confidence

        # 두 검색 모두에서 결과가 있으면 추가 보너스
        if vector_confidence > 0.1 and keyword_confidence > 0.1:
            hybrid_confidence = min(hybrid_confidence * 1.05, 1.0)

        return min(hybrid_confidence, 1.0)

    def _combine_hybrid_results(self, vector_results: List[Dict],
                                keyword_results: List[Dict],
                                top_k: int) -> List[Dict]:
        """비동기 하이브리드 결과 결합 (Stage 2 로직)"""
        VECTOR_WEIGHT = 0.7
        KEYWORD_WEIGHT = 0.3

        all_results = []

        # Vector 결과 처리 - 거리 기반 점수
        for i, result in enumerate(vector_results):
            distance = result.get('distance', 1.0)
            # 거리를 0-1 점수로 변환
            vector_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
            # 순위 기반 보너스
            rank_bonus = max(0.0, (10 - i) / 10.0) if i < 10 else 0.0
            final_vector_score = (vector_score + rank_bonus * 0.1) / 1.1

            all_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'id': result['id'],
                'vector_score': final_vector_score,
                'keyword_score': 0.0,
                'hybrid_score': VECTOR_WEIGHT * final_vector_score,
                'source': 'vector',
                'original_rank': i + 1
            })

        # 키워드 결과 처리 - BM25 점수
        for i, result in enumerate(keyword_results):
            bm25_score = result.get('bm25_score', 0)
            # BM25 점수를 0-1로 정규화
            keyword_score = max(0.0, min(1.0, bm25_score / 15.0))
            # 순위 기반 보너스
            rank_bonus = max(0.0, (10 - i) / 10.0) if i < 10 else 0.0
            final_keyword_score = (keyword_score + rank_bonus * 0.1) / 1.1

            all_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'id': result['id'],
                'vector_score': 0.0,
                'keyword_score': final_keyword_score,
                'hybrid_score': KEYWORD_WEIGHT * final_keyword_score,
                'source': 'keyword',
                'original_rank': i + 1
            })

        # 하이브리드 점수로 정렬
        sorted_results = sorted(
            all_results,
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        return sorted_results[:top_k]

    def search_and_answer_stage3(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        동기 환경에서 비동기 검색 호출

        Args:
            query: 검색 쿼리
            **kwargs: 추가 파라미터

        Returns:
            검색 결과
        """
        return run_async(self.search_and_answer_async(query, **kwargs))

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        return self.advanced_cache.get_stats()

    def search_with_smart_enhancement(self, user_query: str, search_method: str = "hybrid",
                                    top_k: int = 5) -> Dict[str, Any]:
        """
        스마트 향상된 검색 (Stage 3 - 캐싱 적용)

        Args:
            user_query: 사용자 질문
            search_method: 검색 방법
            top_k: 반환할 문서 수

        Returns:
            검색 결과 (캐싱됨)
        """
        # 캐시 키 생성 (스마트 검색용)
        cache_key = f"smart:{search_method}:{user_query}:{top_k}"

        # 캐시 확인
        cached_result = self.advanced_cache.get(cache_key)
        if cached_result is not None:
            print(f"[캐시 히트] 스마트 검색: {user_query[:30]}...")
            cached_result['from_cache'] = True
            return cached_result

        # 캐시 미스 - 부모 클래스의 스마트 검색 호출
        # 하지만 내부적으로 search_and_answer_stage3 사용하도록 수정
        try:
            start_time = time.time()

            # 1. 원본 검색 수행 (Stage 3 메서드 사용)
            original_result = self.search_and_answer_stage3(user_query, search_method=search_method, top_k=top_k)

            # 2. 검색 결과 품질 평가
            needs_enhancement = (
                original_result['confidence'] < 0.6 or
                len(original_result['sources']) < 2
            )

            enhanced_info = {
                'used_enhancement': False,
                'original_confidence': original_result['confidence'],
                'enhancement_suggestions': []
            }

            if needs_enhancement:
                # 3. 쿼리 향상
                enhancement = self.enhance_user_query(user_query)
                enhanced_query = enhancement['enhanced_query']

                # 4. 향상된 쿼리로 재검색 (원본과 다른 경우만)
                if enhanced_query != user_query:
                    enhanced_result = self.search_and_answer_stage3(enhanced_query, search_method=search_method, top_k=top_k)

                    # 5. 더 나은 결과 선택
                    if enhanced_result['confidence'] > original_result['confidence']:
                        result = enhanced_result
                        enhanced_info['used_enhancement'] = True
                        enhanced_info['enhanced_query'] = enhanced_query
                        enhanced_info['confidence_improvement'] = enhanced_result['confidence'] - original_result['confidence']
                    else:
                        result = original_result
                else:
                    result = original_result
            else:
                result = original_result

            # enhancement_info 추가
            result['enhancement_info'] = enhanced_info
            result['response_time'] = time.time() - start_time
            result['from_cache'] = False

            # 캐시에 저장
            self.advanced_cache.set(cache_key, result)

            return result

        except Exception as e:
            print(f"스마트 검색 오류: {str(e)}")
            # 오류 시 기본 Stage 3 검색으로 폴백
            return self.search_and_answer_stage3(user_query, search_method=search_method, top_k=top_k)
