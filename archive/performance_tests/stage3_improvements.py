"""
3단계 성능 개선 구현
- 비동기 OpenAI API 처리
- 고급 캐싱 전략
- 분산 처리 아키텍처
"""
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import json
import hashlib

class AsyncOpenAIProcessor:
    """비동기 OpenAI API 처리기"""

    def __init__(self, api_key: str, max_concurrent: int = 10):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch_async(self, texts: List[str], batch_size: int = 5):
        """비동기 배치 처리"""
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        async with aiohttp.ClientSession() as session:
            tasks = []
            for batch in batches:
                task = self._process_single_batch(session, batch)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

    async def _process_single_batch(self, session: aiohttp.ClientSession, batch: List[str]):
        """단일 배치 비동기 처리"""
        async with self.semaphore:
            try:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }

                payload = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': ' '.join(batch)}],
                    'max_tokens': 1000
                }

                async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        return f"Error: {response.status}"

            except Exception as e:
                return f"Exception: {str(e)}"

class AdvancedCacheManager:
    """고급 캐싱 전략"""

    def __init__(self):
        self.l1_cache = {}  # 메모리 캐시 (빠른 접근)
        self.l2_cache = {}  # 디스크 캐시 (대용량)
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_l1_size = 1000
        self.max_l2_size = 10000

    def get_cache_key(self, query: str, filters: Dict = None) -> str:
        """캐시 키 생성"""
        cache_data = {'query': query, 'filters': filters or {}}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, key: str):
        """캐시에서 데이터 조회"""
        # L1 캐시 확인
        if key in self.l1_cache:
            self.cache_stats['hits'] += 1
            return self.l1_cache[key]

        # L2 캐시 확인
        if key in self.l2_cache:
            self.cache_stats['hits'] += 1
            # L2에서 L1으로 승격
            self._promote_to_l1(key, self.l2_cache[key])
            return self.l2_cache[key]

        self.cache_stats['misses'] += 1
        return None

    def set(self, key: str, value: Any):
        """캐시에 데이터 저장"""
        # L1 캐시에 저장
        if len(self.l1_cache) >= self.max_l1_size:
            self._evict_l1()

        self.l1_cache[key] = value

    def _promote_to_l1(self, key: str, value: Any):
        """L2에서 L1으로 데이터 승격"""
        if len(self.l1_cache) >= self.max_l1_size:
            self._evict_l1()
        self.l1_cache[key] = value

    def _evict_l1(self):
        """L1 캐시에서 가장 오래된 항목 제거"""
        if self.l1_cache:
            oldest_key = next(iter(self.l1_cache))
            evicted_value = self.l1_cache.pop(oldest_key)

            # L2로 이동
            if len(self.l2_cache) >= self.max_l2_size:
                # L2에서도 가장 오래된 항목 제거
                oldest_l2_key = next(iter(self.l2_cache))
                self.l2_cache.pop(oldest_l2_key)

            self.l2_cache[oldest_key] = evicted_value

    def get_hit_rate(self) -> float:
        """캐시 히트율 조회"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        return self.cache_stats['hits'] / total if total > 0 else 0.0

class DistributedProcessor:
    """분산 처리 아키텍처"""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.task_queue = []
        self.results = {}

    def submit_task(self, task_id: str, func, *args, **kwargs):
        """작업 제출"""
        future = self.executor.submit(func, *args, **kwargs)
        self.results[task_id] = future
        return task_id

    def get_result(self, task_id: str, timeout: float = None):
        """결과 조회"""
        if task_id in self.results:
            return self.results[task_id].result(timeout=timeout)
        return None

    def process_parallel_documents(self, file_paths: List[str], processor_func):
        """문서 병렬 처리"""
        tasks = {}

        for i, file_path in enumerate(file_paths):
            task_id = f"doc_{i}"
            tasks[task_id] = self.submit_task(task_id, processor_func, file_path)

        # 모든 작업 완료 대기
        results = []
        for task_id in tasks:
            try:
                result = self.get_result(task_id, timeout=300)  # 5분 타임아웃
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})

        return results

    def shutdown(self):
        """리소스 정리"""
        self.executor.shutdown(wait=True)

class EnhancedRAGSystem:
    """3단계 개선이 적용된 RAG 시스템"""

    def __init__(self, base_rag_system, openai_api_key: str = None):
        self.base_system = base_rag_system
        self.cache_manager = AdvancedCacheManager()
        self.distributed_processor = DistributedProcessor()

        if openai_api_key:
            self.async_processor = AsyncOpenAIProcessor(openai_api_key)
        else:
            self.async_processor = None

    async def enhanced_search_and_answer(self, query: str, **kwargs):
        """캐시와 비동기 처리가 적용된 검색"""
        # 캐시 확인
        cache_key = self.cache_manager.get_cache_key(query, kwargs)
        cached_result = self.cache_manager.get(cache_key)

        if cached_result:
            return cached_result

        # 캐시 미스 시 실제 검색 수행
        start_time = time.time()

        if self.async_processor:
            # 비동기 처리
            result = await self._async_search_and_answer(query, **kwargs)
        else:
            # 기존 동기 처리
            result = self.base_system.search_and_answer(query, **kwargs)

        # 응답 시간 기록
        result['processing_time'] = time.time() - start_time
        result['cache_hit'] = False

        # 캐시에 저장
        self.cache_manager.set(cache_key, result)

        return result

    async def _async_search_and_answer(self, query: str, **kwargs):
        """비동기 검색 및 답변 생성"""
        # 기존 검색 로직 (동기)
        search_results = self.base_system.retriever.search(query, **kwargs)

        # 비동기 답변 생성
        if self.async_processor and search_results:
            contexts = [result['content'] for result in search_results[:3]]
            answer = await self.async_processor.process_batch_async(
                [f"질문: {query}\n컨텍스트: {' '.join(contexts)}"]
            )

            return {
                'answer': answer[0] if answer else "답변 생성 실패",
                'sources': search_results,
                'confidence': 0.85,  # 비동기 처리 기본 신뢰도
                'async_processed': True
            }

        # 폴백: 기존 처리
        return self.base_system.search_and_answer(query, **kwargs)

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        return {
            'cache_hit_rate': self.cache_manager.get_hit_rate(),
            'cache_stats': self.cache_manager.cache_stats,
            'l1_cache_size': len(self.cache_manager.l1_cache),
            'l2_cache_size': len(self.cache_manager.l2_cache),
            'worker_count': self.distributed_processor.num_workers
        }

    def cleanup(self):
        """리소스 정리"""
        self.distributed_processor.shutdown()

# 3단계 개선 사항 적용 함수
def apply_stage3_improvements(base_rag_system, openai_api_key: str = None):
    """기존 RAG 시스템에 3단계 개선사항 적용"""
    return EnhancedRAGSystem(base_rag_system, openai_api_key)

# 성능 벤치마크 함수
def benchmark_stage3_performance(enhanced_system, test_queries: List[str]):
    """3단계 개선 성능 벤치마크"""
    results = {
        'avg_response_time': 0,
        'cache_hit_rate': 0,
        'async_processing_rate': 0,
        'memory_efficiency': 0
    }

    total_time = 0
    async_count = 0

    for query in test_queries:
        start_time = time.time()

        # 비동기 실행을 위한 이벤트 루프
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            enhanced_system.enhanced_search_and_answer(query)
        )

        response_time = time.time() - start_time
        total_time += response_time

        if result.get('async_processed'):
            async_count += 1

    # 통계 계산
    results['avg_response_time'] = total_time / len(test_queries)
    results['cache_hit_rate'] = enhanced_system.cache_manager.get_hit_rate()
    results['async_processing_rate'] = async_count / len(test_queries)

    # 성능 통계
    perf_stats = enhanced_system.get_performance_stats()
    results.update(perf_stats)

    return results