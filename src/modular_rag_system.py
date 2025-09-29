"""
모듈형 RAG 시스템
각 단계별 기능을 켜고 끌 수 있도록 구현
"""
import time
import os
import sqlite3
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import hashlib
import json

from .rag_system import RAGSystem
from .storage.vector_store import VectorStore
from .storage.metadata_store import MetadataStore


class Stage1Optimizer:
    """1단계: SQLite 최적화 + 기본 캐싱"""

    def __init__(self, metadata_db_path: str):
        self.metadata_db_path = metadata_db_path
        self.search_cache = {}  # 기본 검색 캐시
        self.cache_hits = 0
        self.cache_misses = 0

    def optimize_database(self):
        """SQLite 최적화 적용"""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()

            # 1단계 인덱스 생성
            optimization_queries = [
                "CREATE INDEX IF NOT EXISTS idx_agency_s1 ON documents(agency)",
                "CREATE INDEX IF NOT EXISTS idx_business_type_s1 ON documents(business_type)",
                "CREATE INDEX IF NOT EXISTS idx_processed_date_s1 ON documents(processed_date)",
                "CREATE INDEX IF NOT EXISTS idx_budget_s1 ON documents(budget)",
                "CREATE INDEX IF NOT EXISTS idx_deadline_s1 ON documents(deadline_date)",

                # SQLite 성능 최적화
                "PRAGMA journal_mode = WAL",
                "PRAGMA synchronous = NORMAL",
                "PRAGMA cache_size = 10000",
                "PRAGMA temp_store = MEMORY"
            ]

            for query in optimization_queries:
                cursor.execute(query)

            conn.commit()
            conn.close()
            print("1단계 SQLite 최적화 완료")
            return True

        except Exception as e:
            print(f"1단계 SQLite 최적화 오류: {e}")
            return False

    def get_cached_search(self, query: str) -> Optional[Dict]:
        """캐시된 검색 결과 조회"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.search_cache:
            self.cache_hits += 1
            return self.search_cache[cache_key]
        else:
            self.cache_misses += 1
            return None

    def cache_search_result(self, query: str, result: Dict):
        """검색 결과 캐시에 저장"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        # 최대 100개까지만 캐시
        if len(self.search_cache) >= 100:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]

        self.search_cache[cache_key] = result

    def get_cache_stats(self) -> Dict[str, float]:
        """캐시 통계 반환"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }


class Stage2Optimizer:
    """2단계: 병렬 처리 + 벡터 최적화"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = None

    def initialize_parallel_processing(self):
        """병렬 처리 초기화"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        print(f"2단계 병렬 처리 초기화: {self.max_workers}개 워커")

    def process_documents_parallel(self, file_paths: List[str], processor_func):
        """문서 병렬 처리"""
        if not self.executor:
            self.initialize_parallel_processing()

        futures = []
        for file_path in file_paths:
            future = self.executor.submit(processor_func, file_path)
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5분 타임아웃
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})

        return results

    def optimize_vector_search(self, vector_store):
        """벡터 검색 최적화"""
        # ChromaDB HNSW 파라미터 최적화는 이미 현재 시스템에 적용됨
        print("2단계 벡터 검색 최적화 적용")
        return True

    def cleanup(self):
        """리소스 정리"""
        if self.executor:
            self.executor.shutdown(wait=True)


class Stage3Optimizer:
    """3단계: 비동기 API + 고급 캐싱 + 분산 처리"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.l1_cache = {}  # 메모리 캐시
        self.l2_cache = {}  # 디스크 캐시
        self.max_l1_size = 100
        self.max_l2_size = 500
        self.cache_stats = {'hits': 0, 'misses': 0}

    async def async_openai_call(self, prompt: str) -> str:
        """비동기 OpenAI API 호출"""
        if not self.openai_api_key:
            # API 키가 없으면 시뮬레이션
            await asyncio.sleep(0.3)  # API 호출 시뮬레이션
            return f"비동기 응답: {prompt[:50]}..."

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json'
                }

                payload = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500
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
                        return f"API 오류: {response.status}"

        except Exception as e:
            return f"비동기 처리 오류: {str(e)}"

    def get_from_advanced_cache(self, key: str) -> Optional[Any]:
        """고급 캐시에서 데이터 조회"""
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

    def store_in_advanced_cache(self, key: str, value: Any):
        """고급 캐시에 데이터 저장"""
        if len(self.l1_cache) >= self.max_l1_size:
            self._evict_from_l1()
        self.l1_cache[key] = value

    def _promote_to_l1(self, key: str, value: Any):
        """L2에서 L1으로 데이터 승격"""
        if len(self.l1_cache) >= self.max_l1_size:
            self._evict_from_l1()
        self.l1_cache[key] = value

    def _evict_from_l1(self):
        """L1 캐시에서 가장 오래된 항목 제거"""
        if self.l1_cache:
            oldest_key = next(iter(self.l1_cache))
            evicted_value = self.l1_cache.pop(oldest_key)

            # L2로 이동
            if len(self.l2_cache) >= self.max_l2_size:
                oldest_l2_key = next(iter(self.l2_cache))
                self.l2_cache.pop(oldest_l2_key)

            self.l2_cache[oldest_key] = evicted_value

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total * 100) if total > 0 else 0

        return {
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'hit_rate': hit_rate,
            'total_requests': total
        }


class ModularRAGSystem:
    """모듈형 RAG 시스템 - 각 단계를 켜고 끌 수 있음"""

    def __init__(self,
                 vector_db_path: str,
                 metadata_db_path: str,
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 enable_stage1: bool = False,
                 enable_stage2: bool = False,
                 enable_stage3: bool = False,
                 openai_api_key: Optional[str] = None):

        self.vector_db_path = vector_db_path
        self.metadata_db_path = metadata_db_path
        self.chunk_size = chunk_size
        self.overlap = overlap

        # 단계별 활성화 플래그
        self.enable_stage1 = enable_stage1
        self.enable_stage2 = enable_stage2
        self.enable_stage3 = enable_stage3

        # 기본 RAG 시스템 초기화
        self.base_rag = RAGSystem(vector_db_path, metadata_db_path, chunk_size, overlap)

        # 각 단계별 최적화기 초기화
        self.stage1 = Stage1Optimizer(metadata_db_path) if enable_stage1 else None
        self.stage2 = Stage2Optimizer() if enable_stage2 else None
        self.stage3 = Stage3Optimizer(openai_api_key) if enable_stage3 else None

        # 단계별 기능 초기화
        self._initialize_enabled_stages()

    def _initialize_enabled_stages(self):
        """활성화된 단계별 기능 초기화"""
        print(f"모듈형 RAG 시스템 초기화:")
        print(f"  - 1단계 (SQLite+캐싱): {'활성화' if self.enable_stage1 else '비활성화'}")
        print(f"  - 2단계 (병렬+벡터): {'활성화' if self.enable_stage2 else '비활성화'}")
        print(f"  - 3단계 (비동기+고급캐싱): {'활성화' if self.enable_stage3 else '비활성화'}")

        if self.stage1:
            self.stage1.optimize_database()

        if self.stage2:
            self.stage2.initialize_parallel_processing()
            self.stage2.optimize_vector_search(self.base_rag.vector_store)

        if self.stage3:
            print("3단계 고급 캐싱 시스템 초기화")

    def process_document(self, file_path: str):
        """문서 처리 (단계별 최적화 적용)"""
        if self.stage2 and hasattr(self.stage2, 'executor') and self.stage2.executor:
            # 2단계: 병렬 처리 (하지만 단일 파일이므로 기본 처리)
            return self.base_rag.process_document(file_path)
        else:
            # 기본 처리
            return self.base_rag.process_document(file_path)

    def process_documents_batch(self, file_paths: List[str]):
        """배치 문서 처리"""
        if self.stage2:
            # 2단계: 병렬 처리
            return self.stage2.process_documents_parallel(
                file_paths,
                lambda fp: self.base_rag.process_document(fp)
            )
        else:
            # 순차 처리
            results = []
            for file_path in file_paths:
                results.append(self.base_rag.process_document(file_path))
            return results

    def search_and_answer(self, query: str, top_k: int = 3):
        """검색 및 답변 생성 (단계별 최적화 적용)"""
        start_time = time.time()

        # 1단계: 기본 캐싱 확인
        if self.stage1:
            cached_result = self.stage1.get_cached_search(query)
            if cached_result:
                cached_result['from_cache'] = True
                cached_result['processing_time'] = time.time() - start_time
                return cached_result

        # 3단계: 고급 캐싱 확인
        if self.stage3:
            cache_key = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
            cached_result = self.stage3.get_from_advanced_cache(cache_key)
            if cached_result:
                cached_result['from_advanced_cache'] = True
                cached_result['processing_time'] = time.time() - start_time
                return cached_result

        # 캐시 미스 - 실제 검색 수행
        if self.stage3:
            # 3단계: 비동기 처리
            return asyncio.run(self._async_search_and_answer(query, top_k, start_time))
        else:
            # 기본 검색
            result = self.base_rag.search_and_answer(query, top_k)
            result['processing_time'] = time.time() - start_time
            result['from_cache'] = False

            # 결과 캐싱
            if self.stage1:
                self.stage1.cache_search_result(query, result)

            return result

    async def _async_search_and_answer(self, query: str, top_k: int, start_time: float):
        """비동기 검색 및 답변 생성"""
        # 기본 검색 수행
        search_result = self.base_rag.search_and_answer(query, top_k)

        # 비동기 답변 생성 개선
        if self.stage3 and search_result.get('sources'):
            contexts = [src['content'] for src in search_result['sources'][:3]]
            prompt = f"질문: {query}\n컨텍스트: {' '.join(contexts)}"

            enhanced_answer = await self.stage3.async_openai_call(prompt)
            search_result['enhanced_answer'] = enhanced_answer
            search_result['async_processed'] = True

        search_result['processing_time'] = time.time() - start_time
        search_result['from_cache'] = False

        # 결과 캐싱
        if self.stage1:
            self.stage1.cache_search_result(query, search_result)

        if self.stage3:
            cache_key = hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()
            self.stage3.store_in_advanced_cache(cache_key, search_result)

        return search_result

    def get_system_stats(self):
        """시스템 통계 조회"""
        base_stats = self.base_rag.get_system_stats()

        # 단계별 통계 추가
        stage_stats = {
            'enabled_stages': {
                'stage1': self.enable_stage1,
                'stage2': self.enable_stage2,
                'stage3': self.enable_stage3
            }
        }

        if self.stage1:
            stage_stats['stage1_cache'] = self.stage1.get_cache_stats()

        if self.stage2:
            stage_stats['stage2_parallel'] = {
                'max_workers': self.stage2.max_workers,
                'executor_active': self.stage2.executor is not None
            }

        if self.stage3:
            stage_stats['stage3_advanced'] = self.stage3.get_cache_stats()

        base_stats['modular_stages'] = stage_stats
        return base_stats

    def cleanup(self):
        """리소스 정리"""
        if self.stage2:
            self.stage2.cleanup()

        print("모듈형 RAG 시스템 정리 완료")


def create_baseline_system(vector_db_path: str, metadata_db_path: str) -> ModularRAGSystem:
    """패치 전 베이스라인 시스템 생성"""
    return ModularRAGSystem(
        vector_db_path=vector_db_path,
        metadata_db_path=metadata_db_path,
        chunk_size=1000,    # 큰 청크 크기
        overlap=100,        # 적은 오버랩
        enable_stage1=False,
        enable_stage2=False,
        enable_stage3=False
    )

def create_stage1_system(vector_db_path: str, metadata_db_path: str) -> ModularRAGSystem:
    """1단계 시스템 생성"""
    return ModularRAGSystem(
        vector_db_path=vector_db_path,
        metadata_db_path=metadata_db_path,
        chunk_size=800,     # 중간 청크 크기
        overlap=150,        # 중간 오버랩
        enable_stage1=True,
        enable_stage2=False,
        enable_stage3=False
    )

def create_stage2_system(vector_db_path: str, metadata_db_path: str) -> ModularRAGSystem:
    """2단계 시스템 생성"""
    return ModularRAGSystem(
        vector_db_path=vector_db_path,
        metadata_db_path=metadata_db_path,
        chunk_size=600,     # 작은 청크 크기
        overlap=200,        # 많은 오버랩
        enable_stage1=True,
        enable_stage2=True,
        enable_stage3=False
    )

def create_stage3_system(vector_db_path: str, metadata_db_path: str, openai_api_key: str = None) -> ModularRAGSystem:
    """3단계 시스템 생성"""
    return ModularRAGSystem(
        vector_db_path=vector_db_path,
        metadata_db_path=metadata_db_path,
        chunk_size=600,     # 최적화된 청크 크기
        overlap=200,        # 최적화된 오버랩
        enable_stage1=True,
        enable_stage2=True,
        enable_stage3=True,
        openai_api_key=openai_api_key
    )