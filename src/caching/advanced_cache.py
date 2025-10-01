"""
고급 캐싱 시스템 - L1/L2 다단계 캐시
3단계 최적화: 메모리 + 디스크 캐시
"""
import time
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from collections import OrderedDict


class L1Cache:
    """L1 캐시 - 메모리 기반, 초고속"""

    def __init__(self, max_size: int = 100, ttl: int = 300):
        """
        Args:
            max_size: 최대 캐시 항목 수
            ttl: Time To Live (초)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0}

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            value, timestamp = self.cache[key]

            # TTL 체크
            if time.time() - timestamp < self.ttl:
                # LRU: 최근 사용으로 이동
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return value
            else:
                # 만료된 항목 삭제
                del self.cache[key]

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any):
        """캐시에 값 저장"""
        # 크기 제한 체크
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (FIFO)
            self.cache.popitem(last=False)

        self.cache[key] = (value, time.time())

    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%"
        }


class L2Cache:
    """L2 캐시 - 디스크 기반, 대용량"""

    def __init__(self, cache_dir: str = "./cache_l2", max_size: int = 1000, ttl: int = 3600):
        """
        Args:
            cache_dir: 캐시 저장 디렉토리
            max_size: 최대 캐시 파일 수
            ttl: Time To Live (초)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self.stats = {'hits': 0, 'misses': 0}

    def _get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로 생성"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """디스크 캐시에서 값 조회"""
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # TTL 체크
                if time.time() - data['timestamp'] < self.ttl:
                    self.stats['hits'] += 1
                    return data['value']
                else:
                    # 만료된 파일 삭제
                    cache_path.unlink()

            except Exception as e:
                print(f"L2 캐시 읽기 오류: {e}")

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any):
        """디스크 캐시에 값 저장"""
        try:
            # 크기 제한 체크
            cache_files = list(self.cache_dir.glob("*.json"))
            if len(cache_files) >= self.max_size:
                # 가장 오래된 파일 삭제
                oldest_file = min(cache_files, key=lambda p: p.stat().st_mtime)
                oldest_file.unlink()

            # 캐시 저장
            cache_path = self._get_cache_path(key)
            data = {
                'value': value,
                'timestamp': time.time()
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

        except Exception as e:
            print(f"L2 캐시 쓰기 오류: {e}")

    def clear(self):
        """캐시 전체 삭제"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0

        cache_files = list(self.cache_dir.glob("*.json"))

        return {
            'size': len(cache_files),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%"
        }


class AdvancedCacheManager:
    """L1 + L2 통합 캐시 관리자"""

    def __init__(self,
                 l1_max_size: int = 100,
                 l1_ttl: int = 300,
                 l2_max_size: int = 1000,
                 l2_ttl: int = 3600):
        """
        Args:
            l1_max_size: L1 캐시 최대 항목 수
            l1_ttl: L1 캐시 TTL (초)
            l2_max_size: L2 캐시 최대 항목 수
            l2_ttl: L2 캐시 TTL (초)
        """
        self.l1 = L1Cache(max_size=l1_max_size, ttl=l1_ttl)
        self.l2 = L2Cache(max_size=l2_max_size, ttl=l2_ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회 (L1 → L2 순서)

        Returns:
            캐시된 값 또는 None
        """
        # L1 먼저 확인
        value = self.l1.get(key)
        if value is not None:
            return value

        # L2 확인
        value = self.l2.get(key)
        if value is not None:
            # L2 히트 시 L1에도 저장 (캐시 승격)
            self.l1.set(key, value)
            return value

        return None

    def set(self, key: str, value: Any):
        """캐시에 값 저장 (L1과 L2 모두)"""
        self.l1.set(key, value)
        self.l2.set(key, value)

    def clear(self):
        """모든 캐시 삭제"""
        self.l1.clear()
        self.l2.clear()

    def get_stats(self) -> Dict[str, Any]:
        """전체 캐시 통계"""
        l1_stats = self.l1.get_stats()
        l2_stats = self.l2.get_stats()

        total_hits = self.l1.stats['hits'] + self.l2.stats['hits']
        total_misses = self.l1.stats['misses'] + self.l2.stats['misses']
        total_requests = total_hits + total_misses

        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'overall_hit_rate': f"{overall_hit_rate:.1f}%",
            'total_requests': total_requests
        }
