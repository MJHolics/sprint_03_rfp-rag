"""
성능 개선 단계별 비교 벤치마크 (간단 버전)
"""
import time
import psutil
import os
import sqlite3
from pathlib import Path
import sys
import tempfile

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from config.settings import *

def measure_memory_usage():
    """메모리 사용량 측정 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_baseline_performance():
    """패치 전 성능 시뮬레이션 (순차 처리)"""
    print("=" * 60)
    print("패치 전 (기본) 성능 측정")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vector_db = os.path.join(temp_dir, "baseline_vector_db")
        temp_metadata_db = os.path.join(temp_dir, "baseline_metadata.db")

        rag_system = RAGSystem(temp_vector_db, temp_metadata_db, CHUNK_SIZE, CHUNK_OVERLAP)

        # 테스트 파일 (소규모)
        test_files = list(Path("./files").glob("*.pdf"))[:3] + list(Path("./files").glob("*.hwp"))[:3]

        initial_memory = measure_memory_usage()
        start_time = time.time()

        successful = 0
        total_chunks = 0

        # 순차 처리 (병렬 처리 없음)
        for file_path in test_files:
            try:
                result = rag_system.process_document(str(file_path))
                if result.success:
                    successful += 1
                    total_chunks += result.total_chunks
                print(f"처리 완료: {file_path.name}")
            except Exception as e:
                print(f"처리 실패: {file_path.name} - {e}")

        processing_time = time.time() - start_time
        memory_used = measure_memory_usage() - initial_memory

        # DB 성능 측정
        conn = sqlite3.connect(temp_metadata_db)
        cursor = conn.cursor()

        # 인덱스 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()

        # 쿼리 속도 테스트
        query_start = time.time()
        cursor.execute("SELECT COUNT(*) FROM documents")
        cursor.fetchall()
        query_time = time.time() - query_start

        conn.close()

        results = {
            'total_files': len(test_files),
            'successful': successful,
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'memory_used': memory_used,
            'files_per_minute': (successful / processing_time) * 60 if processing_time > 0 else 0,
            'query_time': query_time,
            'indexes_count': len(indexes)
        }

        print(f"결과:")
        print(f"  처리 파일: {successful}/{len(test_files)}")
        print(f"  총 청크: {total_chunks}")
        print(f"  처리 시간: {processing_time:.2f}초")
        print(f"  메모리 사용: {memory_used:.1f}MB")
        print(f"  처리 속도: {results['files_per_minute']:.1f} 파일/분")
        print(f"  쿼리 시간: {query_time:.3f}초")
        print(f"  인덱스 수: {len(indexes)}")

        return results

def test_optimized_performance():
    """2단계 패치 후 성능 측정"""
    print("=" * 60)
    print("2단계 패치 후 성능 측정")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_vector_db = os.path.join(temp_dir, "optimized_vector_db")
        temp_metadata_db = os.path.join(temp_dir, "optimized_metadata.db")

        rag_system = RAGSystem(temp_vector_db, temp_metadata_db, CHUNK_SIZE, CHUNK_OVERLAP)

        # 동일한 테스트 파일
        test_files = list(Path("./files").glob("*.pdf"))[:3] + list(Path("./files").glob("*.hwp"))[:3]

        initial_memory = measure_memory_usage()
        start_time = time.time()

        # 디렉토리 처리 (병렬 처리 포함)
        temp_test_dir = os.path.join(temp_dir, "test_files")
        os.makedirs(temp_test_dir)

        # 테스트 파일들을 임시 디렉토리에 복사
        for file_path in test_files:
            shutil.copy2(file_path, temp_test_dir)

        # 병렬 처리로 실행
        results_dict = rag_system.process_directory(temp_test_dir)

        processing_time = time.time() - start_time
        memory_used = measure_memory_usage() - initial_memory

        # DB 성능 측정
        conn = sqlite3.connect(temp_metadata_db)
        cursor = conn.cursor()

        # 인덱스 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()

        # 쿼리 속도 테스트
        query_start = time.time()
        cursor.execute("SELECT COUNT(*) FROM documents")
        cursor.fetchall()
        query_time = time.time() - query_start

        conn.close()

        results = {
            'total_files': results_dict['total_files'],
            'successful': results_dict['successful'],
            'total_chunks': results_dict['total_chunks'],
            'processing_time': processing_time,
            'memory_used': memory_used,
            'files_per_minute': (results_dict['successful'] / processing_time) * 60 if processing_time > 0 else 0,
            'query_time': query_time,
            'indexes_count': len(indexes)
        }

        print(f"결과:")
        print(f"  처리 파일: {results['successful']}/{results['total_files']}")
        print(f"  총 청크: {results['total_chunks']}")
        print(f"  처리 시간: {processing_time:.2f}초")
        print(f"  메모리 사용: {memory_used:.1f}MB")
        print(f"  처리 속도: {results['files_per_minute']:.1f} 파일/분")
        print(f"  쿼리 시간: {query_time:.3f}초")
        print(f"  인덱스 수: {len(indexes)}")

        return results

def compare_results(baseline, optimized):
    """결과 비교 및 출력"""
    print("\n" + "=" * 60)
    print("성능 개선 효과 분석")
    print("=" * 60)

    # 처리 속도 개선
    if baseline['processing_time'] > 0:
        speed_improvement = baseline['processing_time'] / optimized['processing_time']
        print(f"문서 처리 속도 개선: {speed_improvement:.1f}배")

    # 메모리 개선
    if baseline['memory_used'] > 0:
        memory_improvement = (baseline['memory_used'] - optimized['memory_used']) / baseline['memory_used'] * 100
        print(f"메모리 사용량 개선: {memory_improvement:.1f}% 절약")

    # 처리량 개선
    if baseline['files_per_minute'] > 0:
        throughput_improvement = optimized['files_per_minute'] / baseline['files_per_minute']
        print(f"처리량 개선: {throughput_improvement:.1f}배")

    # DB 쿼리 속도 개선
    if baseline['query_time'] > 0:
        query_improvement = baseline['query_time'] / optimized['query_time']
        print(f"DB 쿼리 속도 개선: {query_improvement:.1f}배")

    # 상세 비교표
    print(f"\n상세 비교표:")
    print(f"{'지표':<15} {'패치 전':<12} {'2단계 후':<12} {'개선 효과'}")
    print("-" * 55)
    print(f"{'처리시간(초)':<15} {baseline['processing_time']:<12.2f} {optimized['processing_time']:<12.2f} {speed_improvement:.1f}배")
    print(f"{'메모리(MB)':<15} {baseline['memory_used']:<12.1f} {optimized['memory_used']:<12.1f} {memory_improvement:.1f}% 절약")
    print(f"{'파일/분':<15} {baseline['files_per_minute']:<12.1f} {optimized['files_per_minute']:<12.1f} {throughput_improvement:.1f}배")
    print(f"{'쿼리시간(초)':<15} {baseline['query_time']:<12.3f} {optimized['query_time']:<12.3f} {query_improvement:.1f}배")
    print(f"{'인덱스 수':<15} {baseline['indexes_count']:<12} {optimized['indexes_count']:<12} +{optimized['indexes_count'] - baseline['indexes_count']}")

def main():
    print("RAG 시스템 성능 개선 단계별 비교")
    print("=" * 60)

    if not Path("./files").exists():
        print("오류: ./files 디렉토리가 없습니다.")
        return

    # 1. 패치 전 성능 측정
    baseline_results = test_baseline_performance()

    print("\n")

    # 2. 2단계 패치 후 성능 측정
    optimized_results = test_optimized_performance()

    # 3. 결과 비교
    compare_results(baseline_results, optimized_results)

if __name__ == "__main__":
    import shutil
    main()