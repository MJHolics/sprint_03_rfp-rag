"""
간단한 성능 비교 테스트
기존 시스템 상태에서 실행하여 현재 성능 확인
"""
import time
import psutil
import os
import sqlite3
from pathlib import Path
import sys

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from config.settings import *

def measure_memory_usage():
    """메모리 사용량 측정 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def analyze_current_system():
    """현재 시스템 분석"""
    print("=" * 60)
    print("현재 시스템 상태 분석")
    print("=" * 60)

    # RAG 시스템 초기화
    rag_system = RAGSystem(
        vector_db_path=str(VECTOR_DB_PATH),
        metadata_db_path=str(METADATA_DB_PATH),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    # 시스템 통계
    stats = rag_system.get_system_stats()

    print(f"벡터 DB 상태:")
    print(f"  처리된 청크: {stats['vector_store'].get('total_chunks', 0)}")
    print(f"  유니크 파일: {stats['vector_store'].get('unique_files', 0)}")

    print(f"메타데이터 DB 상태:")
    print(f"  총 문서: {stats['metadata_store']['total_documents']}")

    print(f"시스템 설정:")
    print(f"  OpenAI API: {'활성화' if stats['openai_enabled'] else '비활성화'}")
    print(f"  지원 형식: {', '.join(stats['processors'])}")

    return stats

def test_database_performance():
    """데이터베이스 성능 테스트"""
    print("\n" + "=" * 60)
    print("데이터베이스 성능 테스트")
    print("=" * 60)

    db_path = str(METADATA_DB_PATH)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 인덱스 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]

        print(f"총 인덱스 수: {len(indexes)}")

        # 최적화된 인덱스 확인
        optimized_indexes = [idx for idx in indexes if any(col in idx.lower() for col in ['agency', 'budget', 'business'])]
        print(f"최적화된 인덱스: {len(optimized_indexes)}")

        # 쿼리 성능 테스트
        test_queries = [
            ("총 문서 수", "SELECT COUNT(*) FROM documents"),
            ("기관별 집계", "SELECT agency, COUNT(*) FROM documents WHERE agency IS NOT NULL GROUP BY agency LIMIT 5"),
            ("예산 필터", "SELECT COUNT(*) FROM documents WHERE budget != '' AND budget IS NOT NULL"),
        ]

        print(f"\n쿼리 성능:")
        total_time = 0

        for name, query in test_queries:
            start_time = time.time()
            try:
                cursor.execute(query)
                results = cursor.fetchall()
                query_time = time.time() - start_time
                total_time += query_time
                print(f"  {name}: {query_time:.3f}초 ({len(results)}건)")
            except Exception as e:
                print(f"  {name}: 오류 - {e}")

        avg_query_time = total_time / len(test_queries)
        print(f"  평균 쿼리 시간: {avg_query_time:.3f}초")

        conn.close()

        return {
            'total_indexes': len(indexes),
            'optimized_indexes': len(optimized_indexes),
            'avg_query_time': avg_query_time,
            'index_optimization_ratio': len(optimized_indexes) / len(indexes) if len(indexes) > 0 else 0
        }

    except Exception as e:
        print(f"DB 테스트 오류: {e}")
        return {}

def test_search_performance():
    """검색 성능 테스트"""
    print("\n" + "=" * 60)
    print("검색 성능 테스트")
    print("=" * 60)

    rag_system = RAGSystem(
        vector_db_path=str(VECTOR_DB_PATH),
        metadata_db_path=str(METADATA_DB_PATH),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    test_queries = [
        "시스템 구축 예산",
        "프로젝트 기간",
        "개발 인력 규모",
        "기술 요구사항",
        "국민연금공단"
    ]

    total_time = 0
    total_confidence = 0
    successful_searches = 0

    for i, query in enumerate(test_queries, 1):
        try:
            initial_memory = measure_memory_usage()
            start_time = time.time()

            result = rag_system.search_and_answer(query, top_k=3)

            search_time = time.time() - start_time
            final_memory = measure_memory_usage()
            memory_used = final_memory - initial_memory

            total_time += search_time
            total_confidence += result['confidence']
            successful_searches += 1

            print(f"  {i}. '{query}'")
            print(f"     응답시간: {search_time:.2f}초")
            print(f"     신뢰도: {result['confidence']:.3f}")
            print(f"     메모리: +{memory_used:.1f}MB")
            print(f"     참조: {len(result.get('sources', []))}개 문서")

        except Exception as e:
            print(f"  {i}. '{query}' - 실패: {e}")

    if successful_searches > 0:
        avg_time = total_time / successful_searches
        avg_confidence = total_confidence / successful_searches

        print(f"\n검색 성능 요약:")
        print(f"  성공률: {successful_searches}/{len(test_queries)}")
        print(f"  평균 응답시간: {avg_time:.2f}초")
        print(f"  평균 신뢰도: {avg_confidence:.3f}")

        return {
            'avg_response_time': avg_time,
            'avg_confidence': avg_confidence,
            'success_rate': successful_searches / len(test_queries)
        }

    return {}

def estimate_improvements():
    """예상 개선 효과 분석"""
    print("\n" + "=" * 60)
    print("2단계 개선 효과 분석")
    print("=" * 60)

    # 현재 파일 수 확인
    files_dir = Path("./files")
    if files_dir.exists():
        pdf_files = len(list(files_dir.glob("*.pdf")))
        hwp_files = len(list(files_dir.glob("*.hwp")))
        total_files = pdf_files + hwp_files

        print(f"처리 대상 파일:")
        print(f"  PDF: {pdf_files}개")
        print(f"  HWP: {hwp_files}개")
        print(f"  총합: {total_files}개")

        # 예상 개선 효과
        print(f"\n예상 성능 개선 (2단계):")
        print(f"  병렬 처리 개선:")
        print(f"    - 4개 워커 동시 처리")
        print(f"    - 예상 속도: 3-4배 향상")
        print(f"    - 대용량 파일에서 더 큰 효과")

        print(f"  벡터 인덱스 최적화:")
        print(f"    - HNSW M=16, ef_construction=200, ef_search=100")
        print(f"    - 검색 정확도 및 속도 향상")

        print(f"  메모리 스트리밍:")
        print(f"    - 페이지별 스트리밍 처리")
        print(f"    - 예상 메모리 절약: 60-80%")

        print(f"  데이터베이스 최적화:")
        print(f"    - 추가 인덱스 적용")
        print(f"    - 쿼리 속도 5-10배 향상")

def main():
    """메인 실행"""
    print("RAG 시스템 현재 성능 분석 및 개선 효과 예측")
    print("=" * 70)

    # 현재 시스템 분석
    system_stats = analyze_current_system()

    # DB 성능 테스트
    db_performance = test_database_performance()

    # 검색 성능 테스트 (데이터가 있는 경우만)
    if system_stats['vector_store'].get('total_chunks', 0) > 0:
        search_performance = test_search_performance()
    else:
        print("\n검색 테스트를 위해 먼저 문서를 처리해주세요.")
        search_performance = {}

    # 개선 효과 예측
    estimate_improvements()

    # 종합 결과
    print("\n" + "=" * 70)
    print("종합 분석 결과")
    print("=" * 70)

    print("현재 시스템 상태:")
    print(f"  처리된 문서: {system_stats['metadata_store']['total_documents']}개")
    print(f"  벡터 청크: {system_stats['vector_store'].get('total_chunks', 0)}개")

    if db_performance:
        print(f"  DB 인덱스: {db_performance['total_indexes']}개")
        print(f"  최적화 비율: {db_performance['index_optimization_ratio']:.1%}")

    if search_performance:
        print(f"  검색 성능: {search_performance['avg_response_time']:.2f}초")
        print(f"  검색 신뢰도: {search_performance['avg_confidence']:.3f}")

    print(f"\n2단계 개선으로 기대되는 효과:")
    print(f"  - 문서 처리: 3-4배 빨라짐")
    print(f"  - 메모리 사용: 60-80% 절약")
    print(f"  - 검색 정확도: 향상")
    print(f"  - DB 쿼리: 5-10배 빨라짐")

if __name__ == "__main__":
    main()