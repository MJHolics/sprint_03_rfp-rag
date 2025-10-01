"""
대시보드 기능 테스트
"""
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_performance_data_loading():
    """성능 데이터 로딩 테스트"""
    print("성능 데이터 로딩 테스트")
    print("=" * 50)

    try:
        # comprehensive_performance_data.json 로딩 테스트
        with open('comprehensive_performance_data.json', 'r', encoding='utf-8') as f:
            comp_data = json.load(f)

        print("✓ comprehensive_performance_data.json 로딩 성공")

        # 필수 키 확인
        required_keys = ['baseline', 'stage1', 'current_stage2', 'stage3', 'graph_data']
        for key in required_keys:
            if key in comp_data:
                print(f"✓ {key} 데이터 존재")
            else:
                print(f"✗ {key} 데이터 누락")
                return False

        # graph_data 키 확인
        graph_data = comp_data['graph_data']
        graph_keys = ['stages', 'search_times', 'query_times', 'files_per_minute', 'concurrent_users', 'cache_hit_rates']
        for key in graph_keys:
            if key in graph_data:
                print(f"✓ graph_data.{key} 존재: {len(graph_data[key])}개 항목")
            else:
                print(f"✗ graph_data.{key} 누락")
                return False

        # 데이터 미리보기
        print("\n성능 데이터 미리보기:")
        print(f"단계: {graph_data['stages']}")
        print(f"검색 시간: {[f'{t:.2f}초' for t in graph_data['search_times']]}")
        print(f"쿼리 시간: {[f'{t:.3f}ms' for t in graph_data['query_times']]}")
        print(f"처리 속도: {[f'{f:.0f} 파일/분' for f in graph_data['files_per_minute']]}")
        print(f"동시 사용자: {graph_data['concurrent_users']}")
        print(f"캐시 히트율: {[f'{c}%' for c in graph_data['cache_hit_rates']]}")

        return True

    except FileNotFoundError:
        print("✗ comprehensive_performance_data.json 파일이 없습니다.")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ JSON 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"✗ 예상치 못한 오류: {e}")
        return False

def test_dashboard_imports():
    """대시보드 모듈 임포트 테스트"""
    print("\n대시보드 모듈 임포트 테스트")
    print("=" * 50)

    try:
        from src.rag_system import RAGSystem
        print("✓ RAGSystem 임포트 성공")

        from config.settings import VECTOR_DB_PATH, METADATA_DB_PATH
        print("✓ 설정 파일 임포트 성공")

        import plotly.graph_objects as go
        print("✓ Plotly 임포트 성공")

        import pandas as pd
        print("✓ Pandas 임포트 성공")

        return True

    except ImportError as e:
        print(f"✗ 임포트 오류: {e}")
        return False
    except Exception as e:
        print(f"✗ 예상치 못한 오류: {e}")
        return False

def test_rag_system_initialization():
    """RAG 시스템 초기화 테스트"""
    print("\nRAG 시스템 초기화 테스트")
    print("=" * 50)

    try:
        from src.rag_system import RAGSystem
        from config.settings import VECTOR_DB_PATH, METADATA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

        rag_system = RAGSystem(
            vector_db_path=str(VECTOR_DB_PATH),
            metadata_db_path=str(METADATA_DB_PATH),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        print("✓ RAG 시스템 초기화 성공")

        # 시스템 통계 확인
        stats = rag_system.get_system_stats()
        print(f"✓ 시스템 통계 조회 성공")
        print(f"  - 총 문서: {stats['metadata_store']['total_documents']}개")
        print(f"  - 총 청크: {stats['vector_store'].get('total_chunks', 0)}개")
        print(f"  - OpenAI 활성화: {stats.get('openai_enabled', False)}")

        return True

    except Exception as e:
        print(f"✗ RAG 시스템 초기화 오류: {e}")
        return False

def main():
    """전체 테스트 실행"""
    print("대시보드 통합 테스트")
    print("=" * 80)

    tests = [
        test_performance_data_loading,
        test_dashboard_imports,
        test_rag_system_initialization
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ 테스트 통과\n")
            else:
                print("✗ 테스트 실패\n")
        except Exception as e:
            print(f"✗ 테스트 중 오류: {e}\n")

    print("=" * 80)
    print(f"테스트 결과: {passed}/{total} 통과")

    if passed == total:
        print("🎉 모든 테스트 통과! 대시보드가 정상적으로 작동할 수 있습니다.")
        print("\n대시보드 접속 정보:")
        print("- 메인 대시보드: http://localhost:8503")
        print("- 성능 비교 페이지: 사이드바에서 '성능 비교 분석' 선택")
        print("\n주요 기능:")
        print("- 4단계 성능 변화 추이 (패치 전 → 1단계 → 2단계 → 3단계)")
        print("- 실제 측정 데이터 기반 그래프")
        print("- 검색 시간, DB 쿼리, 캐시 히트율 등 상세 지표")
        return True
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 대시보드에 문제가 있을 수 있습니다.")
        return False

if __name__ == "__main__":
    main()