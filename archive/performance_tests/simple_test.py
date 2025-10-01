"""
간단한 대시보드 테스트
"""
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_loading():
    """데이터 로딩 테스트"""
    print("성능 데이터 로딩 테스트")
    print("=" * 50)

    try:
        with open('comprehensive_performance_data.json', 'r', encoding='utf-8') as f:
            comp_data = json.load(f)

        print("OK: comprehensive_performance_data.json 로딩 성공")

        # 필수 키 확인
        if 'graph_data' in comp_data:
            graph_data = comp_data['graph_data']
            print("OK: graph_data 존재")

            print(f"단계: {graph_data['stages']}")
            print(f"검색 시간: {[f'{t:.2f}초' for t in graph_data['search_times']]}")
            print(f"처리 속도: {[f'{f:.0f} 파일/분' for f in graph_data['files_per_minute']]}")
            print(f"동시 사용자: {graph_data['concurrent_users']}")
            return True
        else:
            print("ERROR: graph_data 누락")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_rag_system():
    """RAG 시스템 테스트"""
    print("\nRAG 시스템 테스트")
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
        print("OK: RAG 시스템 초기화 성공")

        stats = rag_system.get_system_stats()
        print(f"OK: 총 문서 {stats['metadata_store']['total_documents']}개")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """전체 테스트"""
    print("대시보드 통합 테스트")
    print("=" * 80)

    tests = [test_data_loading, test_rag_system]
    passed = 0

    for test in tests:
        if test():
            passed += 1
            print("PASS\n")
        else:
            print("FAIL\n")

    print("=" * 80)
    print(f"결과: {passed}/{len(tests)} 통과")

    if passed == len(tests):
        print("성공: 모든 테스트 통과")
        print("\n대시보드 접속:")
        print("- URL: http://localhost:8503")
        print("- 사이드바에서 '성능 비교 분석' 선택")
        print("- 4단계 성능 개선 효과 확인 가능")
    else:
        print("실패: 일부 테스트 실패")

if __name__ == "__main__":
    main()