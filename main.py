"""
RFP RAG 시스템 메인 실행 파일
"""
import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from config.settings import *

def main():
    parser = argparse.ArgumentParser(description="RFP RAG 시스템")
    parser.add_argument('--mode', choices=['process', 'serve', 'evaluate', 'stats'],
                       required=True, help="실행 모드")
    parser.add_argument('--data_path', default='./files', help="데이터 디렉토리 경로")
    parser.add_argument('--metadata_csv', default='./data_list.csv', help="메타데이터 CSV 파일")
    parser.add_argument('--rebuild_index', action='store_true', help="검색 인덱스 재구축")

    args = parser.parse_args()

    # RAG 시스템 초기화
    print("RFP RAG 시스템 시작")
    rag_system = RAGSystem(
        vector_db_path=str(VECTOR_DB_PATH),
        metadata_db_path=str(METADATA_DB_PATH),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    if args.mode == 'process':
        process_documents(rag_system, args)
    elif args.mode == 'serve':
        serve_interactive(rag_system, args)
    elif args.mode == 'evaluate':
        evaluate_system(rag_system, args)
    elif args.mode == 'stats':
        show_statistics(rag_system)

def process_documents(rag_system: RAGSystem, args):
    """문서 처리 모드"""
    print(f"\n문서 처리 시작: {args.data_path}")

    if args.rebuild_index:
        print("검색 인덱스 재구축...")
        rag_system.rebuild_search_index()

    # 디렉토리 처리
    if Path(args.data_path).is_dir():
        results = rag_system.process_directory(
            args.data_path,
            args.metadata_csv if Path(args.metadata_csv).exists() else None
        )

        print(f"\n처리 결과:")
        print(f"   총 파일: {results['total_files']}개")
        print(f"   성공: {results['successful']}개")
        print(f"   실패: {results['failed']}개")
        print(f"   총 청크: {results['total_chunks']}개")
        print(f"   처리 시간: {results['processing_time']:.1f}초")

        if results['errors']:
            print(f"\n오류 목록:")
            for error in results['errors'][:5]:  # 최대 5개만 표시
                print(f"   {error['file']}: {error['error']}")

    else:
        # 단일 파일 처리
        result = rag_system.process_document(args.data_path)
        if result.success:
            print(f"처리 완료: {result.total_chunks}개 청크")
        else:
            print(f"처리 실패: {result.error_message}")

def serve_interactive(rag_system: RAGSystem, args):
    """대화형 서비스 모드"""
    print("\nRFP RAG 시스템 대화 모드")
    print("질문을 입력하세요 (종료: 'quit', 통계: 'stats', 도움말: 'help')")

    while True:
        try:
            query = input("\n질문: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', '종료']:
                print("시스템을 종료합니다.")
                break

            elif query.lower() == 'stats':
                show_statistics(rag_system)
                continue

            elif query.lower() == 'help':
                show_help()
                continue

            # 검색 및 답변
            print("검색 중...")
            result = rag_system.search_and_answer(
                query=query,
                search_method="hybrid",
                top_k=DEFAULT_TOP_K
            )

            print(f"\n답변 (신뢰도: {result['confidence']:.3f}):")
            print(f"   {result['answer']}")

            if result['sources']:
                print(f"\n참조 문서:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source['file_name']} ({source['agency']})")
                    print(f"      {source['content_preview']}")

            print(f"\n응답 시간: {result['response_time']:.2f}초")

        except KeyboardInterrupt:
            print("\n시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")

def evaluate_system(rag_system: RAGSystem, args):
    """시스템 평가 모드"""
    print("\n시스템 성능 평가")

    # 테스트 질문들
    test_questions = [
        "시스템 구축 예산은 얼마인가요?",
        "프로젝트 기간은 어떻게 되나요?",
        "필요한 개발 인력은 몇 명인가요?",
        "주요 기술 요구사항을 알려주세요",
        "국민연금공단 관련 사업이 있나요?"
    ]

    total_time = 0
    total_confidence = 0

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")

        result = rag_system.search_and_answer(question)
        total_time += result['response_time']
        total_confidence += result['confidence']

        print(f"   신뢰도: {result['confidence']:.3f}")
        print(f"   응답 시간: {result['response_time']:.2f}초")
        print(f"   답변: {result['answer'][:100]}...")

    print(f"\n전체 평가 결과:")
    print(f"   평균 신뢰도: {total_confidence/len(test_questions):.3f}")
    print(f"   평균 응답 시간: {total_time/len(test_questions):.2f}초")

def show_statistics(rag_system: RAGSystem):
    """시스템 통계 표시"""
    print("\n시스템 통계")
    stats = rag_system.get_system_stats()

    print(f"   벡터 저장소: {stats['vector_store']['total_documents']}개 문서")
    print(f"   메타데이터: {stats['metadata_store']['total_documents']}개 문서")
    print(f"   총 청크: {stats['metadata_store']['total_chunks']}개")
    print(f"   지원 형식: {', '.join(stats['processors'])}")
    print(f"   OpenAI: {'활성화' if stats['openai_enabled'] else '비활성화'}")

    if 'top_agencies' in stats['metadata_store']:
        print(f"\n주요 발주기관:")
        for agency, count in list(stats['metadata_store']['top_agencies'].items())[:5]:
            if agency:
                print(f"   {agency}: {count}개")

def show_help():
    """도움말 표시"""
    print("""
사용 가능한 명령어:
   - quit/exit/종료: 시스템 종료
   - stats: 시스템 통계 표시
   - help: 이 도움말 표시

검색 팁:
   - 구체적인 키워드 사용 (예: "시스템 구축", "예산", "기간")
   - 발주기관명 포함 (예: "국민연금공단 시스템")
   - 여러 키워드 조합 (예: "웹 개발 예산")

예시 질문:
   - "시스템 구축 예산은 얼마인가요?"
   - "프로젝트 기간은 어떻게 되나요?"
   - "국민연금공단 관련 사업을 찾아주세요"
    """)

if __name__ == "__main__":
    main()