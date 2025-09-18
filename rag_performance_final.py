"""
RAG 시스템 성능 테스트 (최종 한글 지원 버전)
"""
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# matplotlib 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 확실한 한글 폰트 설정
def setup_font():
    # Windows Malgun Gothic 강제 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

setup_font()

# 프로젝트 루트 추가
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
from config.settings import *

class RAGPerformanceTester:
    """RAG 시스템 성능 테스트"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.results_dir = Path("performance_results")
        self.results_dir.mkdir(exist_ok=True)

        # 테스트 질문 세트
        self.test_queries = [
            "시스템 구축 예산은 얼마인가요?",
            "프로젝트 기간은 어떻게 되나요?",
            "개발 인력은 몇 명이 필요한가요?",
            "주요 기술 요구사항을 알려주세요",
            "국민연금공단 관련 사업이 있나요?"
        ]

    def run_quick_test(self) -> Dict[str, Any]:
        """빠른 성능 테스트"""
        print("RAG 성능 테스트 시작...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'query_results': []
        }

        # 간단한 테스트
        for i, query in enumerate(self.test_queries, 1):
            print(f"   테스트 {i}/{len(self.test_queries)}: {query}")

            start_time = time.time()
            result = self.rag_system.search_and_answer(query, search_method="hybrid", top_k=3)
            response_time = time.time() - start_time

            results['query_results'].append({
                'query': query,
                'response_time': response_time,
                'confidence': result.get('confidence', 0.0),
                'sources_count': len(result.get('sources', [])),
                'answer_length': len(result.get('answer', ''))
            })

        return results

    def create_simple_chart(self, results: Dict[str, Any]):
        """간단한 성능 차트"""
        print("성능 차트 생성 중...")

        df = pd.DataFrame(results['query_results'])

        # 2x2 레이아웃
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 응답 시간
        axes[0, 0].bar(range(len(df)), df['response_time'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('응답 시간')
        axes[0, 0].set_ylabel('시간(초)')
        axes[0, 0].set_xlabel('쿼리 번호')

        # 2. 신뢰도
        axes[0, 1].bar(range(len(df)), df['confidence'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('신뢰도 점수')
        axes[0, 1].set_ylabel('신뢰도')
        axes[0, 1].set_xlabel('쿼리 번호')

        # 3. 소스 개수
        axes[1, 0].bar(range(len(df)), df['sources_count'], color='orange', alpha=0.7)
        axes[1, 0].set_title('검색된 소스 개수')
        axes[1, 0].set_ylabel('개수')
        axes[1, 0].set_xlabel('쿼리 번호')

        # 4. 응답 시간 vs 신뢰도
        axes[1, 1].scatter(df['response_time'], df['confidence'], color='red', alpha=0.7, s=60)
        axes[1, 1].set_title('응답시간 vs 신뢰도')
        axes[1, 1].set_xlabel('응답시간(초)')
        axes[1, 1].set_ylabel('신뢰도')

        plt.tight_layout()

        # 저장
        chart_path = self.results_dir / f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        print(f"차트 저장: {chart_path}")

        plt.show()

        # 기본 통계 출력
        print("\n=== 성능 요약 ===")
        print(f"평균 응답시간: {df['response_time'].mean():.3f}초")
        print(f"평균 신뢰도: {df['confidence'].mean():.3f}")
        print(f"평균 소스 개수: {df['sources_count'].mean():.1f}개")

    def save_results(self, results: Dict[str, Any]):
        """결과 저장"""
        results_path = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {results_path}")

def main():
    """메인 실행"""
    print("=== RAG 성능 테스트 ===")

    # RAG 시스템 초기화
    rag_system = RAGSystem(
        vector_db_path=str(VECTOR_DB_PATH),
        metadata_db_path=str(METADATA_DB_PATH),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    # 테스터 생성 및 실행
    tester = RAGPerformanceTester(rag_system)
    results = tester.run_quick_test()
    tester.create_simple_chart(results)
    tester.save_results(results)

    print("\n테스트 완료!")

if __name__ == "__main__":
    main()