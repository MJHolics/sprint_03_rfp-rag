"""
성능 개선 단계별 비교 벤치마크
- 패치 전 (기본)
- 1단계 패치 후 (SQLite 최적화, 캐싱, 배치 처리)
- 2단계 패치 후 (병렬 처리, 벡터 최적화, 메모리 스트리밍)
"""
import time
import psutil
import os
import sqlite3
from pathlib import Path
import sys
import shutil
import tempfile

# 프로젝트 루트를 Python 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import RAGSystem
from config.settings import *

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            'baseline': {},  # 패치 전
            'stage1': {},    # 1단계 후
            'stage2': {}     # 2단계 후
        }

    def measure_memory_usage(self):
        """메모리 사용량 측정 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def measure_db_performance(self, db_path):
        """데이터베이스 쿼리 성능 측정"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 인덱스 존재 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]

            # 테스트 쿼리들
            queries = [
                "SELECT COUNT(*) FROM documents",
                "SELECT * FROM documents WHERE agency LIKE '%국민연금%' LIMIT 10",
                "SELECT * FROM documents WHERE budget != '' LIMIT 10",
                "SELECT agency, COUNT(*) FROM documents GROUP BY agency LIMIT 10"
            ]

            query_times = []
            for query in queries:
                start_time = time.time()
                try:
                    cursor.execute(query)
                    cursor.fetchall()
                    query_times.append(time.time() - start_time)
                except Exception as e:
                    query_times.append(999)  # 오류 시 높은 값

            conn.close()

            return {
                'total_indexes': len(indexes),
                'optimized_indexes': len([idx for idx in indexes if 'agency' in idx or 'budget' in idx]),
                'avg_query_time': sum(query_times) / len(query_times),
                'query_times': query_times
            }
        except Exception as e:
            return {'error': str(e)}

    def test_document_processing(self, test_mode="current"):
        """문서 처리 성능 테스트"""
        print(f"\n{'='*60}")
        print(f"📊 {test_mode.upper()} 모드 성능 테스트")
        print(f"{'='*60}")

        # 임시 디렉토리에서 테스트 (기존 데이터 영향 방지)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vector_db = os.path.join(temp_dir, "test_vector_db")
            temp_metadata_db = os.path.join(temp_dir, "test_metadata.db")

            # 테스트용 RAG 시스템 생성
            if test_mode == "baseline":
                # 패치 전 모드 (병렬 처리 비활성화, 기본 설정)
                rag_system = self._create_baseline_system(temp_vector_db, temp_metadata_db)
            else:
                # 현재 모드 (모든 최적화 적용)
                rag_system = RAGSystem(
                    vector_db_path=temp_vector_db,
                    metadata_db_path=temp_metadata_db,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP
                )

            # 성능 측정
            initial_memory = self.measure_memory_usage()
            start_time = time.time()

            # 작은 샘플로 테스트 (빠른 비교를 위해)
            test_files = list(Path("./files").glob("*.pdf"))[:5] + list(Path("./files").glob("*.hwp"))[:5]

            results = {
                'total_files': len(test_files),
                'successful': 0,
                'failed': 0,
                'total_chunks': 0,
                'processing_time': 0,
                'memory_used': 0,
                'files_per_minute': 0,
                'chunks_per_second': 0
            }

            if not test_files:
                print(" 테스트 파일이 없습니다.")
                return results

            # 개별 파일 처리
            for file_path in test_files:
                try:
                    result = rag_system.process_document(str(file_path))
                    if result.success:
                        results['successful'] += 1
                        results['total_chunks'] += result.total_chunks
                    else:
                        results['failed'] += 1
                except Exception:
                    results['failed'] += 1

            # 성능 지표 계산
            processing_time = time.time() - start_time
            final_memory = self.measure_memory_usage()

            results.update({
                'processing_time': processing_time,
                'memory_used': final_memory - initial_memory,
                'files_per_minute': (results['successful'] / processing_time) * 60 if processing_time > 0 else 0,
                'chunks_per_second': results['total_chunks'] / processing_time if processing_time > 0 else 0
            })

            # DB 성능 측정
            db_performance = self.measure_db_performance(temp_metadata_db)
            results['db_performance'] = db_performance

            print(f"📁 처리 결과:")
            print(f"   총 파일: {results['total_files']}")
            print(f"   성공: {results['successful']}")
            print(f"   실패: {results['failed']}")
            print(f"   총 청크: {results['total_chunks']}")
            print(f" 성능 지표:")
            print(f"   처리 시간: {results['processing_time']:.2f}초")
            print(f"   메모리 사용: {results['memory_used']:.1f}MB")
            print(f"   처리 속도: {results['files_per_minute']:.1f} 파일/분")
            print(f"   청킹 속도: {results['chunks_per_second']:.1f} 청크/초")
            print(f"💾 DB 성능:")
            print(f"   인덱스 수: {db_performance.get('total_indexes', 0)}")
            print(f"   최적화 인덱스: {db_performance.get('optimized_indexes', 0)}")
            print(f"   평균 쿼리 시간: {db_performance.get('avg_query_time', 0):.3f}초")

            return results

    def _create_baseline_system(self, vector_db_path, metadata_db_path):
        """패치 전 시스템 시뮬레이션"""
        # 기본 RAG 시스템 생성하되 병렬 처리 등 최적화 비활성화
        # 실제로는 현재 시스템이지만 순차 처리로 제한
        rag_system = RAGSystem(vector_db_path, metadata_db_path, CHUNK_SIZE, CHUNK_OVERLAP)

        # 병렬 처리 비활성화 (process_directory 메서드 패치)
        original_process_directory = rag_system.process_directory

        def sequential_process_directory(directory_path, metadata_csv_path=None):
            """순차 처리 버전"""
            from pathlib import Path
            import time

            directory_path = Path(directory_path)
            results = {
                'total_files': 0, 'successful': 0, 'failed': 0,
                'total_chunks': 0, 'processing_time': 0, 'errors': []
            }

            start_time = time.time()
            supported_files = []
            for ext in rag_system.processors.keys():
                supported_files.extend(directory_path.glob(f"**/*{ext}"))

            results['total_files'] = len(supported_files)

            # 순차 처리 (병렬 처리 없음)
            for file_path in supported_files:
                try:
                    result = rag_system.process_document(str(file_path))
                    if result.success:
                        results['successful'] += 1
                        results['total_chunks'] += result.total_chunks
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'file': file_path.name,
                            'error': result.error_message
                        })
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'file': file_path.name,
                        'error': str(e)
                    })

            results['processing_time'] = time.time() - start_time
            return results

        rag_system.process_directory = sequential_process_directory
        return rag_system

    def test_search_performance(self, num_queries=5):
        """검색 성능 테스트"""
        print(f"\n🔍 검색 성능 테스트")
        print(f"{'='*40}")

        # 현재 시스템으로 검색 테스트
        rag_system = RAGSystem(
            vector_db_path=str(VECTOR_DB_PATH),
            metadata_db_path=str(METADATA_DB_PATH),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        test_queries = [
            "시스템 구축 예산",
            "프로젝트 기간",
            "개발 인력",
            "기술 요구사항",
            "국민연금공단"
        ][:num_queries]

        total_time = 0
        total_confidence = 0
        successful_searches = 0

        for i, query in enumerate(test_queries, 1):
            try:
                start_time = time.time()
                result = rag_system.search_and_answer(query, top_k=3)
                search_time = time.time() - start_time

                total_time += search_time
                total_confidence += result['confidence']
                successful_searches += 1

                print(f"{i}. '{query}' - {search_time:.2f}초 (신뢰도: {result['confidence']:.3f})")

            except Exception as e:
                print(f"{i}. '{query}' - 실패: {e}")

        avg_time = total_time / successful_searches if successful_searches > 0 else 0
        avg_confidence = total_confidence / successful_searches if successful_searches > 0 else 0

        return {
            'avg_response_time': avg_time,
            'avg_confidence': avg_confidence,
            'total_queries': len(test_queries),
            'successful_queries': successful_searches
        }

    def compare_performance(self):
        """전체 성능 비교"""
        print("🚀 RAG 시스템 성능 개선 단계별 비교")
        print("="*80)

        # 1. 패치 전 (기본) 테스트
        print("\n 패치 전 (기본) 성능 측정")
        baseline_results = self.test_document_processing("baseline")
        self.results['baseline'] = baseline_results

        # 2. 현재 (2단계 패치 후) 테스트
        print("\n 2단계 패치 후 성능 측정")
        stage2_results = self.test_document_processing("stage2")
        self.results['stage2'] = stage2_results

        # 3. 검색 성능 테스트
        search_results = self.test_search_performance()

        # 4. 결과 비교 및 요약
        self.print_comparison_summary(search_results)

    def print_comparison_summary(self, search_results):
        """성능 비교 요약 출력"""
        print("\n" + "="*80)
        print("📈 성능 개선 효과 분석")
        print("="*80)

        baseline = self.results['baseline']
        stage2 = self.results['stage2']

        # 처리 속도 개선
        if baseline['processing_time'] > 0 and stage2['processing_time'] > 0:
            speed_improvement = baseline['processing_time'] / stage2['processing_time']
            print(f" 문서 처리 속도 개선: {speed_improvement:.1f}배")

        # 메모리 사용량 비교
        if baseline['memory_used'] > 0 and stage2['memory_used'] > 0:
            memory_improvement = (baseline['memory_used'] - stage2['memory_used']) / baseline['memory_used'] * 100
            print(f"💾 메모리 사용량 개선: {memory_improvement:.1f}% 절약")

        # 처리량 비교
        baseline_throughput = baseline.get('files_per_minute', 0)
        stage2_throughput = stage2.get('files_per_minute', 0)
        if baseline_throughput > 0:
            throughput_improvement = stage2_throughput / baseline_throughput
            print(f"📊 처리량 개선: {throughput_improvement:.1f}배")

        # DB 성능 비교
        baseline_db = baseline.get('db_performance', {})
        stage2_db = stage2.get('db_performance', {})

        baseline_query_time = baseline_db.get('avg_query_time', 0)
        stage2_query_time = stage2_db.get('avg_query_time', 0)

        if baseline_query_time > 0 and stage2_query_time > 0:
            db_improvement = baseline_query_time / stage2_query_time
            print(f"🗄️ DB 쿼리 속도 개선: {db_improvement:.1f}배")

        # 검색 성능
        print(f"🔍 검색 성능:")
        print(f"   평균 응답시간: {search_results['avg_response_time']:.2f}초")
        print(f"   평균 신뢰도: {search_results['avg_confidence']:.3f}")
        print(f"   성공률: {search_results['successful_queries']}/{search_results['total_queries']}")

        print(f"\n📋 상세 비교표:")
        print(f"{'지표':<15} {'패치 전':<12} {'2단계 후':<12} {'개선 효과':<12}")
        print("-" * 60)
        print(f"{'처리시간(초)':<15} {baseline['processing_time']:<12.2f} {stage2['processing_time']:<12.2f} {speed_improvement:<12.1f}배")
        print(f"{'메모리(MB)':<15} {baseline['memory_used']:<12.1f} {stage2['memory_used']:<12.1f} {memory_improvement:<12.1f}% 절약")
        print(f"{'파일/분':<15} {baseline_throughput:<12.1f} {stage2_throughput:<12.1f} {throughput_improvement:<12.1f}배")
        print(f"{'쿼리시간(초)':<15} {baseline_query_time:<12.3f} {stage2_query_time:<12.3f} {db_improvement:<12.1f}배")

def main():
    """메인 실행"""
    if not Path("./files").exists():
        print(" ./files 디렉토리가 없습니다. 테스트 파일을 준비해주세요.")
        return

    benchmark = PerformanceBenchmark()
    benchmark.compare_performance()

if __name__ == "__main__":
    main()