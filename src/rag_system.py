"""
메인 RAG 시스템 - 모든 컴포넌트 통합
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .processors.base import DocumentProcessor, DocumentChunk, ProcessingResult
from .processors.pdf_processor import PDFProcessor
from .processors.hwp_processor import HWPProcessor
from .storage.vector_store import VectorStore
from .storage.metadata_store import MetadataStore
from .retrieval.hybrid_retriever import HybridRetriever

class RAGSystem:
    """입찰메이트 RFP RAG 시스템"""

    def __init__(self,
                 vector_db_path: str = "./vector_db",
                 metadata_db_path: str = "rfp_metadata.db",
                 chunk_size: int = 1000,
                 overlap: int = 200):

        # 컴포넌트 초기화
        self.processors = self._init_processors(chunk_size, overlap)
        self.vector_store = VectorStore(vector_db_path)
        self.metadata_store = MetadataStore(metadata_db_path)
        self.retriever = HybridRetriever(self.vector_store, self.metadata_store)

        # RAG 설정
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.confidence_threshold = 0.3  # 할루시네이션 방지 임계값

        # OpenAI 설정 확인
        self.use_openai = self._check_openai_availability()

        print("RAG 시스템 초기화 완료")
        print(f"   - 지원 파일 형식: {list(self.processors.keys())}")
        print(f"   - OpenAI API: {'사용' if self.use_openai else '미사용'}")

    def _init_processors(self, chunk_size: int, overlap: int) -> Dict[str, DocumentProcessor]:
        """문서 처리기 초기화"""
        return {
            '.pdf': PDFProcessor(chunk_size, overlap),
            '.hwp': HWPProcessor(chunk_size, overlap)
        }

    def _check_openai_availability(self) -> bool:
        """OpenAI API 사용 가능 여부 확인"""
        try:
            import openai
            return bool(os.getenv('OPENAI_API_KEY'))
        except ImportError:
            return False

    def process_document(self, file_path: str, additional_metadata: Dict[str, Any] = None) -> ProcessingResult:
        """
        단일 문서 처리

        Args:
            file_path: 처리할 파일 경로
            additional_metadata: 추가 메타데이터

        Returns:
            ProcessingResult 객체
        """
        try:
            file_path = Path(file_path).resolve()
            extension = file_path.suffix.lower()

            # 지원 형식 확인
            if extension not in self.processors:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    error_message=f"지원하지 않는 파일 형식: {extension}"
                )

            print(f"문서 처리 시작: {file_path.name}")

            # 1. 문서 처리
            processor = self.processors[extension]
            result = processor.process_document(str(file_path), additional_metadata)

            if not result.success:
                print(f"문서 처리 실패: {result.error_message}")
                return result

            # 2. 벡터 저장소에 추가
            if result.chunks:
                vector_success = self.vector_store.add_documents(result.chunks)
                if not vector_success:
                    print("벡터 저장소 추가 실패")

                # 3. 메타데이터 저장소에 추가
                meta_success = self.metadata_store.save_document_metadata(result, str(file_path))
                if not meta_success:
                    print("메타데이터 저장 실패")

                # 4. BM25 인덱스 업데이트 (필요시)
                self._update_bm25_index()

            print(f"문서 처리 완료: {result.total_chunks}개 청크 생성")
            return result

        except Exception as e:
            return ProcessingResult(
                success=False,
                chunks=[],
                total_chunks=0,
                error_message=f"문서 처리 중 오류: {str(e)}"
            )

    def process_directory(self, directory_path: str,
                         metadata_csv_path: str = None) -> Dict[str, Any]:
        """
        디렉토리 내 모든 문서 처리

        Args:
            directory_path: 처리할 디렉토리 경로
            metadata_csv_path: 메타데이터 CSV 파일 경로

        Returns:
            처리 결과 요약
        """
        directory_path = Path(directory_path)

        # CSV 메타데이터 로드
        external_metadata = {}
        if metadata_csv_path and Path(metadata_csv_path).exists():
            imported_count = self.metadata_store.import_from_csv(metadata_csv_path)
            print(f"외부 메타데이터 가져오기: {imported_count}개 항목")

        # 처리 결과 통계
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'processing_time': 0,
            'errors': []
        }

        start_time = time.time()

        # 지원하는 파일들 찾기
        supported_files = []
        for ext in self.processors.keys():
            supported_files.extend(directory_path.glob(f"**/*{ext}"))

        results['total_files'] = len(supported_files)
        print(f"처리할 파일: {results['total_files']}개")

        # 파일별 처리
        for file_path in supported_files:
            try:
                # 외부 메타데이터 추가
                additional_meta = external_metadata.get(file_path.name, {})

                result = self.process_document(str(file_path), additional_meta)

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

        print(f"\n처리 완료:")
        print(f"   성공: {results['successful']}/{results['total_files']}")
        print(f"   총 청크: {results['total_chunks']}개")
        print(f"   처리 시간: {results['processing_time']:.1f}초")

        return results

    def search_and_answer(self, query: str,
                         search_method: str = "hybrid",
                         top_k: int = 5,
                         filters: Dict[str, Any] = None,
                         confidence_threshold: float = None) -> Dict[str, Any]:
        """
        질문 검색 및 답변 생성

        Args:
            query: 사용자 질문
            search_method: 검색 방법 ("hybrid", "vector", "keyword")
            top_k: 반환할 문서 수
            filters: 검색 필터
            confidence_threshold: 신뢰도 임계값

        Returns:
            답변 및 메타데이터
        """
        start_time = time.time()

        try:
            # 1. 검색 수행
            if search_method == "hybrid":
                search_results = self.retriever.hybrid_search(query, k=top_k, filters=filters)
            elif search_method == "vector":
                search_results = self.retriever.vector_only_search(query, k=top_k, filters=filters)
            elif search_method == "keyword":
                search_results = self.retriever.keyword_only_search(query, k=top_k)
            else:
                search_results = self.retriever.hybrid_search(query, k=top_k, filters=filters)

            # 2. 신뢰도 계산
            confidence = self._calculate_confidence(search_results, query)
            threshold = confidence_threshold or self.confidence_threshold

            # 3. 답변 생성
            if confidence < threshold:
                answer = "문서에서 관련 정보를 찾을 수 없습니다. 다른 키워드로 다시 검색해 보세요."
                answer_confidence = "low"
            else:
                answer = self._generate_answer(query, search_results)
                answer_confidence = "high" if confidence > 0.7 else "medium"

            response_time = time.time() - start_time

            # 4. 중복 제거된 소스 준비
            unique_sources = self._deduplicate_sources(search_results[:5])  # 상위 5개에서 중복 제거

            # 5. 응답 구성
            result = {
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'answer_confidence': answer_confidence,
                'search_method': search_method,
                'sources': [
                    {
                        'content_preview': r['content'][:200] + "...",
                        'file_name': r['metadata'].get('file_name', 'Unknown'),
                        'agency': r['metadata'].get('agency', 'Unknown'),
                        'score': r.get('hybrid_score', r.get('vector_score', 0))
                    }
                    for r in unique_sources[:3]  # 중복 제거된 상위 3개
                ],
                'response_time': response_time,
                'total_results': len(search_results)
            }

            # 5. 검색 로그 기록
            self.metadata_store.log_search(
                query=query,
                search_method=search_method,
                results_count=len(search_results),
                confidence_score=confidence,
                response_time=response_time
            )

            return result

        except Exception as e:
            return {
                'query': query,
                'answer': f"검색 중 오류가 발생했습니다: {str(e)}",
                'confidence': 0.0,
                'answer_confidence': "error",
                'search_method': search_method,
                'sources': [],
                'response_time': time.time() - start_time,
                'total_results': 0
            }

    def _generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """검색 결과를 바탕으로 답변 생성"""
        if not search_results:
            return "관련 문서를 찾을 수 없습니다."

        # 중복 제거된 결과 사용
        unique_results = self._deduplicate_sources(search_results[:5])
        context_chunks = [result['content'] for result in unique_results[:3]]
        context = "\n\n".join(context_chunks)

        if self.use_openai:
            return self._generate_openai_answer(query, context)
        else:
            return self._generate_template_answer(query, context_chunks)

    def _generate_openai_answer(self, query: str, context: str) -> str:
        """OpenAI API를 사용한 답변 생성"""
        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            # 디버깅용 로그
            prompt = f"""다음은 RFP(제안요청서) 문서의 관련 내용입니다:

{context}

질문: {query}

위 문서 내용을 바탕으로 정확하고 구체적으로 답변해주세요.
관련 정보가 있다면 구체적으로 설명하고, 정말 관련 정보가 없을 때만 '문서에서 확인할 수 없습니다'라고 답변하세요.

답변:"""

            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4o 모델 사용 (안정적)
                messages=[
                    {"role": "system", "content": "당신은 RFP 문서 분석 전문가입니다. 주어진 문서 내용을 자세히 분석하여 관련 정보를 찾아 답변합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            return f"답변 생성 중 오류: {str(e)}"

    def _deduplicate_sources(self, search_results: List[Dict]) -> List[Dict]:
        """검색 결과에서 중복 파일 제거"""
        seen_files = set()
        unique_results = []

        for result in search_results:
            file_name = result['metadata'].get('file_name', 'Unknown')

            # 파일명 기준으로 중복 체크
            if file_name not in seen_files:
                seen_files.add(file_name)
                unique_results.append(result)

            # 최대 5개까지만
            if len(unique_results) >= 5:
                break

        return unique_results

    def _generate_template_answer(self, query: str, chunks: List[str]) -> str:
        """템플릿 기반 답변 생성 (OpenAI 없을 때)"""
        if not chunks:
            return "관련 정보를 찾을 수 없습니다."

        # 간단한 키워드 매칭 기반 답변
        relevant_text = chunks[0][:500]  # 첫 번째 청크의 앞부분

        return f"문서에서 다음과 같은 관련 내용을 찾았습니다:\n\n{relevant_text}..."

    def _calculate_confidence(self, search_results: List[Dict], query: str) -> float:
        """검색 결과의 신뢰도 계산"""
        if not search_results:
            return 0.0

        # 간단한 신뢰도 계산 (실제로는 더 정교하게 구현)
        scores = []
        for result in search_results:
            score = result.get('hybrid_score', result.get('vector_score', 0))
            scores.append(score)

        if scores:
            # 상위 3개 결과의 평균 점수
            top_scores = sorted(scores, reverse=True)[:3]
            confidence = sum(top_scores) / len(top_scores)
            return min(confidence, 1.0)  # 최대 1.0으로 제한

        return 0.0

    def _update_bm25_index(self):
        """BM25 인덱스 업데이트 (전체 청크 기반)"""
        try:
            # 현재는 간단히 구현, 실제로는 증분 업데이트 필요
            pass
        except Exception as e:
            print(f"BM25 인덱스 업데이트 오류: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        vector_stats = self.vector_store.get_collection_stats()
        metadata_stats = self.metadata_store.get_statistics()
        retriever_stats = self.retriever.get_search_stats()

        return {
            'vector_store': vector_stats,
            'metadata_store': metadata_stats,
            'retriever': retriever_stats,
            'processors': list(self.processors.keys()),
            'openai_enabled': self.use_openai
        }

    def rebuild_search_index(self):
        """검색 인덱스 재구축"""
        try:
            print("검색 인덱스 재구축 시작...")

            # 모든 청크 조회
            stats = self.metadata_store.get_statistics()
            print(f"총 {stats.get('total_chunks', 0)}개 청크 발견")

            # BM25 인덱스 재구축 (필요시 구현)
            print("검색 인덱스 재구축 완료")

        except Exception as e:
            print(f"인덱스 재구축 오류: {e}")

    def export_data(self, output_dir: str = "./exports"):
        """데이터 내보내기"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # 메타데이터 CSV 내보내기
            csv_path = output_path / "processed_documents.csv"
            self.metadata_store.export_to_csv(str(csv_path))

            # 데이터베이스 백업
            backup_path = output_path / f"rfp_metadata_backup_{int(time.time())}.db"
            self.metadata_store.backup_database(str(backup_path))

            print(f"데이터 내보내기 완료: {output_dir}")

        except Exception as e:
            print(f"데이터 내보내기 오류: {e}")