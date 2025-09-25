"""
Vector Database 저장소 - ChromaDB 기반
OpenAI 임베딩 지원
"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ..processors.base import DocumentChunk

class VectorStore:
    """ChromaDB 기반 벡터 저장소"""

    def __init__(self, persist_directory: str = "./vector_db", collection_name: str = "rfp_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_chromadb()

    def _init_chromadb(self):
        """ChromaDB 초기화"""
        try:
            # Persistent client 설정
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # OpenAI 사용 여부 확인 (임시 체크)
            use_openai = bool(os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'YOUR_API_KEY_HERE')

            # 컬렉션 생성 또는 로드
            try:
                self.collection = self.client.get_collection(name=self.collection_name)

                # 기존 컬렉션이 있지만 임베딩 차원이 맞지 않는 경우 처리
                if use_openai:
                    # OpenAI 임베딩 사용 시 기존 컬렉션 확인
                    try:
                        # 테스트용 더미 데이터로 차원 확인
                        test_embedding = [0.0] * 1536  # OpenAI text-embedding-3-small 차원
                        self.collection.add(
                            embeddings=[test_embedding],
                            documents=["test"],
                            ids=["test_dimension_check"]
                        )
                        # 성공하면 테스트 데이터 삭제
                        self.collection.delete(ids=["test_dimension_check"])
                        print(f"기존 ChromaDB 컬렉션 로드됨 (OpenAI 호환): {self.collection_name}")
                    except Exception as dim_error:
                        print(f"기존 컬렉션이 OpenAI 임베딩과 호환되지 않습니다. 새 컬렉션을 생성합니다.")
                        # 기존 컬렉션 삭제 후 재생성
                        self._recreate_collection_for_openai()
                else:
                    print(f"기존 ChromaDB 컬렉션 로드됨 (기본 임베딩): {self.collection_name}")

            except:
                # 컬렉션이 없는 경우 새로 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"새 ChromaDB 컬렉션 생성됨: {self.collection_name}")

        except Exception as e:
            print(f"ChromaDB 초기화 오류: {e}")
            raise

    def _recreate_collection_for_openai(self):
        """OpenAI 임베딩을 위한 컬렉션 재생성"""
        try:
            # 기존 컬렉션 삭제
            self.client.delete_collection(name=self.collection_name)
            print(f"기존 컬렉션 삭제됨: {self.collection_name}")

            # 새 컬렉션 생성
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"OpenAI 호환 컬렉션 생성됨: {self.collection_name}")

        except Exception as e:
            print(f"컬렉션 재생성 오류: {e}")
            raise

    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """문서 청크들을 벡터스토어에 추가"""
        if not chunks:
            return False

        try:
            # OpenAI 임베딩 사용 여부 확인
            use_openai = self._check_openai_availability()

            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            if use_openai:
                print("OpenAI 임베딩으로 추가 중...")
                embeddings = self._create_openai_embeddings(texts)
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                print("ChromaDB 기본 임베딩으로 추가 중...")
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

            print(f"벡터 저장소에 {len(chunks)}개 청크 추가 완료")
            return True

        except Exception as e:
            print(f"벡터 저장소 추가 오류: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """유사도 기반 검색"""
        try:
            # 쿼리 임베딩 생성
            use_openai = self._check_openai_availability()

            if use_openai:
                query_embedding = self._create_openai_embeddings([query])[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=filters
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=filters
                )

            # 결과 정리
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    # 거리를 유사도 점수로 변환 (코사인 거리 -> 유사도)
                    score = 1.0 - distance if distance <= 1.0 else 0.0

                    search_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i],
                        'distance': distance,
                        'score': score
                    })

            return search_results

        except Exception as e:
            print(f"벡터 검색 오류: {e}")
            return []

    def _create_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API로 임베딩 생성"""
        try:
            from openai import OpenAI

            # OpenAI 클라이언트 초기화
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            embeddings = []
            batch_size = 100  # API 제한 고려

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                if len(texts) > batch_size:
                    print(f"임베딩 생성 진행: {len(embeddings)}/{len(texts)}")

            return embeddings

        except Exception as e:
            print(f"OpenAI 임베딩 생성 오류: {e}")
            raise

    def _check_openai_availability(self) -> bool:
        """OpenAI API 사용 가능 여부 확인"""
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            return bool(api_key and api_key != 'YOUR_API_KEY_HERE')
        except ImportError:
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보"""
        try:
            total_chunks = self.collection.count()

            # 고유 파일 수 계산
            unique_files = set()
            if total_chunks > 0:
                # 모든 메타데이터에서 파일명 추출
                results = self.collection.get(include=['metadatas'])
                for metadata in results['metadatas']:
                    if metadata and 'file_name' in metadata:
                        unique_files.add(metadata['file_name'])

            return {
                'total_chunks': total_chunks,
                'total_documents': total_chunks,  # 스트림릿 호환성용
                'unique_files': len(unique_files),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"통계 조회 오류: {e}")
            return {'total_chunks': 0, 'total_documents': 0, 'unique_files': 0}

    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"컬렉션 삭제됨: {self.collection_name}")
        except Exception as e:
            print(f"컬렉션 삭제 오류: {e}")

    def reset_collection(self):
        """컬렉션 초기화"""
        try:
            self.delete_collection()
            self._init_chromadb()
            print(f"컬렉션 초기화 완료: {self.collection_name}")
        except Exception as e:
            print(f"컬렉션 초기화 오류: {e}")