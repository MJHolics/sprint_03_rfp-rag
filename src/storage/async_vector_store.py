"""
비동기 Vector Store - OpenAI API 비동기 처리
3단계 최적화: asyncio 기반 임베딩 생성
"""
import os
import asyncio
from typing import List, Dict, Any
import httpx
from ..processors.base import DocumentChunk

class AsyncVectorStore:
    """비동기 벡터 스토어 - OpenAI API 병렬 처리"""

    def __init__(self, vector_store):
        """
        기존 VectorStore를 래핑하여 비동기 처리 추가

        Args:
            vector_store: 기존 VectorStore 인스턴스
        """
        self.sync_store = vector_store
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.embedding_model = "text-embedding-3-small"

    async def create_embeddings_async(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        비동기로 임베딩 생성 - 배치 병렬 처리

        Args:
            texts: 임베딩 생성할 텍스트 리스트
            batch_size: 배치 크기 (OpenAI API 제한 고려)

        Returns:
            임베딩 벡터 리스트
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

        all_embeddings = []

        # 배치로 나누기
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

        # 비동기 HTTP 클라이언트
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 모든 배치를 병렬로 처리
            tasks = [
                self._process_batch_async(client, batch, batch_idx)
                for batch_idx, batch in enumerate(batches)
            ]

            # 모든 작업 완료 대기
            batch_results = await asyncio.gather(*tasks)

            # 결과 병합
            for batch_embeddings in batch_results:
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _process_batch_async(self, client: httpx.AsyncClient, batch: List[str], batch_idx: int) -> List[List[float]]:
        """단일 배치 비동기 처리"""
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": batch,
            "model": self.embedding_model
        }

        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]

            print(f"  배치 {batch_idx + 1} 완료: {len(embeddings)}개 임베딩 생성")
            return embeddings

        except Exception as e:
            print(f"  배치 {batch_idx + 1} 오류: {e}")
            raise

    async def add_documents_async(self, chunks: List[DocumentChunk]) -> bool:
        """
        비동기로 문서 청크들을 벡터스토어에 추가

        Args:
            chunks: 추가할 DocumentChunk 리스트

        Returns:
            성공 여부
        """
        if not chunks:
            return False

        try:
            print(f"비동기 임베딩 생성 시작: {len(chunks)}개 청크")

            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            # 비동기로 임베딩 생성
            embeddings = await self.create_embeddings_async(texts, batch_size=20)

            # 동기 방식으로 ChromaDB에 저장
            self.sync_store.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            print(f"비동기 추가 완료: {len(chunks)}개 청크")
            return True

        except Exception as e:
            print(f"비동기 벡터 저장소 추가 오류: {e}")
            return False

    async def similarity_search_async(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        비동기 유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 임베딩 비동기 생성
            query_embeddings = await self.create_embeddings_async([query])
            query_embedding = query_embeddings[0]

            # ChromaDB 검색 (동기)
            results = self.sync_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            # 결과 정리
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
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
            print(f"비동기 벡터 검색 오류: {e}")
            return []


def run_async(coro):
    """비동기 코루틴을 동기 환경에서 실행"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
