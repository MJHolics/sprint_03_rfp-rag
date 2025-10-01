"""
RAG 시스템 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 기본 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
METADATA_DB_PATH = PROJECT_ROOT / "rfp_metadata.db"

# 문서 처리 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 검색 설정
DEFAULT_TOP_K = 5
CONFIDENCE_THRESHOLD = 0.3
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# OpenAI 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # 최고 성능 임베딩 (3072차원)
OPENAI_CHAT_MODEL = "gpt-4o"  # 새 API 키로 gpt-4o 재시도

# ChromaDB 설정
CHROMA_COLLECTION_NAME = "rfp_documents"

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# LibreOffice 설정 (HWP 변환용)
LIBREOFFICE_TIMEOUT = 60  # 초

# 성능 설정
MAX_CONCURRENT_PROCESSES = 4
BATCH_SIZE = 100

# 멀티모달 설정
ENABLE_MULTIMODAL = True  # GPT-4V 이미지 분석 활성화
MAX_IMAGES_PER_DOCUMENT = 5  # 문서당 최대 분석할 이미지 수
MULTIMODAL_MAX_PAGES = 10  # 멀티모달 분석할 최대 페이지 수