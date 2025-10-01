# 개인 일정 링크
[Codeit AI 3] _ Part3_2팀 _ 중급 프로젝트 개인 Daily_지민종	https://www.notion.so/26c8c9c2de2280d5ab53cba1a396ad4e?v=26c8c9c2de2280dcb302000ce3c78c0e

[Codeit AI 3] _ Part3_2팀 _ 중급 프로젝트 개인 Daily_조계승	https://www.notion.so/AI03-Project-2-26b1165973bd80328ea6d851451c3bc9?source=copy_link

[Codeit AI 3] _ Part3_2팀 _ 중급 프로젝트 개인 Daily_유영은	https://www.notion.so/daily-26c5954c5686803a8783df86df676a6e?source=copy_link

[Codeit AI 3] _ Part3_2팀 _ 중급 프로젝트 개인 Daily_최우석	https://www.notion.so/270e67a1ff2380a7a010cc16d2599d11?v=270e67a1ff2380eeb88a000c6069db9c&source=copy_link

# RFP RAG 시스템

RFP(제안요청서) 문서를 처리하고 자연어 질의응답을 제공하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- PDF, HWP 문서 자동 처리 및 벡터화
- 하이브리드 검색 (벡터 검색 + 키워드 검색)
- OpenAI GPT 기반 자연어 답변 생성
- 실시간 웹 대시보드 (Streamlit)
- 3단계 최적화 (비동기 API + L1/L2 캐싱)

## 시스템 요구사항

- Python 3.8 이상
- LibreOffice (HWP 파일 변환용)
- OpenAI API Key

## 설치 방법

### 1. LibreOffice 설치

HWP 파일 변환을 위해 LibreOffice 설치가 필요합니다.

- Windows: https://www.libreoffice.org/download
- macOS: `brew install --cask libreoffice`
- Linux: `sudo apt install libreoffice`

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API Key를 설정합니다:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용 방법

### 문서 처리

#### 전체 files 폴더 처리
```bash
python main.py --mode process
```

#### 특정 폴더 처리
```bash
python main.py --mode process --data_path ./your_data_folder
```

### 대화형 검색

```bash
python main.py --mode serve
```

콘솔에서 질문을 입력하면 답변을 받을 수 있습니다.

### Streamlit 대시보드 실행

```bash
streamlit run streamlit_dashboard_final.py
```

웹 브라우저에서 다음 기능을 사용할 수 있습니다:

- 스마트 문서 검색 (하이브리드, 벡터, 키워드 검색)
- 스마트 쿼리 분석 (저수준 쿼리 자동 향상)
- 성능 비교 분석 (단계별 최적화 효과)
- 시스템 모니터링 (문서/벡터/메모리 현황)

### 시스템 통계 확인

```bash
python main.py --mode stats
```

처리된 문서 수, 벡터 청크 수, 데이터베이스 상태를 확인할 수 있습니다.

## 주요 설정

`config/settings.py`에서 다음 값들을 조정할 수 있습니다:

- `CHUNK_SIZE`: 텍스트 청크 크기 (기본: 1000)
- `CHUNK_OVERLAP`: 청크 간 오버랩 (기본: 200)
- `DEFAULT_TOP_K`: 검색 결과 개수 (기본: 5)
- `OPENAI_CHAT_MODEL`: GPT 모델 (기본: gpt-4o)
- `ENABLE_MULTIMODAL`: 이미지 분석 기능 (기본: True)

## 프로젝트 구조

```
project/
├── src/
│   ├── processors/          # 문서 처리 (PDF, HWP)
│   ├── storage/             # 벡터 DB, 메타데이터 DB
│   ├── retrieval/           # 검색 엔진 (하이브리드, BM25)
│   ├── caching/             # L1/L2 캐시 시스템
│   ├── rag_system.py        # Stage 2 RAG 시스템
│   └── rag_system_stage3.py # Stage 3 RAG 시스템
├── config/
│   └── settings.py          # 설정 파일
├── files/                   # 처리할 문서 폴더
├── vector_db/               # 벡터 데이터베이스
├── rfp_metadata.db          # 메타데이터 데이터베이스
├── main.py                  # CLI 진입점
└── streamlit_dashboard_final.py  # 웹 대시보드
```

## 최적화 단계

### Stage 1: 기본 RAG 시스템
- 순차 문서 처리
- 기본 벡터 검색

### Stage 2: 하이브리드 검색 + 인덱싱
- 하이브리드 검색 (벡터 70% + BM25 30%)
- SQLite 인덱스 최적화 (21개)
- 배치 처리

### Stage 3: 비동기 + 캐싱
- 비동기 OpenAI API (httpx)
- L1 메모리 캐시 (100개, 5분 TTL)
- L2 디스크 캐시 (1000개, 1시간 TTL)
- 캐시 히트 시 844배 속도 향상

## 성능

- 문서 처리: 3-4배 향상 (Stage 2 vs Stage 1)
- 검색 응답: 평균 3.39초 (하이브리드), 1.08초 (벡터)
- DB 쿼리: 1500배 향상 (인덱스 최적화)
- 캐시 히트: 844배 향상 (Stage 3)

## 기술 스택

- Python 3.8+
- OpenAI API (text-embedding-3-small, gpt-4o)
- ChromaDB (벡터 데이터베이스)
- SQLite (메타데이터)
- BM25 (키워드 검색)
- Streamlit (웹 대시보드)
- asyncio + httpx (비동기 처리)

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
