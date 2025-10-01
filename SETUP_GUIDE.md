### 1: 필수 소프트웨어 설치
- **LibreOffice** (HWP 파일 변환용)

### 2: Python 가상환경 설정
```bash
# 가상환경 생성
python -m venv venv
```
# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
```bash
pip install -r requirements.txt
```

### 3: 환경변수 설정
프로젝트 루트에 `.env` 파일 생성:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

# 특정 폴더 처리
```bash
python main.py --mode process --data_path ./your_data_folder
```

### 대화형 검색 시작
```bash
python main.py --mode serve
```

### Streamlit 대시보드 실행
```bash
streamlit run streamlit_dashboard_final.py
```

### 시스템 통계 확인
```bash
python main.py --mode stats
```

##  주요 설정값

`config/settings.py`에서 수정 가능:
- `CHUNK_SIZE`: 텍스트 청크 크기 (기본: 1000)
- `DEFAULT_TOP_K`: 검색 결과 개수 (기본: 5)
- `OPENAI_CHAT_MODEL`: 사용할 GPT 모델 (기본: gpt-4o)
- `ENABLE_MULTIMODAL`: 이미지 분석 기능 (기본: True)
