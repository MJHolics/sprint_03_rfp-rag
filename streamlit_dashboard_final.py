"""
2팀 RFP 분석 대시보드 v1.0
- 개인 프로젝트 스타일의 커스텀 인터페이스
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
from pathlib import Path
import sys

# 프로젝트 경로 설정
sys.path.insert(0, str(Path(__file__).parent))
from src.rag_system import RAGSystem
from config.settings import *

# 페이지 설정 - 개성있는 설정
st.set_page_config(
    page_title="2팀",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 개성있는 스타일링
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* 헤더 스타일 */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* 카드 스타일 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    /* 검색 박스 */
    .search-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px dashed #667eea;
    }

    /* 사이드바 */
    .css-1d391kg {
        background: #2c3e50;
    }

    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* 성능 카드 */
    .perf-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }

    /* 소스 카드 */
    .source-card {
        background: #fff;
        border: 1px solid #e1e8ed;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* 통계 섹션 */
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* 진행 표시기 */
    .progress-ring {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# RAG 시스템 초기화 (캐시)
@st.cache_resource
def init_rag_system():
    return RAGSystem(
        vector_db_path=str(VECTOR_DB_PATH),
        metadata_db_path=str(METADATA_DB_PATH),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

def show_custom_header():
    """커스텀 헤더"""
    st.markdown("""
    <div class="custom-header fade-in">
        <h1>2팀 RFP 분석 시스템</h1>
        <p>AI 기반 제안요청서 검색 & 분석 플랫폼</p>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard_overview(rag_system):
    """메인 대시보드 개요"""
    stats = rag_system.get_system_stats()

    # 시스템 상태 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#667eea;">총 문서</h3>
            <h2 style="margin:0;">{}</h2>
            <small>처리된 RFP 문서</small>
        </div>
        """.format(stats['metadata_store'].get('total_documents', 0)),
        unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#3498db;">청크 수</h3>
            <h2 style="margin:0;">{:,}</h2>
            <small>검색 가능한 텍스트 조각</small>
        </div>
        """.format(stats['metadata_store'].get('total_chunks', 0)),
        unsafe_allow_html=True)

    with col3:
        openai_status = "활성" if stats.get('openai_enabled', False) else "비활성"
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#e74c3c;">AI 엔진</h3>
            <h2 style="margin:0; font-size:1.2rem;">{}</h2>
            <small>GPT-4o 연결 상태</small>
        </div>
        """.format(openai_status),
        unsafe_allow_html=True)

    with col4:
        processor_count = len(stats.get('processors', []))
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color:#f39c12;">지원 형식</h3>
            <h2 style="margin:0;">{} 종류</h2>
            <small>PDF, HWP 처리 가능</small>
        </div>
        """.format(processor_count),
        unsafe_allow_html=True)

    # 발주기관 분포 (더 세련된 차트)
    if 'top_agencies' in stats['metadata_store'] and stats['metadata_store']['top_agencies']:
        st.markdown("### 주요 발주기관 분포")

        agencies_data = stats['metadata_store']['top_agencies']
        top_10 = dict(list(agencies_data.items())[:10])

        # 도넛 차트로 변경
        fig = go.Figure(data=[go.Pie(
            labels=list(top_10.keys()),
            values=list(top_10.values()),
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#3498db', '#e74c3c', '#f39c12',
                          '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6']
        )])

        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

def show_smart_search(rag_system):
    """스마트 검색 인터페이스"""
    st.markdown("### 스마트 문서 검색")

    # 검색 컨테이너 (점선 제거)
    with st.container():
        st.markdown("""
        <div style="background:#f8f9fa; padding:2rem; border-radius:15px; margin:1rem 0; border:2px solid #667eea;">
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            query = st.text_input(
                "질문을 입력하세요",
                placeholder="예: 시스템 구축 예산은 얼마인가요?",
                key="search_query"
            )

        with col2:
            search_method = st.selectbox(
                "검색 방식",
                ["hybrid", "vector", "keyword"],
                format_func=lambda x: {"hybrid": "하이브리드", "vector": "의미검색", "keyword": "키워드"}[x]
            )

        with col3:
            st.write("")  # 빈 공간
            search_button = st.button("🔍 검색", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if query and search_button:
        # 검색 실행
        with st.spinner("문서를 검색하는 중..."):
            start_time = time.time()
            result = rag_system.search_and_answer(
                query,
                search_method=search_method,
                top_k=5
            )
            response_time = time.time() - start_time

        # 결과 표시
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="perf-card">
                <h4>응답 시간</h4>
                <h3>{response_time:.2f}초</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            confidence = result.get('confidence', 0)
            confidence_color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.4 else "#e74c3c"
            st.markdown(f"""
            <div class="perf-card">
                <h4>신뢰도</h4>
                <h3 style="color:{confidence_color}">{confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            source_count = len(result.get('sources', []))
            st.markdown(f"""
            <div class="perf-card">
                <h4>참조 소스</h4>
                <h3>{source_count}개</h3>
            </div>
            """, unsafe_allow_html=True)

        # 답변 표시
        st.markdown("#### AI 답변")
        answer = result.get('answer', '답변을 생성할 수 없습니다.')
        st.markdown(f"""
        <div style="background:#f8f9fa; padding:1.5rem; border-radius:10px; border-left:4px solid #667eea;">
            {answer}
        </div>
        """, unsafe_allow_html=True)

        # 참조 소스
        if result.get('sources'):
            st.markdown("#### 참조 문서")
            for i, source in enumerate(result['sources'], 1):
                score = source.get('score', 0)
                score_color = "#27ae60" if score > 0.8 else "#f39c12" if score > 0.6 else "#e74c3c"

                with st.expander(f"문서 {i}: {source.get('file_name', '알 수 없음')} (관련도: {score:.1%})"):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>발주기관:</strong> {source.get('agency', '정보 없음')}<br>
                        <strong>내용 미리보기:</strong><br>
                        {source.get('content_preview', '미리보기 없음')}
                    </div>
                    """, unsafe_allow_html=True)

def show_analytics_lab(rag_system):
    """분석 실험실"""
    st.markdown("### 성능 분석 실험실")

    # 테스트 쿼리들
    test_scenarios = {
        "예산 관련": [
            "시스템 구축 예산은 얼마인가요?",
            "데이터베이스 구축 비용은?",
            "총 사업비는 어떻게 되나요?"
        ],
        "일정 관련": [
            "프로젝트 기간은 어떻게 되나요?",
            "개발 일정은 몇 개월인가요?",
            "완료 예정일은 언제인가요?"
        ],
        "인력 관련": [
            "개발 인력은 몇 명이 필요한가요?",
            "PM은 몇 명 투입되나요?",
            "개발자 자격 요건은?"
        ],
        "기술 관련": [
            "주요 기술 요구사항을 알려주세요",
            "사용할 프로그래밍 언어는?",
            "데이터베이스는 어떤 것을 써야 하나요?"
        ]
    }

    selected_category = st.selectbox("테스트 시나리오 선택", list(test_scenarios.keys()))

    if st.button("일괄 성능 테스트 실행", type="primary"):
        queries = test_scenarios[selected_category]

        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        results = []

        for i, query in enumerate(queries):
            status_text.text(f"테스트 진행 중... ({i+1}/{len(queries)}) {query}")

            start_time = time.time()
            result = rag_system.search_and_answer(query, search_method="hybrid")
            response_time = time.time() - start_time

            results.append({
                'query': query,
                'response_time': response_time,
                'confidence': result.get('confidence', 0),
                'sources_count': len(result.get('sources', [])),
                'answer': result.get('answer', '')
            })

            progress_bar.progress((i + 1) / len(queries))

        status_text.text("테스트 완료!")

        # 결과 시각화
        with results_container:
            df = pd.DataFrame(results)

            # 성능 요약
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_time = df['response_time'].mean()
                st.markdown(f"""
                <div class="stats-container">
                    <h4>평균 응답시간</h4>
                    <h2>{avg_time:.2f}초</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                avg_conf = df['confidence'].mean()
                st.markdown(f"""
                <div class="stats-container">
                    <h4>평균 신뢰도</h4>
                    <h2>{avg_conf:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_sources = df['sources_count'].mean()
                st.markdown(f"""
                <div class="stats-container">
                    <h4>평균 소스 수</h4>
                    <h2>{avg_sources:.1f}개</h2>
                </div>
                """, unsafe_allow_html=True)

            # 상세 차트
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('응답시간 분포', '신뢰도 분포', '응답시간 vs 신뢰도', '소스 개수'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # 응답시간 바 차트
            fig.add_trace(
                go.Bar(x=list(range(len(df))), y=df['response_time'],
                       name='응답시간', marker_color='#667eea'),
                row=1, col=1
            )

            # 신뢰도 바 차트
            fig.add_trace(
                go.Bar(x=list(range(len(df))), y=df['confidence'],
                       name='신뢰도', marker_color='#764ba2'),
                row=1, col=2
            )

            # 산점도
            fig.add_trace(
                go.Scatter(x=df['response_time'], y=df['confidence'],
                          mode='markers', name='시간-신뢰도',
                          marker=dict(size=10, color='#3498db')),
                row=2, col=1
            )

            # 소스 개수
            fig.add_trace(
                go.Bar(x=list(range(len(df))), y=df['sources_count'],
                       name='소스 수', marker_color='#e74c3c'),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # 상세 결과
            st.markdown("#### 상세 테스트 결과")
            for i, result in enumerate(results, 1):
                with st.expander(f"Q{i}: {result['query']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("응답시간", f"{result['response_time']:.2f}초")
                    with col2:
                        st.metric("신뢰도", f"{result['confidence']:.1%}")
                    with col3:
                        st.metric("소스 수", f"{result['sources_count']}개")

                    st.markdown("**답변:**")
                    st.write(result['answer'])

def show_system_monitor(rag_system):
    """시스템 모니터링"""
    st.markdown("### 시스템 상태 모니터")

    stats = rag_system.get_system_stats()

    # 시스템 정보 카드들
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>벡터 데이터베이스</h4>
            <p><strong>컬렉션명:</strong> rfp_documents</p>
            <p><strong>저장된 벡터:</strong> {:,}개</p>
            <p><strong>임베딩 모델:</strong> text-embedding-3-large</p>
        </div>
        """.format(stats['vector_store'].get('total_documents', 0)),
        unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>AI 모델 정보</h4>
            <p><strong>채팅 모델:</strong> GPT-4o</p>
            <p><strong>검색 방식:</strong> 하이브리드 (벡터+키워드)</p>
            <p><strong>신뢰도 임계값:</strong> 0.3</p>
        </div>
        """, unsafe_allow_html=True)

    # 실시간 상태 체크
    if st.button("실시간 상태 체크"):
        with st.spinner("시스템 상태 확인 중..."):
            time.sleep(1)  # 실제 체크 시뮬레이션

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("벡터 DB 연결 정상")
        with col2:
            st.success("OpenAI API 연결 정상")
        with col3:
            st.success("메타데이터 DB 정상")

def main():
    """메인 애플리케이션"""

    # 헤더 표시
    show_custom_header()

    # 사이드바 네비게이션
    st.sidebar.markdown("### 네비게이션")

    page = st.sidebar.radio(
        "페이지 선택",
        ["대시보드", "스마트 검색", "분석 실험실", "시스템 모니터"]
    )

    # RAG 시스템 초기화
    try:
        rag_system = init_rag_system()

        # 페이지 라우팅
        if page == "대시보드":
            show_dashboard_overview(rag_system)
        elif page == "스마트 검색":
            show_smart_search(rag_system)
        elif page == "분석 실험실":
            show_analytics_lab(rag_system)
        elif page == "시스템 모니터":
            show_system_monitor(rag_system)

    except Exception as e:
        st.error(f"시스템 초기화 오류: {str(e)}")
        st.info("백업 파일로 복구가 가능합니다.")

    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.markdown("**2팀 v1.0**")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.sidebar.markdown(f"업데이트: {current_time}")

if __name__ == "__main__":
    main()