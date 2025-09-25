#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
스마트 질의 개선 시스템
- 권장 질문 생성
- 어휘력 차이 극복
- 의미 유추 및 확장
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher
import json

class SmartQuerySystem:
    """지능형 질의 처리 및 개선 시스템"""

    def __init__(self):
        # RFP 도메인 특화 시소러스 (유의어 사전)
        self.domain_thesaurus = {
            # 예산 관련
            '예산': ['비용', '금액', '총액', '소요비용', '사업비', '투자비', '재정', '자금', '예산안'],
            '비용': ['예산', '금액', '총액', '소요비용', '사업비', '투자비', '경비', '지출'],
            '가격': ['단가', '비용', '금액', '요금', '가액', '값'],

            # 일정 관련
            '일정': ['스케줄', '계획', '기간', '납기', '완료일', '종료일', '시간표'],
            '기간': ['일정', '소요시간', '수행기간', '개발기간', '납기'],
            '완료': ['종료', '마감', '끝', '완성', '달성', '수행완료'],

            # 기술 관련
            '기술': ['테크놀로지', '솔루션', '방법론', '기법', '스킬'],
            '시스템': ['플랫폼', '서비스', '솔루션', '체계', '구조'],
            '개발': ['구축', '제작', '구현', '생성', '작성', '설계'],
            '구축': ['개발', '구현', '설치', '설정', '배포', '세팅'],

            # 요구사항 관련
            '요구사항': ['필요사항', '조건', '스펙', '규격', '기준', '조건사항'],
            '규격': ['스펙', '사양', '기준', '표준', '조건'],
            '성능': ['퍼포먼스', '속도', '처리량', '효율성', '능력'],

            # 조직/인력 관련
            '인력': ['인원', '담당자', '개발자', '팀', '조직', '인적자원'],
            '담당자': ['책임자', '매니저', '관리자', 'PM', '리더'],
            '업체': ['회사', '기업', '업체', '사업자', '공급업체', '벤더'],

            # 보안 관련
            '보안': ['시큐리티', '암호화', '인증', '방화벽', '접근제어'],
            '암호화': ['보안', '인크립션', '해시', '키'],

            # 문서 관련
            '제안서': ['제안요청서', 'RFP', '사업계획서', '기획서'],
            '계약': ['협약', '약정', '계약서', '협정'],

            # 품질 관련
            '품질': ['퀄리티', '수준', '등급', '완성도'],
            '테스트': ['시험', '검증', '점검', '확인', '검사'],
        }

        # 레벨별 어휘 매핑 (쉬운 말 -> 어려운 말)
        self.vocabulary_levels = {
            '초급': {
                '돈': '예산', '비용': '예산',
                '만들기': '개발', '만든다': '구축한다',
                '언제': '일정', '얼마나': '기간',
                '뭐': '무엇', '어떤': '어떠한',
                '좋은': '우수한', '빠른': '신속한',
                '쉬운': '간편한', '어려운': '복잡한'
            },
            '중급': {
                '시스템': '플랫폼',
                '프로그램': '애플리케이션',
                '데이터': '정보',
                '서버': '서버시스템'
            },
            '고급': {
                '아키텍처': '시스템구조',
                '인프라': '기반구조',
                '솔루션': '해결방안'
            }
        }

        # 질문 패턴별 추천 템플릿
        self.question_templates = {
            '예산': [
                "총 사업예산은 얼마인가요?",
                "예산 규모는 어느 정도인가요?",
                "투자비용은 얼마나 필요한가요?",
                "사업비 범위를 알려주세요",
                "소요예산 규모는?",
                "총 비용은 얼마나 되나요?",
                "예산 한도는 어떻게 되나요?",
                "경비 규모가 궁금합니다"
            ],
            '일정': [
                "사업 일정은 어떻게 되나요?",
                "개발 기간은 얼마나 걸리나요?",
                "완료 예정일은 언제인가요?",
                "납기는 어떻게 되나요?",
                "수행 기간이 궁금합니다",
                "언제까지 완료해야 하나요?",
                "스케줄 계획을 알려주세요",
                "마감일정이 있나요?"
            ],
            '기술': [
                "어떤 기술을 사용하나요?",
                "기술 스택은 무엇인가요?",
                "개발 기술이 궁금합니다",
                "어떤 솔루션을 쓰나요?",
                "기술적 요구사항은?",
                "플랫폼 환경은 어떻게 되나요?",
                "개발 방법론은 무엇인가요?",
                "기술 아키텍처를 알려주세요"
            ],
            '요구사항': [
                "필수 요구사항은 무엇인가요?",
                "기능 요구사항을 알려주세요",
                "시스템 요구사항은?",
                "어떤 조건들이 있나요?",
                "필요한 기능은 무엇인가요?",
                "성능 요구사항이 궁금합니다",
                "규격 조건은 어떻게 되나요?",
                "어떤 스펙이 필요한가요?"
            ],
            '인력': [
                "필요한 인력 규모는?",
                "팀 구성은 어떻게 되나요?",
                "담당자는 몇 명인가요?",
                "개발 인원이 궁금합니다",
                "프로젝트 팀원 구성은?",
                "인적 자원 계획은?",
                "역할별 담당자는?",
                "조직 구성도를 알려주세요"
            ],
            '보안': [
                "보안 요구사항은 무엇인가요?",
                "보안 조치는 어떻게 되나요?",
                "암호화 방법은?",
                "접근 제어 방식은?",
                "보안 정책이 있나요?",
                "인증 시스템은 어떻게?",
                "방화벽 설정은?",
                "보안 수준은 어느 정도인가요?"
            ]
        }

        # 문맥별 연관 키워드
        self.contextual_associations = {
            '웹시스템': ['웹사이트', '홈페이지', '인터넷', 'UI/UX', '브라우저', '반응형'],
            '모바일': ['앱', '스마트폰', '태블릿', 'iOS', '안드로이드', '크로스플랫폼'],
            '데이터베이스': ['DB', 'SQL', 'NoSQL', '저장', '조회', '백업'],
            'AI': ['인공지능', '머신러닝', '딥러닝', '자동화', '예측', '분석'],
            '클라우드': ['AWS', 'Azure', 'GCP', '호스팅', 'SaaS', 'IaaS'],
        }

    def generate_recommended_questions(self, document_content: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """문서 내용 분석하여 추천 질문 생성"""

        # 1. 문서에서 주요 키워드 추출
        main_keywords = self._extract_main_keywords(document_content)

        # 2. 키워드별 추천 질문 생성
        recommended_questions = []

        for keyword, frequency in main_keywords.items():
            # 해당 키워드 카테고리 찾기
            category = self._find_keyword_category(keyword)

            if category and category in self.question_templates:
                # 템플릿 질문들 가져오기
                templates = self.question_templates[category]

                for template in templates:
                    confidence = self._calculate_question_relevance(template, document_content, frequency)

                    recommended_questions.append({
                        'question': template,
                        'category': category,
                        'keyword': keyword,
                        'confidence': confidence,
                        'frequency': frequency
                    })

        # 3. 문서 특화 질문 생성
        specialized_questions = self._generate_document_specific_questions(document_content)
        recommended_questions.extend(specialized_questions)

        # 4. 신뢰도 순으로 정렬하여 상위 반환
        recommended_questions.sort(key=lambda x: x['confidence'], reverse=True)

        return recommended_questions[:top_k]

    def enhance_user_query(self, user_query: str, document_context: str = "") -> Dict[str, Any]:
        """사용자 질의를 의미적으로 향상"""

        # 1. 원본 쿼리 분석
        original_analysis = self._analyze_query(user_query)

        # 2. 어휘력 수준 보정
        standardized_query = self._standardize_vocabulary(user_query)

        # 3. 유의어 확장
        expanded_query = self._expand_with_synonyms(standardized_query)

        # 4. 문맥 기반 연관어 추가
        contextualized_query = self._add_contextual_terms(expanded_query, document_context)

        # 5. 의도 추론 및 보완
        intent_enhanced_query = self._enhance_with_intent(contextualized_query, original_analysis)

        return {
            'original_query': user_query,
            'enhanced_query': intent_enhanced_query,
            'standardized_query': standardized_query,
            'expanded_terms': self._get_expanded_terms(user_query, intent_enhanced_query),
            'detected_intent': original_analysis['intent'],
            'confidence_improvement': self._calculate_improvement_score(user_query, intent_enhanced_query),
            'suggested_alternatives': self._generate_query_alternatives(user_query)
        }

    def _extract_main_keywords(self, text: str) -> Dict[str, int]:
        """문서에서 주요 키워드 추출"""

        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과',
                    '의', '도', '만', '부터', '까지', '라서', '이므로', '그리고', '또한', '하지만',
                    '그러나', '따라서', '그래서', '즉', '또는', '및', '등', '기타'}

        # 텍스트 정제 및 토큰화
        text = re.sub(r'[^\w\s]', ' ', text)
        words = [word for word in text.split() if len(word) > 1 and word not in stopwords]

        # 빈도수 계산
        word_freq = Counter(words)

        # TF-IDF 가중치 적용 (간단 버전)
        total_words = len(words)
        weighted_keywords = {}

        for word, freq in word_freq.items():
            if freq >= 2:  # 최소 2회 이상 등장
                tf = freq / total_words
                # 도메인 중요도 가중치
                domain_weight = self._get_domain_importance_weight(word)
                weighted_keywords[word] = tf * domain_weight

        # 상위 키워드만 반환
        sorted_keywords = dict(sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_keywords.items())[:30])

    def _find_keyword_category(self, keyword: str) -> str:
        """키워드가 속한 카테고리 찾기"""
        for category, synonyms in self.domain_thesaurus.items():
            if keyword in synonyms or keyword == category:
                # 카테고리를 질문 템플릿 키로 매핑
                if category in ['예산', '비용', '가격']:
                    return '예산'
                elif category in ['일정', '기간', '완료']:
                    return '일정'
                elif category in ['기술', '시스템', '개발', '구축']:
                    return '기술'
                elif category in ['요구사항', '규격', '성능']:
                    return '요구사항'
                elif category in ['인력', '담당자', '업체']:
                    return '인력'
                elif category in ['보안', '암호화']:
                    return '보안'
        return None

    def _calculate_question_relevance(self, question: str, document: str, keyword_freq: float) -> float:
        """질문의 문서 관련성 점수 계산"""

        # 기본 점수 (키워드 빈도 기반)
        base_score = min(keyword_freq * 10, 1.0)

        # 질문 키워드와 문서 매칭도
        question_words = set(re.findall(r'\w+', question.lower()))
        document_words = set(re.findall(r'\w+', document.lower()))

        intersection = len(question_words & document_words)
        union = len(question_words | document_words)
        jaccard_score = intersection / union if union > 0 else 0

        # 최종 점수 (0-1)
        final_score = (base_score * 0.7 + jaccard_score * 0.3)
        return round(final_score, 3)

    def _generate_document_specific_questions(self, document: str) -> List[Dict[str, Any]]:
        """문서 특화 질문 생성"""
        specialized = []

        # 숫자 패턴 감지 (예산, 일정 등)
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', document)
        if numbers:
            specialized.append({
                'question': f"구체적인 수치나 규모는 어떻게 되나요?",
                'category': '구체정보',
                'keyword': '수치',
                'confidence': 0.8,
                'frequency': len(numbers)
            })

        # 날짜 패턴 감지
        dates = re.findall(r'\d{4}년|\d{1,2}월|\d{1,2}일', document)
        if dates:
            specialized.append({
                'question': f"프로젝트 세부 일정은 어떻게 계획되어 있나요?",
                'category': '일정',
                'keyword': '날짜',
                'confidence': 0.9,
                'frequency': len(dates)
            })

        # 조건문 감지 ("필수", "선택", "권장" 등)
        conditions = re.findall(r'필수|선택|권장|조건|기준', document)
        if conditions:
            specialized.append({
                'question': f"필수 조건과 선택 조건을 구분해서 알려주세요",
                'category': '요구사항',
                'keyword': '조건',
                'confidence': 0.85,
                'frequency': len(conditions)
            })

        return specialized

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석 (의도, 키워드 등)"""

        # 의도 분류
        intent = 'general'
        if any(word in query for word in ['얼마', '비용', '예산', '돈']):
            intent = 'budget'
        elif any(word in query for word in ['언제', '일정', '기간', '완료']):
            intent = 'schedule'
        elif any(word in query for word in ['어떻게', '방법', '기술', '시스템']):
            intent = 'technical'
        elif any(word in query for word in ['누구', '담당자', '인력', '팀']):
            intent = 'personnel'
        elif any(word in query for word in ['보안', '암호화', '인증']):
            intent = 'security'

        # 키워드 추출
        keywords = re.findall(r'\w+', query)

        return {
            'intent': intent,
            'keywords': keywords,
            'length': len(query),
            'complexity': len(keywords) / len(query) if query else 0
        }

    def _standardize_vocabulary(self, query: str) -> str:
        """어휘력 차이 보정"""
        standardized = query

        # 초급 -> 표준 어휘 변환
        for level_vocab in self.vocabulary_levels.values():
            for casual, formal in level_vocab.items():
                standardized = re.sub(r'\b' + casual + r'\b', formal, standardized)

        return standardized

    def _expand_with_synonyms(self, query: str) -> str:
        """유의어로 쿼리 확장"""
        expanded_terms = []
        query_words = re.findall(r'\w+', query)

        for word in query_words:
            # 유의어 찾기
            synonyms = self._find_synonyms(word)
            if synonyms:
                expanded_terms.extend(synonyms[:2])  # 상위 2개만

        if expanded_terms:
            return query + " " + " ".join(expanded_terms)
        return query

    def _find_synonyms(self, word: str) -> List[str]:
        """단어의 유의어 찾기"""
        for main_word, synonyms in self.domain_thesaurus.items():
            if word == main_word or word in synonyms:
                return [s for s in synonyms if s != word][:3]
        return []

    def _add_contextual_terms(self, query: str, context: str) -> str:
        """문맥 기반 연관어 추가"""
        if not context:
            return query

        # 문맥에서 주요 기술 영역 감지
        for tech_area, related_terms in self.contextual_associations.items():
            if any(term in context.lower() for term in related_terms):
                # 쿼리에 관련 용어가 없으면 추가
                if not any(term in query.lower() for term in related_terms):
                    query += f" {tech_area}"
                break

        return query

    def _enhance_with_intent(self, query: str, analysis: Dict) -> str:
        """의도 기반 쿼리 보강"""
        intent = analysis['intent']

        # 의도별 보강 키워드
        intent_keywords = {
            'budget': ['예산', '비용', '금액'],
            'schedule': ['일정', '기간', '완료'],
            'technical': ['기술', '시스템', '방법'],
            'personnel': ['인력', '담당자', '팀'],
            'security': ['보안', '암호화', '인증']
        }

        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            # 해당 의도의 키워드가 쿼리에 없으면 추가
            if not any(kw in query for kw in keywords):
                query += f" {keywords[0]}"

        return query

    def _get_expanded_terms(self, original: str, enhanced: str) -> List[str]:
        """확장된 용어들 반환"""
        original_words = set(re.findall(r'\w+', original))
        enhanced_words = set(re.findall(r'\w+', enhanced))
        return list(enhanced_words - original_words)

    def _calculate_improvement_score(self, original: str, enhanced: str) -> float:
        """개선 점수 계산"""
        original_len = len(re.findall(r'\w+', original))
        enhanced_len = len(re.findall(r'\w+', enhanced))

        if original_len == 0:
            return 0.0

        improvement = (enhanced_len - original_len) / original_len
        return min(improvement, 1.0)

    def _generate_query_alternatives(self, query: str) -> List[str]:
        """대안 질의 생성"""
        alternatives = []
        analysis = self._analyze_query(query)
        intent = analysis['intent']

        if intent in self.question_templates:
            # 같은 의도의 다른 질문들 추천
            templates = self.question_templates[intent]
            alternatives = templates[:3]  # 상위 3개

        return alternatives

    def _get_domain_importance_weight(self, word: str) -> float:
        """도메인 중요도 가중치"""
        # RFP 도메인에서 중요한 단어들
        high_importance = ['예산', '비용', '일정', '기간', '기술', '시스템', '요구사항', '성능']
        medium_importance = ['개발', '구축', '관리', '운영', '테스트', '품질']

        if word in high_importance:
            return 3.0
        elif word in medium_importance:
            return 2.0
        elif any(word in synonyms for synonyms in self.domain_thesaurus.values()):
            return 1.5
        else:
            return 1.0

    def suggest_better_questions(self, user_query: str, search_results: List[Dict],
                               confidence_threshold: float = 0.5) -> List[str]:
        """검색 결과가 좋지 않을 때 더 나은 질문 제안"""

        if not search_results:
            return ["더 구체적인 키워드를 사용해보세요"]

        avg_confidence = sum(r.get('score', 0) for r in search_results) / len(search_results)

        if avg_confidence < confidence_threshold:
            # 분석 및 개선 제안
            analysis = self._analyze_query(user_query)
            suggestions = []

            if analysis['intent'] == 'general':
                suggestions.append("더 구체적인 질문을 해보세요. 예: '예산은 얼마인가요?'")

            if len(analysis['keywords']) < 2:
                suggestions.append("관련 키워드를 추가해보세요")

            # 유의어 제안
            enhanced = self.enhance_user_query(user_query)
            if enhanced['enhanced_query'] != user_query:
                suggestions.append(f"이렇게 질문해보세요: '{enhanced['enhanced_query']}'")

            return suggestions

        return []