"""
CNN 기반 메타데이터 분석 시스템
문서의 레이아웃, 구조, 이미지 등을 체계적으로 분석하여 메타데이터 추출
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
from .advanced_layout_analyzer import AdvancedLayoutAnalyzer

@dataclass
class StructuralElement:
    """문서 구조 요소"""
    element_type: str  # 'title', 'subtitle', 'paragraph', 'table', 'image', 'list'
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text_content: str
    font_size: float
    font_weight: str  # 'normal', 'bold'
    confidence: float
    page_number: int

@dataclass
class DocumentStructure:
    """문서 전체 구조"""
    elements: List[StructuralElement]
    page_count: int
    document_type: str  # 'rfp', 'proposal', 'contract', 'technical_doc'
    main_categories: List[str]
    metadata: Dict[str, Any]

class CNNMetadataAnalyzer:
    """CNN 기반 문서 구조 및 메타데이터 분석기"""

    def __init__(self, use_advanced_analysis: bool = True):
        self.use_advanced_analysis = use_advanced_analysis
        self.advanced_analyzer = AdvancedLayoutAnalyzer() if use_advanced_analysis else None

        self.title_patterns = [
            r'^\d+\.\s*.+',  # 1. 제목
            r'^[가-힣]+\s*:',  # 한글: 형태
            r'^【.+】',  # 【제목】 형태
            r'^\s*[◆◇▣▢■□●○]\s*.+',  # 특수기호 제목
        ]

        self.category_keywords = {
            '사업개요': ['사업', '개요', '목적', '배경', '필요성'],
            '기술요구사항': ['기술', '요구사항', '규격', '성능', '기능'],
            '일정': ['일정', '기간', '계획', '스케줄', '완료'],
            '예산': ['예산', '비용', '가격', '금액', '총액'],
            '인력': ['인력', '인원', '담당자', '책임자', '개발자'],
            '시스템': ['시스템', '플랫폼', '서버', '데이터베이스', 'API'],
            '보안': ['보안', '암호화', '인증', '방화벽', '접근제어']
        }

    def analyze_document_structure(self, file_path: str, extracted_content: Dict[str, Any]) -> DocumentStructure:
        """문서 구조 분석 및 메타데이터 추출"""

        # 고급 분석기 사용 가능한 경우
        if self.use_advanced_analysis and self.advanced_analyzer:
            try:
                return self._analyze_with_advanced_models(file_path, extracted_content)
            except Exception as e:
                print(f"고급 분석 실패, 기본 방식으로 전환: {e}")
                # 실패시 기본 방식으로 fallback

        # 기본 분석 방식
        return self._analyze_with_basic_rules(file_path, extracted_content)

    def _analyze_with_advanced_models(self, file_path: str, extracted_content: Dict[str, Any]) -> DocumentStructure:
        """고급 CNN + OCR 모델을 사용한 분석"""

        # 1. 고급 레이아웃 분석 수행
        page_layouts = self.advanced_analyzer.analyze_document_advanced(file_path)

        # 2. 고급 메타데이터 생성
        advanced_metadata = self.advanced_analyzer.generate_enhanced_metadata(page_layouts, file_path)

        # 3. LayoutElement를 StructuralElement로 변환
        structural_elements = []
        for layout in page_layouts:
            for elem in layout.elements:
                structural_elem = StructuralElement(
                    element_type=elem.element_type,
                    bbox=elem.bbox,
                    text_content=elem.text_content,
                    font_size=elem.font_info.get('estimated_size', 12.0),
                    font_weight=elem.font_info.get('weight', 'normal'),
                    confidence=elem.confidence,
                    page_number=layout.page_number
                )
                structural_elements.append(structural_elem)

        # 4. 문서 타입 및 카테고리 분석
        doc_type = self._classify_document_type(structural_elements)
        categories = self._extract_main_categories(structural_elements)

        # 5. 메타데이터 통합
        enhanced_metadata = self._generate_enhanced_metadata(
            file_path, structural_elements, doc_type, categories
        )

        # 고급 분석 결과 추가
        enhanced_metadata.update({
            'advanced_analysis': True,
            'ocr_confidence': advanced_metadata.get('avg_ocr_confidence', 0.0),
            'layout_types': advanced_metadata.get('layout_types', []),
            'reading_order_available': True,
            'visual_features_extracted': True
        })

        return DocumentStructure(
            elements=structural_elements,
            page_count=len(page_layouts),
            document_type=doc_type,
            main_categories=categories,
            metadata=enhanced_metadata
        )

    def _analyze_with_basic_rules(self, file_path: str, extracted_content: Dict[str, Any]) -> DocumentStructure:
        """기본 규칙 기반 분석 (기존 방식)"""

        # 1. 텍스트 기반 구조 분석
        text_elements = self._analyze_text_structure(extracted_content.get('text', ''))

        # 2. 이미지 기반 레이아웃 분석 (PDF 페이지 이미지가 있는 경우)
        if 'page_images' in extracted_content:
            layout_elements = self._analyze_layout_from_images(extracted_content['page_images'])
            text_elements = self._merge_text_and_layout(text_elements, layout_elements)

        # 3. 문서 타입 분류
        doc_type = self._classify_document_type(text_elements)

        # 4. 주요 카테고리 추출
        categories = self._extract_main_categories(text_elements)

        # 5. 강화된 메타데이터 생성
        enhanced_metadata = self._generate_enhanced_metadata(
            file_path, text_elements, doc_type, categories
        )

        enhanced_metadata['advanced_analysis'] = False

        return DocumentStructure(
            elements=text_elements,
            page_count=extracted_content.get('total_pages', 1),
            document_type=doc_type,
            main_categories=categories,
            metadata=enhanced_metadata
        )

    def _analyze_text_structure(self, text: str) -> List[StructuralElement]:
        """텍스트 기반 구조 분석"""
        elements = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            element_type = self._classify_text_element(line)
            font_info = self._estimate_font_properties(line, element_type)

            element = StructuralElement(
                element_type=element_type,
                bbox=(0, i*20, len(line)*10, (i+1)*20),  # 추정 bbox
                text_content=line,
                font_size=font_info['size'],
                font_weight=font_info['weight'],
                confidence=font_info['confidence'],
                page_number=1  # TODO: 실제 페이지 번호 계산
            )
            elements.append(element)

        return elements

    def _classify_text_element(self, text: str) -> str:
        """텍스트 요소 분류"""
        import re

        # 제목 패턴 확인
        for pattern in self.title_patterns:
            if re.match(pattern, text):
                if len(text) < 50:  # 짧으면 제목
                    return 'title'
                else:
                    return 'subtitle'

        # 리스트 항목 확인
        if re.match(r'^\s*[-*•]\s+', text):
            return 'list'

        # 표 데이터 확인 (탭이나 많은 공백으로 분리된 경우)
        if '\t' in text or len(re.findall(r'\s{3,}', text)) > 2:
            return 'table'

        # 기본은 단락
        return 'paragraph'

    def _estimate_font_properties(self, text: str, element_type: str) -> Dict[str, Any]:
        """폰트 속성 추정"""
        base_confidence = 0.7

        if element_type == 'title':
            return {
                'size': 16.0,
                'weight': 'bold',
                'confidence': base_confidence + 0.2
            }
        elif element_type == 'subtitle':
            return {
                'size': 14.0,
                'weight': 'bold',
                'confidence': base_confidence + 0.1
            }
        else:
            return {
                'size': 12.0,
                'weight': 'normal',
                'confidence': base_confidence
            }

    def _analyze_layout_from_images(self, page_images: List[np.ndarray]) -> List[StructuralElement]:
        """이미지 기반 레이아웃 분석 (향후 CNN 모델 적용 예정)"""
        # TODO: 실제 CNN 모델을 사용한 레이아웃 분석
        # 현재는 기본적인 컴퓨터 비전 기법 사용

        layout_elements = []

        for page_idx, image in enumerate(page_images):
            if image is None:
                continue

            # 간단한 텍스트 영역 감지
            text_regions = self._detect_text_regions(image)

            for region in text_regions:
                element = StructuralElement(
                    element_type='text_region',
                    bbox=region['bbox'],
                    text_content='',
                    font_size=region.get('estimated_font_size', 12.0),
                    font_weight='normal',
                    confidence=region.get('confidence', 0.8),
                    page_number=page_idx + 1
                )
                layout_elements.append(element)

        return layout_elements

    def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """텍스트 영역 감지 (OpenCV 사용)"""
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 텍스트 영역 감지를 위한 전처리
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # 텍스트 영역으로 추정되는 크기 필터링
                if w > 50 and h > 10 and w < image.shape[1] * 0.8:
                    regions.append({
                        'bbox': (x, y, x+w, y+h),
                        'estimated_font_size': max(10.0, h * 0.8),
                        'confidence': 0.7
                    })

            return regions

        except Exception as e:
            print(f"텍스트 영역 감지 오류: {e}")
            return []

    def _merge_text_and_layout(self, text_elements: List[StructuralElement],
                              layout_elements: List[StructuralElement]) -> List[StructuralElement]:
        """텍스트와 레이아웃 정보 병합"""
        # TODO: 실제 위치 기반 매칭 알고리즘 구현
        # 현재는 단순히 텍스트 요소 우선 반환
        return text_elements

    def _classify_document_type(self, elements: List[StructuralElement]) -> str:
        """문서 타입 분류"""
        text_content = ' '.join([elem.text_content for elem in elements])

        rfp_keywords = ['제안요청서', '입찰', '사업계획', '과업지시서', '규격서']
        proposal_keywords = ['제안서', '기술제안', '사업제안']
        contract_keywords = ['계약서', '협약서', '약정서']

        rfp_count = sum(1 for keyword in rfp_keywords if keyword in text_content)
        proposal_count = sum(1 for keyword in proposal_keywords if keyword in text_content)
        contract_count = sum(1 for keyword in contract_keywords if keyword in text_content)

        if rfp_count >= proposal_count and rfp_count >= contract_count:
            return 'rfp'
        elif proposal_count >= contract_count:
            return 'proposal'
        elif contract_count > 0:
            return 'contract'
        else:
            return 'technical_doc'

    def _extract_main_categories(self, elements: List[StructuralElement]) -> List[str]:
        """주요 카테고리 추출"""
        categories = []
        text_content = ' '.join([elem.text_content for elem in elements])

        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_content)
            if score >= 2:  # 최소 2개 키워드 매칭
                categories.append(category)

        return categories

    def _generate_enhanced_metadata(self, file_path: str, elements: List[StructuralElement],
                                  doc_type: str, categories: List[str]) -> Dict[str, Any]:
        """강화된 메타데이터 생성"""
        file_path_obj = Path(file_path)

        # 제목들 추출
        titles = [elem.text_content for elem in elements
                 if elem.element_type in ['title', 'subtitle']]

        # 주요 키워드 추출 (제목과 카테고리에서)
        keywords = []
        for title in titles:
            words = title.split()
            keywords.extend([word for word in words if len(word) > 2])
        keywords.extend(categories)

        # 복잡도 계산
        complexity_score = self._calculate_complexity_score(elements)

        return {
            'file_name': file_path_obj.name,
            'file_path': str(file_path_obj.absolute()),
            'file_extension': file_path_obj.suffix,
            'document_type': doc_type,
            'main_categories': categories,
            'extracted_titles': titles[:5],  # 상위 5개 제목
            'keywords': list(set(keywords))[:10],  # 중복 제거 후 상위 10개
            'complexity_score': complexity_score,
            'total_elements': len(elements),
            'title_count': len([e for e in elements if e.element_type == 'title']),
            'paragraph_count': len([e for e in elements if e.element_type == 'paragraph']),
            'table_count': len([e for e in elements if e.element_type == 'table']),
            'analysis_method': 'cnn_enhanced',
            'confidence_score': np.mean([elem.confidence for elem in elements])
        }

    def _calculate_complexity_score(self, elements: List[StructuralElement]) -> float:
        """문서 복잡도 점수 계산"""
        if not elements:
            return 0.0

        # 요소 타입 다양성
        element_types = set(elem.element_type for elem in elements)
        diversity_score = len(element_types) / 6.0  # 최대 6가지 타입

        # 구조적 깊이 (제목 레벨)
        title_count = len([e for e in elements if e.element_type in ['title', 'subtitle']])
        structure_score = min(title_count / 20.0, 1.0)  # 최대 20개 제목 가정

        # 전체 길이
        total_text_length = sum(len(elem.text_content) for elem in elements)
        length_score = min(total_text_length / 10000.0, 1.0)  # 최대 10,000자 가정

        return (diversity_score + structure_score + length_score) / 3.0

    def extract_key_information(self, structure: DocumentStructure) -> Dict[str, Any]:
        """주요 정보 추출 (예산, 일정, 기술요구사항 등)"""
        key_info = {
            'budget_info': [],
            'schedule_info': [],
            'technical_requirements': [],
            'personnel_info': []
        }

        for element in structure.elements:
            text = element.text_content.lower()

            # 예산 정보 추출
            if any(keyword in text for keyword in ['예산', '비용', '금액', '원']):
                key_info['budget_info'].append({
                    'text': element.text_content,
                    'confidence': element.confidence,
                    'element_type': element.element_type
                })

            # 일정 정보 추출
            if any(keyword in text for keyword in ['일정', '기간', '완료', '월', '일']):
                key_info['schedule_info'].append({
                    'text': element.text_content,
                    'confidence': element.confidence,
                    'element_type': element.element_type
                })

            # 기술 요구사항 추출
            if any(keyword in text for keyword in ['기술', '시스템', '플랫폼', 'api', '데이터베이스']):
                key_info['technical_requirements'].append({
                    'text': element.text_content,
                    'confidence': element.confidence,
                    'element_type': element.element_type
                })

        return key_info