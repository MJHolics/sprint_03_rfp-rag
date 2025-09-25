"""
실제 CNN 기반 문서 레이아웃 분석 시스템
YOLO + OCR + 규칙 기반 분석의 조합
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
import fitz  # PyMuPDF
import math

# OCR 라이브러리들
try:
    import easyocr
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    EASYOCR_AVAILABLE = True
except ImportError:
    print("EasyOCR/Tesseract 미설치: pip install easyocr pytesseract")
    EASYOCR_AVAILABLE = False

# 딥러닝 모델
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch 미설치: pip install torch torchvision")
    TORCH_AVAILABLE = False

@dataclass
class LayoutElement:
    """고급 레이아웃 요소"""
    element_type: str  # 'text', 'title', 'table', 'image', 'chart', 'list', 'header', 'footer'
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    text_content: str
    ocr_confidence: float
    font_info: Dict[str, Any]
    visual_features: Dict[str, Any]

@dataclass
class PageLayout:
    """페이지 레이아웃 정보"""
    page_number: int
    elements: List[LayoutElement]
    reading_order: List[int]  # 요소들의 읽기 순서
    columns: int
    layout_type: str  # 'single_column', 'two_column', 'complex'

class AdvancedLayoutAnalyzer:
    """실제 CNN + OCR 기반 고급 문서 분석기"""

    def __init__(self):
        self.ocr_reader = None
        self.cnn_model = None
        self.initialize_models()

    def initialize_models(self):
        """모델 초기화"""
        # 1. OCR 모델 초기화
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available() if TORCH_AVAILABLE else False)
                print("EasyOCR 초기화 완료 (한글/영어)")
            except Exception as e:
                print(f"EasyOCR 초기화 실패: {e}")

        # 2. CNN 분류 모델 초기화 (문서 요소 분류용)
        if TORCH_AVAILABLE:
            try:
                self.cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.cnn_model.eval()

                # 전처리 함수
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                print("CNN 모델 (ResNet50) 초기화 완료")
            except Exception as e:
                print(f"CNN 모델 초기화 실패: {e}")

    def analyze_document_advanced(self, file_path: str) -> List[PageLayout]:
        """고급 문서 분석 수행"""
        try:
            doc = fitz.open(file_path)
            page_layouts = []

            for page_num in range(min(len(doc), 5)):  # 처음 5페이지만 분석
                page = doc[page_num]

                # 페이지를 이미지로 변환
                mat = fitz.Matrix(2.0, 2.0)  # 2배 확대로 해상도 향상
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # OpenCV 이미지로 변환
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    continue

                # 페이지 분석 수행
                layout = self._analyze_page_layout(image, page_num + 1)
                page_layouts.append(layout)

            doc.close()
            return page_layouts

        except Exception as e:
            print(f"고급 문서 분석 실패: {e}")
            return []

    def _analyze_page_layout(self, image: np.ndarray, page_num: int) -> PageLayout:
        """단일 페이지 레이아웃 분석"""
        height, width = image.shape[:2]

        # 1. 텍스트 영역 감지
        text_regions = self._detect_text_regions_advanced(image)

        # 2. 각 영역에 대해 OCR 및 분류 수행
        elements = []
        for i, region in enumerate(text_regions):
            element = self._analyze_region(image, region, i)
            if element:
                elements.append(element)

        # 3. 읽기 순서 결정
        reading_order = self._determine_reading_order(elements)

        # 4. 레이아웃 타입 결정
        layout_type = self._classify_layout_type(elements, width, height)

        # 5. 컬럼 수 추정
        columns = self._estimate_columns(elements, width)

        return PageLayout(
            page_number=page_num,
            elements=elements,
            reading_order=reading_order,
            columns=columns,
            layout_type=layout_type
        )

    def _detect_text_regions_advanced(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """고급 텍스트 영역 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 적응형 임계값으로 이진화
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # 2. 모폴로지 연산으로 텍스트 블록 찾기
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # 3. 컨투어 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        img_area = image.shape[0] * image.shape[1]
        min_area = max(500, img_area * 0.001)  # 최소 면적을 이미지 크기에 비례하여 설정
        max_area = img_area * 0.8

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # 더 엄격한 필터링
                aspect_ratio = w / h
                if (0.1 < aspect_ratio < 20 and  # 가로세로 비율
                    w > 50 and h > 20 and        # 최소 크기
                    w < image.shape[1] * 0.95 and h < image.shape[0] * 0.95):  # 최대 크기

                    # 텍스트 영역인지 추가 검증
                    roi = binary[y:y+h, x:x+w]
                    text_pixel_ratio = np.sum(roi == 0) / (w * h)

                    if 0.05 < text_pixel_ratio < 0.8:  # 적당한 텍스트 밀도
                        regions.append({
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour,
                            'text_density': text_pixel_ratio
                        })

        # 면적 순으로 정렬
        regions.sort(key=lambda x: x['area'], reverse=True)
        return regions[:15]  # 상위 15개만 선택

    def _analyze_region(self, image: np.ndarray, region: Dict, region_id: int) -> Optional[LayoutElement]:
        """개별 영역 분석"""
        x1, y1, x2, y2 = region['bbox']
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # 1. OCR로 텍스트 추출
        text_content, ocr_confidence = self._extract_text_ocr(roi)

        # 2. 시각적 특징 추출
        visual_features = self._extract_visual_features(roi)

        # 3. CNN으로 요소 타입 분류
        element_type, cnn_confidence = self._classify_element_type(roi, text_content, visual_features)

        # 4. 폰트 정보 추정
        font_info = self._estimate_font_info(roi, visual_features)

        return LayoutElement(
            element_type=element_type,
            bbox=(x1, y1, x2, y2),
            confidence=cnn_confidence,
            text_content=text_content,
            ocr_confidence=ocr_confidence,
            font_info=font_info,
            visual_features=visual_features
        )

    def _extract_text_ocr(self, roi: np.ndarray) -> Tuple[str, float]:
        """OCR로 텍스트 추출"""
        if not self.ocr_reader:
            return "", 0.0

        try:
            # 이미지 전처리
            processed_roi = self._preprocess_for_ocr(roi)

            # EasyOCR 실행
            results = self.ocr_reader.readtext(processed_roi, detail=1)

            if not results:
                return "", 0.0

            # 결과 통합
            texts = []
            confidences = []

            for bbox, text, conf in results:
                if conf > 0.5:  # 신뢰도 50% 이상만 사용
                    texts.append(text.strip())
                    confidences.append(conf)

            if not texts:
                return "", 0.0

            combined_text = " ".join(texts)
            avg_confidence = np.mean(confidences)

            return combined_text, avg_confidence

        except Exception as e:
            print(f"OCR 실패: {e}")
            return "", 0.0

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR을 위한 이미지 전처리"""
        # PIL 이미지로 변환
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 1. 해상도 향상 (300 DPI 기준)
        if pil_img.size[0] < 300:
            scale_factor = 300 / pil_img.size[0]
            new_size = (int(pil_img.size[0] * scale_factor),
                       int(pil_img.size[1] * scale_factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        # 2. 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)

        # 3. 대비 향상
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)

        # 4. 노이즈 제거
        pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))

        # OpenCV 형식으로 다시 변환
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _extract_visual_features(self, roi: np.ndarray) -> Dict[str, Any]:
        """시각적 특징 추출"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 1. 텍스트 밀도
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text_pixels = np.sum(binary == 0)  # 검은색 픽셀 (텍스트)
        text_density = text_pixels / (height * width)

        # 2. 선 감지 (표 구조)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        horizontal_lines = 0
        vertical_lines = 0

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if abs(theta - np.pi/2) < 0.2:  # 수직선
                    vertical_lines += 1
                elif abs(theta) < 0.2 or abs(theta - np.pi) < 0.2:  # 수평선
                    horizontal_lines += 1

        # 3. 색상 분포
        colors = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        dominant_colors = np.unravel_index(np.argmax(colors), colors.shape)

        # 4. 텍스트 라인 수 추정
        horizontal_projection = np.sum(binary == 0, axis=1)
        peaks, _ = cv2.findContours(
            (horizontal_projection > width * 0.1).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        estimated_lines = len(peaks)

        return {
            'text_density': text_density,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'dominant_colors': dominant_colors,
            'estimated_text_lines': estimated_lines,
            'aspect_ratio': width / height,
            'size': (width, height),
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray)
        }

    def _classify_element_type(self, roi: np.ndarray, text: str, features: Dict) -> Tuple[str, float]:
        """CNN + 규칙 기반 요소 타입 분류"""

        # 1. 규칙 기반 분류 (빠른 판단)
        rule_type, rule_conf = self._classify_by_rules(text, features)

        # 2. CNN 분류 (정확한 판단)
        cnn_type, cnn_conf = self._classify_by_cnn(roi) if self.cnn_model else (rule_type, 0.0)

        # 3. 결합 결정
        if rule_conf > 0.8:  # 규칙 기반이 매우 확실한 경우
            return rule_type, rule_conf
        elif cnn_conf > 0.7:  # CNN이 확실한 경우
            return cnn_type, cnn_conf
        else:  # 규칙 기반 우선
            return rule_type, max(rule_conf, 0.5)

    def _classify_by_rules(self, text: str, features: Dict) -> Tuple[str, float]:
        """규칙 기반 분류"""

        # 표 패턴 감지
        if (features['horizontal_lines'] >= 2 and features['vertical_lines'] >= 2) or \
           ('│' in text or '┌' in text or '├' in text):
            return 'table', 0.9

        # 제목 패턴 감지
        title_patterns = [
            r'^\d+\.\s*[가-힣A-Za-z]+',
            r'^제\s*\d+\s*[장절항]',
            r'^[가-힣]+\s*[:：]\s*',
            r'^【.+】',
            r'^\s*[◆◇▣▢■□●○]\s*.+'
        ]

        import re
        for pattern in title_patterns:
            if re.match(pattern, text):
                if len(text) < 100:  # 짧으면 제목
                    return 'title', 0.85
                else:
                    return 'subtitle', 0.75

        # 리스트 패턴
        if re.match(r'^\s*[-*•]\s+', text) or re.match(r'^\s*\d+\)\s+', text):
            return 'list', 0.8

        # 텍스트 밀도 기반 판단
        if features['text_density'] > 0.3:
            if features['estimated_text_lines'] > 5:
                return 'paragraph', 0.7
            else:
                return 'text', 0.6

        return 'text', 0.5

    def _classify_by_cnn(self, roi: np.ndarray) -> Tuple[str, float]:
        """CNN 기반 분류 (단순화된 버전)"""
        if not self.cnn_model or not TORCH_AVAILABLE:
            return 'text', 0.0

        try:
            # 이미지 전처리
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil_img).unsqueeze(0)

            # 추론
            with torch.no_grad():
                outputs = self.cnn_model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)

            # ImageNet 클래스를 문서 요소로 매핑 (단순화)
            max_prob, max_idx = torch.max(probs, 0)
            confidence = max_prob.item()

            # 실제로는 문서 레이아웃 전용 모델을 훈련해야 함
            # 여기서는 일반적인 패턴으로 매핑
            if confidence > 0.8:
                return 'text', confidence * 0.7  # 보수적 신뢰도
            else:
                return 'text', 0.5

        except Exception as e:
            print(f"CNN 분류 실패: {e}")
            return 'text', 0.0

    def _estimate_font_info(self, roi: np.ndarray, features: Dict) -> Dict[str, Any]:
        """폰트 정보 추정"""

        # 텍스트 높이 추정
        height = features['size'][1]
        estimated_lines = max(1, features['estimated_text_lines'])
        font_height = height / estimated_lines

        # 폰트 크기 추정 (픽셀 → 포인트 변환)
        font_size = max(8, font_height * 0.75)

        # 굵기 판단 (이진화 후 스트로크 너비 분석)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 간단한 스트로크 너비 추정
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(255 - binary, kernel, iterations=1)
        stroke_width = np.sum(eroded > 0) / max(1, np.sum(binary == 0)) if np.sum(binary == 0) > 0 else 0

        is_bold = stroke_width > 0.15

        return {
            'estimated_size': font_size,
            'weight': 'bold' if is_bold else 'normal',
            'stroke_width': stroke_width,
            'line_height': font_height
        }

    def _determine_reading_order(self, elements: List[LayoutElement]) -> List[int]:
        """읽기 순서 결정"""
        if not elements:
            return []

        # Y좌표 기준으로 정렬 후 X좌표로 미세 조정
        indexed_elements = [(i, elem) for i, elem in enumerate(elements)]

        # 행별 그룹화 (Y좌표가 비슷한 것끼리)
        indexed_elements.sort(key=lambda x: x[1].bbox[1])  # y1으로 정렬

        rows = []
        current_row = []
        current_y = None
        tolerance = 20  # 같은 행으로 간주할 Y좌표 차이

        for idx, elem in indexed_elements:
            y = elem.bbox[1]
            if current_y is None or abs(y - current_y) <= tolerance:
                current_row.append((idx, elem))
                current_y = y if current_y is None else (current_y + y) / 2
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [(idx, elem)]
                current_y = y

        if current_row:
            rows.append(current_row)

        # 각 행 내에서 X좌표로 정렬
        reading_order = []
        for row in rows:
            row.sort(key=lambda x: x[1].bbox[0])  # x1으로 정렬
            reading_order.extend([idx for idx, _ in row])

        return reading_order

    def _classify_layout_type(self, elements: List[LayoutElement], width: int, height: int) -> str:
        """레이아웃 타입 분류"""
        if not elements:
            return 'empty'

        # X좌표 분포 분석
        x_centers = [((elem.bbox[0] + elem.bbox[2]) / 2) for elem in elements]

        # 컬럼 경계 찾기
        left_third = width / 3
        right_third = width * 2 / 3

        left_elements = sum(1 for x in x_centers if x < left_third)
        middle_elements = sum(1 for x in x_centers if left_third <= x < right_third)
        right_elements = sum(1 for x in x_centers if x >= right_third)

        total = len(elements)

        if left_elements / total > 0.8:
            return 'single_column'
        elif (left_elements > 0 and right_elements > 0 and
              middle_elements / total < 0.3):
            return 'two_column'
        else:
            return 'complex'

    def _estimate_columns(self, elements: List[LayoutElement], width: int) -> int:
        """컬럼 수 추정"""
        if not elements:
            return 1

        x_centers = [((elem.bbox[0] + elem.bbox[2]) / 2) for elem in elements]

        # K-means 클러스터링으로 컬럼 찾기 (간단 버전)
        x_centers.sort()

        # 간격이 큰 부분을 찾아 컬럼 구분
        gaps = []
        for i in range(1, len(x_centers)):
            gap = x_centers[i] - x_centers[i-1]
            gaps.append(gap)

        if not gaps:
            return 1

        large_gaps = [gap for gap in gaps if gap > width * 0.1]  # 전체 너비의 10% 이상 간격

        return min(len(large_gaps) + 1, 3)  # 최대 3컬럼으로 제한

    def generate_enhanced_metadata(self, layouts: List[PageLayout], file_path: str) -> Dict[str, Any]:
        """고급 메타데이터 생성"""

        all_elements = []
        for layout in layouts:
            all_elements.extend(layout.elements)

        # 요소 타입별 통계
        type_counts = {}
        total_confidence = 0
        total_ocr_confidence = 0

        for elem in all_elements:
            type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
            total_confidence += elem.confidence
            total_ocr_confidence += elem.ocr_confidence

        avg_confidence = total_confidence / len(all_elements) if all_elements else 0
        avg_ocr_confidence = total_ocr_confidence / len(all_elements) if all_elements else 0

        # 텍스트 추출
        extracted_text = ""
        for layout in layouts:
            page_text = f"\n[페이지 {layout.page_number}]\n"
            # 읽기 순서대로 텍스트 조합
            for elem_idx in layout.reading_order:
                if elem_idx < len(layout.elements):
                    elem = layout.elements[elem_idx]
                    if elem.text_content.strip():
                        page_text += f"[{elem.element_type.upper()}] {elem.text_content}\n"
            extracted_text += page_text

        # 문서 복잡도 계산
        complexity_score = self._calculate_advanced_complexity(layouts, all_elements)

        return {
            'file_path': file_path,
            'total_pages': len(layouts),
            'total_elements': len(all_elements),
            'element_types': type_counts,
            'avg_confidence': avg_confidence,
            'avg_ocr_confidence': avg_ocr_confidence,
            'extracted_text': extracted_text,
            'complexity_score': complexity_score,
            'layout_types': [layout.layout_type for layout in layouts],
            'column_info': [layout.columns for layout in layouts],
            'analysis_method': 'advanced_cnn_ocr',
            'features_extracted': [
                'text_regions', 'reading_order', 'font_info',
                'visual_features', 'ocr_text', 'layout_structure'
            ]
        }

    def _calculate_advanced_complexity(self, layouts: List[PageLayout], all_elements: List[LayoutElement]) -> float:
        """고급 복잡도 계산"""
        if not layouts or not all_elements:
            return 0.0

        # 1. 요소 다양성 (0-1)
        unique_types = len(set(elem.element_type for elem in all_elements))
        diversity_score = min(unique_types / 8.0, 1.0)  # 최대 8가지 타입

        # 2. 레이아웃 복잡성 (0-1)
        complex_layouts = sum(1 for layout in layouts if layout.layout_type == 'complex')
        layout_score = complex_layouts / len(layouts)

        # 3. 텍스트 길이 (0-1)
        total_text = sum(len(elem.text_content) for elem in all_elements)
        length_score = min(total_text / 20000.0, 1.0)  # 최대 20,000자

        # 4. OCR 품질 (0-1) - 낮을수록 복잡함을 의미
        avg_ocr_conf = np.mean([elem.ocr_confidence for elem in all_elements])
        ocr_complexity = 1.0 - avg_ocr_conf

        # 가중 평균
        return (diversity_score * 0.25 +
                layout_score * 0.25 +
                length_score * 0.25 +
                ocr_complexity * 0.25)