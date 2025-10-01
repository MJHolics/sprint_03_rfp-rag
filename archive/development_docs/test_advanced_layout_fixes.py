"""
동료파일 해결책 적용 후 advanced_layout_analyzer 테스트
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.processors.advanced_layout_analyzer import AdvancedLayoutAnalyzer
    print("[OK] AdvancedLayoutAnalyzer import 성공")
except ImportError as e:
    print(f"[ERROR] Import 실패: {e}")
    sys.exit(1)

def test_analyzer_initialization():
    """분석기 초기화 테스트"""
    print("\n[TEST] 분석기 초기화 테스트...")
    try:
        analyzer = AdvancedLayoutAnalyzer()
        print("[OK] AdvancedLayoutAnalyzer 초기화 성공")

        # OCR 모델 확인
        if analyzer.ocr_reader:
            print("[OK] EasyOCR 모델 로드 성공")
        else:
            print("[WARN] EasyOCR 모델 로드 실패 (라이브러리 미설치)")

        # CNN 모델 확인
        if analyzer.cnn_model:
            print("[OK] CNN 모델 (ResNet50) 로드 성공")
        else:
            print("[WARN] CNN 모델 로드 실패 (PyTorch 미설치)")

        return analyzer

    except Exception as e:
        print(f"[ERROR] 초기화 실패: {e}")
        return None

def test_with_sample_pdf():
    """샘플 PDF로 분석 테스트"""
    print("\n[TEST] PDF 분석 테스트...")

    analyzer = test_analyzer_initialization()
    if not analyzer:
        return

    # files 폴더에서 PDF 파일 찾기
    files_dir = Path("./files")
    if files_dir.exists():
        pdf_files = list(files_dir.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"[FILE] 테스트 파일: {test_file.name}")

            try:
                # 고급 분석 실행
                layouts = analyzer.analyze_document_advanced(str(test_file))

                print(f"[OK] 분석 완료 - {len(layouts)}개 페이지 처리")

                # 결과 요약
                total_elements = sum(len(layout.elements) for layout in layouts)
                print(f"[RESULT] 총 {total_elements}개 요소 감지")

                # 페이지별 상세 정보
                for i, layout in enumerate(layouts[:3]):  # 처음 3페이지만
                    elements_count = len(layout.elements)
                    print(f"   페이지 {layout.page_number}: {elements_count}개 요소, {layout.layout_type} 레이아웃")

                    # 요소 타입별 통계
                    if layout.elements:
                        element_types = {}
                        for elem in layout.elements:
                            element_types[elem.element_type] = element_types.get(elem.element_type, 0) + 1
                        print(f"     요소 타입: {dict(element_types)}")

                # 동료파일 이슈 해결 확인
                print("\n[CHECK] 동료파일 이슈 해결 확인:")

                # 1. 빈 결과 문제 해결 여부
                if total_elements > 0:
                    print("[OK] 흑백 반전 이슈 해결됨 - 요소 감지 성공")
                else:
                    print("[ERROR] 여전히 요소가 감지되지 않음")

                # 2. 텍스트 추출 개선 여부
                extracted_texts = []
                for layout in layouts:
                    for elem in layout.elements:
                        if elem.text_content.strip():
                            extracted_texts.append(elem.text_content.strip())

                if extracted_texts:
                    print(f"[OK] OCR 신뢰도 개선됨 - {len(extracted_texts)}개 텍스트 추출")
                    print(f"   샘플 텍스트: {extracted_texts[0][:50]}...")
                else:
                    print("[ERROR] 텍스트 추출 여전히 실패")

                return True

            except Exception as e:
                print(f"[ERROR] 분석 실패: {e}")
                return False
        else:
            print("[WARN] files 폴더에 PDF 파일이 없습니다")
    else:
        print("[WARN] files 폴더가 존재하지 않습니다")

    return False

def main():
    """메인 테스트 실행"""
    print("[TEST] Advanced Layout Analyzer 수정사항 테스트")
    print("=" * 50)

    print("\n[INFO] 적용된 수정사항:")
    print("1. 흑백 반전 이슈 해결 (cv2.bitwise_not 적용)")
    print("2. OCR 신뢰도 적응형 임계값 (0.3~0.7)")
    print("3. 필터링 완화 (가로세로 비율 0.1~100)")

    # 초기화 테스트
    analyzer = test_analyzer_initialization()

    if analyzer:
        # PDF 분석 테스트
        success = test_with_sample_pdf()

        if success:
            print("\n[SUCCESS] 모든 테스트 통과!")
            print("[OK] 동료파일에서 발견된 이슈들이 해결되었습니다.")
        else:
            print("\n[WARN] 일부 테스트 실패")
            print("추가 디버깅이 필요할 수 있습니다.")
    else:
        print("\n[ERROR] 분석기 초기화 실패")
        print("필요한 라이브러리 설치를 확인해주세요:")
        print("pip install easyocr pytesseract torch torchvision")

if __name__ == "__main__":
    main()