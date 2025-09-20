#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HWP 프로세서의 _screenshot_table_html 메서드 테스트
test.xhtml 파일의 모든 표를 스크린샷으로 저장
"""
import os
import sys
from pathlib import Path
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from bs4 import BeautifulSoup
    from src.processors.hwp_processor import HWPProcessor
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    print("필요한 패키지들이 설치되어 있는지 확인해주세요.")
    sys.exit(1)

def test_screenshot_table_html():
    """_screenshot_table_html 메서드 테스트"""
    print("=" * 70)
    print("HWP 프로세서 _screenshot_table_html 메서드 테스트")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    output_dir = project_root / "output_screenshot"
    output_dir.mkdir(exist_ok=True)
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # HWP 프로세서 초기화 (표 이미지 처리 기능 활성화)
    try:
        processor = HWPProcessor(extract_table_images=True)
    except Exception as e:
        print(f"❌ HWP 프로세서 초기화 실패: {e}")
        return
    
    # test.xhtml 파일 경로
    xhtml_path = project_root / "data" / "xhtml" / "test.xhtml"
    
    if not xhtml_path.exists():
        print(f"❌ XHTML 파일을 찾을 수 없습니다: {xhtml_path}")
        print("data/xhtml/test.xhtml 파일이 존재하는지 확인해주세요.")
        return
    
    print(f"📄 XHTML 파일: {xhtml_path}")
    
    try:
        # XHTML 파일 읽기
        with open(xhtml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # 모든 표 요소 찾기
        tables = soup.find_all('table')
        print(f"🔍 발견된 표 개수: {len(tables)}개")
        
        if not tables:
            print("⚠️ XHTML 파일에서 표를 찾을 수 없습니다.")
            return
        
        # 각 표에 대해 스크린샷 생성 (처음 5개만 테스트)
        success_count = 0
        total_size = 0
        test_count = min(20, len(tables))  # 처음 20개만 테스트
        
        for i, table in enumerate(tables[:test_count]):
            print(f"\n📊 표 {i+1}/{test_count} 처리 중...")
            
            try:
                # Standalone HTML 생성
                standalone_html = processor._create_standalone_table_html(table, i)
                
                # 스크린샷 생성
                start_time = time.time()
                image_data = processor._screenshot_table_html(standalone_html)
                process_time = time.time() - start_time
                
                if image_data and len(image_data) > 0:
                    # 파일로 저장
                    image_path = output_dir / f"table_{i+1:03d}.png"
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    # 통계 업데이트
                    success_count += 1
                    file_size = len(image_data)
                    total_size += file_size
                    
                    print(f"    ✅ 저장 완료: {image_path.name}")
                    print(f"    📏 파일 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                    print(f"    ⏱️ 처리 시간: {process_time:.2f}초")
                    
                    # 이미지 크기 정보 (PIL로 확인)
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(image_data))
                        print(f"    🖼️ 이미지 크기: {img.width}x{img.height} pixels")
                        print(f"    🎨 이미지 모드: {img.mode}")
                    except Exception as e:
                        print(f"    ⚠️ 이미지 정보 확인 실패: {e}")
                        
                else:
                    print(f"    ❌ 스크린샷 생성 실패: 빈 이미지 데이터")
                    
            except Exception as e:
                print(f"    ❌ 표 {i+1} 처리 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 최종 결과 요약
        print("\n" + "=" * 70)
        print("📊 처리 결과 요약")
        print("=" * 70)
        print(f"✅ 성공: {success_count}개 / {test_count}개")
        print(f"📁 저장 위치: {output_dir}")
        print(f"💾 총 파일 크기: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        
        if success_count > 0:
            print(f"📏 평균 파일 크기: {total_size/success_count:,.0f} bytes")
            
    except Exception as e:
        print(f"❌ 전체 처리 실패: {e}")
        import traceback
        traceback.print_exc()

def test_single_table_html():
    """단일 표 HTML로 간단 테스트"""
    print("\n" + "=" * 70)
    print("단일 표 HTML 테스트")
    print("=" * 70)
    
    # 출력 디렉토리
    output_dir = project_root / "output_screenshot"
    output_dir.mkdir(exist_ok=True)
    
    # HWP 프로세서 초기화
    try:
        processor = HWPProcessor(extract_table_images=True)
    except Exception as e:
        print(f"❌ HWP 프로세서 초기화 실패: {e}")
        return
    
    # 테스트용 간단한 HTML
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: 'Malgun Gothic', Arial, sans-serif; 
                margin: 10px; 
                background-color: white;
                font-size: 16px;
            }
            table { 
                border-collapse: collapse; 
                width: auto;   
                margin: 0 auto;
            }
            th, td { 
                border: 1px solid #333; 
                padding: 6px 10px;
                text-align: left;
                vertical-align: middle;
                word-wrap: break-word;
            }
            th { 
                background-color: #f5f5f5; 
                font-weight: bold; 
            }
        </style>
    </head>
    <body>
        <table>
            <tr>
                <th>항목</th>
                <th>내용</th>
                <th>비고</th>
            </tr>
            <tr>
                <td>테스트 1</td>
                <td>스크린샷 테스트</td>
                <td>성공</td>
            </tr>
            <tr>
                <td>테스트 2</td>
                <td>이미지 생성</td>
                <td>확인</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    try:
        print("📊 단일 테스트 표 스크린샷 생성 중...")
        
        # 스크린샷 생성
        start_time = time.time()
        image_data = processor._screenshot_table_html(test_html)
        process_time = time.time() - start_time
        
        if image_data and len(image_data) > 0:
            # 파일로 저장
            image_path = output_dir / "test_table_simple.png"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            print(f"✅ 저장 완료: {image_path}")
            print(f"📏 파일 크기: {len(image_data):,} bytes")
            print(f"⏱️ 처리 시간: {process_time:.2f}초")
            
            # 이미지 정보
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                print(f"🖼️ 이미지 크기: {img.width}x{img.height} pixels")
            except Exception as e:
                print(f"⚠️ 이미지 정보 확인 실패: {e}")
        else:
            print("❌ 스크린샷 생성 실패")
            
    except Exception as e:
        print(f"❌ 단일 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """필요한 의존성 확인"""
    print("🔧 의존성 라이브러리 확인")
    print("-" * 30)
    
    dependencies = [
        ("selenium", "selenium"),
        ("webdriver_manager", "webdriver-manager"), 
        ("bs4", "beautifulsoup4"),
        ("PIL", "pillow")
    ]
    
    missing = []
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {import_name}: 설치됨")
        except ImportError:
            print(f"❌ {import_name}: 설치 필요")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ 설치 필요한 라이브러리: {', '.join(missing)}")
        print("pip install selenium webdriver-manager beautifulsoup4 pillow 명령으로 설치해주세요.")
        return False
    
    return True

def test_table_splitting():
    """표 분할 기능 테스트 - 실제 XHTML 파일 사용"""
    print("\n" + "=" * 70)
    print("표 분할 기능 테스트 (XHTML 파일 기반)")
    print("=" * 70)
    
    # 출력 디렉토리
    output_dir = project_root / "output_screenshot"
    output_dir.mkdir(exist_ok=True)
    
    # HWP 프로세서 초기화
    try:
        processor = HWPProcessor(extract_table_images=True)
        print(f"📏 분할 높이 기준: {processor.CROP_HEIGHT_PX}px")
        print(f"🔄 겹침 높이: {processor.OVERLAP_HEIGHT}px")
    except Exception as e:
        print(f"❌ HWP 프로세서 초기화 실패: {e}")
        return
    
    # test.xhtml 파일 경로
    xhtml_path = project_root / "data" / "xhtml" / "test.xhtml"
    
    if not xhtml_path.exists():
        print(f"❌ XHTML 파일을 찾을 수 없습니다: {xhtml_path}")
        return
    
    print(f"📄 XHTML 파일: {xhtml_path}")
    
    try:
        # XHTML 파일 읽기
        with open(xhtml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # 모든 표 요소 찾기
        tables = soup.find_all('table')
        print(f"🔍 발견된 표 개수: {len(tables)}개")
        
        if not tables:
            print("⚠️ XHTML 파일에서 표를 찾을 수 없습니다.")
            return
        
        # 처음 30개 표 테스트 (분할 기능 중심)
        test_count = min(30, len(tables))
        print(f"📊 테스트할 표 개수: {test_count}개")
        
        split_count = 0
        single_count = 0
        total_parts = 0
        
        for i, table in enumerate(tables[:test_count]):
            print(f"\n📊 표 {i+1}/{test_count} 분할 테스트 중...")
            
            try:
                # Standalone HTML 생성
                standalone_html = processor._create_standalone_table_html(table, i)
                
                # 분할 기능 테스트
                start_time = time.time()
                image_parts = processor._split_table_by_pixels(standalone_html)
                process_time = time.time() - start_time
                
                is_split = len(image_parts) > 1
                
                if is_split:
                    split_count += 1
                    total_parts += len(image_parts)
                    print(f"    ✂️ 분할됨: {len(image_parts)}개 이미지")
                    
                    # 분할된 이미지들 저장
                    for part_idx, image_data in enumerate(image_parts):
                        if image_data and len(image_data) > 0:
                            image_path = output_dir / f"table_{i+1:03d}_part_{part_idx+1:02d}.png"
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                            
                            # 이미지 크기 확인
                            try:
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(image_data))
                                
                                # 높이 체크
                                if img.height > processor.CROP_HEIGHT_PX + 100:
                                    print(f"        ⚠️ Part {part_idx+1}: 높이 초과 {img.height}px")
                                else:
                                    print(f"        ✅ Part {part_idx+1}: {img.width}x{img.height}px ({len(image_data)/1024:.1f}KB)")
                                    
                            except Exception as e:
                                print(f"        ❌ Part {part_idx+1}: 이미지 정보 확인 실패")
                else:
                    single_count += 1
                    total_parts += 1
                    print(f"    � 단일 이미지: {len(image_parts[0])/1024:.1f}KB")
                    
                    # 단일 이미지 저장
                    image_path = output_dir / f"table_{i+1:03d}_single.png"
                    with open(image_path, 'wb') as f:
                        f.write(image_parts[0])
                
                print(f"    ⏱️ 처리 시간: {process_time:.2f}초")
                
            except Exception as e:
                print(f"    ❌ 표 {i+1} 처리 실패: {e}")
                continue
        
        # 최종 결과 요약
        print("\n" + "=" * 70)
        print("📊 분할 테스트 결과 요약")
        print("=" * 70)
        print(f"📄 테스트된 표: {test_count}개")
        print(f"✂️ 분할된 표: {split_count}개")
        print(f"📄 단일 표: {single_count}개")
        print(f"🖼️ 총 생성된 이미지: {total_parts}개")
        print(f"📁 저장 위치: {output_dir}")
        
        if split_count > 0:
            avg_parts = (total_parts - single_count) / split_count
            print(f"📏 분할된 표 평균 부분 수: {avg_parts:.1f}개")
            
    except Exception as e:
        print(f"❌ 분할 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 실행 함수"""
    print("🚀 HWP 프로세서 표 스크린샷 + 분할 테스트 시작")
    print(f"📂 프로젝트 루트: {project_root}")
    
    # 의존성 확인
    if not check_dependencies():
        return
    
    print()
    
    # 1. 단일 표 테스트 (빠른 확인)
    test_single_table_html()
    
    # 2. 분할 기능 테스트 (새로 추가!)
    test_table_splitting()
    
    print()
    
    # 3. 실제 XHTML 파일 테스트 (일부만)
    # test_screenshot_table_html()  # 주석처리 (시간 절약)
    
    print("\n🎉 테스트 완료!")
    print("📁 결과 확인: output_screenshot/ 폴더의 이미지들을 확인해보세요!")
    print("   - test_table_simple.png: 단일 표 테스트")
    print("   - table_XXX_single.png: 분할되지 않은 표들")
    print("   - table_XXX_part_XX.png: 분할된 표들")

if __name__ == "__main__":
    main()