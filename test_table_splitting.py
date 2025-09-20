#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
표 분할 기능 테스트
"""
import sys
sys.path.append('src')

from processors.hwp_processor import HWPProcessor
from processors.base import TableImageChunk

def test_table_splitting():
    """표 분할 기능 테스트"""
    
    # 긴 표 HTML 생성 (A4 높이를 초과하도록)
    long_table_html = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                font-family: 'Malgun Gothic', sans-serif;
                font-size: 16px;
            }
            th, td {
                border: 1px solid #333;
                padding: 6px 10px;
                text-align: left;
                vertical-align: top;
            }
            th {
                background-color: #f5f5f5;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <table>
            <thead>
                <tr>
                    <th>번호</th>
                    <th>항목명</th>
                    <th>설명</th>
                    <th>비고</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # 많은 행 추가 (긴 표 만들기)
    for i in range(1, 101):  # 100개 행
        long_table_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>항목 {i}</td>
                    <td>이것은 항목 {i}에 대한 상세한 설명입니다. 여러 줄에 걸친 내용이 포함될 수 있습니다.</td>
                    <td>비고 {i}</td>
                </tr>
        """
    
    long_table_html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    # HWPProcessor 초기화
    processor = HWPProcessor(extract_table_images=True)
    
    print("표 분할 기능 테스트 시작...")
    
    try:
        # 표 스크린샷 (분할 기능 포함)
        image_parts = processor._screenshot_table_html(long_table_html)
        
        print(f"생성된 이미지 개수: {len(image_parts)}")
        
        # 이미지들을 파일로 저장
        import os
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for i, img_data in enumerate(image_parts):
            if img_data:
                filename = f"{output_dir}/table_part_{i+1}.png"
                with open(filename, 'wb') as f:
                    f.write(img_data)
                saved_files.append(filename)
                print(f"   - 이미지 {i+1}: {len(img_data)} bytes -> {filename}")
            else:
                print(f"   - 이미지 {i+1}: 생성 실패")
        
        if len(image_parts) > 1:
            print("✅ 표 분할 기능이 정상 작동합니다!")
            print(f"   - 총 {len(image_parts)}개의 이미지로 분할됨")
        else:
            print("⚠️ 표 분할이 발생하지 않음 (단일 이미지)")
            
        # 이미지 뷰어로 열기 시도
        if saved_files:
            print(f"\n📁 저장된 이미지 파일들:")
            for file in saved_files:
                print(f"   - {file}")
            
            # Windows에서 첫 번째 이미지 열기
            try:
                import subprocess
                subprocess.run(['start', saved_files[0]], shell=True, check=True)
                print(f"\n👀 첫 번째 이미지를 기본 뷰어로 열었습니다: {saved_files[0]}")
            except:
                print(f"\n💡 수동으로 이미지를 확인하세요: {output_dir} 폴더")
            
            # matplotlib으로 모든 이미지 표시
            try:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                from PIL import Image
                import io
                
                # 이미지 개수에 따라 subplot 구성
                num_images = len(image_parts)
                if num_images > 0:
                    fig, axes = plt.subplots(num_images, 1, figsize=(12, 4*num_images))
                    if num_images == 1:
                        axes = [axes]  # 단일 이미지인 경우 리스트로 변환
                    
                    for i, img_data in enumerate(image_parts):
                        if img_data:
                            # bytes를 PIL Image로 변환
                            img = Image.open(io.BytesIO(img_data))
                            axes[i].imshow(img)
                            axes[i].set_title(f'표 분할 이미지 {i+1}/{num_images}')
                            axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    print("\n📊 matplotlib으로 모든 이미지를 표시했습니다.")
                    
            except ImportError:
                print("\n💡 matplotlib이 설치되지 않아 이미지 표시를 건너뜁니다.")
                print("   설치하려면: pip install matplotlib pillow")
            except Exception as e:
                print(f"\n⚠️ 이미지 표시 중 오류: {e}")
            
        # TableImageChunk 생성 테스트
        test_chunk = TableImageChunk(
            content="테스트 표 설명",
            metadata={},
            chunk_id="test_chunk",
            document_id="test_doc",
            chunk_index=0,
            table_html=long_table_html,
            image_data=image_parts[0] if image_parts else b"",
            gpt_description="테스트 GPT 설명",
            image_parts=image_parts,
            is_split_table=len(image_parts) > 1,
            total_parts=len(image_parts)
        )
        
        print(f"\nTableImageChunk 테스트:")
        print(f"  - is_split_table: {test_chunk.is_split_table}")
        print(f"  - total_parts: {test_chunk.total_parts}")
        print(f"  - main_image 크기: {len(test_chunk.main_image)} bytes")
        print(f"  - get_all_images() 개수: {len(test_chunk.get_all_images())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # WebDriver 정리
        if hasattr(processor, 'driver') and processor.driver:
            processor.driver.quit()

if __name__ == "__main__":
    test_table_splitting()