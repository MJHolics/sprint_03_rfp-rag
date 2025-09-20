#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HWP 프로세서의 문서 텍스트 추출 테스트
data/sample의 HWP 파일로 텍스트 추출 및 요약 기능 테스트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.processors.hwp_processor import HWPProcessor
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    print("필요한 패키지들이 설치되어 있는지 확인해주세요.")
    sys.exit(1)

def test_document_text_extraction():
    """문서 텍스트 추출 테스트"""
    print("=" * 70)
    print("HWP 문서 텍스트 추출 및 요약 테스트")
    print("=" * 70)
    
    # HWP 프로세서 초기화 (표 이미지 처리 없이)
    try:
        processor = HWPProcessor(extract_table_images=False)
    except Exception as e:
        print(f"❌ HWP 프로세서 초기화 실패: {e}")
        return
    
    # sample 파일 경로
    sample_dir = project_root / "data" / "sample"
    hwp_files = list(sample_dir.glob("*.hwp"))
    
    if not hwp_files:
        print(f"❌ {sample_dir}에서 HWP 파일을 찾을 수 없습니다.")
        return
    
    hwp_file = hwp_files[0]  # 첫 번째 HWP 파일 사용
    print(f"📄 테스트 파일: {hwp_file.name}")
    
    try:
        # 1. 텍스트 추출
        print("\n🔍 1단계: 텍스트 추출 중...")
        text_content = processor._extract_hwp_text(str(hwp_file))
        
        if text_content:
            print(f"✅ 텍스트 추출 성공!")
            print(f"📏 텍스트 길이: {len(text_content):,}자")
            print(f"📄 라인 수: {len(text_content.split('\n')):,}줄")
            
            # 첫 500자 미리보기
            preview = text_content[:500]
            print(f"\n📖 텍스트 미리보기 (첫 500자):")
            print("-" * 50)
            print(preview)
            if len(text_content) > 500:
                print("...")
            print("-" * 50)
            
        else:
            print("❌ 텍스트 추출 실패: 빈 내용")
            return
            
        # 2. 문서 요약 테스트
        print(f"\n🔍 2단계: 문서 요약 테스트...")
        summary = processor._summarize_document_if_needed(text_content)
        
        print(f"📏 요약 결과:")
        print(f"   - 원본: {len(text_content):,}자")
        print(f"   - 요약: {len(summary):,}자")
        print(f"   - 압축률: {len(summary)/len(text_content)*100:.1f}%")
        
        if summary != text_content:
            print(f"\n📖 요약 내용 미리보기 (첫 300자):")
            print("-" * 50)
            print(summary[:300])
            if len(summary) > 300:
                print("...")
            print("-" * 50)
        else:
            print("📝 문서가 짧아서 요약하지 않음")
            
        # 3. 키워드 분석
        print(f"\n🔍 3단계: 키워드 분석...")
        keywords = ['제안', '사업', '프로젝트', '개발', '시스템', '서비스', '예산', '일정']
        found_keywords = []
        
        for keyword in keywords:
            count = text_content.count(keyword)
            if count > 0:
                found_keywords.append(f"{keyword}({count})")
        
        if found_keywords:
            print(f"🔑 발견된 키워드: {', '.join(found_keywords)}")
        else:
            print("🔑 지정된 키워드를 찾을 수 없음")
            
        # 4. GPT 입력 준비 상태 확인
        print(f"\n🔍 4단계: GPT 입력 준비 상태 확인...")
        
        # 예상 토큰 수 계산 (대략 한글 1토큰 = 2-3자)
        estimated_tokens = len(summary) // 2
        print(f"📊 예상 토큰 수: {estimated_tokens:,} tokens")
        
        if estimated_tokens > 100000:  # 100K 토큰
            print("⚠️ 토큰 수가 많아서 GPT 입력 시 제한될 수 있음")
        elif estimated_tokens > 50000:  # 50K 토큰
            print("⚠️ 토큰 수가 다소 많음 - 요약 권장")
        else:
            print("✅ GPT 입력에 적합한 크기")
            
        print(f"\n🎉 문서 텍스트 추출 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 실행 함수"""
    print("🚀 HWP 문서 텍스트 추출 테스트 시작")
    print(f"📂 프로젝트 루트: {project_root}")
    
    test_document_text_extraction()

if __name__ == "__main__":
    main()