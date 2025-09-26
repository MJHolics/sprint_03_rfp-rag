#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 재구축 스크립트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
import time

def rebuild_database():
    print("=== 데이터베이스 재구축 시작 ===")

    # RAG 시스템 초기화
    rag_system = RAGSystem()

    # 처리 가능한 파일 확인
    files_dir = Path('./files')
    pdf_files = list(files_dir.glob('*.pdf'))
    hwp_files = list(files_dir.glob('*.hwp'))

    print(f"PDF 파일: {len(pdf_files)}개")
    print(f"HWP 파일: {len(hwp_files)}개")

    # 일부 파일만 처리 (테스트용)
    test_files = pdf_files[:2] + hwp_files[:5]  # 7개 파일로 테스트

    processed = 0
    failed = 0

    for i, file_path in enumerate(test_files, 1):
        try:
            print(f"[{i}/{len(test_files)}] 처리 중: {file_path.name[:50]}...")
            start_time = time.time()

            result = rag_system.process_document(str(file_path))

            if result and result.chunks:
                chunks_count = len(result.chunks)
                processing_time = time.time() - start_time
                print(f"  [OK] {chunks_count}개 청크 생성 ({processing_time:.1f}초)")
                processed += 1
            else:
                print(f"  [FAIL] 처리 결과 없음")
                failed += 1

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            failed += 1

    print(f"\n=== 처리 완료 ===")
    print(f"성공: {processed}개")
    print(f"실패: {failed}개")

    # 최종 통계 확인
    stats = rag_system.get_system_stats()
    print(f"\n=== 최종 통계 ===")
    print(f"총 문서: {stats['metadata_store'].get('total_documents', 0)}개")
    print(f"총 청크: {stats['metadata_store'].get('total_chunks', 0)}개")
    print(f"벡터 청크: {stats['vector_store'].get('total_chunks', 0)}개")

    return processed > 0

if __name__ == "__main__":
    success = rebuild_database()
    if success:
        print("\n[SUCCESS] 데이터베이스 재구축 완료!")
    else:
        print("\n[ERROR] 데이터베이스 재구축 실패!")