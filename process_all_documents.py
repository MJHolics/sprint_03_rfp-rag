#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 문서 처리 스크립트 - 100개 문서 모두 처리
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem
import sqlite3
import time
from datetime import datetime

def get_remaining_files():
    """처리되지 않은 파일 목록 반환"""
    # 이미 처리된 파일 목록
    conn = sqlite3.connect('rfp_metadata.db')
    cursor = conn.cursor()
    cursor.execute('SELECT file_name FROM documents')
    processed_files = [row[0] for row in cursor.fetchall()]
    conn.close()

    # 전체 파일 목록
    files_dir = Path('./files')
    all_files = []
    for f in files_dir.iterdir():
        if f.suffix in ['.pdf', '.hwp']:
            all_files.append(f)

    # 남은 파일들
    remaining_files = [f for f in all_files if f.name not in processed_files]

    return remaining_files, len(processed_files), len(all_files)

def process_all_documents():
    """모든 문서 처리"""
    print("=== 전체 문서 처리 시작 ===")
    start_time = datetime.now()

    # RAG 시스템 초기화
    rag = RAGSystem()

    # 처리 전 상태
    remaining_files, processed_count, total_count = get_remaining_files()

    print(f"전체 파일: {total_count}개")
    print(f"이미 처리됨: {processed_count}개")
    print(f"남은 파일: {len(remaining_files)}개")
    print(f"목표: {total_count}개 모두 처리\n")

    if not remaining_files:
        print("모든 파일이 이미 처리되었습니다!")
        return True

    # 배치 처리
    batch_size = 5  # 5개씩 처리
    total_batches = (len(remaining_files) + batch_size - 1) // batch_size

    successful = 0
    failed = 0

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]

        print(f"\n=== 배치 {batch_idx + 1}/{total_batches} 처리 ({len(batch_files)}개 파일) ===")

        for i, file_path in enumerate(batch_files, 1):
            try:
                relative_idx = batch_start + i
                file_name = file_path.name
                display_name = file_name[:60] + "..." if len(file_name) > 60 else file_name

                print(f"[{relative_idx}/{len(remaining_files)}] 처리 중: {display_name}")

                # 파일 처리
                file_start_time = time.time()
                result = rag.process_document(str(file_path))
                processing_time = time.time() - file_start_time

                if result and result.chunks:
                    chunk_count = len(result.chunks)
                    print(f"  [OK] 성공: {chunk_count}개 청크 생성 ({processing_time:.1f}초)")
                    successful += 1
                else:
                    print(f"  [FAIL] 실패: 결과 없음")
                    failed += 1

            except Exception as e:
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                print(f"  [ERROR] 오류: {error_msg}")
                failed += 1

        # 배치 완료 후 중간 통계
        current_stats = rag.get_system_stats()
        current_docs = current_stats['metadata_store'].get('total_documents', 0)
        current_chunks = current_stats['metadata_store'].get('total_chunks', 0)

        print(f"\n배치 {batch_idx + 1} 완료:")
        print(f"  현재 총 문서: {current_docs}개")
        print(f"  현재 총 청크: {current_chunks}개")
        print(f"  성공: {successful}개")
        print(f"  실패: {failed}개")

        # 진행률 표시
        progress = ((batch_idx + 1) / total_batches) * 100
        print(f"  전체 진행률: {progress:.1f}%")

        # 짧은 휴식 (API 레이트 리밋 방지)
        if batch_idx < total_batches - 1:
            print("  다음 배치까지 5초 대기...")
            time.sleep(5)

    # 최종 결과
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    final_stats = rag.get_system_stats()
    final_docs = final_stats['metadata_store'].get('total_documents', 0)
    final_chunks = final_stats['metadata_store'].get('total_chunks', 0)

    print(f"\n=== 처리 완료 ===")
    print(f"처리 시간: {total_time/60:.1f}분")
    print(f"최종 문서 수: {final_docs}개")
    print(f"최종 청크 수: {final_chunks:,}개")
    print(f"성공: {successful}개")
    print(f"실패: {failed}개")
    print(f"성공률: {(successful/(successful+failed)*100):.1f}%")

    if final_docs >= 100:
        print("\n[SUCCESS] 100개 문서 처리 목표 달성!")
    else:
        print(f"\n[WARNING] 목표 미달: {100-final_docs}개 더 필요")

    return successful > 0

if __name__ == "__main__":
    try:
        success = process_all_documents()
        if success:
            print("\n[COMPLETE] 문서 처리가 완료되었습니다!")
        else:
            print("\n[FAILED] 문서 처리에 실패했습니다.")
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류: {str(e)}")