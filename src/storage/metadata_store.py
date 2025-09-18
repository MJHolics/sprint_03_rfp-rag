"""
SQLite 기반 메타데이터 저장소
체계적인 문서 및 청크 메타데이터 관리
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from ..processors.base import DocumentChunk, ProcessingResult

class MetadataStore:
    """SQLite 기반 메타데이터 저장소"""

    def __init__(self, db_path: str = "rfp_metadata.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """데이터베이스 및 테이블 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 문서 메타데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_name TEXT,
                title TEXT,
                agency TEXT,
                budget TEXT,
                deadline TEXT,
                business_type TEXT,
                business_name TEXT,
                total_pages INTEGER,
                file_size INTEGER,
                source_type TEXT,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                metadata_json TEXT
            )
        ''')

        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agency ON documents(agency)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_business_type ON documents(business_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_date ON documents(processed_date)')

        # 청크 정보 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                content_preview TEXT,
                chunk_size INTEGER,
                section_index INTEGER,
                metadata_json TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')

        # 청크 테이블 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_index ON chunks(chunk_index)')

        # 검색 통계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                search_method TEXT,
                results_count INTEGER,
                confidence_score REAL,
                response_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 검색 통계 테이블 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON search_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_method ON search_logs(search_method)')

        conn.commit()
        conn.close()

    def save_document_metadata(self, processing_result: ProcessingResult, file_path: str) -> bool:
        """문서 처리 결과를 메타데이터 저장소에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 문서 메타데이터 추출
            if processing_result.chunks:
                first_chunk = processing_result.chunks[0]
                metadata = first_chunk.metadata
                document_id = first_chunk.document_id
            else:
                metadata = processing_result.extracted_metadata
                document_id = metadata.get('document_id', file_path)

            # 문서 정보 저장
            cursor.execute('''
                INSERT OR REPLACE INTO documents
                (id, file_path, file_name, title, agency, budget, deadline,
                 business_type, business_name, total_pages, file_size, source_type,
                 processing_time, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                file_path,
                metadata.get('file_name', Path(file_path).name),
                metadata.get('title', ''),
                metadata.get('agency', ''),
                metadata.get('budget', ''),
                metadata.get('deadline', ''),
                metadata.get('business_type', ''),
                metadata.get('business_name', ''),
                metadata.get('total_pages', 0),
                metadata.get('file_size', 0),
                metadata.get('source_type', ''),
                processing_result.processing_time,
                json.dumps(metadata, ensure_ascii=False)
            ))

            # 청크 정보 저장
            for chunk in processing_result.chunks:
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, document_id, chunk_index, content_preview,
                     chunk_size, section_index, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content[:200],  # 미리보기용 200자
                    len(chunk.content),
                    chunk.metadata.get('section_index', 0),
                    json.dumps(chunk.metadata, ensure_ascii=False)
                ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"메타데이터 저장 오류: {e}")
            return False

    def search_documents_by_filter(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """필터 조건으로 문서 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 기본 쿼리
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        # 필터 조건 추가
        if 'agency' in filters and filters['agency']:
            query += " AND (agency LIKE ? OR title LIKE ?)"
            agency_term = f"%{filters['agency']}%"
            params.extend([agency_term, agency_term])

        if 'business_type' in filters and filters['business_type']:
            query += " AND business_type LIKE ?"
            params.append(f"%{filters['business_type']}%")

        if 'budget_min' in filters and filters['budget_min']:
            # 예산 범위 필터링 (추후 구현)
            pass

        if 'date_from' in filters and filters['date_from']:
            query += " AND processed_date >= ?"
            params.append(filters['date_from'])

        if 'date_to' in filters and filters['date_to']:
            query += " AND processed_date <= ?"
            params.append(filters['date_to'])

        # 정렬
        query += " ORDER BY processed_date DESC"

        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def search_by_agency(self, agency_name: str) -> List[str]:
        """발주 기관으로 문서 검색 (하위 호환성)"""
        documents = self.search_documents_by_filter({'agency': agency_name})
        return [doc['file_path'] for doc in documents]

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """특정 문서의 모든 청크 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
        ''', (document_id,))

        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """저장소 통계 정보"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 문서 통계
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['total_documents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        stats['total_chunks'] = cursor.fetchone()[0]

        # 소스 타입별 통계
        cursor.execute('''
            SELECT source_type, COUNT(*)
            FROM documents
            GROUP BY source_type
        ''')
        stats['by_source_type'] = dict(cursor.fetchall())

        # 발주기관별 통계 (상위 10개)
        cursor.execute('''
            SELECT agency, COUNT(*) as count
            FROM documents
            WHERE agency IS NOT NULL AND agency != ''
            GROUP BY agency
            ORDER BY count DESC
            LIMIT 10
        ''')
        stats['top_agencies'] = dict(cursor.fetchall())

        # 최근 처리 통계
        cursor.execute('''
            SELECT DATE(processed_date) as date, COUNT(*)
            FROM documents
            WHERE processed_date >= date('now', '-30 days')
            GROUP BY DATE(processed_date)
            ORDER BY date DESC
        ''')
        stats['recent_processing'] = dict(cursor.fetchall())

        conn.close()
        return stats

    def log_search(self, query: str, search_method: str, results_count: int,
                   confidence_score: float = None, response_time: float = None):
        """검색 로그 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO search_logs
                (query, search_method, results_count, confidence_score, response_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (query, search_method, results_count, confidence_score, response_time))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"검색 로그 저장 오류: {e}")

    def import_from_csv(self, csv_path: str) -> int:
        """CSV 파일에서 메타데이터 가져오기"""
        try:
            df = pd.read_csv(csv_path)
            imported_count = 0

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    # CSV 컬럼명을 데이터베이스 컬럼에 매핑
                    file_name = row.get('파일명', row.get('file_name', ''))

                    cursor.execute('''
                        INSERT OR IGNORE INTO documents
                        (id, file_path, file_name, agency, business_name, budget, deadline, business_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        file_name,  # ID로 파일명 사용
                        f"./files/{file_name}",  # 가정된 경로
                        file_name,
                        row.get('발주기관', row.get('agency', '')),
                        row.get('사업명', row.get('business_name', '')),
                        row.get('예산', row.get('budget', '')),
                        row.get('마감일', row.get('deadline', '')),
                        row.get('사업분야', row.get('business_type', ''))
                    ))
                    imported_count += 1

                except Exception as e:
                    print(f"행 처리 오류: {e}")
                    continue

            conn.commit()
            conn.close()

            return imported_count

        except Exception as e:
            print(f"CSV 가져오기 오류: {e}")
            return 0

    def export_to_csv(self, output_path: str) -> bool:
        """메타데이터를 CSV로 내보내기"""
        try:
            conn = sqlite3.connect(self.db_path)

            df = pd.read_sql_query('''
                SELECT file_name, agency, business_name, budget, deadline,
                       business_type, total_pages, source_type, processed_date
                FROM documents
                ORDER BY processed_date DESC
            ''', conn)

            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            conn.close()

            return True

        except Exception as e:
            print(f"CSV 내보내기 오류: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """데이터베이스 백업"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"데이터베이스 백업 오류: {e}")
            return False