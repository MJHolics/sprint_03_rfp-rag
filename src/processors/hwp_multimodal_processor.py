"""
HWP 파일 직접 멀티모달 처리기
olefile을 사용하여 HWP 파일에서 이미지를 직접 추출하고 GPT-4V로 분석
"""
import os
import io
import base64
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL

try:
    import olefile
    import openai
    from PIL import Image
except ImportError as e:
    print(f"필수 라이브러리 누락: {e}")
    print("pip install olefile pillow openai 실행 필요")

class HWPMultimodalProcessor:
    """HWP 파일 직접 멀티모달 처리기"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def extract_images_from_hwp(self, file_path: str, max_images: int = 5) -> List[Dict[str, Any]]:
        """HWP 파일에서 이미지 직접 추출"""
        images = []

        try:
            if not olefile.isOleFile(file_path):
                print(f"HWP 파일이 아닙니다: {file_path}")
                return []

            ole = olefile.OleFileIO(file_path)

            # HWP 파일 구조에서 이미지 스트림 찾기
            # HWP는 OLE 복합 문서 형식으로 이미지가 별도 스트림에 저장됨
            listdir = ole.listdir()

            image_count = 0
            for entry in listdir:
                if image_count >= max_images:
                    break

                # 이미지 스트림 패턴 찾기
                # HWP에서 이미지는 주로 'BinData', 'Pictures', 'Images' 등의 이름으로 저장
                stream_name = '/'.join(entry)

                if any(keyword in stream_name.lower() for keyword in ['bindata', 'picture', 'image', 'img']):
                    try:
                        # 스트림 데이터 읽기 (수정된 방식)
                        with ole.openstream(entry) as stream:
                            stream_data = stream.read()

                            # 이미지 데이터인지 확인 (매직 바이트 체크)
                            if self._is_image_data(stream_data):
                                try:
                                    # PIL로 이미지 읽기 시도
                                    img = Image.open(io.BytesIO(stream_data))

                                    # PNG로 변환하여 base64 인코딩
                                    img_buffer = io.BytesIO()
                                    # RGBA 모드로 변환 (투명도 지원)
                                    if img.mode not in ('RGB', 'RGBA'):
                                        img = img.convert('RGB')
                                    img.save(img_buffer, format='PNG')
                                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                                    images.append({
                                        'index': image_count,
                                        'stream_name': stream_name,
                                        'width': img.size[0],
                                        'height': img.size[1],
                                        'format': img.format or 'Unknown',
                                        'mode': img.mode,
                                        'size': len(stream_data),
                                        'base64': img_base64
                                    })

                                    image_count += 1
                                    print(f"HWP 이미지 추출: {stream_name} ({img.size[0]}x{img.size[1]})")

                                except Exception as img_error:
                                    print(f"이미지 처리 실패 {stream_name}: {img_error}")
                                    continue

                    except Exception as stream_error:
                        print(f"스트림 읽기 실패 {stream_name}: {stream_error}")
                        continue

            ole.close()
            return images

        except Exception as e:
            print(f"HWP 이미지 추출 실패: {e}")
            return []

    def _is_image_data(self, data: bytes) -> bool:
        """데이터가 이미지인지 매직 바이트로 확인"""
        if len(data) < 8:
            return False

        # 주요 이미지 포맷 매직 바이트
        image_signatures = [
            b'\xFF\xD8\xFF',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'BM',  # BMP
            b'RIFF',  # WebP (RIFF container)
            b'\x00\x00\x01\x00',  # ICO
        ]

        for signature in image_signatures:
            if data.startswith(signature):
                return True

        return False

    def analyze_hwp_images(self, file_path: str, max_images: int = 5) -> List[Dict[str, Any]]:
        """HWP 이미지 분석 및 텍스트 추출"""
        if not self.client:
            print("OpenAI 클라이언트가 없습니다")
            return []

        try:
            # HWP에서 이미지 추출
            images = self.extract_images_from_hwp(file_path, max_images)
            analyzed_images = []

            for img_data in images:
                try:
                    # GPT-4V로 이미지 분석
                    analysis = self._analyze_image_with_gpt4v(img_data['base64'])

                    analyzed_images.append({
                        **img_data,
                        "analysis": analysis,
                        "extracted_text": analysis.get("extracted_text", ""),
                        "description": analysis.get("description", ""),
                        "content_type": analysis.get("content_type", "unknown")
                    })

                    print(f"HWP 이미지 분석 완료: {img_data['stream_name']}")

                except Exception as e:
                    print(f"이미지 분석 실패 {img_data['stream_name']}: {e}")
                    continue

            return analyzed_images

        except Exception as e:
            print(f"HWP 이미지 분석 실패: {e}")
            return []

    def _analyze_image_with_gpt4v(self, img_base64: str) -> Dict[str, str]:
        """GPT-4V로 이미지 분석 (PDF와 동일한 로직)"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """이 HWP 문서의 이미지를 분석해주세요. 다음 정보를 JSON 형태로 제공해주세요:
1. extracted_text: 이미지에서 추출한 모든 텍스트 (표, 차트의 숫자와 라벨, 한글 텍스트 포함)
2. description: 이미지 내용에 대한 상세한 설명 (차트 유형, 데이터 트렌드, 표 구조, 다이어그램 설명 등)
3. content_type: 이미지 타입 (table, chart, diagram, flowchart, text, screenshot, other)

RFP(제안요청서) 문서의 이미지이므로 다음에 특히 주의해주세요:
- 기술 요구사항, 시스템 구조도
- 예산 관련 표와 수치
- 일정표, 프로젝트 계획
- 업무 흐름도, 프로세스 다이어그램
- 조직도, 역할 분담표

한글 텍스트 추출에 특별히 신경써주세요."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # JSON 파싱 시도
            try:
                import json
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content

                parsed = json.loads(json_content)
                return {
                    "extracted_text": parsed.get("extracted_text", ""),
                    "description": parsed.get("description", ""),
                    "content_type": parsed.get("content_type", "other")
                }
            except:
                return {
                    "extracted_text": content,
                    "description": content,
                    "content_type": "unknown"
                }

        except Exception as e:
            print(f"GPT-4V 분석 실패: {e}")
            return {
                "extracted_text": "",
                "description": f"분석 실패: {str(e)}",
                "content_type": "error"
            }

    def enhance_hwp_content_with_images(self, content: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """HWP 콘텐츠에 이미지 분석 결과 통합"""
        try:
            print(f"HWP 멀티모달 분석 시작: {os.path.basename(file_path)}")

            # 이미지 분석 수행
            image_analyses = self.analyze_hwp_images(file_path)

            # 이미지에서 추출한 텍스트를 메인 텍스트에 추가
            enhanced_text = content.get("text", "")

            for img_analysis in image_analyses:
                stream_name = img_analysis.get("stream_name", "알 수 없음")
                extracted_text = img_analysis["extracted_text"]
                description = img_analysis["description"]
                content_type = img_analysis["content_type"]

                if extracted_text:
                    enhanced_text += f"\n\n[HWP 이미지 분석: {stream_name} - {content_type}]\n"
                    enhanced_text += f"추출된 텍스트: {extracted_text}\n"
                    if description != extracted_text:
                        enhanced_text += f"상세 설명: {description}\n"

            # 기존 content 업데이트
            enhanced_content = content.copy()
            enhanced_content["text"] = enhanced_text
            enhanced_content["hwp_image_analyses"] = image_analyses
            enhanced_content["total_analyzed_hwp_images"] = len(image_analyses)

            print(f"HWP 멀티모달 분석 완료: {len(image_analyses)}개 이미지 분석")
            return enhanced_content

        except Exception as e:
            print(f"HWP 멀티모달 콘텐츠 강화 실패: {e}")
            return content