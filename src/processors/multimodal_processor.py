"""
멀티모달 문서 처리기 - GPT-4V를 활용한 이미지 분석
"""
import os
import base64
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
from pathlib import Path
import openai
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL

class MultimodalProcessor:
    """멀티모달 문서 처리기 - 이미지 분석 및 OCR"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def analyze_pdf_images(self, file_path: str, max_images: int = 5) -> List[Dict[str, Any]]:
        """PDF 이미지 분석 및 텍스트 추출"""
        if not self.client:
            return []

        try:
            doc = fitz.open(file_path)
            analyzed_images = []

            for page_num in range(min(len(doc), 10)):  # 최대 10페이지만 처리
                page = doc[page_num]
                images = page.get_images(full=True)

                for img_index, img in enumerate(images[:max_images]):
                    if len(analyzed_images) >= max_images:
                        break

                    try:
                        # 이미지 추출
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # RGB로 변환 (CMYK인 경우)
                        if pix.n >= 5:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # 이미지를 base64로 인코딩
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()

                        # GPT-4V로 이미지 분석
                        analysis = self._analyze_image_with_gpt4v(img_base64)

                        analyzed_images.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "width": pix.width,
                            "height": pix.height,
                            "analysis": analysis,
                            "extracted_text": analysis.get("extracted_text", ""),
                            "description": analysis.get("description", "")
                        })

                        pix = None  # 메모리 해제

                    except Exception as e:
                        print(f"이미지 {img_index} 분석 실패: {e}")
                        continue

            doc.close()
            return analyzed_images

        except Exception as e:
            print(f"PDF 이미지 분석 실패: {e}")
            return []

    def _analyze_image_with_gpt4v(self, img_base64: str) -> Dict[str, str]:
        """GPT-4V로 이미지 분석"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4V 지원 모델
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """이 이미지를 분석해주세요. 다음 정보를 JSON 형태로 제공해주세요:
1. extracted_text: 이미지에서 추출한 모든 텍스트 (표, 차트의 숫자와 라벨 포함)
2. description: 이미지 내용에 대한 상세한 설명 (차트 유형, 데이터 트렌드, 표 구조 등)
3. content_type: 이미지 타입 (table, chart, diagram, text, other)

RFP(제안요청서) 문서의 이미지이므로 기술 요구사항, 예산, 일정, 시스템 구조도 등에 중점을 두어 분석해주세요."""
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
                # JSON 블록 추출
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
                # JSON 파싱 실패 시 텍스트 그대로 반환
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

    def enhance_content_with_images(self, content: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """기존 텍스트 콘텐츠에 이미지 분석 결과 통합"""
        try:
            # 이미지 분석 수행
            image_analyses = self.analyze_pdf_images(file_path)

            # 이미지에서 추출한 텍스트를 메인 텍스트에 추가
            enhanced_text = content.get("text", "")

            for img_analysis in image_analyses:
                page_num = img_analysis["page"]
                extracted_text = img_analysis["extracted_text"]
                description = img_analysis["description"]
                content_type = img_analysis["content_type"]

                if extracted_text:
                    enhanced_text += f"\n\n[페이지 {page_num} {content_type} 분석]\n"
                    enhanced_text += f"추출된 텍스트: {extracted_text}\n"
                    if description != extracted_text:
                        enhanced_text += f"상세 설명: {description}\n"

            # 기존 content 업데이트
            enhanced_content = content.copy()
            enhanced_content["text"] = enhanced_text
            enhanced_content["image_analyses"] = image_analyses
            enhanced_content["total_analyzed_images"] = len(image_analyses)

            return enhanced_content

        except Exception as e:
            print(f"멀티모달 콘텐츠 강화 실패: {e}")
            return content