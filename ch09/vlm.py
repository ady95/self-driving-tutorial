"""09장 공통: OpenAI 호환 API로 VLM에 카메라 영상 한 장과 질문을 보냅니다.

같은 코드로 두 가지를 씁니다.
  - ollama : 내 GPU에서 오픈 VLM 실행 (기본: qwen3.5:9b). `ollama pull qwen3.5:9b` 후 사용
  - openai : OpenAI API. 환경 변수 OPENAI_API_KEY 필요, 모델은 --model 또는 OPENAI_MODEL
"""
import base64
import os
import time

import cv2
from openai import OpenAI

DATA = "data/carla_urban"


def make_client(backend, model=None):
    if backend == "ollama":
        return OpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama"), \
            model or "qwen3.5:9b"
    return OpenAI(), model or os.getenv("OPENAI_MODEL", "gpt-6-luna")


def load_frame(weather, frame):
    """carla_urban 데이터(03장)에서 한 프레임을 꺼낸다."""
    cap = cv2.VideoCapture(f"{DATA}/{weather}/rgb.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    if not ok:
        raise FileNotFoundError(f"{DATA}/{weather}/rgb.mp4 의 {frame}번 프레임을 읽지 못했습니다")
    return img


def ask(client, model, img, prompt, think=False, api="chat"):
    """영상 + 질문 → (답, 걸린 초, 출력 토큰 수). think=False면 모델의 '생각' 단계를 끈다."""
    url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()
    start = time.perf_counter()
    if api == "responses":                      # OpenAI Responses API 형식
        extra = {} if think else {"reasoning": {"effort": "none"}}
        r = client.responses.create(model=model, input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": url}]}], **extra)
        text, out_tokens = r.output_text, r.usage.output_tokens
    else:                                       # Chat Completions 형식 (Ollama·vLLM·OpenAI 공통)
        extra = {} if think else {"reasoning_effort": "none"}
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}], **extra)
        text, out_tokens = r.choices[0].message.content, r.usage.completion_tokens
    return (text or "").strip(), time.perf_counter() - start, out_tokens
