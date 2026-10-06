"""09-2: VLM에게 주행 장면을 보여 주고 운전자처럼 질문합니다.

python ch09/ask_scene.py                                   # 맑은 낮 200번 프레임, 로컬 qwen3.5:9b
python ch09/ask_scene.py --weather night --frame 200
python ch09/ask_scene.py --backend openai                  # OpenAI API
python ch09/ask_scene.py --think                           # 모델의 '생각' 단계를 켜고 (느림)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import ask, load_frame, make_client  # noqa: E402

QUESTIONS = [
    "이 장면을 운전자의 입장에서 두세 문장으로 설명해 줘.",
    "앞쪽의 차량들은 각각 무엇을 하고 있어?",
    "지금 왼쪽 차로로 차선을 변경해도 될까? 이유와 함께 답해 줘.",
    "지금 이 장면에서 운전자가 주의해야 할 위험 요소를 중요한 순서대로 알려 줘.",
    "자차는 지금 어떻게 해야 해? 출발·감속·정지 중 하나를 고르고 이유를 한 문장으로 말해 줘.",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--api", default="chat", choices=["chat", "responses"])
    ap.add_argument("--weather", default="clear", choices=["clear", "night", "rain"])
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()
    client, model = make_client(args.backend, args.model)
    img = load_frame(args.weather, args.frame)
    print(f"[{model}] {args.weather} {args.frame}번 프레임, think={args.think}\n")
    for q in QUESTIONS:
        answer, sec, tokens = ask(client, model, img, q, think=args.think, api=args.api)
        print(f"Q. {q}\nA. {answer}\n   ({sec:.1f}초, 출력 {tokens}토큰)\n", flush=True)
