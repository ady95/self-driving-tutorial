"""11-3: 자연어 명령 → 주행 매개변수. LLM이 해석하고, 규칙이 허용 범위를 정합니다.

LLM의 답을 그대로 차에 넣지 않습니다. 정해진 항목만, 정해진 범위 안에서만 받아들입니다(10-4의 '규칙의 울타리').

python ch11/command.py                       # 예시 명령들을 로컬 qwen3.5:9b로 해석
python ch11/command.py --backend openai --api responses
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch09"))
from vlm import make_client  # noqa: E402

DEFAULT = {"cruise_kmh": 30.0, "time_gap_s": 2.0, "pedestrian_margin_m": 3.0, "allow_bypass": True}
LIMITS = {"cruise_kmh": (10.0, 30.0),             # 도로 제한 속도(30km/h)를 넘을 수 없다
          "time_gap_s": (1.5, 4.0),               # 앞차 간격은 1.5초보다 좁힐 수 없다
          "pedestrian_margin_m": (3.0, 6.0)}      # 보행자 주의 폭은 3m보다 좁힐 수 없다
PROMPT = """너는 자율주행차의 명령 해석기다. 탑승자의 말을 아래 JSON 하나로만 바꿔라. 다른 말은 쓰지 마라.
{"cruise_kmh": 순항 속도 km/h (기본 30),
 "time_gap_s": 앞차와의 시간 간격 초 (기본 2.0, 안전거리를 더 두라면 크게),
 "pedestrian_margin_m": 경로에서 이 거리 안의 보행자에게 감속할 폭 m (기본 3.0, 보행자 조심하라면 크게),
 "allow_bypass": 멈춰 선 장애물·공사 구간을 옆 차로로 피해 가도 되는지 true/false (기본 true),
 "reply": 탑승자에게 할 한 문장 대답}
탑승자의 말: """
EXAMPLES = [
    "앞 차량과 안전거리를 유지하면서 목적지까지 가.",
    "앞에 공사 구간이 있으니 안전하게 피해 목적지까지 가라.",
    "아이들이 많은 동네니까 천천히, 사람 조심해서 가 줘.",
    "늦었어! 최대한 빨리, 시속 80으로 밟아.",
    "앞차에 바짝 붙어서 따라가.",
    "차선 변경은 하지 말고 그냥 가.",
]


def clamp(got):
    """LLM이 낸 값을 허용 범위 안의 유한한 숫자로. 숫자가 아니거나 NaN·무한대면 기본값으로 되돌린다."""
    params, notes = dict(DEFAULT), []
    if not isinstance(got, dict):                       # JSON 객체가 아니면 전부 기본값
        return params, ["JSON 객체가 아님 → 모두 기본값"] if got else []
    for k, (lo, hi) in LIMITS.items():
        if k not in got:
            continue
        v = got[k]
        try:
            v = float(v) if not isinstance(v, bool) else None   # true/false는 숫자로 받지 않는다
        except (TypeError, ValueError):
            v = None
        if v is None or not math.isfinite(v):                    # "abc", "NaN", Infinity 등
            notes.append(f"{k} {got[k]!r} → 기본값 {DEFAULT[k]:g} (유한한 숫자가 아님)")
            continue
        clipped = min(max(v, lo), hi)
        if clipped != v:
            notes.append(f"{k} {v:g} → {clipped:g} (허용 범위 {lo:g}~{hi:g})")
        params[k] = clipped
    if isinstance(got.get("allow_bypass"), bool):
        params["allow_bypass"] = got["allow_bypass"]
    assert all(math.isfinite(params[k]) and LIMITS[k][0] <= params[k] <= LIMITS[k][1] for k in LIMITS)
    return params, notes


def parse_command(text, client, model, api="chat"):
    """명령 → (매개변수, 바뀐 항목 설명, 대답, LLM 원문)."""
    if api == "responses":
        r = client.responses.create(model=model, input=PROMPT + text, reasoning={"effort": "none"})
        raw = r.output_text
    else:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": PROMPT + text}],
                                           reasoning_effort="none")
        raw = r.choices[0].message.content
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        got = {}
    params, notes = clamp(got)
    reply = got.get("reply", "") if isinstance(got, dict) else ""
    return params, notes, reply, raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--api", default="chat", choices=["chat", "responses"])
    args = ap.parse_args()
    client, model = make_client(args.backend, args.model)
    print(f"[{model}]")
    for text in EXAMPLES:
        params, notes, reply, _ = parse_command(text, client, model, args.api)
        changed = {k: v for k, v in params.items() if v != DEFAULT[k]}
        print(f"\n명령: {text}\n  → 바뀐 매개변수 {changed or '없음'}" + (f"\n  → 규칙이 고침: {notes}" if notes else "")
              + f"\n  → 대답: {reply}")
