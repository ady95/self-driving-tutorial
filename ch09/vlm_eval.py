"""09-2: VLM의 장면 이해와 행동 판단을 CARLA 정답과 맞춰 채점합니다.

VLM에게 정해진 JSON 형식으로 답하게 하고(날씨, 차량 수, 보행자 수, 신호, 행동), 03장 carla_urban 데이터의
정답과 비교합니다.
  - 차량·보행자 수 : boxes.csv. 크게 보이는 것(strict)은 다 세야 하고, 조금이라도 보이는 것(loose)보다 많으면 환각
  - 신호          : 140번 프레임부터는 앞 교차로의 빨간불이 또렷이 보인다 (영상으로 확인). 그 전은 채점하지 않음
  - 행동          : 0~20번은 GO (녹화 당시 교차로를 그대로 통과), 180번부터는 STOP (빨간불에 정지선에서 대기).
                    그 사이(빨간불 교차로로 다가가는 중)는 분포만 본다

python ch09/vlm_eval.py                         # 로컬 qwen3.5:9b, 날씨 3종 x 20프레임
python ch09/vlm_eval.py --backend openai --api responses
python ch09/vlm_eval.py --hint                  # 신호등 읽는 법을 프롬프트에 덧붙여서
python ch09/vlm_eval.py --from-csv outputs/ch09/eval_qwen3.5-9b.csv     # 저장한 결과만 다시 집계
결과: outputs/ch09/eval_<모델>.csv
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm import DATA, ask, load_frame, make_client  # noqa: E402

OUT = Path("outputs/ch09")
PROMPT = """너는 자율주행차의 판단 보조 AI다. 전방 카메라 영상을 보고 아래 JSON 하나로만 답하라. 다른 말은 쓰지 마라.
{"weather": "clear" | "night" | "rain",
 "vehicles": 영상에 보이는 차량(승용차·버스·트럭) 수 (정수),
 "pedestrians": 영상에 보이는 보행자 수 (정수),
 "traffic_light": 자차 진행 방향 신호 "red" | "yellow" | "green" | "none",
 "action": 자차가 지금 할 행동 "GO" | "SLOW_DOWN" | "STOP",
 "reason": 행동의 이유 한 문장}"""
HINT = """
참고: 신호등은 세로로 세 개의 등이 있고 맨 위가 빨강, 가운데가 노랑, 맨 아래가 초록이다.
낮에는 빨간 등이 주황색처럼 보일 수 있으니 색보다 켜진 등의 위치로 판단하라.
옆이나 뒤를 보고 있어 등이 보이지 않는 신호등은 자차의 신호가 아니다."""
VEHICLE = {"car", "bus", "truck"}
STRICT, LOOSE = 1000, 50                        # 정답 상자 픽셀 수 기준: 크게 보임 / 조금이라도 보임


def ground_truth(weather):
    """프레임별 정답: 차량·보행자 수(strict, loose), 자차 속도."""
    gt = defaultdict(lambda: {"veh": [0, 0], "ped": [0, 0]})
    for r in csv.DictReader(open(f"{DATA}/{weather}/boxes.csv")):
        key = "veh" if r["class"] in VEHICLE else "ped"
        px = int(r["pixels"])
        g = gt[int(r["frame"])][key]
        g[0] += px >= STRICT
        g[1] += px >= LOOSE
    speed = {int(r["frame"]): float(r["speed_kmh"]) for r in csv.DictReader(open(f"{DATA}/{weather}/ego.csv"))}
    return gt, speed


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def judge_count(pred, strict, loose):
    if pred > loose:
        return "over"                           # 없는 것을 셌다 (환각)
    if pred < strict:
        return "under"                          # 크게 보이는 것을 놓쳤다
    return "ok"


def summarize(rows, title):
    ok = [r for r in rows if r["valid"]]
    secs = [float(r["sec"]) for r in rows]
    print(f"\n[{title}] {len(rows)}장, JSON 형식 지킴 {len(ok)}/{len(rows)}")
    print(f"  응답 시간 중앙값 {statistics.median(secs):.1f}초, 최대 {max(secs):.1f}초, "
          f"출력 토큰 중앙값 {statistics.median(int(r['tokens']) for r in rows):.0f}")
    for w in ["clear", "night", "rain"]:
        print(f"  날씨 {w}: {dict(Counter(r['p_weather'] for r in ok if r['weather'] == w))}")
    for key, s, l in [("p_veh", "veh_strict", "veh_loose"), ("p_ped", "ped_strict", "ped_loose")]:
        c = Counter(judge_count(as_int(r[key]), int(r[s]), int(r[l])) for r in ok if as_int(r[key]) is not None)
        print(f"  {'차량' if key == 'p_veh' else '보행자'} 수: 맞음 {c['ok']}, 과대(환각) {c['over']}, 과소(놓침) {c['under']}")
    red = [r for r in ok if int(r["frame"]) >= 140]
    print(f"  신호 (140번~, 정답 red) {len(red)}장: {dict(Counter(r['p_light'] for r in red))}")
    for name, lo, hi, want in [("통과 구간 0~20번", 0, 20, "GO"), ("접근 구간 40~160번", 40, 160, None),
                               ("정지 구간 180번~", 180, 399, "STOP")]:
        part = [r for r in ok if lo <= int(r["frame"]) <= hi]
        acts = Counter(r["p_action"] for r in part)
        hit = f", 정답({want}) {acts[want]}/{len(part)}" if want else ""
        print(f"  행동 — {name} {len(part)}장: {dict(acts)}{hit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--api", default="chat", choices=["chat", "responses"])
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--step", type=int, default=20, help="몇 프레임마다 물을지 (400프레임 중)")
    ap.add_argument("--hint", action="store_true", help="신호등 읽는 법을 프롬프트에 덧붙인다")
    ap.add_argument("--from-csv", default=None, help="저장한 결과 CSV를 다시 집계만 한다")
    args = ap.parse_args()
    if args.from_csv:
        rows = list(csv.DictReader(open(args.from_csv, encoding="utf-8")))
        for r in rows:
            r["valid"] = r["valid"] == "True"
        summarize(rows, Path(args.from_csv).stem)
        sys.exit()
    client, model = make_client(args.backend, args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for weather in ["clear", "night", "rain"]:
        gt, speed = ground_truth(weather)
        for f in range(0, 400, args.step):
            prompt = PROMPT + (HINT if args.hint else "")
            text, sec, tokens = ask(client, model, load_frame(weather, f), prompt, think=args.think, api=args.api)
            p = parse(text) or {}
            g = gt[f]
            row = {"weather": weather, "frame": f, "sec": round(sec, 2), "tokens": tokens, "valid": bool(p),
                   "p_weather": p.get("weather"), "p_veh": p.get("vehicles"), "p_ped": p.get("pedestrians"),
                   "p_light": p.get("traffic_light"), "p_action": p.get("action"), "reason": p.get("reason"),
                   "veh_strict": g["veh"][0], "veh_loose": g["veh"][1], "ped_strict": g["ped"][0],
                   "ped_loose": g["ped"][1], "speed": speed[f]}
            rows.append(row)
            print(f"{weather} {f:3d}  {sec:5.1f}s  {p.get('action')!s:9} 신호 {p.get('traffic_light')!s:6} "
                  f"차 {p.get('vehicles')}/{g['veh'][0]}~{g['veh'][1]}  사람 {p.get('pedestrians')}/{g['ped'][0]}~"
                  f"{g['ped'][1]}  {p.get('reason')}", flush=True)
    name = model.replace(":", "-").replace("/", "-") + ("_think" if args.think else "") + ("_hint" if args.hint else "")
    with open(OUT / f"eval_{name}.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summarize(rows, f"{model} think={args.think} hint={args.hint}")
    print(f"저장: {OUT / f'eval_{name}.csv'}")
