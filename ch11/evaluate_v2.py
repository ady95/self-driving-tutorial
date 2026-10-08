"""11-4: v1 설정과 v2를 같은 조건·같은 시험 상황에서 달려 10장 지표로 비교합니다.

  기본 주행 : 경로 0·1 x 맑은 낮·밤·폭우 (다른 차 40대, 보행자 30명)
  시험 상황 : 보행자 횡단, 고장 차, 공사 구간 (모두 경로 1) x 맑은 낮·밤·폭우
              상황의 효과만 보려고 다른 차와 보행자는 뺀다 (다른 차가 공사 구간에 막혀 서 버리는 등 결과가 섞이므로)
  지표      : Driving Score(10-3), 충돌, 빨간불 통과, 최소 TTC, 보행자 최소 거리, 신호 판독 정확도, 결과

python ch11/evaluate_v2.py --tm-port 8300                 # 30회, 약 2시간
python ch11/evaluate_v2.py --only scenarios --weathers clear
python ch11/evaluate_v2.py --resume           # 중간에 멈췄다면 끝난 조건은 건너뛰고 이어서
결과: outputs/ch11/eval_<이름>.csv, 주행마다 로그·결과 outputs/ch11/agent_v2_<조건>_<이름>_*
같은 --name의 결과가 이미 있으면 덮어쓰지 않습니다 (--resume으로 이어 가거나 --name을 바꾸세요).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch10"))
import agent_v2  # noqa: E402
from evaluate import driving_score, load_rows  # noqa: E402

OUT = Path("outputs/ch11")
BASE = [(0, "none"), (1, "none")]
SCENARIOS = [(1, "pedestrian"), (1, "stalled_car"), (1, "construction")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--only", default="all", choices=["all", "base", "scenarios"])
    ap.add_argument("--weathers", nargs="+", default=["clear", "night", "rain"])
    ap.add_argument("--modes", nargs="+", default=["v1", "v2"])
    ap.add_argument("--name", default="v1_vs_v2")
    ap.add_argument("--resume", action="store_true", help="같은 --name의 결과에서 끝난 조건은 건너뛴다")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = (BASE if args.only != "scenarios" else []) + (SCENARIOS if args.only != "base" else [])
    rows = load_rows(OUT / f"eval_{args.name}.csv", args.resume)
    done = {(r["mode"], r["route"], r["scenario"], r["weather"]) for r in rows}
    for route, scenario in jobs:
        for weather in args.weathers:
            for mode in args.modes:
                if (mode, route, scenario, weather) in done:
                    continue
                print(f"\n=== {mode}, 경로 {route}, {weather}, 상황 {scenario}", flush=True)
                start = time.perf_counter()
                traffic = [] if scenario == "none" else ["--npc", "0", "--walkers", "0"]
                r = agent_v2.main(agent_v2.get_args(["--tm-port", str(args.tm_port), "--route", str(route),
                                                     "--weather", weather, "--scenario", scenario, "--mode", mode,
                                                     "--tag", f"_{args.name}"] + traffic))   # 로그 이름에 평가 이름을 붙인다
                ds, ip = driving_score(r)
                rows.append({"mode": mode, "route": route, "scenario": scenario, "weather": weather,
                             "result": r["result"], "time_s": r["time_s"], "progress": r["progress"],
                             "collisions": "|".join(r["collisions"]), "red_run": r["red_run"], "solid": r["solid"],
                             "min_ttc": r["min_ttc"], "min_lead": r["min_lead"], "ped_min": r["ped_min"],
                             "light_acc": r["light_acc"], "light_unknown": r["light_unknown"],
                             "bypasses": r["bypasses"], "penalty": round(ip, 3), "driving_score": round(ds, 3),
                             "wall_s": round(time.perf_counter() - start)})
                print(f"→ Driving Score {ds:.3f}", flush=True)
                with open(OUT / f"eval_{args.name}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)

    print(f"\n{'':4}{'상황':>13} {'날씨':>6} {'결과':>10} {'시간':>7} {'충돌':>18} {'빨간불':>4} {'TTC':>6} "
          f"{'신호':>6} {'DS':>6}")
    for r in rows:
        print(f"{r['mode']:4}{r['scenario']:>13} {r['weather']:>6} {r['result']:>10} {r['time_s']:>6.1f}s "
              f"{(r['collisions'] or '-')[:18]:>18} {r['red_run']:>4} {str(r['min_ttc']):>6} "
              f"{(format(r['light_acc'], '.1%') if r['light_acc'] is not None else '-'):>6} "
              f"{r['driving_score']:>6.3f}")
    for mode in args.modes:
        part = [r for r in rows if r["mode"] == mode]
        if part:
            print(f"{mode} 평균 Driving Score {sum(r['driving_score'] for r in part) / len(part):.3f} ({len(part)}회), "
                  f"최솟값 {min(r['driving_score'] for r in part):.3f}")
