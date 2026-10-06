"""10-3: 07-4의 Modular Driving Agent v1을 날씨·시간대·교통량을 바꿔 가며 평가하고 Driving Score를 매깁니다.

Driving Score (CARLA Leaderboard 방식) = Route Completion x Infraction Penalty
  - Route Completion : 경로를 몇 %나 갔는가 (0~1)
  - Infraction Penalty: 위반마다 계수를 곱한다 (1에서 시작)
      보행자 충돌 0.50, 차량 충돌 0.60, 정적 물체 충돌 0.65, 빨간불 통과 0.70
  - 실선 침범은 Leaderboard에서는 감점하지 않으므로 따로 센다

python ch10/evaluate.py                       # 경로 2 x 날씨 3 x 교통량 3 = 18회 (약 1시간)
python ch10/evaluate.py --routes 1 --weathers clear --traffic normal --seeds 0 1 2    # 같은 조건 반복
결과: outputs/ch10/eval_<이름>.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
import agent_v1  # noqa: E402

OUT = Path("outputs/ch10")
TRAFFIC = {"light": (10, 10), "normal": (40, 30), "heavy": (80, 60)}      # (다른 차, 보행자)
PENALTY = {"walker": 0.50, "vehicle": 0.60, "static": 0.65, "red_light": 0.70}


def driving_score(r):
    rc = 1.0 if r["result"] == "도착" else r["progress"]           # 목적지 5m 안에 들어오면 완주
    ip = 1.0
    for actor in r["collisions"]:
        ip *= PENALTY.get(actor.split(".")[0], PENALTY["static"])
    ip *= PENALTY["red_light"] ** r["red_run"]
    return rc * ip, ip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--routes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--weathers", nargs="+", default=["clear", "night", "rain"])
    ap.add_argument("--traffic", nargs="+", default=["light", "normal", "heavy"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--name", default="matrix")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for route in args.routes:
        for weather in args.weathers:
            for traffic in args.traffic:
                for seed in args.seeds:
                    npc, walkers = TRAFFIC[traffic]
                    run = argparse.Namespace(host=args.host, tm_port=args.tm_port, route=route, weather=weather,
                                             npc=npc, walkers=walkers, seed=seed, curve_limit=True, video=False)
                    print(f"\n=== 경로 {route}, {weather}, 교통량 {traffic}({npc}대·{walkers}명), seed {seed}", flush=True)
                    start = time.perf_counter()
                    r = agent_v1.main(run)
                    tag = f"r{route}_{weather}" + ("" if (npc, walkers, seed) == (40, 30, 0) else f"_n{npc}w{walkers}s{seed}")
                    log = Path(f"outputs/ch07/agent_v1_{tag}_log.csv")
                    if log.exists():                                # 실행마다 상태 로그를 따로 보관
                        log.replace(OUT / f"log_{args.name}_{len(rows):02d}_{tag}.csv")
                    ds, ip = driving_score(r)
                    rows.append({"route": route, "weather": weather, "traffic": traffic, "seed": seed,
                                 "result": r["result"], "time_s": r["time_s"], "progress": r["progress"],
                                 "collisions": len(r["collisions"]), "collision_with": "|".join(r["collisions"]),
                                 "red_run": r["red_run"], "solid": r["solid"], "avg_kmh": r["avg_kmh"],
                                 "lat_max": r["lat_max"], "penalty": round(ip, 3), "driving_score": round(ds, 3),
                                 "emergency": r["states"].get("EMERGENCY", 0), "wall_s": round(time.perf_counter() - start)})
                    print(f"→ Driving Score {ds:.3f} (완주 {r['progress']:.0%} x 감점 {ip:.2f})", flush=True)
                    with open(OUT / f"eval_{args.name}.csv", "w", newline="") as f:          # 한 번 돌 때마다 저장
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)

    print(f"\n{'경로':>4} {'날씨':>6} {'교통량':>7} {'seed':>4} {'결과':>6} {'시간':>7} {'충돌':>4} {'빨간불':>5} "
          f"{'실선':>4} {'DS':>6}")
    for r in rows:
        print(f"{r['route']:>4} {r['weather']:>6} {r['traffic']:>7} {r['seed']:>4} {r['result']:>6} "
              f"{r['time_s']:>6.1f}s {r['collisions']:>4} {r['red_run']:>5} {r['solid']:>4} {r['driving_score']:>6.3f}")
    print(f"평균 Driving Score {sum(r['driving_score'] for r in rows) / len(rows):.3f} ({len(rows)}회)")
