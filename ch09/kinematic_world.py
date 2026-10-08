"""09-4: 가장 단순한 World Model — "이렇게 조작하면 차가 어디로 갈까"를 자전거 모델(06-3)로 예측합니다.

08장에서 기록한 앞으로 2초 동안의 조향·속도를 그대로 넣어 0.5~2초 뒤 위치를 굴려 보고, CARLA가 실제로 움직인
위치와 비교합니다. 미래의 실제 조작을 미리 주는 '재생 시험'이며, 지금 센서만으로 예측하는 시험이 아닙니다.
비교: 조향만 넣고 속도는 지금 값으로 고정 / 아무 조작도 없이 지금 속도로 똑바로
지금 상태 + 행동 → 다음 상태를 예측하는 것이 World Model의 핵심이고, 학습형 World Model은 이것을
영상 전체에 대해 데이터로 배웁니다.

python ch09/kinematic_world.py                       # 08-2에서 직접 모은 data/e2e/Town03
python ch09/kinematic_world.py --data data/e2e_ref   # 책의 표를 낸 기준 라벨 (download_samples.py --e2e-ref)
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

WHEELBASE, MAX_STEER, DT = 2.86, math.radians(70), 0.05
HORIZON = [10, 20, 40]                                   # 0.5·1.0·2.0초 뒤


def rollout(x, y, yaw, steers, speeds):
    """자전거 모델: 조향·속도를 차례로 넣어 위치를 굴려 본다."""
    path = []
    for s, v in zip(steers, speeds):
        yaw += v / WHEELBASE * math.tan(s * MAX_STEER) * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        path.append((x, y))
    return path


def load_labels(data):
    path = Path(data) / "Town03" / "labels.csv"
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 08-2의 collect.py로 모으거나, "
                         f"python scripts/download_samples.py --e2e-ref 후 --data data/e2e_ref로 실행하세요")
    return list(csv.DictReader(open(path)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/e2e")
    rows = load_labels(ap.parse_args().data)
    val = {k: np.array([float(r[k]) for r in rows]) for k in ["steer", "speed", "x", "y", "yaw"]}
    errs = {h: [] for h in HORIZON}
    straight = {h: [] for h in HORIZON}
    steer_only = {h: [] for h in HORIZON}
    for j in range(0, len(rows) - HORIZON[-1], 5):
        seg = slice(j + 1, j + HORIZON[-1] + 1)
        if np.max(np.hypot(np.diff(val["x"][j:seg.stop]), np.diff(val["y"][j:seg.stop]))) > 2.0:
            continue                                     # 수집 중 재배치 구간 제외
        if val["speed"][j] < 1.0:
            continue
        x0, y0, yaw0 = val["x"][j], val["y"][j], math.radians(val["yaw"][j])
        path = rollout(x0, y0, yaw0, val["steer"][seg], val["speed"][seg])          # 조향·속도 모두 기록값
        path_s = rollout(x0, y0, yaw0, val["steer"][seg], np.full(HORIZON[-1], val["speed"][j]))  # 속도는 지금 값
        line = [(x0 + val["speed"][j] * (h * DT) * math.cos(yaw0), y0 + val["speed"][j] * (h * DT) * math.sin(yaw0))
                for h in HORIZON]                        # 비교: 지금 속도로 똑바로
        for h, (lx, ly) in zip(HORIZON, line):
            tx, ty = val["x"][j + h], val["y"][j + h]
            errs[h].append(math.hypot(path[h - 1][0] - tx, path[h - 1][1] - ty))
            steer_only[h].append(math.hypot(path_s[h - 1][0] - tx, path_s[h - 1][1] - ty))
            straight[h].append(math.hypot(lx - tx, ly - ty))
    print(f"예측 {len(errs[HORIZON[0]])}번 (Town03, 달리는 중)")
    print("            조향·속도 넣음      |  조향만 (속도 고정)  |  등속 직진 (조작 없음)   평균 (95%)")
    for h in HORIZON:
        e, o, s = np.array(errs[h]), np.array(steer_only[h]), np.array(straight[h])
        print(f"  {h * DT:.1f}초 뒤:  {e.mean():.2f}m ({np.percentile(e, 95):.2f}m)  |  "
              f"{o.mean():.2f}m ({np.percentile(o, 95):.2f}m)  |  {s.mean():.2f}m ({np.percentile(s, 95):.2f}m)")
