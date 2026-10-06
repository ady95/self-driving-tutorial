"""09-4: 가장 단순한 World Model — "이렇게 조작하면 차가 어디로 갈까"를 자전거 모델(06-3)로 예측합니다.

08장에서 기록한 조향·속도를 그대로 넣어 0.5~2초 뒤 위치를 예측하고, CARLA가 실제로 움직인 위치와 비교합니다.
지금 상태 + 행동 → 다음 상태를 예측하는 것이 World Model의 핵심이고, 학습형 World Model은 이것을
영상 전체에 대해 데이터로 배웁니다.

python ch09/kinematic_world.py        # 08-2에서 모은 data/e2e/Town03 필요
"""
import csv
import math

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


if __name__ == "__main__":
    rows = list(csv.DictReader(open("data/e2e/Town03/labels.csv")))
    val = {k: np.array([float(r[k]) for r in rows]) for k in ["steer", "speed", "x", "y", "yaw"]}
    errs = {h: [] for h in HORIZON}
    straight = {h: [] for h in HORIZON}
    for j in range(0, len(rows) - HORIZON[-1], 5):
        seg = slice(j + 1, j + HORIZON[-1] + 1)
        if np.max(np.hypot(np.diff(val["x"][j:seg.stop]), np.diff(val["y"][j:seg.stop]))) > 2.0:
            continue                                     # 수집 중 재배치 구간 제외
        if val["speed"][j] < 1.0:
            continue
        x0, y0, yaw0 = val["x"][j], val["y"][j], math.radians(val["yaw"][j])
        path = rollout(x0, y0, yaw0, val["steer"][seg], val["speed"][seg])
        line = [(x0 + val["speed"][j] * (h * DT) * math.cos(yaw0), y0 + val["speed"][j] * (h * DT) * math.sin(yaw0))
                for h in HORIZON]                        # 비교: 지금 속도로 똑바로
        for h, (lx, ly) in zip(HORIZON, line):
            tx, ty = val["x"][j + h], val["y"][j + h]
            errs[h].append(math.hypot(path[h - 1][0] - tx, path[h - 1][1] - ty))
            straight[h].append(math.hypot(lx - tx, ly - ty))
    print(f"예측 {len(errs[HORIZON[0]])}번 (Town03, 달리는 중)")
    for h in HORIZON:
        e, s = np.array(errs[h]), np.array(straight[h])
        print(f"  {h * DT:.1f}초 뒤: 자전거 모델 평균 {e.mean():.2f}m (95% {np.percentile(e, 95):.2f}m)  |  "
              f"등속 직진 평균 {s.mean():.2f}m (95% {np.percentile(s, 95):.2f}m)")
