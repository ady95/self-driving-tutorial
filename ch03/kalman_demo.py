"""03-4 실습 1: 칼만 필터로 흔들리는 검출 위치를 다듬고, 가려진 동안의 위치를 예측합니다.

CARLA 정답에서 가장 오래 보이는 차의 화면 x좌표(상자 중심)를 가져와,
검출기처럼 흔들림(잡음)을 넣고 중간 20프레임을 지운 뒤(가려짐) 칼만 필터로 따라갑니다.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/carla_urban/clear")
NOISE_PX = 8.0                       # 검출 상자 중심의 흔들림 (표준편차, 픽셀)
HIDE = range(150, 170)               # 이 프레임들 동안 검출이 사라진다 (가려짐)

# 1. 정답 궤적: 가장 많은 프레임에 나온 차의 상자 중심 x
track = defaultdict(dict)
with open(ROOT / "boxes.csv") as f:
    for r in csv.DictReader(f):
        if r["class"] == "car":
            track[int(r["object_id"])][int(r["frame"])] = (float(r["x1"]) + float(r["x2"])) / 2
oid = max(track, key=lambda i: len(track[i]))
frames = sorted(track[oid])
truth = np.array([track[oid][k] for k in frames])
print(f"객체 {oid}: {len(frames)} 프레임 (#{frames[0]}~#{frames[-1]})")

# 2. 검출 흉내: 잡음을 더하고, 가려진 구간은 None
rng = np.random.default_rng(0)
meas = [None if k in HIDE else x + rng.normal(0, NOISE_PX) for k, x in zip(frames, truth)]

# 3. 등속 모델 칼만 필터: 상태 = [위치, 속도]
dt = 1.0
F = np.array([[1, dt], [0, 1]])      # 다음 위치 = 위치 + 속도
H = np.array([[1, 0]])               # 관측하는 것은 위치뿐
Q = np.diag([0.5, 0.5])              # 모델이 틀릴 수 있는 정도 (가속·감속)
R = np.array([[NOISE_PX ** 2]])      # 관측 잡음
x = np.array([meas[0], 0.0])
P = np.diag([100.0, 100.0])
est = []
for z in meas:
    x = F @ x                        # 예측
    P = F @ P @ F.T + Q
    if z is not None:                # 관측이 있으면 보정
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y).ravel()
        P = (np.eye(2) - K @ H) @ P
    est.append(x[0])
est = np.array(est)

seen = np.array([z is not None for z in meas])
raw_err = np.abs(np.array([z for z in meas if z is not None]) - truth[seen])
kf_err = np.abs(est - truth)
hidden = np.array([k in HIDE for k in frames])
print(f"보이는 구간 평균 오차: 검출 그대로 {raw_err.mean():.1f}px → 칼만 필터 {kf_err[seen].mean():.1f}px")
print(f"가려진 20프레임 동안 예측 오차: 평균 {kf_err[hidden].mean():.1f}px, 최대 {kf_err[hidden].max():.1f}px")
print(f"가려진 동안 정답이 움직인 거리: {abs(truth[hidden][-1] - truth[hidden][0]):.1f}px")
