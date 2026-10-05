"""05-3: 추측 항법(바퀴 속도 + IMU 자이로), GPS, 그리고 둘을 합친 칼만 필터로 위치를 추정해 정답 궤적과 비교합니다.

data/carla_drive/ 의 ego.csv(정답)와 imu.csv(CARLA IMU, 잡음 포함)를 씁니다.
바퀴 속도와 GPS는 정답에 잡음을 더해 흉내 냅니다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("data/carla_drive")
OUT = Path("outputs/ch05")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.05                                     # 20 FPS
rng = np.random.default_rng(0)

ego = np.loadtxt(ROOT / "ego.csv", delimiter=",", skiprows=1)
imu = np.loadtxt(ROOT / "imu.csv", delimiter=",", skiprows=1)
gt_xy = ego[:, 2:4]
gt_yaw = np.unwrap(np.radians(ego[:, 7]))
n = len(ego)

# 센서 흉내: 바퀴 속도는 2% 크게 재고(타이어 마모 등) 잡음 0.1m/s, GPS는 1초에 한 번·오차 2m
speed = ego[:, 8] / 3.6 * 1.02 + rng.normal(0, 0.1, n)
gyro = imu[:, 7]                              # CARLA IMU z축 각속도 (잡음·편향 포함)
GPS_EVERY, GPS_SIGMA = 20, 2.0
gps = {k: gt_xy[k] + rng.normal(0, GPS_SIGMA, 2) for k in range(0, n, GPS_EVERY)}

# 1. 추측 항법: 처음 위치·방향만 알고, 속도와 회전 속도를 적분한다
dr = np.zeros((n, 2))
dr[0], yaw = gt_xy[0], gt_yaw[0]
for k in range(1, n):
    yaw += gyro[k] * DT
    dr[k] = dr[k - 1] + speed[k] * DT * np.array([np.cos(yaw), np.sin(yaw)])

# 2. GPS만: 측정 사이에는 마지막 값을 유지
gps_only = np.array([gps[(k // GPS_EVERY) * GPS_EVERY] for k in range(n)])

# 3. 확장 칼만 필터(EKF): 상태 [x, y, yaw], 예측은 추측 항법, 보정은 GPS
x = np.array([*gt_xy[0], gt_yaw[0]])
P = np.diag([1.0, 1.0, 0.01])
Q = np.diag([0.02, 0.02, 0.0005])             # 한 스텝 동안 모델이 틀릴 수 있는 정도
R = np.eye(2) * GPS_SIGMA ** 2
H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
ekf = np.zeros((n, 2))
for k in range(n):
    if k > 0:                                 # 예측
        v, w = speed[k], gyro[k]
        x = x + np.array([v * DT * np.cos(x[2]), v * DT * np.sin(x[2]), w * DT])
        F = np.array([[1, 0, -v * DT * np.sin(x[2])], [0, 1, v * DT * np.cos(x[2])], [0, 0, 1]])
        P = F @ P @ F.T + Q
    if k in gps:                              # 보정
        y = gps[k] - H @ x
        S = H @ P @ H.T + R
        Kg = P @ H.T @ np.linalg.inv(S)
        x = x + Kg @ y
        P = (np.eye(3) - Kg @ H) @ P
    ekf[k] = x[:2]

dist = np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1))
print(f"주행 {n / 20:.0f}초, {dist:.1f} m, GPS {len(gps)}회(1Hz, 오차 {GPS_SIGMA}m)\n")
print(f"{'방법':<18}{'평균 오차':>9}{'최대 오차':>9}{'마지막 오차':>11}")
for name, est in [("추측 항법", dr), ("GPS만", gps_only), ("EKF (추측 항법+GPS)", ekf)]:
    e = np.linalg.norm(est - gt_xy, axis=1)
    print(f"{name:<18}{e.mean():8.2f}m{e.max():8.2f}m{e[-1]:10.2f}m")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(gt_xy[:, 1], gt_xy[:, 0], "k-", lw=3, label="ground truth")
ax.plot(dr[:, 1], dr[:, 0], "-", c="tab:orange", label="dead reckoning")
g = np.array(list(gps.values()))
ax.plot(g[:, 1], g[:, 0], ".", c="tab:blue", ms=5, label="GPS (1 Hz)")
ax.plot(ekf[:, 1], ekf[:, 0], "-", c="tab:green", label="EKF")
ax.set_xlabel("y (m)")
ax.set_ylabel("x (m)")
ax.set_aspect("equal")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "localize.png", dpi=90)
print(f"\n궤적 그림: {OUT / 'localize.png'}")
