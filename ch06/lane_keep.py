"""06-4: PID로 차를 차선 중앙에 유지합니다 — 잡음과 지연이 있는 '진짜 센서'를 붙였을 때.

02-2의 차선 인식기는 차선 중앙에서 벗어난 정도(offset)와 차선의 기울기를 알려 줍니다.
이 두 값을 센서로 쓴다고 보고 다음을 넣습니다. 속도는 54km/h, 1m 왼쪽에서 출발합니다.
  - offset 잡음 : 표준편차 0.05m (02-2 challenge.mp4의 중앙 흔들림 9.9px를 미터로 바꾼 정도)
  - 방향 잡음   : 표준편차 1도 (차선 기울기로 구한 차와 차선 사이 각도)
  - 지연        : 2프레임(0.1초) (영상 처리 시간 + 카메라 노출)
"""
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vehicle import DT, MAX_STEER, Bicycle, cross_track_error, load_route

OUT = Path("outputs/ch06")
OUT.mkdir(parents=True, exist_ok=True)
rx, ry, ryaw, _ = load_route()
curv = np.abs(np.gradient(ryaw)) / 0.34 > 0.01                 # 곡률 0.01/m 이상(반경 100m 이하) = 커브
SPEED = 15.0


class PID:
    """offset e(왼쪽 +)만 보고 조향한다. 왼쪽으로 벗어나면 오른쪽(+)으로 꺾는다."""

    def __init__(self, kp, ki, kd, d_filter=1.0):
        self.kp, self.ki, self.kd, self.a = kp, ki, kd, d_filter   # d_filter < 1이면 D항을 부드럽게
        self.integral, self.prev, self.d = 0.0, None, 0.0

    def __call__(self, e, h):
        self.integral = np.clip(self.integral + e * DT, -2.0, 2.0)  # 적분 누적 상한(와인드업 방지)
        raw = 0.0 if self.prev is None else (e - self.prev) / DT
        self.d = self.a * raw + (1 - self.a) * self.d                # 지수 이동 평균 필터
        self.prev = e
        return self.kp * e + self.ki * self.integral + self.kd * self.d


class OffsetHeading:
    """offset은 PI로, 차와 차선 사이 각도 h(오른쪽을 향하면 +)는 직접 되먹인다."""

    def __init__(self, kp, ki, kh):
        self.kp, self.ki, self.kh, self.integral = kp, ki, kh, 0.0

    def __call__(self, e, h):
        self.integral = np.clip(self.integral + e * DT, -2.0, 2.0)
        return self.kp * e + self.ki * self.integral - self.kh * h


def drive(ctrl, noise=0.0, delay=0, seed=0):
    rng = np.random.default_rng(seed)
    car = Bicycle(rx[0] + 1.0 * np.sin(ryaw[0]), ry[0] - 1.0 * np.cos(ryaw[0]), ryaw[0], SPEED)
    buf = deque([(1.0, 0.0)] * (delay + 1), maxlen=delay + 1)
    errors, steers, in_curve, i = [], [], [], 0
    while i < len(rx) - 30:
        e, i = cross_track_error(car.x, car.y, rx, ry, ryaw, i)
        if abs(e) > 5.0:                                         # 차로를 완전히 벗어났다
            return np.array(errors), np.array(steers), np.array(in_curve), False
        h = (car.yaw - ryaw[i] + np.pi) % (2 * np.pi) - np.pi
        buf.append((e + rng.normal(0, noise), h + rng.normal(0, np.radians(20 * noise))))
        delta = np.clip(ctrl(*buf[0]), -MAX_STEER, MAX_STEER)    # 지연 버퍼의 가장 오래된 측정값으로 조향
        car.step(delta)
        errors.append(e)
        steers.append(delta)
        in_curve.append(curv[i])
    return np.array(errors), np.array(steers), np.array(in_curve), True


CASES = [  # (이름, 제어기, 잡음, 지연)   잡음 0.05 → offset 0.05m, 방향 1도
    ("P (Kp=0.3)", lambda: PID(0.3, 0, 0), 0.0, 0),
    ("PD (Kd=0.1)", lambda: PID(0.3, 0, 0.1), 0.0, 0),
    ("PD + 잡음·지연", lambda: PID(0.3, 0, 0.1), 0.05, 2),
    ("PD + 잡음·지연 + D필터", lambda: PID(0.3, 0, 0.1, d_filter=0.15), 0.05, 2),
    ("P+방향 + 잡음·지연", lambda: OffsetHeading(0.3, 0, 1.0), 0.05, 2),
    ("PI+방향 + 잡음·지연", lambda: OffsetHeading(0.3, 0.05, 1.0), 0.05, 2),
]

if __name__ == "__main__":
    print(f"속도 {SPEED * 3.6:.0f}km/h, 1m 벗어난 곳에서 출발, 경로 중 커브 비율 {curv.mean():.1%}\n")
    print(f"{'설정':<22}{'결과':<9}{'RMS':>7}{'최대':>7}{'커브 치우침':>11}{'조향 떨림':>10}")
    traces = {}
    for name, make, noise, delay in CASES:
        e, s, c, ok = drive(make(), noise, delay)
        status = "완주" if ok else f"이탈({len(e) * DT:.0f}초)"
        print(f"{name:<22}{status:<9}{np.sqrt(np.mean(e ** 2)):6.3f}m{np.max(np.abs(e)):6.2f}m"
              f"{np.mean(e[c]):+10.3f}m{np.degrees(np.mean(np.abs(np.diff(s)))):9.2f}도")
        traces[name] = (e, s)

    show = {"P (Kp=0.3)": "P only", "PD + 잡음·지연": "PD, noise+delay",
            "P+방향 + 잡음·지연": "P+heading, noise+delay"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, label in show.items():
        e, s = traces[name]
        n = min(len(e), int(20 / DT))
        axes[0].plot(np.arange(n) * DT, e[:n], label=label, lw=1)
        axes[1].plot(np.arange(n) * DT, np.degrees(s[:n]), label=label, lw=1)
    axes[0].set_ylabel("cross-track error (m)")
    axes[1].set_ylabel("steering (deg)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lane_keep.png", dpi=90)
    print(f"\n그래프: {OUT / 'lane_keep.png'} (처음 20초)")
