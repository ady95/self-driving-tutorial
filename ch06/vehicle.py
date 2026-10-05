"""06장 공통: 운동학적 자전거 모델(Kinematic Bicycle Model)과 주행 경로 도구.

앞바퀴 두 개, 뒷바퀴 두 개를 각각 하나로 합쳐 '자전거'처럼 다루는 가장 단순한 차량 모델입니다.
상태: 뒷바퀴 축 중심의 위치 (x, y), 방향 yaw(rad), 속도 v(m/s). 입력: 조향각 delta, 가속도 a.
"""
from pathlib import Path

import numpy as np

DT = 0.05                      # 20Hz (CARLA 데이터와 같다)
WHEELBASE = 2.9                # 앞뒤 바퀴 축 사이 거리 (m), 중형 세단 수준
MAX_STEER = np.radians(35)     # 최대 조향각


class Bicycle:
    def __init__(self, x, y, yaw, v):
        self.x, self.y, self.yaw, self.v = x, y, yaw, v

    def step(self, delta, a=0.0, dt=DT):
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / WHEELBASE * np.tan(delta) * dt     # 조향각이 클수록, 빠를수록 빨리 돈다
        self.v = max(0.0, self.v + a * dt)
        return self


def load_route(path="data/carla_drive/ego.csv", every=1):
    """carla_drive의 정답 궤적을 '따라갈 경로'로 읽는다: x, y, yaw(rad, 연속), 속도(m/s)."""
    e = np.loadtxt(Path(path), delimiter=",", skiprows=1)[::every]
    return e[:, 2], e[:, 3], np.unwrap(np.radians(e[:, 7])), e[:, 8] / 3.6


def cross_track_error(x, y, rx, ry, ryaw, hint=0, window=200):
    """(x, y)와 경로 사이의 가로 오차(m, 왼쪽이 +)와 가장 가까운 경로 점 번호.
    CARLA 좌표는 y가 오른쪽이므로, 진행 방향의 왼쪽은 (sin yaw, -cos yaw) 방향이다."""
    lo, hi = max(0, hint - 20), min(len(rx), hint + window)
    d = np.hypot(rx[lo:hi] - x, ry[lo:hi] - y)
    i = lo + int(np.argmin(d))
    dx, dy = x - rx[i], y - ry[i]
    left = np.sin(ryaw[i]) * dx - np.cos(ryaw[i]) * dy
    return left, i
