"""06-3: 같은 경로를 PID, Pure Pursuit, 간단한 MPC로 따라가 보고 비교합니다.

경로는 carla_drive의 실제 주행 궤적(823m, 누적 회전 459도)입니다. 자전거 모델 차량이 이 경로를
일정한 속도로 따라가며, 매 순간 경로에서 옆으로 벗어난 거리(가로 오차)를 잽니다.

python ch06/control.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vehicle import DT, MAX_STEER, WHEELBASE, Bicycle, cross_track_error, load_route

OUT = Path("outputs/ch06")
OUT.mkdir(parents=True, exist_ok=True)
rx, ry, ryaw, _ = load_route()


class PID:
    """가로 오차 e(왼쪽 +)가 0이 되도록 조향한다. 왼쪽으로 벗어나면 오른쪽(+)으로 꺾는다."""

    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral, self.prev = 0.0, None

    def __call__(self, e, dt=DT):
        self.integral += e * dt
        de = 0.0 if self.prev is None else (e - self.prev) / dt
        self.prev = e
        return self.kp * e + self.ki * self.integral + self.kd * de


def pure_pursuit(car, i, lookahead):
    """경로 위에서 lookahead만큼 앞의 점을 겨냥해, 그 점을 지나는 원호의 조향각을 구한다."""
    d = np.cumsum(np.hypot(np.diff(rx[i:i + 400]), np.diff(ry[i:i + 400])))
    j = i + 1 + int(np.searchsorted(d, lookahead))
    j = min(j, len(rx) - 1)
    alpha = np.arctan2(ry[j] - car.y, rx[j] - car.x) - car.yaw          # 목표점이 차 기준 얼마나 오른쪽인가
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return np.arctan2(2 * WHEELBASE * np.sin(alpha), lookahead)


def simple_mpc(car, i, horizon=1.0, candidates=31):
    """조향각 후보를 여러 개 놓고, 각각 horizon초 동안 미리 달려 본 뒤 가장 경로에 가까운 것을 고른다."""
    best, best_cost = 0.0, np.inf
    for delta in np.linspace(-MAX_STEER, MAX_STEER, candidates):
        sim = Bicycle(car.x, car.y, car.yaw, car.v)
        cost, hint = 0.0, i
        for _ in range(int(horizon / DT)):
            sim.step(delta)
            e, hint = cross_track_error(sim.x, sim.y, rx, ry, ryaw, hint)
            cost += e ** 2
        cost += 0.1 * delta ** 2                                        # 큰 조향에 작은 벌점
        if cost < best_cost:
            best, best_cost = delta, cost
    return best


def run(controller, speed, start_offset=0.5, measure=None):
    """경로 시작점에서 왼쪽으로 start_offset만큼 벗어나 출발해 끝까지 달린다."""
    car = Bicycle(rx[0] + start_offset * np.sin(ryaw[0]), ry[0] - start_offset * np.cos(ryaw[0]),
                  ryaw[0], speed)
    errors, steers, i = [], [], 0
    while i < len(rx) - 30 and len(errors) < 20000:
        e, i = cross_track_error(car.x, car.y, rx, ry, ryaw, i)
        if abs(e) > 5.0:                                                 # 차로를 완전히 벗어났다
            return np.array(errors), np.array(steers), False
        delta = controller(car, e if measure is None else measure(e), i)
        car.step(delta)
        errors.append(e)
        steers.append(np.clip(delta, -MAX_STEER, MAX_STEER))
    return np.array(errors), np.array(steers), True


def summary(name, speed, errors, steers, ok):
    jerk = np.degrees(np.mean(np.abs(np.diff(steers)))) if len(steers) > 1 else 0
    status = "완주" if ok else f"이탈({len(errors) * DT:.0f}초)"
    print(f"{name:<22}{speed * 3.6:5.0f}km/h  {status:<10}{np.sqrt(np.mean(errors ** 2)):7.3f}m"
          f"{np.max(np.abs(errors)):7.2f}m{jerk:9.3f}도")


if __name__ == "__main__":
    print(f"{'제어기':<22}{'속도':>9}  {'결과':<10}{'RMS 오차':>8}{'최대':>7}{'조향 변화':>10}")
    traces = {}
    for speed in [8.0, 15.0]:                                            # 약 29km/h, 54km/h
        ctrls = {
            "P (Kp=0.3)": lambda car, e, i, c=PID(0.3, 0, 0): c(e),
            "PD (Kp=0.3, Kd=0.1)": lambda car, e, i, c=PID(0.3, 0, 0.1): c(e),
            "Pure Pursuit (6m)": lambda car, e, i: pure_pursuit(car, i, 6.0),
            "Simple MPC (1s)": lambda car, e, i: simple_mpc(car, i),
        }
        for name, ctrl in ctrls.items():
            errors, steers, ok = run(ctrl, speed)
            summary(name, speed, errors, steers, ok)
            if speed == 15.0:
                traces[name] = errors
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, e in traces.items():
        ax.plot(np.arange(len(e)) * DT, e, label=name, lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cross-track error (m, left +)")
    ax.set_title("Path tracking at 54 km/h")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "control_compare.png", dpi=90)
