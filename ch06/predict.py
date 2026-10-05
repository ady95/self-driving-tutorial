"""06-1: 차의 미래 위치를 단순한 운동 모델로 예측하고, 정답과 비교합니다.

carla_drive의 자차 궤적을 '주변 차량 하나'라고 생각하고, 매 시점마다 지금까지의 위치만 보고
1·2·3초 뒤 위치를 예측합니다. 주변 차량 예측도 같은 계산입니다(관측하는 센서만 다를 뿐).
  CV   (Constant Velocity)               : 지금 속도와 방향 그대로 간다
  CTRV (Constant Turn Rate and Velocity) : 지금 속도와 회전 속도 그대로 간다
"""
import numpy as np

from vehicle import DT, load_route

x, y, yaw, v = load_route()
yaw_rate = np.gradient(yaw) / DT
HORIZONS = [1.0, 2.0, 3.0]                        # 예측 시간(초)


def predict_cv(i, T):
    return x[i] + v[i] * T * np.cos(yaw[i]), y[i] + v[i] * T * np.sin(yaw[i])


def predict_ctrv(i, T):
    w = yaw_rate[i]
    if abs(w) < 1e-3:                             # 거의 직진이면 CV와 같다
        return predict_cv(i, T)
    th = yaw[i] + w * T
    return (x[i] + v[i] / w * (np.sin(th) - np.sin(yaw[i])),
            y[i] + v[i] / w * (np.cos(yaw[i]) - np.cos(th)))


turning = np.abs(yaw_rate) > np.radians(5)        # 초당 5도 넘게 도는 순간
print(f"평가 시점: {len(x)}개 중 회전 중 {turning.mean():.1%}\n")
print(f"{'예측 시간':<8}{'CV 전체':>9}{'CTRV 전체':>11}{'CV 회전 중':>12}{'CTRV 회전 중':>14}")
for T in HORIZONS:
    n = int(T / DT)
    idx = np.arange(0, len(x) - n)
    e_cv = np.array([np.hypot(*(np.array(predict_cv(i, T)) - [x[i + n], y[i + n]])) for i in idx])
    e_ct = np.array([np.hypot(*(np.array(predict_ctrv(i, T)) - [x[i + n], y[i + n]])) for i in idx])
    t = turning[idx]
    print(f"{T:4.1f}초  {np.mean(e_cv):8.2f}m{np.mean(e_ct):10.2f}m{np.mean(e_cv[t]):11.2f}m{np.mean(e_ct[t]):13.2f}m")

# 어디서 틀리는가: 1초 예측 오차를 그 1초 동안의 속도 변화로 나눠 본다
n = int(1.0 / DT)
idx = np.arange(0, len(x) - n)
dv = np.abs(v[idx + n] - v[idx])
e_ct = np.array([np.hypot(*(np.array(predict_ctrv(i, 1.0)) - [x[i + n], y[i + n]])) for i in idx])
print("\nCTRV 1초 예측 오차, 그 1초 동안의 속도 변화별")
for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, np.inf)]:
    m = (dv >= lo) & (dv < hi)
    print(f"  속도 변화 {lo}~{hi} m/s: 시점의 {m.mean():5.1%}, 평균 오차 {e_ct[m].mean():.2f}m")
