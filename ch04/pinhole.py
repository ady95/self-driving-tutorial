"""04-1: 핀홀 카메라 모델로 3D 점을 화면에 투영하고, 반대로 상자 높이에서 거리를 추정합니다."""
import math

import numpy as np

from carla_data import CALIB, K, depth_gt, gt_boxes, object_distance_gt, semantic_gt

# 1. 내부 행렬은 화각(FOV)과 해상도에서 나온다
W, H = CALIB["image_size"]
f = W / (2 * math.tan(math.radians(CALIB["fov_deg"]) / 2))
print(f"해상도 {W}x{H}, 화각 {CALIB['fov_deg']}도 → 초점 거리 f = {f:.1f}px")
print("K =\n", np.array(K).round(1))

# 2. 투영: 앞 Z미터, 옆 X미터에 있는 폭 1.8m 차는 화면에서 몇 픽셀인가
print("\n앞 거리   차 폭(px)  차 높이 1.5m(px)")
for Z in [5, 10, 20, 40, 80]:
    print(f"{Z:5d} m   {f * 1.8 / Z:8.1f}   {f * 1.5 / Z:10.1f}")

# 3. 거꾸로: 상자 높이 h(px)를 알면 거리 = f * 실제 높이 / h
CAR_HEIGHT = 1.5                       # 승용차 높이를 1.5m라고 '가정'한다
rows = []
boxes = gt_boxes("clear")
for k in range(0, 400, 10):            # 깊이 정답이 있는 프레임
    depth, sem = depth_gt("clear", k), semantic_gt("clear", k)
    for cls, (x1, y1, x2, y2) in boxes[k]:
        if cls != "car":
            continue
        true = object_distance_gt(depth, sem, (x1, y1, x2, y2))
        if true is None:
            continue
        est = f * CAR_HEIGHT / (y2 - y1 + 1)
        rows.append((true, est))

rows = np.array(rows)
err = np.abs(rows[:, 1] - rows[:, 0]) / rows[:, 0]
print(f"\n상자 높이로 거리 추정: 승용차 {len(rows)}개")
print(f"상대 오차 중앙값 {np.median(err):.1%}, 오차 20% 이내 {np.mean(err < 0.2):.1%}")
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 100)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if m.any():
        print(f"  실제 {lo:>2}~{hi:<3}m: {m.sum():3d}개, 상대 오차 중앙값 {np.median(err[m]):.1%}")
