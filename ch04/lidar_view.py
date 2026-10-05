"""04-3: LiDAR 포인트 클라우드를 열어 보고, 위에서 내려다본 그림(BEV)으로 그립니다."""
from pathlib import Path

import cv2
import numpy as np

from carla_data import lidar_points

OUT = Path("outputs/ch04")
OUT.mkdir(parents=True, exist_ok=True)

pts = lidar_points("clear", 100)                 # (N, 4): x 앞, y 오른쪽, z 위, 반사 강도
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
r = np.hypot(x, y)
print(f"포인트 수     : {len(pts):,}개 (한 바퀴, 1/20초)")
print(f"x 범위(앞뒤)  : {x.min():6.1f} ~ {x.max():5.1f} m")
print(f"y 범위(좌우)  : {y.min():6.1f} ~ {y.max():5.1f} m")
print(f"z 범위(높이)  : {z.min():6.1f} ~ {z.max():5.1f} m  (LiDAR가 지면 위 2.2m에 있음)")
print(f"거리 중앙값   : {np.median(r):.1f} m")

print("\n거리 구간별 포인트 수와 밀도")
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
    m = (r >= lo) & (r < hi)
    area = np.pi * (hi ** 2 - lo ** 2)
    print(f"  {lo:>2}~{hi:<3}m: {m.sum():6d}개  ({m.sum() / area:6.2f}개/m²)")

# BEV: 차를 중심으로 앞 40m, 좌우 25m를 0.1m 격자로 그린다. 색 = 높이
res, fwd, side = 0.1, 40.0, 25.0
H, W = int(fwd * 2 / res), int(side * 2 / res)
bev = np.zeros((H, W, 3), np.uint8)
m = (np.abs(x) < fwd) & (np.abs(y) < side)
rows = ((fwd - x[m]) / res).astype(int)          # 앞이 위쪽
cols = ((y[m] + side) / res).astype(int)         # 오른쪽이 오른쪽
height = z[m] + 2.2                              # 지면에서의 높이 (m)
colors = cv2.applyColorMap((np.clip(height / 4.0, 0, 1) * 255).astype(np.uint8).reshape(-1, 1),
                           cv2.COLORMAP_TURBO)[:, 0]
for r_, c_, h_, col in zip(rows, cols, height, colors):
    if h_ < 0.3:                                 # 지면 점은 어두운 회색 점으로
        bev[r_, c_] = (70, 70, 70)
    else:                                        # 물체(차·건물·기둥)는 높이 색으로 굵게
        cv2.circle(bev, (int(c_), int(r_)), 2, tuple(int(v) for v in col), -1)
cv2.rectangle(bev, (W // 2 - 9, H // 2 - 24), (W // 2 + 9, H // 2 + 24), (255, 255, 255), -1)  # 자차
cv2.imwrite(str(OUT / "lidar_bev.png"), bev)
print(f"\nBEV 그림: {OUT / 'lidar_bev.png'} ({W}x{H}, 1픽셀 = {res}m)")
