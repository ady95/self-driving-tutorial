"""05-2 실습 1·2: 카메라 영상을 위에서 본 영상(Top View)으로 바꾸고, 주행 가능 영역을 BEV 지도로 옮겨 채점합니다."""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bev import BEV_H, BEV_W, RES, X_MAX, ipm, visible_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from carla_data import frame_at, semantic_gt  # noqa: E402

OUT = Path("outputs/ch05")
OUT.mkdir(parents=True, exist_ok=True)
ROAD_TAGS = [1, 24]                           # CARLA 정답: 도로 + 차선
MAX_RANGE = 30.0                              # 채점은 앞 30m까지

# 실습 1: 한 프레임을 Top View로
frame = frame_at("clear", 100)
top = ipm(frame)
vis = visible_mask()
print(f"BEV 격자 {BEV_W}x{BEV_H} (한 칸 {RES}m, 앞 {X_MAX:.0f}m · 좌우 {BEV_W * RES / 2:.0f}m)")
print(f"카메라에 보이는 칸: {vis.mean():.1%}, 가장 가까이 보이는 지면: "
      f"{X_MAX - np.where(vis.any(axis=1))[0].max() * RES:.2f}m 앞")
for x in (5, 10, 20, 40):                     # 거리별로 화면 한 줄(1px)이 지면 몇 m인가
    v = 640 * 1.6 / x + 360
    print(f"  {x:>2}m 앞 지면: 화면 v={v:6.1f}px, 화면 1줄 = 지면 {(640 * 1.6 / (v - 360 - 1) - x):.2f}m")
for x in (10, 20, 30):                        # 10m마다 거리 눈금
    r = int((X_MAX - x) / RES)
    cv2.line(top, (0, r), (BEV_W - 1, r), (0, 255, 255), 1)
    cv2.putText(top, f"{x}m", (4, r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.imwrite(str(OUT / "topview.jpg"), np.hstack([cv2.resize(frame, (711, 400)), top]))

# 실습 2: 의미 분할(03-3) → BEV, 정답 BEV와 IoU
model = YOLO("yolo26n-sem.pt")
rows = np.arange(BEV_H)[:, None] * np.ones((1, BEV_W))
in_range = vis & ((X_MAX - (rows + 0.5) * RES) <= MAX_RANGE)
print(f"\n{'날씨':<7}{'영상 IoU':>9}{'BEV IoU':>9}")
for weather in ["clear", "night", "rain"]:
    img_i = img_u = bev_i = bev_u = 0
    for k in range(0, 400, 5):
        f = frame_at(weather, k)
        pred = (model(f, verbose=False)[0].semantic_mask.data.cpu().numpy() == 0).astype(np.uint8)
        gt = np.isin(semantic_gt(weather, k), ROAD_TAGS).astype(np.uint8)
        img_i += np.sum(pred & gt)
        img_u += np.sum(pred | gt)
        pb = ipm(pred, cv2.INTER_NEAREST).astype(bool) & in_range
        gb = ipm(gt, cv2.INTER_NEAREST).astype(bool) & in_range
        bev_i += np.sum(pb & gb)
        bev_u += np.sum(pb | gb)
        if weather == "clear" and k == 100:
            cv2.imwrite(str(OUT / "road_bev.png"),
                        np.hstack([gb.astype(np.uint8) * 255, pb.astype(np.uint8) * 255]))
    print(f"{weather:<7}{img_i / img_u:9.3f}{bev_i / bev_u:9.3f}")
