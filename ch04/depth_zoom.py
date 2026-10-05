"""04-2 실습 3: 같은 장면을 확대(초점 거리 2배)해서 넣으면 단안 깊이 모델의 거리 판단이 어떻게 바뀌는지 봅니다.

영상 가운데 640x360 영역에 대해 두 가지 예측을 같은 정답과 비교합니다.
  full : 원본 영상 전체를 넣고, 예측에서 가운데 영역만 잘라 낸 것
  zoom : 가운데 영역을 잘라 2배로 키운 영상(망원 렌즈처럼 보임)을 넣은 것
"""
import cv2
import numpy as np
from ultralytics import YOLO

from carla_data import depth_gt, frame_at
from depth_mono import MAX_DEPTH, delta1

model = YOLO("yolo26n-depth.pt")
CROP = (slice(180, 540), slice(320, 960))              # 가운데 640x360
res = {"full": [], "zoom": []}

for k in range(0, 400, 10):
    frame, gt = frame_at("clear", k), depth_gt("clear", k)[CROP]
    full = model(frame, verbose=False)[0].depth.data.cpu().numpy()[CROP]
    zoom_in = cv2.resize(frame[CROP], (1280, 720))       # 화각 90도 → 약 53도
    zoom = cv2.resize(model(zoom_in, verbose=False)[0].depth.data.cpu().numpy(), (640, 360))
    m = (gt > 0.5) & (gt < MAX_DEPTH)
    for key, pred in [("full", full), ("zoom", zoom)]:
        p, g = pred[m], gt[m]
        res[key].append((delta1(p, g), np.mean(np.abs(p - g) / g), np.median(g) / np.median(p)))

for key, v in res.items():
    v = np.array(v)
    print(f"{key:<5} δ<1.25 {v[:, 0].mean():.1%}, AbsRel {v[:, 1].mean():.3f}, "
          f"스케일(정답/예측) {v[:, 2].mean():.2f}")
