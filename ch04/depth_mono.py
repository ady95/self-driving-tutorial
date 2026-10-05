"""04-2: 카메라 한 대로 깊이 지도(Depth Map)를 만들고, CARLA 깊이 정답으로 채점합니다.

python ch04/depth_mono.py                     # yolo26n-depth, 날씨별 채점
python ch04/depth_mono.py yolo26s-depth.pt    # 더 큰 모델
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from carla_data import depth_gt, frame_at

MAX_DEPTH = 80.0                       # 이보다 먼 곳(하늘·먼 건물)은 채점에서 뺀다
OUT = Path("outputs/ch04")


def predict_depth(model, frame):
    r = model(frame, verbose=False)[0]
    return r.depth.data.cpu().numpy(), r.speed["inference"]   # (H, W) 미터


def colorize(depth, max_d=MAX_DEPTH):
    """가까울수록 밝게(노랑), 멀수록 어둡게(보라) 칠한다."""
    norm = np.clip(1 - depth / max_d, 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


BANDS = [(0.5, 10), (10, 20), (20, 40), (40, MAX_DEPTH)]


def delta1(p, g):
    return float(np.mean(np.maximum(p / g, g / p) < 1.25))      # 오차 25% 이내 픽셀 비율


def metrics(pred, gt):
    m = (gt > 0.5) & (gt < MAX_DEPTH)
    p, g = pred[m], gt[m]
    s = np.median(g) / np.median(p)                             # 전체 크기가 어긋난 정도
    out = {
        "absrel": float(np.mean(np.abs(p - g) / g)),            # 평균 상대 오차
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),           # 평균 제곱근 오차 (m)
        "delta1": delta1(p, g),
        "scale": float(s),
        "delta1_aligned": delta1(p * s, g),                     # 크기를 맞춘 뒤의 δ (모양만 평가)
    }
    for lo, hi in BANDS:                                        # 거리 구간별 δ
        b = (g >= lo) & (g < hi)
        out[f"band{lo:g}"] = delta1(p[b], g[b]) if b.sum() > 100 else np.nan
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n-depth.pt"
    model = YOLO(name)
    print(f"모델 {name}, 채점 범위 0.5~{MAX_DEPTH:.0f}m\n")
    bands = []
    print(f"{'날씨':<7}{'AbsRel':>8}{'RMSE':>8}{'δ<1.25':>8}{'스케일':>8}{'맞춘 δ':>8}{'추론':>8}")
    for weather in ["clear", "night", "rain"]:
        acc, ms = [], []
        for k in range(0, 400, 10):
            frame = frame_at(weather, k)
            pred, t = predict_depth(model, frame)
            acc.append(metrics(pred, depth_gt(weather, k)))
            ms.append(t)
            if k == 100:
                gt_vis = colorize(depth_gt(weather, k))
                cv2.imwrite(str(OUT / f"depth_{weather}.jpg"),
                            np.vstack([frame, colorize(pred), gt_vis]))
        avg = {key: np.nanmean([a[key] for a in acc]) for key in acc[0]}
        bands.append((weather, avg))
        print(f"{weather:<7}{avg['absrel']:8.3f}{avg['rmse']:7.2f}m{avg['delta1']:8.1%}"
              f"{avg['scale']:8.2f}{avg['delta1_aligned']:8.1%}{np.median(ms):6.1f}ms")

    print("\n거리 구간별 δ<1.25")
    print(f"{'날씨':<7}" + "".join(f"{f'{lo:g}~{hi:g}m':>10}" for lo, hi in BANDS))
    for weather, avg in bands:
        print(f"{weather:<7}" + "".join(f"{avg[f'band{lo:g}']:10.1%}" for lo, _ in BANDS))
