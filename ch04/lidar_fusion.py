"""04-4: LiDAR 포인트를 카메라 영상에 투영하고, YOLO 상자와 합쳐 차까지의 거리를 잽니다.

카메라는 "무엇이" 있는지, LiDAR는 "얼마나 멀리" 있는지를 잘 압니다.
YOLO 상자 안에 떨어진 LiDAR 포인트로 거리를 재는 것이 가장 단순한 센서 퓨전입니다.
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from carla_data import (depth_gt, frame_at, gt_boxes, lidar_points, lidar_to_camera,
                        object_distance_gt, project, semantic_gt)

OUT = Path("outputs/ch04")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720


def lidar_on_image(points):
    """LiDAR 점을 화면 픽셀로 옮긴다. 카메라 뒤쪽과 화면 밖 점은 버린다."""
    cam = lidar_to_camera(points)
    cam = cam[cam[:, 0] > 0.5]
    u, v, d = project(cam)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok], v[ok], d[ok]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def box_distance(u, v, d, box):
    """상자 안 LiDAR 점들의 거리 중 가까운 쪽 30% 값. 상자 안의 배경 점을 피하려는 것.
    함께 돌려주는 점 수는 그 거리 ±1m 안의 점, 즉 실제로 차에 맞았다고 볼 수 있는 점의 수다."""
    x1, y1, x2, y2 = box
    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if m.sum() < 3:
        return None, 0
    dist = float(np.percentile(d[m], 30))
    return dist, int(np.sum(np.abs(d[m] - dist) < 1.0))


if __name__ == "__main__":
    # 1. 투영이 맞는지 확인: 투영된 LiDAR 거리 vs 같은 픽셀의 깊이 정답
    u, v, d = lidar_on_image(lidar_points("clear", 100))
    gt = depth_gt("clear", 100)
    err = np.abs(gt[v.astype(int), u.astype(int)] - d)
    print(f"[투영 확인] 화면 안 LiDAR 점 {len(d):,}개, 깊이 정답과 차이 중앙값 {np.median(err):.3f}m, "
          f"0.5m 이내 {np.mean(err < 0.5):.1%}")

    frame = frame_at("clear", 100)
    vis = frame.copy()
    colors = cv2.applyColorMap((np.clip(1 - d / 60, 0, 1) * 255).astype(np.uint8).reshape(-1, 1),
                               cv2.COLORMAP_TURBO)[:, 0]
    for uu, vv, c in zip(u.astype(int), v.astype(int), colors):
        cv2.circle(vis, (uu, vv), 2, tuple(int(x) for x in c), -1)
    cv2.imwrite(str(OUT / "lidar_on_image.jpg"), vis)

    # 2. 퓨전: YOLO 상자 + LiDAR 거리, 그리고 단안 깊이 모델과 비교
    det = YOLO("yolo26n.pt")
    depth_model = YOLO("yolo26n-depth.pt")
    rows = []
    for weather in ["clear", "night", "rain"]:
        boxes = gt_boxes(weather)
        for k in range(0, 400, 10):
            frame = frame_at(weather, k)
            dgt, sem = depth_gt(weather, k), semantic_gt(weather, k)
            u, v, d = lidar_on_image(lidar_points(weather, k))
            mono = depth_model(frame, verbose=False)[0].depth.data.cpu().numpy()
            r = det(frame, classes=[2, 5, 7], conf=0.25, verbose=False)[0]
            for pb in r.boxes.xyxy.tolist():
                # 정답 상자와 짝지어 정답 거리를 구한다
                match = [gb for c, gb in boxes[k] if c in ("car", "truck", "bus") and iou(pb, gb) >= 0.5]
                if not match:
                    continue
                true = object_distance_gt(dgt, sem, match[0])
                if true is None:
                    continue
                lid, n = box_distance(u, v, d, [int(t) for t in pb])
                x1, y1, x2, y2 = [int(t) for t in pb]
                mono_d = float(np.median(mono[y1:y2, x1:x2]))
                rows.append((weather, true, lid, n, mono_d))

    print(f"\n[퓨전] YOLO가 찾고 정답과 짝지어진 차량 {len(rows)}개")
    print(f"{'날씨':<7}{'차량':>5}{'LiDAR 점 3개 이상':>18}{'LiDAR 오차':>12}{'단안 깊이 오차':>14}")
    for weather in ["clear", "night", "rain"]:
        w = [r for r in rows if r[0] == weather]
        with_lidar = [r for r in w if r[2] is not None]
        le = np.median([abs(r[2] - r[1]) / r[1] for r in with_lidar])
        me = np.median([abs(r[4] - r[1]) / r[1] for r in w])
        print(f"{weather:<7}{len(w):5d}{len(with_lidar) / len(w):>17.1%}{le:>12.1%}{me:>14.1%}")
    print("\n차에 맞은 LiDAR 점 수 (거리 구간별 중앙값)")
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 100)]:
        b = [r[3] for r in rows if lo <= r[1] < hi]
        if b:
            print(f"  {lo:>2}~{hi:<3}m: 차량 {len(b):3d}대, 점 {np.median(b):5.0f}개")
