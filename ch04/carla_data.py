"""04장 공통: CARLA 도심 데이터 읽기와 카메라 투영 함수.

좌표계 (CARLA, 왼손 좌표계)
  차량·LiDAR : x 앞, y 오른쪽, z 위 (미터)
  카메라 영상 : u 오른쪽, v 아래 (픽셀)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/carla_urban")
CALIB = json.loads((ROOT / "calib.json").read_text(encoding="utf-8"))
K = np.array(CALIB["K"])                      # 카메라 내부 행렬 3x3
CAM = CALIB["camera_in_vehicle"]              # 카메라 장착 위치 (차량 기준)
LIDAR = CALIB["lidar_in_vehicle"]             # LiDAR 장착 위치 (차량 기준)
CAR_TAGS = [14, 15, 16]                       # CARLA 의미 분할: Car, Truck, Bus


def frame_at(weather, k):
    cap = cv2.VideoCapture(str(ROOT / weather / "rgb.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, frame = cap.read()
    cap.release()
    return frame


def depth_gt(weather, k):
    """깊이 정답 (미터). 10프레임마다 있다."""
    return cv2.imread(str(ROOT / weather / "depth" / f"{k:04d}.png"), cv2.IMREAD_UNCHANGED) / 100.0


def semantic_gt(weather, k):
    return cv2.imread(str(ROOT / weather / "semantic" / f"{k:04d}.png"), cv2.IMREAD_UNCHANGED)


def lidar_points(weather, k):
    """LiDAR 포인트 (N, 4): x, y, z, 반사 강도. LiDAR 기준 좌표."""
    return np.load(ROOT / weather / "lidar" / f"{k:04d}.npy").astype(np.float32)


def gt_boxes(weather, min_pixels=400):
    boxes = defaultdict(list)                 # 프레임 → [(클래스, 상자)]
    with open(ROOT / weather / "boxes.csv") as f:
        for r in csv.DictReader(f):
            if int(r["pixels"]) >= min_pixels:
                boxes[int(r["frame"])].append(
                    (r["class"], [int(r[c]) for c in ("x1", "y1", "x2", "y2")]))
    return boxes


def lidar_to_camera(points):
    """LiDAR 기준 점 → 카메라 기준 점. 두 센서 모두 회전 없이 장착되어 이동만 하면 된다."""
    xyz = points[:, :3].copy()
    xyz[:, 0] += LIDAR["x"] - CAM["x"]
    xyz[:, 1] += LIDAR["y"] - CAM["y"]
    xyz[:, 2] += LIDAR["z"] - CAM["z"]
    return xyz


def project(xyz_cam):
    """카메라 기준 3D 점 (앞 x, 오른쪽 y, 위 z) → 픽셀 (u, v)와 깊이(앞 방향 거리)."""
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    u = K[0, 0] * y / x + K[0, 2]
    v = K[1, 1] * (-z) / x + K[1, 2]
    return u, v, x


def object_distance_gt(depth, sem, box):
    """정답 상자 안에서 차량 클래스 픽셀의 깊이 중앙값 = 그 차까지의 거리(앞 방향)."""
    x1, y1, x2, y2 = box
    d = depth[y1:y2 + 1, x1:x2 + 1]
    m = np.isin(sem[y1:y2 + 1, x1:x2 + 1], CAR_TAGS)
    return float(np.median(d[m])) if m.sum() > 20 else None
