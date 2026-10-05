"""05장 공통: 카메라 영상과 LiDAR를 위에서 본 지도(BEV)로 옮기는 함수들.

BEV 격자: 자차(카메라 바로 아래 지면)를 원점으로 앞 0~40m, 좌우 ±20m, 한 칸 0.1m.
BEV 이미지에서 위쪽이 앞, 오른쪽이 오른쪽이다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from carla_data import CAM, K, lidar_to_camera  # noqa: E402  04장의 공통 함수를 그대로 쓴다

RES = 0.1                                     # 한 칸 = 0.1m
X_MAX, Y_MAX = 40.0, 20.0                     # 앞 40m, 좌우 20m
BEV_H, BEV_W = int(X_MAX / RES), int(2 * Y_MAX / RES)
CAM_H = CAM["z"]                              # 카메라 높이 1.6m


def bev_grid():
    """BEV 각 칸 중심의 지면 좌표 (X 앞, Y 오른쪽)."""
    rows, cols = np.mgrid[0:BEV_H, 0:BEV_W]
    X = X_MAX - (rows + 0.5) * RES
    Y = (cols + 0.5) * RES - Y_MAX
    return X, Y


def ipm_maps():
    """역원근 변환(IPM)용 좌표표: BEV 칸마다, 그 지면 점이 찍히는 영상 픽셀 (u, v)."""
    X, Y = bev_grid()
    X = np.maximum(X, 0.1)
    u = K[0, 0] * Y / X + K[0, 2]
    v = K[1, 1] * CAM_H / X + K[1, 2]         # 지면은 카메라보다 CAM_H만큼 아래
    return u.astype(np.float32), v.astype(np.float32)


MAP_U, MAP_V = ipm_maps()


def ipm(image, interpolation=cv2.INTER_LINEAR):
    """카메라 영상(또는 라벨 영상) → BEV. 화면 밖이거나 카메라에 안 보이는 칸은 0."""
    return cv2.remap(image, MAP_U, MAP_V, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def visible_mask():
    """카메라 화각 안에 들어오는 BEV 칸 (가까운 2.84m 이내 지면은 보이지 않는다)."""
    return (MAP_U >= 0) & (MAP_U < 1280) & (MAP_V >= 0) & (MAP_V < 720)


def to_bev_cells(X, Y):
    """지면 좌표 → BEV (행, 열). 격자 밖은 None."""
    r = ((X_MAX - X) / RES).astype(int)
    c = ((Y + Y_MAX) / RES).astype(int)
    ok = (r >= 0) & (r < BEV_H) & (c >= 0) & (c < BEV_W)
    return r[ok], c[ok], ok


def depth_to_points(depth):
    """깊이 영상 → 카메라 기준 3D 점 (X 앞, Y 오른쪽, Z 지면 위 높이)."""
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    X = depth
    Y = (u - K[0, 2]) * X / K[0, 0]
    Z = CAM_H - (v - K[1, 2]) * X / K[1, 1]
    return X, Y, Z


def lidar_occupancy(points, h_min=0.3, h_max=2.5):
    """LiDAR → 점유 격자. 0 = 모름, 1 = 비어 있음, 2 = 막힘.

    지면에서 h_min~h_max 높이의 점이 떨어진 칸은 막힘, 센서에서 그 점까지 레이저가 지나간 칸은
    비어 있음, 레이저가 닿지 않은 칸은 모름으로 둔다."""
    xyz = lidar_to_camera(points)
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2] + CAM_H      # Z: 지면 위 높이
    grid = np.zeros((BEV_H, BEV_W), np.uint8)
    # 1) 레이저가 지나간 길: 센서(원점)에서 점까지 칸들을 '비어 있음'으로
    origin = (int(Y_MAX / RES), int(X_MAX / RES))          # 원점의 (열, 행)
    front = X > 0
    rows = ((X_MAX - X[front]) / RES).astype(int)          # 격자 밖 점도 쓴다 (cv2.line이 잘라 줌)
    cols = ((Y[front] + Y_MAX) / RES).astype(int)
    for rr, cc in zip(rows, cols):
        cv2.line(grid, origin, (int(cc), int(rr)), 1, 1)
    # 2) 장애물 높이의 점이 떨어진 칸은 '막힘'
    obst = (Z > h_min) & (Z < h_max)
    r, c, _ = to_bev_cells(X[obst], Y[obst])
    grid[r, c] = 2
    return grid


def colorize_occupancy(grid):
    out = np.zeros((*grid.shape, 3), np.uint8)
    out[grid == 1] = (200, 200, 200)                        # 비어 있음: 밝은 회색
    out[grid == 2] = (0, 0, 220)                            # 막힘: 빨강
    return out
