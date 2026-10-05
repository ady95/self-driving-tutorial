"""05-2 실습 3: LiDAR로 점유 격자(Occupancy Grid)를 만들고, 깊이·의미 분할 정답으로 만든 격자와 비교합니다."""
import sys
from pathlib import Path

import cv2
import numpy as np

from bev import BEV_H, BEV_W, colorize_occupancy, depth_to_points, lidar_occupancy, to_bev_cells, visible_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from carla_data import depth_gt, lidar_points, semantic_gt  # noqa: E402

OUT = Path("outputs/ch05")
OUT.mkdir(parents=True, exist_ok=True)
GROUND_TAGS = [1, 2, 10, 24, 25]               # 도로, 인도, 땅, 차선, 지면


def gt_occupancy(depth, sem, h_min=0.3, h_max=2.5):
    """정답 깊이로 카메라 픽셀을 3D로 되돌려 만든 점유 격자 (카메라에 보이는 곳만)."""
    X, Y, Z = depth_to_points(depth)
    grid = np.zeros((BEV_H, BEV_W), np.uint8)
    ground = np.isin(sem, GROUND_TAGS) & (Z < h_min)
    r, c, _ = to_bev_cells(X[ground], Y[ground])
    grid[r, c] = 1
    obst = (Z > h_min) & (Z < h_max) & (X < 80)
    r, c, _ = to_bev_cells(X[obst], Y[obst])
    grid[r, c] = 2
    return grid


def near(mask, cells):
    """mask의 각 칸에서 cells칸(0.1m 단위) 이내에 True가 있는지."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


if __name__ == "__main__":
    TOL = 3                                        # 허용 거리 3칸 = 0.3m
    vis = visible_mask()
    hit_g = n_g = hit_l = n_l = n_known = n_unknown = 0
    for k in range(0, 400, 10):
        gt = gt_occupancy(depth_gt("clear", k), semantic_gt("clear", k))
        li = lidar_occupancy(lidar_points("clear", k))
        known = vis & (gt > 0)                     # 정답이 정해진 칸만 채점
        g_occ, l_occ = (gt == 2) & known, (li == 2) & vis
        hit_g += np.sum(g_occ & near(li == 2, TOL))
        n_g += g_occ.sum()
        hit_l += np.sum(l_occ & near(gt == 2, TOL))
        n_l += l_occ.sum()
        n_known += known.sum()
        n_unknown += np.sum((li == 0) & known)
        if k == 100:
            cv2.imwrite(str(OUT / "occupancy.png"),
                        np.hstack([colorize_occupancy(gt), np.full((BEV_H, 6, 3), 255, np.uint8),
                                   colorize_occupancy(li)]))
            print(f"[프레임 100] LiDAR 격자: 막힘 {np.mean(li == 2):.1%}, 비어 있음 {np.mean(li == 1):.1%}, "
                  f"모름 {np.mean(li == 0):.1%}")
    print(f"맑은 날 40프레임, 카메라에 보이는 칸 기준 (허용 거리 {TOL * 0.1:.1f}m)")
    print(f"  정답 '막힘' {n_g}칸 중 근처에 LiDAR '막힘'이 있는 비율(재현율): {hit_g / n_g:.1%}")
    print(f"  LiDAR '막힘' {n_l}칸 중 근처에 정답 '막힘'이 있는 비율(정밀도): {hit_l / n_l:.1%}")
    print(f"  정답이 있는 칸 중 LiDAR가 '모름'으로 남긴 비율: {n_unknown / n_known:.1%}")
