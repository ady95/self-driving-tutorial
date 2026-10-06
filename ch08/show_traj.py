"""08-4: 검증 데이터에서 정답 궤적(초록)과 예측 궤적(빨강)을 카메라 영상 위에 그립니다.

궤적은 차 좌표 (앞, 오른쪽) 미터 단위이고 노면 위의 점이므로, 카메라(앞 1.5m, 높이 1.6m, 화각 90도)로
투영하면 영상 위에 그릴 수 있습니다(04-1의 핀홀 카메라 모델).

python ch08/show_traj.py
결과: outputs/ch08/traj_samples.jpg
"""
import argparse
import math

import cv2
import numpy as np
import torch

from train_bc import to_tensor
from train_traj import OUT, build_traj_model, load_items

W, H = 320, 180
SCALE = 1.5                                         # 그림을 키워서 그린다
F = W / 2 / math.tan(math.radians(90) / 2)          # 초점 거리(픽셀) = 160
CAM_X, CAM_Z = 1.5, 1.6


def project(pts):
    """차 좌표 (앞, 오른쪽) 노면 점 → 영상 픽셀 (u, v)."""
    fwd = np.maximum(pts[:, 0] - CAM_X, 0.5)
    return (SCALE * np.stack([W / 2 + F * pts[:, 1] / fwd, H / 2 + F * CAM_Z / fwd], 1)).astype(int)


def draw(img, wps, color):
    pts = project(np.vstack([[CAM_X + 1.0, 0.0], wps]))
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
    for p in pts[1:]:
        cv2.circle(img, tuple(p), 3, color, -1, cv2.LINE_AA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_traj_model().to(device)
    model.load_state_dict(torch.load(OUT / "traj.pt", map_location=device))
    model.eval()
    _, val = load_items()
    lateral = np.array([abs(w[-1, 1]) for _, _, w in val])
    moving = np.array([sp > 3.0 for _, sp, _ in val])             # 달리는 장면만 (정지·출발 제외)
    turns = np.flatnonzero(moving & (lateral > 1.5) & (lateral < 4.0))   # 2초 뒤 옆으로 1.5~4m (회전)
    straight = np.flatnonzero(moving & (lateral < 0.3))
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(turns, args.n // 2, replace=False),
                           rng.choice(straight, args.n - args.n // 2, replace=False)])
    tiles = []
    for i in pick:
        path, speed, gt = val[i]
        img = cv2.imread(str(path))
        with torch.no_grad():
            pred = model(to_tensor(img)[None].to(device), torch.tensor([[speed]], device=device))[0].cpu().numpy()
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE)
        draw(img, gt, (0, 200, 0))
        draw(img, pred, (0, 0, 255))
        err = np.linalg.norm(pred[-1] - gt[-1])
        cv2.putText(img, f"{speed * 3.6:.0f}km/h  2s err {err:.2f}m", (8, 22), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    cols = args.n - args.n // 2
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite(str(OUT / "traj_samples.jpg"), np.vstack(rows))
    print(f"저장: {OUT / 'traj_samples.jpg'} (위: 회전 {args.n // 2}장, 아래: 직진 {args.n - args.n // 2}장)")
