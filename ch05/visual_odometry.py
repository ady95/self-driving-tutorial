"""05-4: 단안 Visual Odometry — 영상만으로 차의 회전과 이동 방향을 추정하고, 정답 궤적과 비교합니다.

특징점을 광류(Optical Flow)로 추적 → Essential Matrix로 두 프레임 사이 카메라 움직임(R, t) 추정.
단안 카메라는 이동 '거리'(스케일)를 알 수 없으므로, 거리만 바퀴 속도에서 가져옵니다.

python ch05/visual_odometry.py        # 모든 프레임
python ch05/visual_odometry.py 2      # 2프레임마다 (10 FPS)
python ch05/visual_odometry.py 1 --no-check   # 타당성 검사 없이
"""
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("data/carla_drive")
OUT = Path("outputs/ch05")
OUT.mkdir(parents=True, exist_ok=True)
K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])   # carla_urban/calib.json과 같은 카메라
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
CHECK = "--no-check" not in sys.argv          # 타당성 검사 끄기 (비교용)
MAX_TURN = 5.0 if CHECK else 360.0            # 한 프레임에 허용하는 최대 회전(도)
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

ego = np.loadtxt(ROOT / "ego.csv", delimiter=",", skiprows=1)
gt_xy = ego[:, 2:4]
gt_yaw = np.unwrap(np.radians(ego[:, 7]))
speed = ego[:, 8] / 3.6 * 1.02 + np.random.default_rng(0).normal(0, 0.1, len(ego))  # 05-3과 같은 바퀴 속도


def detect(gray):
    p = cv2.goodFeaturesToTrack(gray, maxCorners=1500, qualityLevel=0.01, minDistance=8)
    return p.reshape(-1, 2) if p is not None else np.zeros((0, 2), np.float32)


cap = cv2.VideoCapture(str(ROOT / "rgb.mp4"))
ok, frame = cap.read()
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
pts = detect(prev)
R_w = np.eye(3)                                # 카메라 방향 (OpenCV 축: x 오른쪽, y 아래, z 앞)
pos = [gt_xy[0].copy()]                        # 지면 위치 (정답 좌표계)
yaw0 = gt_yaw[0]
headings, n_inliers, skipped, rejected = [yaw0], [], 0, 0
k, start = 0, time.perf_counter()

while True:
    for _ in range(STRIDE):                    # STRIDE만큼 건너뛴다
        ok, frame = cap.read()
        k += 1
    if not ok or k >= len(ego):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts.astype(np.float32), None, **LK)
    good = st.ravel() == 1
    p1, p2 = pts[good], nxt.reshape(-1, 2)[good]
    step = sum(speed[k - STRIDE + 1:k + 1]) * 0.05     # 이번 구간의 이동 거리 (바퀴 속도)
    flow = np.median(np.linalg.norm(p2 - p1, axis=1)) if len(p1) else 0
    if len(p1) > 8 and flow > 0.5 and step > 0.05:
        E, mask = cv2.findEssentialMat(p2, p1, K, cv2.RANSAC, 0.999, 1.0)
        R, t = np.eye(3), np.array([0, 0, 1.0])       # 기본값: 회전 없이 앞으로
        if E is not None and E.shape == (3, 3):
            _, R_est, t_est, mask2 = cv2.recoverPose(E, p2, p1, K, mask=mask)
            n_inliers.append(int(np.count_nonzero(mask2)))
            # 타당성 검사: 한 스텝에 MAX_TURN도 넘게 돌 수는 없다 (20 FPS에서 초당 100도)
            if np.degrees(np.linalg.norm(cv2.Rodrigues(R_est)[0])) <= MAX_TURN * STRIDE:
                R, t = R_est, t_est.ravel()
            else:
                rejected += 1
        # (p2, p1) 순서로 넣었으므로 R, t는 '이전 카메라 기준으로 본 현재 카메라'의 자세다
        d = R_w @ t                                    # 이번 이동 방향을 누적 방향으로 옮긴다
        d = d / (np.linalg.norm(d) + 1e-9)
        R_w = R_w @ R                                  # 카메라가 돈 만큼 누적
        fwd = np.array([d[2], d[0]])                   # 지면 성분: (앞, 오른쪽)
        c, s = np.cos(yaw0), np.sin(yaw0)              # 처음 방향을 정답 좌표계에 맞춘다
        pos.append(pos[-1] + step * np.array([c * fwd[0] - s * fwd[1], s * fwd[0] + c * fwd[1]]))
    else:                                              # 거의 멈춰 있으면 움직임 추정을 건너뛴다
        skipped += 1
        pos.append(pos[-1].copy())
    if k == 300 and STRIDE == 1:                       # 광류 추적 모습을 한 장 저장한다
        vis = frame.copy()
        for a, b in zip(p1[::3], p2[::3]):
            cv2.arrowedLine(vis, tuple(int(v) for v in a), tuple(int(v) for v in b + (b - a) * 4),
                            (0, 255, 0), 1, tipLength=0.3)   # 화살표는 5배로 늘려 그림
        cv2.imwrite(str(OUT / "vo_flow.jpg"), cv2.resize(vis, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    head = R_w @ np.array([0, 0, 1.0])
    headings.append(yaw0 + np.arctan2(head[0], head[2]))
    prev = gray
    pts = p2 if len(p2) > 800 else detect(gray)        # 특징점이 줄면 새로 찾는다

elapsed = time.perf_counter() - start
pos = np.array(pos)
idx = np.arange(0, len(pos)) * STRIDE
err = np.linalg.norm(pos - gt_xy[idx], axis=1)
yaw_err = np.abs((np.degrees(np.array(headings) - gt_yaw[idx]) + 180) % 360 - 180)   # 차이를 ±180도로 감는다
print(f"{STRIDE}프레임마다 처리 ({20 / STRIDE:.0f} FPS), {len(pos)}스텝, 처리 {len(pos) / elapsed:.0f} 스텝/초")
print(f"건너뛴 스텝(거의 정지·특징 부족): {skipped}, RANSAC 인라이어 중앙값 {int(np.median(n_inliers))}개")
print(f"타당성 검사 {'켬' if CHECK else '끔'}: 버린 회전 추정 {rejected}개")
print(f"위치 오차   평균 {err.mean():.2f}m, 최대 {err.max():.2f}m, 마지막 {err[-1]:.2f}m")
print(f"방향 오차   평균 {yaw_err.mean():.2f}도, 마지막 {yaw_err[-1]:.2f}도")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(gt_xy[:, 1], gt_xy[:, 0], "k-", lw=3, label="ground truth")
ax.plot(pos[:, 1], pos[:, 0], "-", c="tab:purple", label=f"visual odometry (stride {STRIDE})")
ax.set_xlabel("y (m)")
ax.set_ylabel("x (m)")
ax.set_aspect("equal")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"vo_s{STRIDE}{'' if CHECK else '_nocheck'}.png", dpi=90)
