"""11-2: 카메라로 신호 읽기 — 지도가 알려 준 신호등 위치를 영상에 투영하고, 켜진 등의 '위치'로 색을 정합니다.

  1. 어디를 볼까: HD 지도에는 신호등 위치가 있다. 내 차로 신호등의 등(head) 상자를 카메라 영상에 투영(04-1)
  2. 무슨 색일까: 투영한 영역을 세로 3칸으로 나눠 가장 밝은 칸이 위 = RED, 가운데 = YELLOW, 아래 = GREEN
     09-2에서 VLM이 낮에 주황색으로 보이는 빨간 등을 노랑으로 읽은 것을 피하려고, 색이 아니라 위치로 판단
신호 '상태'는 카메라로만 읽는다. 지도는 '위치'만 준다.
"""
import numpy as np

STATES = ["RED", "YELLOW", "GREEN"]
W, H = 1280, 720
K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])     # 화각 90도 카메라 (07-3과 같음)


def project_box(corners_world, cam_matrix_inv):
    """세계 좌표의 상자 꼭짓점 8개 → 영상 위 사각형 (x1, y1, x2, y2). 카메라 뒤면 None."""
    p = np.c_[corners_world, np.ones(len(corners_world))] @ cam_matrix_inv.T     # 카메라 좌표 (앞, 오른쪽, 위)
    if np.any(p[:, 0] < 1.0):
        return None
    u = K[0, 2] + K[0, 0] * p[:, 1] / p[:, 0]
    v = K[1, 2] - K[1, 1] * p[:, 2] / p[:, 0]
    x1, x2, y1, y2 = int(u.min()), int(u.max()), int(v.min()), int(v.max())
    if x2 < 0 or y2 < 0 or x1 >= W or y1 >= H:
        return None
    return max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)


def lamp_state(crop, min_contrast=25):
    """신호등 등 하나의 영역 → (상태, 대비). 세 칸 밝기 차이가 작으면 None (등이 안 보임)."""
    h, w = crop.shape[:2]
    if h < 9 or w < 3:
        return None, 0.0
    t = int(h * 0.12)                                  # 위아래 가장자리는 버린다: 상자 위쪽에 하늘이 섞여 들어온다
    v = crop[t:h - t, w // 4: w - w // 4].max(axis=2).astype(np.float32)    # 가운데 세로 띠의 밝기
    h = v.shape[0]
    bright = [float(np.percentile(v[i * h // 3:(i + 1) * h // 3], 90)) for i in range(3)]
    i = int(np.argmax(bright))
    contrast = bright[i] - float(np.median(bright))
    return (STATES[i] if contrast >= min_contrast else None), contrast


def read_light(img, rois):
    """투영한 등 영역들로 투표 → 상태 하나. 큰(가까운) 영역의 표가 더 무겁다."""
    votes = {}
    for x1, y1, x2, y2 in rois:
        if (y2 - y1) < 9 or (x2 - x1) > (y2 - y1):    # 너무 작거나, 화면 가장자리에서 잘린 영역은 버림
            continue
        state, contrast = lamp_state(img[y1:y2, x1:x2])
        if state is not None:
            votes[state] = votes.get(state, 0.0) + (y2 - y1)
    return max(votes, key=votes.get) if votes else None


class LightFilter:
    """프레임마다 흔들리는 판독을 안정시킨다.
    - 유효한 판독끼리 같은 상태가 n번 이어지면 바꾼다. 판독이 없는 프레임(None)은 건너뛰고 세지 않는다
      (예: RED, None, RED, None, RED도 RED 3번으로 본다)
    - 아직 상태가 없을 때(처음, 또는 hold 프레임 넘게 못 본 뒤)는 첫 판독을 바로 받는다
    - 판독이 없으면 hold 프레임(1초) 동안은 직전 상태를 유지한다"""

    def __init__(self, n=3, hold=20):
        self.n, self.hold = n, hold
        self.state, self.cand, self.count, self.missing = None, None, 0, 0

    def update(self, reading):
        if reading is None:
            self.missing += 1
            if self.missing > self.hold:            # 1초(20스텝) 넘게 안 보이면 모름
                self.state = None
            return self.state
        self.missing = 0
        if reading == self.cand:
            self.count += 1
        else:
            self.cand, self.count = reading, 1
        if self.count >= self.n or self.state is None:
            self.state = reading
        return self.state
