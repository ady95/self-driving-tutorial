"""02-3: 차선 검출을 일부러 실패시켜 봅니다.

깨끗한 영상(solidWhiteRight)의 검출 결과를 기준으로 삼고, 같은 프레임에
야간·비·그림자·가려짐 조건을 합성해 결과가 얼마나 어긋나는지 잽니다.
python ch02/break_lanes.py            → 기본 Canny 임계값 (50, 150)
python ch02/break_lanes.py 20 60      → 낮은 임계값으로 다시 측정
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import lane_detect as ld

VIDEO = Path("data/videos/solidWhiteRight.mp4")
OUT = Path("outputs/ch02/break")
TOLERANCE = 30                         # 기준 결과와 이만큼(px) 이내면 '맞게 찾았다'고 본다


def night(frame, i):
    """빛을 70% 줄이고, 어두울 때 커지는 센서 노이즈를 더한다."""
    rng = np.random.default_rng(i)
    dark = frame.astype(np.float32) * 0.3
    noise = rng.normal(0, 6, frame.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def rain(frame, i):
    """대비를 낮추고(물안개), 빗줄기를 긋고, 살짝 흐리게 한다."""
    rng = np.random.default_rng(i)
    out = cv2.addWeighted(frame, 0.6, np.full_like(frame, 150), 0.4, 0)
    h, w = frame.shape[:2]
    for _ in range(250):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(out, (x, y), (x - 6, y + 25), (200, 200, 200), 1)
    return cv2.GaussianBlur(out, (5, 5), 0)


def _leaf_blobs(seed=0, n=60):
    """나뭇잎 그림자를 이룰 타원들의 위치·크기·각도 (한 번만 만든다)."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0, 1), rng.uniform(0, 2), rng.uniform(15, 60),
             rng.uniform(5, 25), rng.uniform(0, 180)) for _ in range(n)]


LEAVES = _leaf_blobs()


def shadow(frame, i):
    """나뭇잎 그림자: 경계가 뚜렷한 어두운 얼룩들이 차가 달리는 만큼 아래로 흘러간다."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for fx, fy, ax, ay, ang in LEAVES:
        y = int(h * 0.55 + ((fy * h * 0.45 + i * 6) % (h * 0.9)))
        cv2.ellipse(mask, (int(fx * w), y), (int(ax), int(ay)), ang, 0, 360, 255, -1)
    out = frame.copy()
    out[mask > 0] = (out[mask > 0] * 0.4).astype(np.uint8)
    return out


def _car_patch():
    """challenge.mp4 첫 프레임에서 앞차를 잘라 온다 (번호판은 흐리게)."""
    cap = cv2.VideoCapture("data/videos/challenge.mp4")
    ok, f = cap.read()
    cap.release()
    car = f[400:508, 815:980].copy()
    car[42:62, 86:128] = cv2.GaussianBlur(car[42:62, 86:128], (21, 21), 0)
    return cv2.resize(car, None, fx=2.0, fy=2.0)


CAR = _car_patch()


def occlusion(frame, i):
    """60~160번 프레임 동안 오른쪽 차로의 차가 내 차로로 끼어들며 오른쪽 차선을 가린다."""
    out = frame.copy()
    if 60 <= i < 160:
        h, w = frame.shape[:2]
        ch, cw = CAR.shape[:2]
        x = int(w * 0.80 - (i - 60) * 2.0)            # 오른쪽에서 왼쪽으로 조금씩 이동
        y = h - ch - 10
        out[y:y + ch, x:x + cw] = CAR[:, :max(0, min(cw, w - x))]
    return out


CONDITIONS = {"night": night, "rain": rain, "shadow": shadow, "occlusion": occlusion}


def lane_error(lane, ref):
    """두 차선의 아래·위 끝점 x좌표 차이의 평균(px)."""
    return (abs(lane[0] - ref[0]) + abs(lane[2] - ref[2])) / 2


def evaluate(infos, reference):
    detected = correct = 0
    errors = []
    for info, ref in zip(infos, reference):
        for side in ("left", "right"):
            lane, base = info[side], ref[side]
            if base is None or lane is None:      # 기준에 없거나, 못 찾았거나
                continue
            detected += 1
            err = lane_error(lane, base)
            errors.append(err)
            correct += err <= TOLERANCE
    total = sum((r["left"] is not None) + (r["right"] is not None) for r in reference)
    return detected / total, correct / total, float(np.median(errors)) if errors else float("nan")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reference = ld.run_video(VIDEO)                      # 기본 설정 + 깨끗한 영상의 결과 = 기준

    if len(sys.argv) == 3:                               # 임계값을 바꿔 다시 측정할 때
        ld.CANNY_LOW, ld.CANNY_HIGH = int(sys.argv[1]), int(sys.argv[2])
    tag = f"canny{ld.CANNY_LOW}-{ld.CANNY_HIGH}"
    print(f"Canny 임계값 ({ld.CANNY_LOW}, {ld.CANNY_HIGH}), 허용 오차 {TOLERANCE}px\n")
    print(f"{'조건':<10}{'검출률':>8}{'정확도':>8}{'오차 중앙값':>12}{'흔들림':>9}")
    for name, fn in [("clean", None)] + list(CONDITIONS.items()):
        infos = ld.run_video(VIDEO, OUT / f"{name}_{tag}.mp4", frame_filter=fn)
        det, acc, med = evaluate(infos, reference)
        jitter = ld.summarize(infos)["jitter"]
        med_s = "-" if np.isnan(med) else f"{med:.1f}px"
        jit_s = "-" if np.isnan(jitter) else f"{jitter:.1f}px"
        print(f"{name:<10}{det:8.1%}{acc:8.1%}{med_s:>12}{jit_s:>9}")

        # 조건마다 대표 프레임(100번)을 저장해 둔다
        cap = cv2.VideoCapture(str(OUT / f"{name}_{tag}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ok, frame = cap.read()
        cap.release()
        cv2.imwrite(str(OUT / f"{name}_{tag}_f100.jpg"), frame)
