"""03-2 실습 4: 02-3에서 차선 검출을 무너뜨린 조건(야간·비·그림자)을 YOLO에 그대로 넣어 봅니다.

깨끗한 영상의 검출 결과를 기준으로, 같은 프레임에 조건을 합성했을 때
기준 객체를 얼마나 다시 찾는지(재현율)와 없던 객체를 얼마나 만들어 내는지(가짜 검출)를 잽니다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
from break_lanes import night, rain, shadow   # 02-3의 조건 함수를 그대로 쓴다

VIDEO = Path("data/videos/challenge.mp4")
VEHICLES = [2, 5, 7]                          # car, bus, truck
IOU_MATCH = 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def match(boxes, refs):
    """기준 박스마다 IoU가 가장 큰 미사용 박스를 짝지어, (찾은 수, 남은 박스 수)를 돌려준다."""
    used = set()
    found = 0
    for r in refs:
        best, best_j = 0.0, None
        for j, b in enumerate(boxes):
            if j not in used and iou(r, b) > best:
                best, best_j = iou(r, b), j
        if best >= IOU_MATCH:
            used.add(best_j)
            found += 1
    return found, len(boxes) - len(used)


def detect(model, frame):
    r = model(frame, classes=VEHICLES, conf=0.25, verbose=False)[0]
    return r.boxes.xyxy.tolist()


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    reference = [detect(model, f) for f in frames]          # 깨끗한 영상 = 기준
    n_ref = sum(len(r) for r in reference)
    print(f"기준: {VIDEO.name} {len(frames)} 프레임, 차량 {n_ref}개 (프레임당 {n_ref / len(frames):.2f})\n")
    print(f"{'조건':<8}{'재현율':>8}{'가짜 검출':>10}")

    out = Path("outputs/ch03")
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [("night", night), ("rain", rain), ("shadow", shadow)]:
        found = extra = 0
        for i, (f, ref) in enumerate(zip(frames, reference)):
            g = fn(f, i)
            boxes = detect(model, g)
            a, b = match(boxes, ref)
            found += a
            extra += b
            if i == 150:                                     # 대표 프레임 저장
                cv2.imwrite(str(out / f"cond_{name}.jpg"), model(g, classes=VEHICLES, verbose=False)[0].plot())
        print(f"{name:<8}{found / n_ref:8.1%}{extra:>10}")
