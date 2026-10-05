"""03-2 실습 5: CARLA 도심 영상의 정답 상자로 YOLO를 날씨별로 채점합니다.

data/carla_urban/{clear,night,rain}/rgb.mp4 와 boxes.csv(정답)를 씁니다.
같은 시드로 녹화한 세 영상은 같은 장면이므로, 날씨만 바뀌었을 때 성능이 어떻게 변하는지 볼 수 있습니다.
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("data/carla_urban")
IOU_MATCH = 0.5
MIN_GT_PIXELS = 400                 # 이보다 작게 보이는 정답은 채점에서 뺀다 (너무 멀거나 거의 가려짐)
# CARLA 정답 클래스 → 비교할 그룹, YOLO(COCO) 클래스 → 비교할 그룹
GT_GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "pedestrian": "person"}
YOLO_GROUP = {2: "vehicle", 5: "vehicle", 7: "vehicle", 0: "person"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def load_gt(path):
    """채점할 정답과, 너무 작아 채점에서 빼는 정답(무시 영역)을 나눠 읽는다."""
    gt, ignore = defaultdict(list), defaultdict(list)     # 프레임 → [(그룹, 상자)]
    with open(path) as f:
        for r in csv.DictReader(f):
            g = GT_GROUP.get(r["class"])
            if g:
                box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                (gt if int(r["pixels"]) >= MIN_GT_PIXELS else ignore)[int(r["frame"])].append((g, box))
    return gt, ignore


def evaluate(model, weather):
    gt, ignore = load_gt(ROOT / weather / "boxes.csv")
    stats = {g: {"tp": 0, "fn": 0, "fp": 0} for g in ("vehicle", "person")}
    cap = cv2.VideoCapture(str(ROOT / weather / "rgb.mp4"))
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, classes=list(YOLO_GROUP), conf=0.25, verbose=False)[0]
        preds = [(YOLO_GROUP[int(c)], b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        for g in stats:
            P = [b for gg, b in preds if gg == g]
            G = [b for gg, b in gt[k] if gg == g]
            used = set()
            for gb in G:
                best, bj = 0.0, None
                for j, pb in enumerate(P):
                    if j not in used and iou(gb, pb) > best:
                        best, bj = iou(gb, pb), j
                if best >= IOU_MATCH:
                    used.add(bj)
                    stats[g]["tp"] += 1
                else:
                    stats[g]["fn"] += 1
            # 짝이 없는 예측 중, 채점에서 뺀 작은 정답과 겹치는 것은 가짜로 세지 않는다
            small = [b for gg, b in ignore[k] if gg == g]
            stats[g]["fp"] += sum(1 for j, pb in enumerate(P)
                                  if j not in used and not any(iou(pb, sb) >= IOU_MATCH for sb in small))
        k += 1
    cap.release()
    return stats


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    print(f"{'날씨':<7}{'그룹':<9}{'정답 수':>7}{'재현율':>8}{'정밀도':>8}")
    for weather in ["clear", "night", "rain"]:
        for g, s in evaluate(model, weather).items():
            n = s["tp"] + s["fn"]
            recall = s["tp"] / n if n else 0
            precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
            print(f"{weather:<7}{g:<9}{n:>7}{recall:8.1%}{precision:8.1%}")
