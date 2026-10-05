"""03-4 실습 2: CARLA 정답 ID로 추적기를 채점합니다.

정답 상자(boxes.csv)의 object_id는 프레임이 바뀌어도 유지됩니다. 추적기가 같은 객체에
같은 ID를 계속 붙이는지(ID 전환 수), 그리고 전체 품질(MOTA)을 잽니다.

python ch03/track_eval_carla.py                 # ByteTrack, 모든 프레임
python ch03/track_eval_carla.py 4 botsort.yaml  # 4프레임마다 (5 FPS), BoT-SORT
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path("data/carla_urban/clear")
MIN_GT_PIXELS = 400
GT_GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "pedestrian": "person"}
YOLO_GROUP = {2: "vehicle", 5: "vehicle", 7: "vehicle", 0: "person"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def load_gt():
    """채점할 정답과, 너무 작아 채점에서 빼는 정답(무시 영역)을 나눠 읽는다."""
    gt, ignore = defaultdict(list), defaultdict(list)     # 프레임 → [(그룹, 정답 ID, 상자)]
    with open(ROOT / "boxes.csv") as f:
        for r in csv.DictReader(f):
            g = GT_GROUP.get(r["class"])
            if g:
                box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                item = (g, int(r["object_id"]), box)
                (gt if int(r["pixels"]) >= MIN_GT_PIXELS else ignore)[int(r["frame"])].append(item)
    return gt, ignore


if __name__ == "__main__":
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tracker = sys.argv[2] if len(sys.argv) > 2 else "bytetrack.yaml"
    model = YOLO("yolo26n.pt")
    gt, ignore = load_gt()

    tp = fn = fp = idsw = 0
    last_track = {}                                 # 정답 ID → 마지막으로 짝지어진 추적 ID
    used_tracks = defaultdict(set)                  # 정답 ID → 지금까지 붙은 추적 ID들
    cap = cv2.VideoCapture(str(ROOT / "rgb.mp4"))
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % stride == 0:
            r = model.track(frame, persist=True, tracker=tracker, classes=list(YOLO_GROUP),
                            conf=0.25, verbose=False)[0]
            preds = []
            if r.boxes.id is not None:
                preds = [(YOLO_GROUP[int(c)], int(t), b) for c, t, b in
                         zip(r.boxes.cls.tolist(), r.boxes.id.tolist(), r.boxes.xyxy.tolist())]
            matched = set()
            for g, gid, gbox in gt[k]:
                best, bj = 0.0, None
                for j, (pg, tid, pbox) in enumerate(preds):
                    if j not in matched and pg == g and iou(gbox, pbox) > best:
                        best, bj = iou(gbox, pbox), j
                if best >= 0.5:
                    matched.add(bj)
                    tp += 1
                    tid = preds[bj][1]
                    if gid in last_track and last_track[gid] != tid:
                        idsw += 1                   # 같은 객체인데 추적 ID가 바뀌었다
                    last_track[gid] = tid
                    used_tracks[gid].add(tid)
                else:
                    fn += 1
            # 짝이 없는 예측 중, 채점에서 뺀 작은 정답과 겹치는 것은 가짜로 세지 않는다
            fp += sum(1 for j, (pg, _, pbox) in enumerate(preds) if j not in matched and
                      not any(sg == pg and iou(pbox, sb) >= 0.5 for sg, _, sb in ignore[k]))
        k += 1
    cap.release()

    n_gt = tp + fn
    followed = [g for g in used_tracks]
    clean = sum(len(used_tracks[g]) == 1 for g in followed)
    print(f"추적기 {tracker}, {stride}프레임마다 처리 (실효 {20 / stride:.0f} FPS)")
    print(f"정답 {n_gt}개, 맞춤 {tp}, 놓침 {fn}, 가짜 {fp}, ID 전환 {idsw}")
    print(f"MOTA          : {1 - (fn + fp + idsw) / n_gt:.3f}")
    print(f"ID 한 번도 안 바뀐 객체: {clean}/{len(followed)} ({clean / len(followed):.1%})")
