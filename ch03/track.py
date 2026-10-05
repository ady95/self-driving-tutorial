"""03-4: YOLO + ByteTrack으로 차량에 ID를 붙여 추적합니다.

python ch03/track.py data/videos/solidYellowLeft.mp4          # 모든 프레임
python ch03/track.py data/videos/solidYellowLeft.mp4 3        # 3프레임마다 하나만 (낮은 FPS 흉내)
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLES = [2, 5, 7]                               # car, bus, truck

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidYellowLeft.mp4")
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tracker = sys.argv[3] if len(sys.argv) > 3 else "bytetrack.yaml"
out = Path("outputs/ch03")
out.mkdir(parents=True, exist_ok=True)
name = path.stem if path.stem != "rgb" else f"carla_{path.parent.name}"

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
tracks = defaultdict(list)                         # ID → [(프레임 번호, 중심 x, 중심 y), ...]
writer = None
idx = used = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % stride == 0:
        # persist=True: 이전 프레임의 추적 상태를 이어서 쓴다
        r = model.track(frame, persist=True, tracker=tracker, classes=VEHICLES,
                        conf=0.25, verbose=False)[0]
        used += 1
        if r.boxes.id is not None:
            for tid, (x1, y1, x2, y2) in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                tracks[tid].append((idx, (x1 + x2) / 2, (y1 + y2) / 2))
        vis = r.plot()
        for tid, pts in tracks.items():            # 최근 궤적을 꼬리처럼 그린다
            recent = [(int(x), int(y)) for f, x, y in pts if idx - f < fps * 2]
            if len(recent) > 1:
                cv2.polylines(vis, [np.array(recent)], False, (0, 255, 255), 2)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out / f"{name}_track_s{stride}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
        writer.write(vis)
    idx += 1
cap.release()
writer.release()

lengths = sorted((len(p) for p in tracks.values()), reverse=True)
long_tracks = [n for n in lengths if n >= used * 0.1]   # 처리한 프레임의 10% 이상 이어진 궤적
print(f"영상          : {path.name}, {used}/{idx} 프레임 사용 (실효 {fps / stride:.1f} FPS)")
print(f"추적기        : {tracker}")
print(f"부여된 ID 수  : {len(tracks)}")
print(f"긴 궤적 수    : {len(long_tracks)} (처리 프레임의 10% 이상)")
print(f"궤적 길이     : 최장 {lengths[0]}, 중앙값 {int(np.median(lengths))} 프레임")
print(f"결과 영상     : {out / f'{name}_track_s{stride}.mp4'}")
