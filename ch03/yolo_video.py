"""03-2 실습 2: 주행 영상 전체에서 차량·보행자·신호등을 찾고, 클래스별 개수와 속도를 잽니다.

python ch03/yolo_video.py data/videos/challenge.mp4            # GPU가 있으면 GPU
python ch03/yolo_video.py data/videos/challenge.mp4 cpu        # CPU로 강제
"""
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

# 자율주행에 필요한 COCO 클래스만 고른다
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
           5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"}

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/challenge.mp4")
device = sys.argv[2] if len(sys.argv) > 2 else None
out = Path("outputs/ch03")
out.mkdir(parents=True, exist_ok=True)
# CARLA 데이터(data/carla_urban/clear/rgb.mp4)는 파일명이 같으므로 폴더 이름(clear)을 붙인다
name = path.stem if path.stem != "rgb" else f"carla_{path.parent.name}"

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = None
counts = Counter()
frames = 0
infer_ms = []
start = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, classes=list(CLASSES), conf=0.25, device=device, verbose=False)[0]
    infer_ms.append(r.speed["inference"])
    counts.update(CLASSES[int(c)] for c in r.boxes.cls.tolist())
    frames += 1

    vis = r.plot()
    if writer is None:
        h, w = vis.shape[:2]
        writer = cv2.VideoWriter(str(out / f"{name}_yolo.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(vis)

elapsed = time.perf_counter() - start
cap.release()
writer.release()

print(f"영상        : {path.name} ({frames} 프레임)")
print(f"장치        : {model.predictor.device}")   # model.device는 추론 후에도 cpu로 남는다
print(f"추론 시간   : 평균 {sum(infer_ms) / len(infer_ms):.1f}ms/프레임 (첫 프레임 {infer_ms[0]:.0f}ms)")
print(f"전체 처리   : {frames / elapsed:.1f} FPS (읽기·그리기·저장 포함)")
print("프레임당 평균 객체 수:")
for cls_name, n in counts.most_common():
    print(f"  {cls_name:<14}{n / frames:5.2f}")
print(f"결과 영상   : {out / (name + '_yolo.mp4')}")
