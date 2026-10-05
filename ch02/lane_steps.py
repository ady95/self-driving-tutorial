"""02-2: 차선 인식의 중간 단계를 한 장의 그림으로 확인합니다."""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lane_detect as ld

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidWhiteRight.mp4")
out = Path("outputs/ch02")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(path))
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

edges = ld.edge_image(frame)
vertices = ld.roi_vertices(w, h)
masked = ld.region_of_interest(edges, vertices)
segments = ld.detect_segments(masked)
left_seg, right_seg = ld.split_left_right(segments, w)
result, info = ld.process_frame(frame)

# Hough 선분을 왼쪽(파랑)/오른쪽(초록)/버림(회색)으로 색을 나눠 그린다
seg_img = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
for s in segments:
    color = (255, 0, 0) if s in left_seg else (0, 255, 0) if s in right_seg else (128, 128, 128)
    cv2.line(seg_img, s[:2], s[2:], color, 3)

print(f"Canny 엣지 픽셀    : {cv2.countNonZero(edges)}")
print(f"ROI 안 엣지 픽셀   : {cv2.countNonZero(masked)}")
print(f"Hough 선분         : {len(segments)}개 (왼쪽 {len(left_seg)}, 오른쪽 {len(right_seg)}, "
      f"버림 {len(segments) - len(left_seg) - len(right_seg)})")
print(f"왼쪽 차선 (아래→위): {info['left']}")
print(f"오른쪽 차선        : {info['right']}")
print(f"차선 중앙 offset   : {info['offset']:+.1f} px")

panels = [
    ("1. canny edges", edges, "gray"),
    ("2. ROI mask applied", masked, "gray"),
    ("3. hough segments (blue=L, green=R)", cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB), None),
    ("4. fitted lanes + center", cv2.cvtColor(result, cv2.COLOR_BGR2RGB), None),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (title, img, cmap) in zip(axes.ravel(), panels):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out / "lane_steps.png", dpi=80)
print(f"그림 저장          : {out / 'lane_steps.png'}")
