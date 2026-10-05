"""02-1: 차선 검출 전처리 단계(흑백 → 블러 → 엣지)를 한 프레임에 차례로 적용해 봅니다."""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidWhiteRight.mp4")
out = Path("outputs/ch02")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(path))
ok, frame = cap.read()                     # 첫 프레임 (BGR)
cap.release()

# 1. 흑백 변환: 3채널 → 1채널
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 2. 가우시안 블러: 작은 노이즈를 뭉개 가짜 엣지를 줄인다
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Sobel: 가로 방향 밝기 변화량(미분)
sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# 4. Canny: 변화량이 큰 곳만 남겨 가는 선으로 만든다
edges = cv2.Canny(blur, 50, 150)

print(f"원본 프레임   : {frame.shape}, {frame.dtype}")
print(f"흑백 프레임   : {gray.shape}, 값 범위 {gray.min()}~{gray.max()}")

# 블러 유무와 Canny 임계값에 따라 엣지 픽셀이 얼마나 달라지는지 센다
total = gray.size
print("\n[Canny 엣지 픽셀 수]")
for name, src in [("블러 없음", gray), ("블러 5x5", blur)]:
    for low, high in [(10, 50), (50, 150), (100, 200)]:
        n = int(np.count_nonzero(cv2.Canny(src, low, high)))
        print(f"  {name:<8} 임계값 ({low:>3}, {high:>3}) : {n:6d} 픽셀 ({n / total:5.1%})")

# 단계별 결과를 한 장의 그림으로 저장한다
panels = [
    ("1. original (BGR->RGB)", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None),
    ("mistake: BGR shown as RGB", frame, None),
    ("2. grayscale", gray, "gray"),
    ("3. gaussian blur 5x5", blur, "gray"),
    ("4. sobel x", sobel_x, "gray"),
    ("5. canny (50, 150)", edges, "gray"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for ax, (title, img, cmap) in zip(axes.ravel(), panels):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out / "basics.png", dpi=80)
print(f"\n그림 저장: {out / 'basics.png'}")
