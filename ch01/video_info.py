"""01-4 실습 1: 주행 영상의 기본 정보를 확인합니다."""
import sys
import time
from pathlib import Path

import cv2

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidWhiteRight.mp4")
cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    sys.exit(f"영상을 열 수 없습니다: {path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"파일        : {path.name}")
print(f"해상도      : {width} x {height}")
print(f"FPS         : {fps:.2f}")
print(f"프레임 수   : {count} (메타데이터)")
print(f"재생 시간   : {count / fps:.1f} 초")

# 첫 프레임으로 배열 구조를 확인한다
ok, frame = cap.read()
print(f"프레임 배열 : shape={frame.shape}, dtype={frame.dtype}")
print(f"좌상단 픽셀 : {frame[0, 0]} (B, G, R 순서)")

# 끝까지 읽어 보며 실제 프레임 수와 읽기 속도를 잰다
start = time.perf_counter()
n = 1
while cap.read()[0]:
    n += 1
elapsed = time.perf_counter() - start
cap.release()

print(f"실제 읽은 수: {n} 프레임")
print(f"읽기 속도   : {n / elapsed:.0f} FPS (실시간의 {n / elapsed / fps:.1f}배)")
