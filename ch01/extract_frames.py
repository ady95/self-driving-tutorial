"""01-4 실습 2: 영상을 일정 간격으로 잘라 프레임 이미지로 저장하고, 한 장의 요약 이미지로 모읍니다."""
import sys
from pathlib import Path

import cv2
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidWhiteRight.mp4")
every_sec = 1.0                      # 몇 초마다 한 장씩 저장할지
out_dir = Path("outputs/ch01") / path.stem
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
step = int(round(fps * every_sec))

saved = []
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % step == 0:
        t = idx / fps
        name = out_dir / f"frame_{idx:05d}.png"
        cv2.imwrite(str(name), frame)
        # 요약 이미지용 썸네일에 시각을 적어 둔다
        thumb = cv2.resize(frame, (320, 180))
        cv2.putText(thumb, f"t={t:4.1f}s  #{idx}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        saved.append(thumb)
    idx += 1
cap.release()
print(f"{idx} 프레임 중 {len(saved)}장 저장 → {out_dir}")

# 썸네일을 4열 격자로 이어 붙인다 (빈 칸은 검은색)
cols = 4
rows = (len(saved) + cols - 1) // cols
blank = np.zeros_like(saved[0])
saved += [blank] * (rows * cols - len(saved))
grid = np.vstack([np.hstack(saved[r * cols:(r + 1) * cols]) for r in range(rows)])
sheet = Path("outputs/ch01") / f"{path.stem}_contact.jpg"
cv2.imwrite(str(sheet), grid)
print(f"요약 이미지 : {sheet} ({grid.shape[1]} x {grid.shape[0]})")
