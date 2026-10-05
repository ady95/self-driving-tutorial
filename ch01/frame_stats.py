"""01-4 실습 3: 프레임마다 밝기를 재서, 영상 세 개의 '조건 차이'를 숫자와 그래프로 비교합니다."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")                # 화면이 없는 서버에서도 그래프를 파일로 저장
import matplotlib.pyplot as plt

videos = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]
out = Path("outputs/ch01")
out.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 4))
print(f"{'영상':<22}{'평균 밝기':>8}{'최소':>7}{'최대':>7}{'도로 영역':>10}")

for name in videos:
    cap = cv2.VideoCapture(str(Path("data/videos") / name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    whole, road = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        whole.append(gray.mean())
        road.append(gray[int(h * 0.6):, :].mean())   # 화면 아래 40% = 도로가 찍히는 곳
    cap.release()

    t = [i / fps for i in range(len(road))]
    ax.plot(t, road, label=name)
    print(f"{name:<22}{sum(whole) / len(whole):8.1f}{min(road):7.1f}{max(road):7.1f}"
          f"{sum(road) / len(road):10.1f}")

ax.set_xlabel("time (s)")
ax.set_ylabel("road-region brightness (0-255)")
ax.set_title("Brightness of the lower 40% of each frame")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "brightness.png", dpi=120)
print(f"그래프 저장 : {out / 'brightness.png'}")
