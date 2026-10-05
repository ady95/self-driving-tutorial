"""실습용 샘플 주행 영상을 data/videos/ 에 내려받습니다.

영상 출처: Udacity CarND-LaneLines-P1 (MIT License)
https://github.com/udacity/CarND-LaneLines-P1
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/"
VIDEOS = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]

out_dir = Path(__file__).resolve().parent.parent / "data" / "videos"
out_dir.mkdir(parents=True, exist_ok=True)

for name in VIDEOS:
    dst = out_dir / name
    if dst.exists():
        print(f"[skip] {name} (이미 있음)")
        continue
    print(f"[down] {name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(BASE + name, dst)
    print(f"{dst.stat().st_size / 1e6:.1f} MB")

print(f"저장 위치: {out_dir}")
