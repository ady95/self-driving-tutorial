"""실습용 샘플 데이터를 data/ 에 내려받습니다.

python scripts/download_samples.py           # 고속도로 주행 영상 3개 (01~02장)
python scripts/download_samples.py --carla   # + CARLA 데이터 약 160MB (도심 장면 03장~, 장거리 주행 05장~)
python scripts/download_samples.py --e2e-ref # + 08장 주행 기준 라벨 (09-3·09-4 표 재현용, 영상 없음)

영상 출처: Udacity CarND-LaneLines-P1 (MIT License)
https://github.com/udacity/CarND-LaneLines-P1
CARLA 데이터: 이 저장소의 scripts/record_carla_urban.py로 생성 (CARLA 에셋 CC-BY 4.0)
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/"
VIDEOS = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]
CARLA_URL = "https://github.com/ady95/self-driving-tutorial/releases/download/data-v1/carla_urban.zip"

data = Path(__file__).resolve().parent.parent / "data"
out_dir = data / "videos"
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

if "--e2e-ref" in sys.argv:                       # 08-2에서 저자가 모은 라벨 (data/e2e_ref/<도시>/labels.csv)
    if (data / "e2e_ref" / "Town03" / "labels.csv").exists():
        print("[skip] e2e_ref (이미 있음)")
    else:
        zip_path = data / "e2e_ref.zip"
        print("[down] e2e_ref.zip ...", end=" ", flush=True)
        urllib.request.urlretrieve(CARLA_URL.replace("carla_urban", "e2e_ref"), zip_path)
        print(f"{zip_path.stat().st_size / 1e6:.1f} MB")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data)
        zip_path.unlink()
        print(f"저장 위치: {data / 'e2e_ref'}")

if "--carla" in sys.argv:
    for name, check, size in [("carla_urban", "calib.json", "약 100MB"),     # 03~05장
                              ("carla_drive", "ego.csv", "약 60MB")]:        # 05장, 08장
        if (data / name / check).exists():
            print(f"[skip] {name} (이미 있음)")
            continue
        zip_path = data / f"{name}.zip"
        print(f"[down] {name}.zip ({size}) ...", end=" ", flush=True)
        urllib.request.urlretrieve(CARLA_URL.replace("carla_urban", name), zip_path)
        print(f"{zip_path.stat().st_size / 1e6:.1f} MB")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data)
        zip_path.unlink()
    print(f"저장 위치: {data}")
