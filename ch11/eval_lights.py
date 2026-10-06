"""11-2: 카메라 신호 판독기를 CARLA 정답으로 채점합니다 (collect_lights.py로 모은 영상).

같은 영역(지도로 투영한 내 차로 신호등)을 두 방법으로 읽어 비교합니다.
  - 위치: 켜진 등이 위·가운데·아래 중 어디인가 (lights.py)
  - 색  : 켜진 등의 색(Hue)이 빨강·노랑·초록 중 무엇인가
python ch11/eval_lights.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lights import read_light  # noqa: E402

DATA = Path("data/lights")


def read_light_by_color(img, rois):
    votes = Counter()
    for x1, y1, x2, y2 in rois:
        if (y2 - y1) < 9 or (x2 - x1) > (y2 - y1):
            continue
        hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        lit = hsv[(hsv[..., 2] > 180) & (hsv[..., 1] > 60)]           # 밝고 채도 있는 픽셀 = 켜진 등
        if len(lit) < 3:
            continue
        hue = float(np.median(lit[:, 0]))                              # OpenCV Hue 0~180
        votes["RED" if hue < 12 or hue > 165 else "YELLOW" if hue < 35 else "GREEN"] += 1
    return votes.most_common(1)[0][0] if votes else None


if __name__ == "__main__":
    rows = list(csv.DictReader(open(DATA / "labels.csv")))
    table = {m: defaultdict(Counter) for m in ["위치", "색"]}
    for r in rows:
        img = cv2.imread(str(DATA / r["file"]))
        rois = json.loads(r["rois"])
        for method, fn in [("위치", read_light), ("색", read_light_by_color)]:
            table[method][r["weather"]][(r["state"], fn(img, rois))] += 1
    print(f"신호 영상 {len(rows)}장 (내 차로 정지선 50m 이내), 정답: "
          f"{dict(Counter(r['state'] for r in rows))}")
    for method, t in table.items():
        print(f"\n[{method}으로 판단]       맞음    모름    틀림   (빨강→초록)")
        for weather in ["clear", "night", "rain"]:
            c = t[weather]
            n = sum(c.values())
            ok = sum(v for (g, p), v in c.items() if g == p)
            unknown = sum(v for (g, p), v in c.items() if p is None)
            print(f"  {weather:6s} {n:4d}장  {ok / n:6.1%}  {unknown / n:6.1%}  {(n - ok - unknown) / n:6.1%}   "
                  f"({c[('RED', 'GREEN')]}장)")
        wrong = sum((Counter({k: v for k, v in c.items() if k[1] is not None and k[0] != k[1]}) for c in t.values()),
                    Counter())
        print(f"  틀린 경우 (정답, 판독): {dict(wrong)}")
