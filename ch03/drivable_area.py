"""03-3 실습 1·2: 사전학습 의미 분할 모델(YOLO26, Cityscapes 19클래스)로 주행 가능 영역(road)을 찾습니다.

python ch03/drivable_area.py data/videos/challenge.mp4     # 실제 영상에 칠해 보기
python ch03/drivable_area.py carla                         # CARLA 정답으로 날씨별 채점
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROAD = 0                                   # Cityscapes 'road' 클래스 번호
CARLA_ROAD = [1, 24]                       # CARLA 정답에서 도로(Roads)와 차선(RoadLine)
OUT = Path("outputs/ch03")


def road_mask(model, frame):
    r = model(frame, verbose=False)[0]
    return r.semantic_mask.data.cpu().numpy() == ROAD, r


def paint(frame, mask, color=(0, 200, 0)):
    over = frame.copy()
    over[mask] = (0.5 * over[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return over


def run_video(model, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer, ratios, ms = None, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, r = road_mask(model, frame)
        ratios.append(mask.mean())
        ms.append(r.speed["inference"])
        vis = paint(frame, mask)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(OUT / f"{path.stem}_road.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(vis)
    cap.release()
    writer.release()
    print(f"{path.name}: 도로 비율 평균 {np.mean(ratios):.1%} (최소 {np.min(ratios):.1%}, "
          f"최대 {np.max(ratios):.1%}), 추론 {np.median(ms):.1f}ms/프레임")


def iou_stats(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter, union, pred.sum(), gt.sum()


def run_carla(model, root=Path("data/carla_urban")):
    print(f"{'날씨':<7}{'IoU':>7}{'정밀도':>8}{'재현율':>8}")
    for weather in ["clear", "night", "rain"]:
        cap = cv2.VideoCapture(str(root / weather / "rgb.mp4"))
        tot = np.zeros(4)
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gt_path = root / weather / "semantic" / f"{k:04d}.png"
            if gt_path.exists():                       # 정답은 5프레임마다 있다
                gt = np.isin(cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED), CARLA_ROAD)
                pred, _ = road_mask(model, frame)
                tot += iou_stats(pred, gt)
                if k == 200:
                    cv2.imwrite(str(OUT / f"road_{weather}.jpg"), paint(frame, pred))
            k += 1
        cap.release()
        inter, union, p, g = tot
        print(f"{weather:<7}{inter / union:7.3f}{inter / p:8.1%}{inter / g:8.1%}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo26n-sem.pt")
    arg = sys.argv[1] if len(sys.argv) > 1 else "data/videos/challenge.mp4"
    if arg == "carla":
        run_carla(model)
    else:
        run_video(model, Path(arg))
