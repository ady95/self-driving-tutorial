"""03-2 실습 1: 사전학습 YOLO로 주행 영상 한 프레임에서 객체를 찾습니다."""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/challenge.mp4")
out = Path("outputs/ch03")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(path))
ok, frame = cap.read()
cap.release()

model = YOLO("yolo26n.pt")           # 처음 실행하면 가중치(약 5MB)를 자동으로 내려받는다
print(f"모델 클래스 수: {len(model.names)} (COCO)")

result = model(frame, verbose=False)[0]   # BGR numpy 배열을 그대로 넣어도 된다
print(f"처리 시간: 전처리 {result.speed['preprocess']:.1f}ms, "
      f"추론 {result.speed['inference']:.1f}ms, 후처리 {result.speed['postprocess']:.1f}ms")
print(f"찾은 객체: {len(result.boxes)}개\n")

print(f"{'클래스':<14}{'신뢰도':>6}   x1    y1    x2    y2")
for cls, conf, (x1, y1, x2, y2) in zip(result.boxes.cls.tolist(),
                                        result.boxes.conf.tolist(),
                                        result.boxes.xyxy.tolist()):
    print(f"{model.names[int(cls)]:<14}{conf:6.2f} {x1:5.0f} {y1:5.0f} {x2:5.0f} {y2:5.0f}")

cv2.imwrite(str(out / "yolo_image.jpg"), result.plot())
print(f"\n결과 그림: {out / 'yolo_image.jpg'}")
