"""02-2: OpenCV 차선 인식 — Canny → ROI → Hough → 좌우 차선 → 차선 중앙.

python ch02/lane_detect.py data/videos/solidWhiteRight.mp4
→ outputs/ch02/<영상이름>_lane.mp4 와 프레임별 통계를 만든다.
02-3에서는 이 파일의 함수들을 import 해서 쓴다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# 사람이 정한 규칙(파라미터). 02-3에서 이 값들이 어디서 무너지는지 본다.
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH = dict(rho=2, theta=np.pi / 180, threshold=40, minLineLength=30, maxLineGap=100)
MIN_SLOPE = 0.5                       # 이보다 완만한 선은 차선이 아니라고 본다
Y_TOP_RATIO = 0.62                    # 차선을 그릴 위쪽 끝 (화면 높이 비율)


def roi_vertices(width, height):
    """차선이 나타날 사다리꼴 영역. 화면 크기에 비례해 잡는다."""
    return np.array([[
        (int(width * 0.10), height),
        (int(width * 0.45), int(height * Y_TOP_RATIO)),
        (int(width * 0.55), int(height * Y_TOP_RATIO)),
        (int(width * 0.95), height),
    ]], dtype=np.int32)


def edge_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)


def region_of_interest(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_segments(masked):
    lines = cv2.HoughLinesP(masked, **HOUGH)
    return [] if lines is None else [tuple(l[0]) for l in lines]


def split_left_right(segments, width):
    """기울기 부호와 화면 위치로 왼쪽/오른쪽 차선 후보를 나눈다.
    이미지 좌표는 y가 아래로 커지므로 왼쪽 차선의 기울기가 음수다."""
    left, right = [], []
    for x1, y1, x2, y2 in segments:
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_SLOPE:
            continue
        if slope < 0 and max(x1, x2) < width * 0.55:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and min(x1, x2) > width * 0.45:
            right.append((x1, y1, x2, y2))
    return left, right


def fit_lane(segments, y_bottom, y_top):
    """선분들을 길이 가중 평균한 직선 하나로 합치고, 화면 아래~위 끝점을 돌려준다."""
    if not segments:
        return None
    slopes, intercepts, weights = [], [], []
    for x1, y1, x2, y2 in segments:
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
        intercepts.append(y1 - slope * x1)
        weights.append(np.hypot(x2 - x1, y2 - y1))
    m = np.average(slopes, weights=weights)
    b = np.average(intercepts, weights=weights)
    x_bottom = int((y_bottom - b) / m)
    x_top = int((y_top - b) / m)
    return (x_bottom, y_bottom, x_top, y_top)


def process_frame(frame):
    """프레임 하나를 처리해 (결과 그림, 정보 dict)를 돌려준다."""
    h, w = frame.shape[:2]
    y_top = int(h * Y_TOP_RATIO)
    vertices = roi_vertices(w, h)

    edges = edge_image(frame)
    masked = region_of_interest(edges, vertices)
    segments = detect_segments(masked)
    left_seg, right_seg = split_left_right(segments, w)
    left = fit_lane(left_seg, h, y_top)
    right = fit_lane(right_seg, h, y_top)

    info = {"segments": len(segments), "left": left, "right": right, "offset": None}
    overlay = frame.copy()
    cv2.polylines(overlay, vertices, True, (255, 200, 0), 1)          # ROI (하늘색 테두리)
    for lane in (left, right):
        if lane:
            cv2.line(overlay, lane[:2], lane[2:], (0, 0, 255), 8)     # 차선 (빨강)
    if left and right:
        lane_center = (left[0] + right[0]) / 2                        # 화면 맨 아래에서의 차선 중앙
        info["offset"] = lane_center - w / 2                          # +면 차가 차선의 왼쪽에 치우침
        cv2.line(overlay, (int(lane_center), h), (int(lane_center), h - 40), (0, 255, 0), 4)
        cv2.line(overlay, (w // 2, h), (w // 2, h - 25), (255, 255, 255), 2)
        text = f"offset {info['offset']:+.0f}px"
    else:
        text = "lane lost"
    cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0), info


def run_video(path, out_path=None, frame_filter=None):
    """영상 전체를 처리하고 통계를 돌려준다. frame_filter는 02-3에서 조건을 바꿀 때 쓴다."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    infos = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_filter is not None:
            frame = frame_filter(frame, len(infos))
        result, info = process_frame(frame)
        infos.append(info)
        if out_path is not None:
            if writer is None:
                h, w = result.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(result)
    cap.release()
    if writer is not None:
        writer.release()
    return infos


def summarize(infos):
    """검출률과 차선 중앙의 프레임 간 흔들림을 계산한다."""
    n = len(infos)
    both = [i for i in infos if i["offset"] is not None]
    offsets = np.array([i["offset"] for i in both])
    jitter = float(np.mean(np.abs(np.diff(offsets)))) if len(offsets) > 1 else float("nan")
    return {
        "frames": n,
        "left_rate": sum(i["left"] is not None for i in infos) / n,
        "right_rate": sum(i["right"] is not None for i in infos) / n,
        "both_rate": len(both) / n,
        "offset_mean": float(offsets.mean()) if len(offsets) else float("nan"),
        "jitter": jitter,
    }


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/videos/solidWhiteRight.mp4")
    out_dir = Path("outputs/ch02")
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = run_video(path, out_dir / f"{path.stem}_lane.mp4")
    s = summarize(infos)
    print(f"영상          : {path.name} ({s['frames']} 프레임)")
    print(f"왼쪽 차선 검출: {s['left_rate']:6.1%}")
    print(f"오른쪽 차선   : {s['right_rate']:6.1%}")
    print(f"양쪽 모두     : {s['both_rate']:6.1%}")
    print(f"평균 offset   : {s['offset_mean']:+.1f} px")
    print(f"중앙 흔들림   : {s['jitter']:.1f} px/프레임")
    print(f"결과 영상     : {out_dir / (path.stem + '_lane.mp4')}")
