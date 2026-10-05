"""03-3 실습 3: 도로와 차선을 픽셀 단위로 나누는 작은 모델을 CARLA 데이터로 직접 학습시킵니다.

맑은 날 영상의 앞부분(0~299프레임)으로 학습하고, 뒷부분(300~399프레임)으로
맑은 날·야간·비에서 채점합니다. --aug를 주면 밝기·대비를 흔드는 데이터 증강을 켭니다.

python ch03/train_lane.py           # 증강 없이
python ch03/train_lane.py --aug     # 밝기·대비 증강
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

ROOT = Path("data/carla_urban")
OUT = Path("outputs/ch03")
SIZE = (640, 360)                                   # 학습 해상도 (가로, 세로)
CLASSES = ["background", "road", "lane"]
SPLIT = 300                                         # 이 프레임 전은 학습, 후는 시험
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(weather, train):
    """rgb.mp4에서 정답이 있는 프레임(5프레임마다)을 골라 (영상, 라벨) 목록을 만든다."""
    cap = cv2.VideoCapture(str(ROOT / weather / "rgb.mp4"))
    items, k = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = ROOT / weather / "semantic" / f"{k:04d}.png"
        if p.exists() and ((k < SPLIT) == train):
            sem = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            label = np.zeros_like(sem)
            label[sem == 1] = 1                     # CARLA Roads
            label[sem == 24] = 2                    # CARLA RoadLine (차선)
            items.append((cv2.resize(frame, SIZE),
                          cv2.resize(label, SIZE, interpolation=cv2.INTER_NEAREST)))
        k += 1
    cap.release()
    return items


def to_tensor(img):
    x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def augment(img, rng):
    """밝기를 0.2~1.2배, 대비를 0.6~1.2배로 무작위로 바꾼다 (야간·흐린 날 흉내)."""
    gain = rng.uniform(0.2, 1.2)
    contrast = rng.uniform(0.6, 1.2)
    m = img.mean()
    out = (img.astype(np.float32) - m) * contrast + m
    return np.clip(out * gain, 0, 255).astype(np.uint8)


def build_model():
    model = lraspp_mobilenet_v3_large(weights="DEFAULT")          # 사전학습 가중치에서 시작
    model.classifier.low_classifier = nn.Conv2d(40, len(CLASSES), 1)
    model.classifier.high_classifier = nn.Conv2d(128, len(CLASSES), 1)
    return model


def predict(model, img, device):
    with torch.no_grad():
        out = model(to_tensor(img)[None].to(device))["out"]
    return out.argmax(1)[0].cpu().numpy()


def score(model, items, device):
    """클래스별 IoU를 모든 시험 프레임에 걸쳐 계산한다."""
    inter = np.zeros(len(CLASSES))
    union = np.zeros(len(CLASSES))
    for img, label in items:
        pred = predict(model, img, device)
        for c in range(len(CLASSES)):
            inter[c] += np.logical_and(pred == c, label == c).sum()
            union[c] += np.logical_or(pred == c, label == c).sum()
    return inter / union


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    train = load("clear", train=True)
    tests = {w: load(w, train=False) for w in ["clear", "night", "rain"]}
    print(f"학습 {len(train)}장 (clear), 시험 각 {len(tests['clear'])}장, 장치 {device}, 증강 {args.aug}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0], device=device))  # 가는 차선에 가중치

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), 8):                       # 배치 크기 8
            batch = [train[j] for j in order[i:i + 8]]
            imgs = [augment(b[0], rng) if args.aug else b[0] for b in batch]
            x = torch.stack([to_tensor(im) for im in imgs]).to(device)
            y = torch.stack([torch.from_numpy(b[1].astype(np.int64)) for b in batch]).to(device)
            loss = loss_fn(model(x)["out"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss {total / len(train):.4f}  ({time.perf_counter() - start:.0f}s)")

    model.eval()
    print(f"\n{'시험':<7}{'road IoU':>10}{'lane IoU':>10}")
    for w, items in tests.items():
        iou = score(model, items, device)
        print(f"{w:<7}{iou[1]:10.3f}{iou[2]:10.3f}")

    tag = "aug" if args.aug else "base"
    torch.save(model.state_dict(), OUT / f"lane_lraspp_{tag}.pt")

    # 시뮬레이터에서만 배운 모델을 실제 고속도로 영상에 적용해 본다
    cap = cv2.VideoCapture("data/videos/challenge.mp4")
    ok, frame = cap.read()
    cap.release()
    img = cv2.resize(frame, SIZE)
    pred = predict(model, img, device)
    vis = img.copy()
    vis[pred == 1] = (0.5 * vis[pred == 1] + [0, 100, 0]).astype(np.uint8)
    vis[pred == 2] = (0, 0, 255)
    cv2.imwrite(str(OUT / f"lane_real_{tag}.jpg"), vis)
    for w in ["night", "rain"]:
        img = tests[w][10][0]
        pred = predict(model, img, device)
        vis = img.copy()
        vis[pred == 1] = (0.5 * vis[pred == 1] + [0, 100, 0]).astype(np.uint8)
        vis[pred == 2] = (0, 0, 255)
        cv2.imwrite(str(OUT / f"lane_{w}_{tag}.jpg"), vis)
    print(f"결과 그림: {OUT}/lane_*_{tag}.jpg")
