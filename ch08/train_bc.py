"""08-2: Behavioral Cloning — 카메라 영상 한 장으로 조향각을 예측하는 모델을 학습합니다.

python ch08/train_bc.py              # 가운데 카메라만
python ch08/train_bc.py --side       # + 좌우 카메라에 조향 보정값을 붙여 데이터 3배
python ch08/train_bc.py --resume     # 중간에 멈췄다면 마지막으로 끝난 에폭 다음부터 이어서

좌우 카메라 보정값: 0.8m 옆으로 벗어난 차가 8m 앞에서 차로 중앙으로 돌아오려면 Pure Pursuit(06-3)로
조향각 atan(2 x 2.86 x sin(atan(0.8/8)) / 8) = 4.1도가 필요하고, 최대 조향각 70도로 나누면 약 0.06이다.
왼쪽 카메라 영상은 '차가 왼쪽으로 치우친 장면'이므로 오른쪽(+)으로 0.06 더 꺾는 것을 정답으로 삼는다.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path("data/e2e")
OUT = Path("outputs/ch08")
TOWNS = ["Town03", "Town10HD_Opt"]
CORRECTION = 0.06
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(bgr):
    x = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class SteerSet(Dataset):
    def __init__(self, items):
        self.items = items                            # [(이미지 경로, 조향 정답)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, steer = self.items[i]
        return to_tensor(cv2.imread(str(path))), torch.tensor([steer], dtype=torch.float32)


def load_items(side, val_ratio=0.1):
    train, val = [], []
    for town in TOWNS:
        rows = list(csv.DictReader(open(ROOT / town / "labels.csv")))
        n_val = int(len(rows) * val_ratio)
        for j, r in enumerate(rows):
            f, s = f"{int(r['frame']):05d}.jpg", float(r["steer"])
            if j >= len(rows) - n_val:                # 각 도시 주행의 마지막 10%는 검증용
                val.append((ROOT / town / "center" / f, s))
                continue
            train.append((ROOT / town / "center" / f, s))
            if side:
                train.append((ROOT / town / "left" / f, s + CORRECTION))
                train.append((ROOT / town / "right" / f, s - CORRECTION))
    return train, val


def save_checkpoint(path, model, opt, sched, epoch):
    """에폭이 끝날 때마다 이어서 학습하는 데 필요한 것을 모두 저장한다."""
    torch.save({"epoch": epoch, "total": sched.T_max, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "rng": torch.get_rng_state()}, path)


def resume(path, model, opt, sched):
    """체크포인트가 있으면 불러오고 다음 에폭 번호를 돌려준다. 없으면 1."""
    if not path.exists():
        print(f"체크포인트 {path} 없음 → 처음부터 학습")
        return 1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["total"] != sched.T_max:                       # 학습률 일정이 달라지므로 같은 --epochs로만 이어 간다
        raise SystemExit(f"체크포인트는 --epochs {ck['total']} 설정입니다. 같은 값으로 이어서 학습하세요")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    torch.set_rng_state(ck["rng"])
    print(f"체크포인트 {path}: {ck['epoch']}에폭까지 학습됨 → {ck['epoch'] + 1}에폭부터 이어서")
    return ck["epoch"] + 1


def build_model():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)    # ImageNet 사전학습
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
    return m


def evaluate(model, loader, device):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, y in loader:
            pred.append(model(x.to(device)).cpu())
            true.append(y)
    p, t = torch.cat(pred).numpy().ravel(), torch.cat(true).numpy().ravel()
    turn = np.abs(t) > 0.1
    return np.mean(np.abs(p - t)), np.mean(np.abs(p[turn] - t[turn])), np.mean(np.abs(p[~turn] - t[~turn]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="에폭마다 저장한 체크포인트에서 이어서 학습")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    train, val = load_items(args.side)
    tl = DataLoader(SteerSet(train), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(SteerSet(val), batch_size=128, num_workers=4)
    print(f"학습 {len(train)}장, 검증 {len(val)}장 (좌우 카메라 {'사용' if args.side else '안 씀'}), 장치 {device}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    name = "bc_side" if args.side else "bc_center"
    first = resume(OUT / f"{name}_ckpt.pt", model, opt, sched) if args.resume else 1
    start = time.perf_counter()
    for epoch in range(first, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tl:
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        mae, mae_turn, mae_straight = evaluate(model, vl, device)
        print(f"epoch {epoch}  train MSE {total / len(train):.5f}  검증 MAE {mae:.4f} "
              f"(회전 {mae_turn:.4f}, 직진 {mae_straight:.4f})  {time.perf_counter() - start:.0f}s", flush=True)
        save_checkpoint(OUT / f"{name}_ckpt.pt", model, opt, sched, epoch)
    torch.save(model.state_dict(), OUT / f"{name}.pt")
    print(f"저장: {OUT / name}.pt")
