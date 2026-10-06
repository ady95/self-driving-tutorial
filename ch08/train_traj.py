"""08-4: Sensor-to-Trajectory — 카메라 영상과 속도로 앞으로 2초 동안의 궤적을 예측하는 작은 Transformer 모델.

  영상 → ResNet18 특징 지도(6x10) → 토큰 60개 + 속도 토큰 → Transformer 인코더
       → 궤적 지점 4개(0.5·1.0·1.5·2.0초 뒤)를 질의로 하는 Transformer 디코더 → (앞, 오른쪽) 좌표
예측한 궤적은 Pure Pursuit(06-3)로 따라가 조향으로 바꿉니다(steer_from_waypoints).

python ch08/train_traj.py
"""
import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from train_bc import ROOT, TOWNS, to_tensor

OUT = Path("outputs/ch08")
HORIZON = [10, 20, 30, 40]                       # 0.5·1.0·1.5·2.0초 뒤 (20 FPS)
WHEELBASE, MAX_STEER = 2.86, math.radians(70)


def future_waypoints(rows, j):
    """j번째 프레임 기준, 미래 위치들을 차 좌표 (앞, 오른쪽)으로. 순간 이동(수집 중 재배치)이 끼면 None."""
    x0, y0, yaw = float(rows[j]["x"]), float(rows[j]["y"]), math.radians(float(rows[j]["yaw"]))
    pts, prev = [], (x0, y0)
    for k in range(j + 1, j + HORIZON[-1] + 1):
        x, y = float(rows[k]["x"]), float(rows[k]["y"])
        if math.hypot(x - prev[0], y - prev[1]) > 2.0:          # 0.05초에 2m 넘게 이동 = 재배치
            return None
        prev = (x, y)
        if k - j in HORIZON:
            dx, dy = x - x0, y - y0
            pts.append((dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)))
    return np.array(pts, np.float32)


class TrajSet(Dataset):
    def __init__(self, items):
        self.items = items                                      # [(이미지 경로, 속도, 미래 궤적 4x2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, speed, wps = self.items[i]
        return to_tensor(cv2.imread(str(path))), torch.tensor([speed]), torch.from_numpy(wps)


def load_items(val_ratio=0.1):
    train, val = [], []
    for town in TOWNS:
        rows = list(csv.DictReader(open(ROOT / town / "labels.csv")))
        n_val = int(len(rows) * val_ratio)
        for j in range(len(rows) - HORIZON[-1]):
            wps = future_waypoints(rows, j)
            if wps is None:
                continue
            item = (ROOT / town / "center" / f"{int(rows[j]['frame']):05d}.jpg", float(rows[j]["speed"]), wps)
            (val if j >= len(rows) - n_val else train).append(item)
    return train, val


class TrajModel(nn.Module):
    def __init__(self, d=128, n_wp=len(HORIZON)):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(r.children())[:-2])          # (B, 512, 6, 10)
        self.proj = nn.Conv2d(512, d, 1)
        self.pos = nn.Parameter(torch.zeros(1, 60, d))                   # 토큰 위치 정보 (학습)
        self.speed = nn.Linear(1, d)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, batch_first=True), 2)
        self.queries = nn.Parameter(torch.randn(1, n_wp, d) * 0.02)      # 궤적 지점마다 질의 하나
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 256, batch_first=True), 2)
        self.head = nn.Linear(d, 2)

    def forward(self, img, speed):
        f = self.proj(self.backbone(img)).flatten(2).transpose(1, 2)     # (B, 60, d)
        tokens = torch.cat([f + self.pos, self.speed(speed)[:, None]], 1)
        memory = self.encoder(tokens)
        out = self.decoder(self.queries.expand(len(img), -1, -1), memory)
        return self.head(out)                                            # (B, 4, 2) 미터


def build_traj_model():
    return TrajModel()


def steer_from_waypoints(wps, speed):
    """예측 궤적 위에서 전방 주시 거리만큼 앞의 점을 Pure Pursuit로 겨냥한다."""
    pts = np.vstack([[0.0, 0.0], wps])
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    ld = float(np.clip(0.6 * speed + 3.0, 4.0, max(arc[-1], 4.0)))
    i = min(int(np.searchsorted(arc, ld)), len(pts) - 1)
    fx, ry = pts[i]
    alpha = math.atan2(ry, fx)
    return math.atan2(2 * WHEELBASE * math.sin(alpha), max(math.hypot(fx, ry), 1.0)) / MAX_STEER


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_items()
    tl = DataLoader(TrajSet(train), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(TrajSet(val), batch_size=128, num_workers=4)
    print(f"학습 {len(train)}장, 검증 {len(val)}장, 장치 {device}")
    model = build_traj_model().to(device)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for img, sp, wps in tl:
            loss = nn.functional.l1_loss(model(img.to(device), sp.to(device)), wps.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(img)
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for img, sp, wps in vl:
                errs.append(torch.linalg.norm(model(img.to(device), sp.to(device)).cpu() - wps, dim=-1))
        e = torch.cat(errs)                                              # (N, 4) 지점별 거리 오차
        print(f"epoch {epoch}  train L1 {total / len(train):.3f}  검증 오차 0.5/1.0/1.5/2.0초: "
              + " / ".join(f"{v:.2f}m" for v in e.mean(0).tolist())
              + f"  {time.perf_counter() - start:.0f}s", flush=True)
    torch.save(model.state_dict(), OUT / "traj.pt")
    print(f"저장: {OUT / 'traj.pt'}")
