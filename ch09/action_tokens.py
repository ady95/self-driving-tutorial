"""09-3: 행동을 토큰으로 — 08장의 조향값과 미래 궤적을 몇 칸으로 나눠 토큰으로 바꾸고, 되돌릴 때의 오차를 잽니다.

RT-2·OpenVLA 같은 VLA는 연속적인 행동 값을 구간(bin)으로 나눠 토큰 번호로 바꾼 뒤,
언어 모델이 단어를 생성하듯 그 번호를 생성합니다.

python ch09/action_tokens.py                         # 08-2에서 직접 모은 data/e2e/Town03
python ch09/action_tokens.py --data data/e2e_ref     # 책의 표를 낸 기준 라벨 (download_samples.py --e2e-ref)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch08"))
from train_traj import future_waypoints  # noqa: E402


def to_token(v, lo, hi, bins):
    """값 → 0 ~ bins-1 사이의 토큰 번호."""
    return np.clip(np.round((v - lo) / (hi - lo) * (bins - 1)), 0, bins - 1).astype(int)


def from_token(t, lo, hi, bins):
    """토큰 번호 → 그 칸을 대표하는 값."""
    return lo + t / (bins - 1) * (hi - lo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/e2e")
    path = Path(ap.parse_args().data) / "Town03" / "labels.csv"
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 08-2의 collect.py로 모으거나, "
                         f"python scripts/download_samples.py --e2e-ref 후 --data data/e2e_ref로 실행하세요")
    rows = list(csv.DictReader(open(path)))
    steer = np.array([float(r["steer"]) for r in rows])
    print(f"조향값 {len(steer)}개 (범위 -1 ~ 1, 1 = 70도)")
    for bins in [3, 15, 16, 255, 256]:
        tok = to_token(steer, -1, 1, bins)
        err = np.abs(from_token(tok, -1, 1, bins) - steer)
        print(f"  {bins:3d}칸: 평균 오차 {err.mean():.4f}, 최대 {err.max():.4f} "
              f"({math.degrees(err.max() * math.radians(70)):.2f}도), 실제로 쓰인 토큰 {len(set(tok))}개")

    wps = np.array([w for w in (future_waypoints(rows, j) for j in range(0, len(rows) - 40, 10)) if w is not None])
    print(f"\n미래 궤적 {len(wps)}개 (지점 4개 x 앞·옆), 앞 0~{wps[..., 0].max():.1f}m, "
          f"옆 {wps[..., 1].min():.1f}~{wps[..., 1].max():.1f}m")
    for bins in [15, 255]:
        fwd = from_token(to_token(wps[..., 0], 0, 50, bins), 0, 50, bins)
        lat = from_token(to_token(wps[..., 1], -20, 20, bins), -20, 20, bins)
        err = np.hypot(fwd - wps[..., 0], lat - wps[..., 1])
        print(f"  {bins:3d}칸: 지점 오차 평균 {err.mean():.3f}m, 최대 {err.max():.3f}m, 궤적 하나 = 토큰 {wps.shape[1] * 2}개")

    turn = wps[np.argmax(np.abs(wps[:, -1, 1]) > 3)]
    print("\n회전 궤적 하나를 토큰으로 (255칸):")
    print("  값  :", ", ".join(f"({x:.1f}, {y:.1f})" for x, y in turn))
    print("  토큰:", " ".join(f"<a{a}><a{b}>" for a, b in zip(to_token(turn[:, 0], 0, 50, 255),
                                                         to_token(turn[:, 1], -20, 20, 255))))
