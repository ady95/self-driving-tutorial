"""06-2: 격자 지도 위에서 Dijkstra, A*, RRT로 장애물을 피하는 경로를 찾고 비교합니다.

python ch06/planning.py          # 합성 지도 + 05-2의 LiDAR 점유 격자
"""
import heapq
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path("outputs/ch06")
OUT.mkdir(parents=True, exist_ok=True)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]   # 8방향 이동


def grid_search(blocked, start, goal, use_heuristic):
    """Dijkstra(use_heuristic=False) 또는 A*(True). 경로(칸 목록)와 확장한 칸 수를 돌려준다."""
    H, W = blocked.shape
    h = (lambda p: np.hypot(p[0] - goal[0], p[1] - goal[1])) if use_heuristic else (lambda p: 0.0)
    cost = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), start)]
    expanded = 0
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            break
        expanded += 1
        for dr, dc, step in MOVES:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < H and 0 <= nxt[1] < W) or blocked[nxt]:
                continue
            c = cost[cur] + step
            if c < cost.get(nxt, np.inf):
                cost[nxt] = c
                parent[nxt] = cur
                heapq.heappush(heap, (c + h(nxt), nxt))
    if goal not in parent:
        return None, expanded
    path, p = [], goal
    while p is not None:
        path.append(p)
        p = parent[p]
    return path[::-1], expanded


def segment_free(blocked, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    rr = np.linspace(a[0], b[0], n).round().astype(int)
    cc = np.linspace(a[1], b[1], n).round().astype(int)
    return not blocked[rr, cc].any()


def rrt(blocked, start, goal, seed, step=8.0, iters=20000, goal_bias=0.1):
    """RRT: 무작위 점을 향해 나무를 조금씩 뻗다가 목표에 닿으면 멈춘다."""
    rng = np.random.default_rng(seed)
    H, W = blocked.shape
    nodes, parent = [np.array(start, float)], [-1]
    for _ in range(iters):
        target = np.array(goal, float) if rng.random() < goal_bias else rng.uniform([0, 0], [H - 1, W - 1])
        d = np.linalg.norm(np.array(nodes) - target, axis=1)
        i = int(np.argmin(d))
        new = nodes[i] + (target - nodes[i]) * min(1.0, step / (d[i] + 1e-9))
        if not segment_free(blocked, nodes[i], new):
            continue
        nodes.append(new)
        parent.append(i)
        if np.linalg.norm(new - goal) < step and segment_free(blocked, new, goal):
            nodes.append(np.array(goal, float))
            parent.append(len(nodes) - 2)
            path, j = [], len(nodes) - 1
            while j >= 0:
                path.append(tuple(nodes[j]))
                j = parent[j]
            return path[::-1], len(nodes)
    return None, len(nodes)


def path_length(path, res):
    p = np.array(path, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) * res)


def compare(name, blocked, start, goal, res):
    print(f"\n[{name}] {blocked.shape[1]}x{blocked.shape[0]}칸 (한 칸 {res}m), 막힌 칸 {blocked.mean():.1%}")
    print(f"{'방법':<10}{'경로 길이':>10}{'탐색한 칸/노드':>15}{'시간':>10}")
    results = {}
    for label, fn in [("Dijkstra", lambda: grid_search(blocked, start, goal, False)),
                      ("A*", lambda: grid_search(blocked, start, goal, True))]:
        t = time.perf_counter()
        path, n = fn()
        ms = (time.perf_counter() - t) * 1000
        results[label] = path
        length = f"{path_length(path, res):8.1f}m" if path else "  실패   "
        print(f"{label:<10}{length:>10}{n:>15,}{ms:>8.0f}ms")
    lengths, counts, times = [], [], []
    for seed in range(20):                        # RRT는 무작위라 20번 돌려 중앙값을 본다
        t = time.perf_counter()
        path, n = rrt(blocked, start, goal, seed)
        times.append((time.perf_counter() - t) * 1000)
        if path:
            lengths.append(path_length(path, res))
            counts.append(n)
            results.setdefault("RRT", path)
    if lengths:
        print(f"{'RRT':<10}{np.median(lengths):8.1f}m{int(np.median(counts)):>15,}{np.median(times):>8.0f}ms"
              f"  (20회 중앙값, 성공 {len(lengths)}/20, 길이 {min(lengths):.1f}~{max(lengths):.1f}m)")
    else:
        print(f"{'RRT':<10}  실패 (20회 모두)")
    return results


def draw(blocked, results, start, goal, fname, unknown=None):
    img = np.full((*blocked.shape, 3), 255, np.uint8)
    img[blocked] = (60, 60, 60)                                    # 막힘: 짙은 회색
    if unknown is not None:
        img[unknown & ~blocked] = (200, 200, 200)                  # 모름(막힘으로 안 본 경우): 옅은 회색
    colors = {"Dijkstra": (255, 150, 0), "A*": (0, 0, 255), "RRT": (0, 160, 0)}
    for label, path in results.items():
        if path:
            pts = np.array([(int(c), int(r)) for r, c in path], np.int32)
            cv2.polylines(img, [pts], False, colors[label], 2)
    cv2.circle(img, (start[1], start[0]), 5, (0, 200, 255), -1)
    cv2.circle(img, (goal[1], goal[0]), 5, (255, 0, 255), -1)
    cv2.imwrite(str(OUT / fname), img)


if __name__ == "__main__":
    # 1. 합성 지도: 100m x 100m, 한 칸 0.5m, 무작위 직사각형 장애물
    rng = np.random.default_rng(8)
    synth = np.zeros((200, 200), bool)
    for _ in range(45):
        r, c = rng.integers(0, 190, 2)
        synth[r:r + rng.integers(5, 30), c:c + rng.integers(5, 30)] = True
    start, goal = (195, 5), (5, 195)                               # 왼쪽 아래 → 오른쪽 위
    synth[190:200, 0:10] = False                                   # 출발·도착 주변은 비워 둔다
    synth[0:10, 190:200] = False
    draw(synth, compare("합성 지도", synth, start, goal, 0.5), start, goal, "plan_synthetic.png")

    # 2. 05-2의 LiDAR 점유 격자: 30m 앞까지 가는 경로. 차 폭의 절반(1m)만큼 장애물을 부풀린다
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch05"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
    from bev import lidar_occupancy              # noqa: E402
    from carla_data import lidar_points          # noqa: E402
    occ = lidar_occupancy(lidar_points("clear", 100))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))     # 반지름 10칸 = 1.0m
    inflated = cv2.dilate((occ == 2).astype(np.uint8), k).astype(bool)
    start, goal = (370, 200), (100, 120)                           # 앞 3m → 앞 30m·왼쪽 8m (두 차로 왼쪽)
    for label, blocked in [("LiDAR 격자, 모름=빈칸", inflated),
                           ("LiDAR 격자, 모름=막힘", inflated | (occ == 0))]:
        blocked = blocked.copy()
        blocked[start] = blocked[goal] = False
        res = compare(label, blocked, start, goal, 0.1)
        draw(blocked, res, start, goal, f"plan_lidar_{'free' if '빈칸' in label else 'blocked'}.png",
             unknown=(occ == 0))
