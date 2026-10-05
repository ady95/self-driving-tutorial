"""07-4: CARLA 도로망으로 그래프를 만들고 A*(06-2)로 출발지 → 목적지 경로를 찾습니다 (Global Planning).

CARLA 지도의 topology는 '차로 구간'의 목록입니다. 구간의 시작·끝을 노드로, 구간을 간선으로 두면
도로망이 그래프가 되고, 06-2의 격자 대신 이 그래프 위에서 같은 A*를 돌리면 됩니다.
"""
import heapq

import numpy as np

STEP = 2.0                                      # 경로 점 간격 (m)


def key(loc):
    return (round(loc.x, 1), round(loc.y, 1))   # 구간 끝과 다음 구간 시작을 같은 노드로 묶는다


def build_graph(carla_map):
    graph = {}                                  # 노드 → [(이웃 노드, 길이, 구간의 점들)]
    pos = {}
    for a, b in carla_map.get_topology():
        pts = [a] + a.next_until_lane_end(STEP)
        # 도로 높이(z)도 함께 저장한다: 오르막·내리막에서 장애물과 노면을 구별하는 데 쓴다
        xy = [(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in pts]
        xy.append((b.transform.location.x, b.transform.location.y, b.transform.location.z))
        length = float(np.sum(np.hypot(*np.diff(np.array(xy)[:, :2], axis=0).T)))
        ka, kb = key(a.transform.location), key(b.transform.location)
        graph.setdefault(ka, []).append((kb, length, xy))
        graph.setdefault(kb, [])
        pos[ka], pos[kb] = xy[0], xy[-1]
    return graph, pos


def nearest(pos, x, y):
    return min(pos, key=lambda k: (pos[k][0] - x) ** 2 + (pos[k][1] - y) ** 2)


def astar(graph, pos, start, goal):
    """06-2와 같은 A*. 휴리스틱 = 남은 직선거리."""
    h = lambda k: np.hypot(pos[k][0] - pos[goal][0], pos[k][1] - pos[goal][1])
    cost, parent = {start: 0.0}, {start: (None, None)}
    heap, expanded = [(h(start), start)], 0
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            break
        expanded += 1
        for nxt, length, xy in graph[cur]:
            c = cost[cur] + length
            if c < cost.get(nxt, np.inf):
                cost[nxt], parent[nxt] = c, (cur, xy)
                heapq.heappush(heap, (c + h(nxt), nxt))
    if goal not in parent:
        return None, expanded
    pieces, k = [], goal
    while parent[k][0] is not None:
        pieces.append(parent[k][1])
        k = parent[k][0]
    route = [p for xy in reversed(pieces) for p in xy]
    return np.array(route), expanded


def nearest_edge(graph, x, y):
    """(x, y)에 가장 가까운 점을 가진 구간과, 그 구간 안에서의 점 번호."""
    best = (np.inf, None, None, 0)
    for a, edges in graph.items():
        for b, _, xy in edges:
            d = np.hypot(*(np.array(xy)[:, :2] - [x, y]).T)
            i = int(np.argmin(d))
            if d[i] < best[0]:
                best = (d[i], a, (b, xy), i)
    return best[1:]


def plan(carla_map, start_loc, goal_loc):
    """출발지가 놓인 구간의 중간에서 시작해, 도착지가 놓인 구간의 중간에서 끝나는 경로 (N, 3): x, y, 도로 높이 z."""
    graph, pos = build_graph(carla_map)
    _, (s_next, s_xy), si = nearest_edge(graph, start_loc.x, start_loc.y)
    g_prev, (_, g_xy), gi = nearest_edge(graph, goal_loc.x, goal_loc.y)
    middle, expanded = astar(graph, pos, s_next, g_prev)
    if middle is None:
        return None, len(graph), expanded
    route = np.vstack([np.array(s_xy[si:]), middle, np.array(g_xy[:gi + 1])])
    keep = np.r_[True, np.hypot(*np.diff(route[:, :2], axis=0).T) > 0.1]       # 겹친 점 제거
    return route[keep], len(graph), expanded
