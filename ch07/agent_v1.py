"""07-4: Modular Driving Agent v1 — 목적지까지 스스로 주행하는 자동차.

  Global Planning : CARLA 도로망 그래프 + A* (route.py, 06-2)
  Perception      : LiDAR 점 중 '내 경로 위' 장애물만 골라 앞 장애물까지 거리 (05-2)
  Behavior        : 순항 / 추종 / 신호 정지 / 긴급 제동 상태 기계 (06-1)
  Control         : 조향 = Pure Pursuit (06-3), 속도 = PI 제어
신호등 상태만은 CARLA가 주는 정답 정보를 씁니다(11장에서 카메라 인지로 바꿉니다).

python ch07/agent_v1.py --route 0 --tm-port 8300
"""
import argparse
import math
import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np

from route import plan

OUT = Path("outputs/ch07")
OUT.mkdir(parents=True, exist_ok=True)
FPS, DT = 20, 0.05
ROUTES = [(7, 154), (14, 22)]                 # Town03 스폰 지점 번호 (출발, 도착)
V_SET = 30 / 3.6                              # 순항 속도 30km/h
HALF_WIDTH = 1.4                              # 경로 좌우로 이 안에 있는 점만 '내 앞 장애물' (m)
TIME_GAP, D_STOP = 2.0, 6.0                   # 앞차와 2초 간격, 6m 안이면 긴급 제동
A_LAT, STEP_M = 2.0, 2.0                      # 커브에서 허용하는 횡가속도(m/s²), 경로 점 간격(m)


class SpeedPI:
    def __init__(self, kp=0.5, ki=0.1):
        self.kp, self.ki, self.i = kp, ki, 0.0

    def __call__(self, target, v):
        e = target - v
        self.i = np.clip(self.i + e * DT, -5, 5)
        u = self.kp * e + self.ki * self.i
        if u >= 0:
            return float(min(u, 0.75)), 0.0                   # (가속, 제동)
        self.i *= 0.9                                         # 감속 중에는 적분을 풀어 준다
        return 0.0, float(min(-u * 0.5, 1.0))


def to_vehicle(xy, tf):
    """세계 좌표 (x, y) → 차량 좌표 (앞, 오른쪽)."""
    yaw = math.radians(tf.rotation.yaw)
    dx, dy = xy[:, 0] - tf.location.x, xy[:, 1] - tf.location.y
    return np.stack([dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)], 1)


def dist_to_polyline(p, pts):
    """점 p에서 꺾은선 pts(경로)까지의 최단 거리 = 경로에서 옆으로 벗어난 정도."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-9), 0, 1)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - p).T)))


def lead_distance(points_world, route_ahead):
    """LiDAR 점(세계 좌표) 중 경로 통로(좌우 HALF_WIDTH) 안에 있고, 그 지점의 도로 면보다
    0.3~2.5m 높은 점까지의 '경로를 따라 잰' 최소 거리. 도로 높이는 지도(경로의 z)에서 가져온다."""
    if len(points_world) == 0 or len(route_ahead) < 2:
        return np.inf, np.zeros((0, 3))
    d = np.hypot(points_world[:, None, 0] - route_ahead[None, :, 0], points_world[:, None, 1] - route_ahead[None, :, 1])
    j = d.argmin(1)
    near = d[np.arange(len(points_world)), j] < HALF_WIDTH
    height = points_world[:, 2] - route_ahead[j, 2]                       # 도로 면에서의 높이
    hit = near & (height > 0.3) & (height < 2.5) & (j > 0)
    if not hit.any():
        return np.inf, np.zeros((0, 3))
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(route_ahead[:, :2], axis=0).T))])
    return float(s[j[hit]].min()), points_world[hit]


def main(args):
    client = carla.Client(args.host, 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town03")
    weather = {"clear": carla.WeatherParameters.ClearNoon, "night": carla.WeatherParameters.ClearNight,
               "rain": carla.WeatherParameters.HardRainNoon}[args.weather]
    world.set_weather(weather)
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    world.set_pedestrians_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cmap, bp = world.get_map(), world.get_blueprint_library()
    spawns = cmap.get_spawn_points()
    a, b = ROUTES[args.route]

    t0 = time.perf_counter()
    route, n_nodes, expanded = plan(cmap, spawns[a].location, spawns[b].location)
    route_len = float(np.sum(np.hypot(*np.diff(route[:, :2], axis=0).T)))
    print(f"경로 {a}→{b}: {route_len:.0f}m, 도로망 노드 {n_nodes}개 중 {expanded}개 탐색, "
          f"{(time.perf_counter() - t0) * 1000:.0f}ms")

    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[a])
    phys = ego.get_physics_control()
    max_steer = math.radians(phys.wheels[0].max_steer_angle)
    # 바퀴 위치는 세계 좌표(cm)라서 x 차이가 아니라 두 점 사이 거리로 축간거리를 구한다
    wheelbase = phys.wheels[0].position.distance(phys.wheels[2].position) / 100.0
    print(f"차량: 축간거리 {wheelbase:.2f}m, 앞바퀴 최대 조향각 {math.degrees(max_steer):.0f}도")
    cars = [x for x in bp.filter("vehicle.*") if int(x.get_attribute("number_of_wheels")) == 4]
    free = [sp for i, sp in enumerate(spawns) if i not in (a, b) and sp.location.distance(spawns[a].location) > 15]
    npcs = [v for v in (world.try_spawn_actor(cars[int(rng.integers(len(cars)))], free[int(i)])
                        for i in rng.choice(len(free), args.npc, replace=False)) if v]
    for v in npcs:
        v.set_autopilot(True, tm.get_port())
    walkers, ctrls = [], []
    for _ in range(args.walkers):
        loc = world.get_random_location_from_navigation()
        wk = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[int(rng.integers(10))], carla.Transform(loc))
        if wk:
            walkers.append(wk)
    world.tick()
    for wk in walkers:
        c = world.spawn_actor(bp.find("controller.ai.walker"), carla.Transform(), attach_to=wk)
        ctrls.append(c)
    world.tick()
    for c in ctrls:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())

    def cam(w, h, tf):
        cb = bp.find("sensor.camera.rgb")
        cb.set_attribute("image_size_x", str(w))
        cb.set_attribute("image_size_y", str(h))
        return world.spawn_actor(cb, tf, attach_to=ego)

    lb = bp.find("sensor.lidar.ray_cast")
    for k, v in {"channels": "32", "range": "50", "rotation_frequency": str(FPS),
                 "points_per_second": "600000", "upper_fov": "5", "lower_fov": "-25"}.items():
        lb.set_attribute(k, v)
    sensors = {"chase": cam(960, 540, carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-12))),
               "lidar": world.spawn_actor(lb, carla.Transform(carla.Location(z=2.2)), attach_to=ego)}
    qs = {k: queue.Queue() for k in sensors}
    for k, sen in sensors.items():
        sen.listen(qs[k].put)
    col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    inv = world.spawn_actor(bp.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
    collisions, invasions = [], []
    col.listen(lambda e: collisions.append((e.frame, e.other_actor.type_id)))
    inv.listen(lambda e: invasions.append((e.frame, [str(m.type) for m in e.crossed_lane_markings])))

    tag = f"r{args.route}_{args.weather}" + ("" if args.curve_limit else "_nocurve")
    if (args.npc, args.walkers, args.seed) != (40, 30, 0):              # 기본 조건이 아니면 파일 이름에 표시
        tag += f"_n{args.npc}w{args.walkers}s{args.seed}"
    writer = cv2.VideoWriter(str(OUT / f"agent_v1_{tag}_raw.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540)) if args.video else None
    pi, idx, step, state = SpeedPI(), 0, 0, "CRUISE"
    states, lat_err, speeds, red_run, loop_ms, log, line_pos = {}, [], [], set(), [], [], {}
    max_steps = int(route_len / 2.0 / DT)                    # 평균 2m/s도 못 내면 시간 초과
    result = "시간 초과"
    try:
        while step < max_steps:
            t_loop = time.perf_counter()
            fid = world.tick()
            data = {}
            for k, q in qs.items():
                d = q.get(timeout=10)
                while d.frame < fid:
                    d = q.get(timeout=10)
                data[k] = d
            step += 1
            tf, vel = ego.get_transform(), ego.get_velocity()
            v = math.hypot(vel.x, vel.y)
            pos = np.array([tf.location.x, tf.location.y])

            # 경로 위 현재 위치 (앞쪽 30개 점 안에서만 찾는다)
            win = route[idx:idx + 30, :2]
            idx += int(np.argmin(np.hypot(*(win - pos).T)))
            lat_err.append(dist_to_polyline(pos, route[max(0, idx - 3):idx + 4, :2]))
            if np.hypot(*(route[-1, :2] - pos)) < 5.0:
                result = "도착"
                break

            # 인지: 경로 앞 45m의 통로 안 장애물
            ahead = to_vehicle(route[idx:idx + 25, :2], tf)
            pts = np.frombuffer(data["lidar"].raw_data, np.float32).reshape(-1, 4)[:, :3]
            pts = pts[(np.abs(pts[:, 0]) > 2.6) | (np.abs(pts[:, 1]) > 1.2)]    # 내 차에 맞은 점은 뺀다
            M = np.array(sensors["lidar"].get_transform().get_matrix())         # 센서 → 세계 (차체 기울기 포함)
            pts_w = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
            dist, hit_w = lead_distance(pts_w, route[idx:idx + 25])
            hit_pts = to_vehicle(hit_w[:, :2], tf) if len(hit_w) else np.zeros((0, 2))

            # 행동: 상태 기계. 신호는 CARLA 정답 정보지만, 40m 앞의 내 차로 신호까지 미리 본다
            light, d_stop = None, np.inf
            fwd = tf.get_forward_vector()
            for tl in world.get_traffic_lights_from_waypoint(cmap.get_waypoint(tf.location), 40.0):
                for wp in tl.get_stop_waypoints():
                    sl = wp.transform.location
                    past = (tf.location.x - sl.x) * fwd.x + (tf.location.y - sl.y) * fwd.y    # 정지선 지나면 +
                    side = abs(-(sl.x - tf.location.x) * fwd.y + (sl.y - tf.location.y) * fwd.x)
                    if side > 2.0:                                # 내 차로의 정지선만
                        continue
                    before = line_pos.get((tl.id, wp.id))
                    if tl.get_state() == carla.TrafficLightState.Red and before is not None and before <= 0 < past:
                        red_run.add(tl.id)                        # 빨간불인 동안 정지선을 넘는 순간
                    line_pos[(tl.id, wp.id)] = past
                    if -past < d_stop and past < 0:
                        light, d_stop = tl.get_state(), -past
            must_stop = light == carla.TrafficLightState.Red or (
                light == carla.TrafficLightState.Yellow and d_stop > v * v / (2 * 4.0) + 1)   # 멈출 수 있으면 멈춘다
            if dist < D_STOP:
                state, target = "EMERGENCY", 0.0
            elif must_stop and d_stop < 40:
                state = "STOP_LIGHT"                              # 정지선 3m 앞에 서도록 2.5m/s²로 감속
                target = float(min(V_SET, math.sqrt(max(0.0, 2 * 2.5 * (d_stop - 3.0)))))
            elif dist < V_SET * TIME_GAP + D_STOP + 5:
                state, target = "FOLLOW", float(np.clip((dist - D_STOP) / TIME_GAP, 0, V_SET))
            else:
                state, target = "CRUISE", V_SET
            # 커브 속도 제한: 앞 15m 경로의 최대 곡률 κ에서 횡가속도가 A_LAT를 넘지 않게 v ≤ √(A_LAT/κ)
            if args.curve_limit and len(ahead) > 8:
                h = np.unwrap(np.arctan2(*np.diff(ahead[:9], axis=0)[:, ::-1].T))
                kappa = np.max(np.abs(np.diff(h))) / STEP_M
                if kappa > 1e-3:
                    target = min(target, math.sqrt(A_LAT / kappa))
            states[state] = states.get(state, 0) + 1

            # 제어: Pure Pursuit 조향 + PI 속도
            ld = float(np.clip(0.6 * v + 3.0, 4.0, 12.0))
            arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ahead, axis=0).T))])
            tx, ty = ahead[min(int(np.searchsorted(arc, ld)), len(ahead) - 1)]
            alpha = math.atan2(ty, tx)
            delta = math.atan2(2 * wheelbase * math.sin(alpha), ld)
            throttle, brake = pi(target, v)
            if state == "EMERGENCY":
                throttle, brake = 0.0, 1.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake,
                                                   steer=float(np.clip(delta / max_steer, -1, 1))))
            speeds.append(v)
            loop_ms.append((time.perf_counter() - t_loop) * 1000)
            log.append((fid, round(step * DT, 2), idx, round(pos[0], 2), round(pos[1], 2), round(v * 3.6, 1), state,
                        round(lat_err[-1], 2), round(min(dist, 99), 1), str(light).split(".")[-1]))

            # 영상: 추적 카메라 + 상태 + 미니 BEV
            if writer is None:
                continue
            img = np.frombuffer(data["chase"].raw_data, np.uint8).reshape(540, 960, 4)[:, :, :3].copy()
            bev = np.zeros((200, 160, 3), np.uint8)
            for p, color in [(ahead, (0, 200, 0)), (hit_pts, (0, 0, 255))]:
                for fx, ry in p:
                    r_, c_ = int(190 - fx * 4), int(80 + ry * 4)
                    if 0 <= r_ < 200 and 0 <= c_ < 160:
                        cv2.circle(bev, (c_, r_), 2, color, -1)
            cv2.rectangle(bev, (76, 182), (84, 198), (255, 255, 255), -1)
            img[10:210, 790:950] = bev
            text = f"{state:<10} {v * 3.6:4.1f}km/h  target {target * 3.6:4.1f}  lead {dist:5.1f}m"
            cv2.putText(img, text, (12, 30), 0, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"route {idx * 100 // len(route)}%  t={step * DT:5.1f}s", (12, 60), 0, 0.7,
                        (0, 255, 255), 2)
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()
        for sen in list(sensors.values()) + [col, inv]:
            sen.stop()
        for c in ctrls:
            c.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in
                            list(sensors.values()) + [col, inv] + ctrls + walkers + npcs + [ego]])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    with open(OUT / f"agent_v1_{tag}_log.csv", "w") as f:
        f.write("frame,t,route_idx,x,y,speed_kmh,state,lateral_m,lead_m,light\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in log)
    events = sorted({f // FPS: t for f, t in collisions}.items())        # 1초 안의 연속 충돌은 한 번으로
    solid_events = [f for f, ms in invasions if any("Solid" in m for m in ms)]
    solid = len(solid_events)
    if solid:
        print("실선 침범 시각(프레임 번호):", solid_events[:5])
    print(f"결과: {result}, 주행 {step * DT:.1f}초, 경로 진행 {min(100, idx * 100 / (len(route) - 1)):.0f}%, "
          f"평균 속도 {np.mean(speeds) * 3.6:.1f}km/h")
    print(f"충돌 {len(events)}회 {[t for _, t in events][:5]}, 실선 침범 {solid}회, "
          f"빨간불 정지선 통과 {len(red_run)}회")
    print(f"경로 이탈: 평균 {np.mean(lat_err):.2f}m, 최대 {np.max(lat_err):.2f}m")
    print("상태 비율: " + ", ".join(f"{k} {v / step:.0%}" for k, v in sorted(states.items(), key=lambda x: -x[1])))
    print(f"루프 처리 시간 중앙값 {np.median(loop_ms):.0f}ms (CARLA tick 포함)")
    return {"result": result, "time_s": round(step * DT, 1), "route_m": round(route_len),
            "progress": round(min(1.0, idx / (len(route) - 1)), 3), "avg_kmh": round(float(np.mean(speeds)) * 3.6, 1),
            "collisions": [t for _, t in events], "solid": solid, "red_run": len(red_run),
            "lat_mean": round(float(np.mean(lat_err)), 2), "lat_max": round(float(np.max(lat_err)), 2),
            "states": {k: round(v / step, 3) for k, v in states.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--route", type=int, default=0)
    ap.add_argument("--weather", default="clear", choices=["clear", "night", "rain"])
    ap.add_argument("--npc", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curve-limit", dest="curve_limit", action="store_false",
                    help="커브 속도 제한 끄기 (비교용)")
    ap.add_argument("--no-video", dest="video", action="store_false", help="영상 저장 끄기 (반복 평가용)")
    main(ap.parse_args())
