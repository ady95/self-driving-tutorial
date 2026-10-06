"""11장: Driving Agent v2 — 07-4의 v1을 바탕으로 10장에서 드러난 약점을 고친 최종 에이전트.

  v1에서 바뀐 것
  - 신호   : CARLA 정답 대신 카메라로 읽는다 (지도로 신호등 위치를 투영 + 켜진 등의 위치, lights.py)
  - 보행자 : YOLO(03-2) + LiDAR 거리(04-4)로 경로 근처 보행자를 일찍 보고 감속·양보
  - 막힘   : 멈춘 장애물 앞에서 15초 넘게 서 있으면 옆 차로로 우회, 못 하면 40초 뒤 최소 위험 상태(비상등 정지)
  - 언어   : 자연어 명령 → 매개변수 (command.py), VLM 조언자는 속도를 낮추기만 (advisor.py)
  - 지표   : 충돌까지 남은 시간(TTC) 최솟값, 보행자 최소 거리, 신호 판독 정확도를 함께 잰다

python ch11/agent_v2.py --route 1 --tm-port 8300
python ch11/agent_v2.py --route 1 --scenario construction --command "앞에 공사 구간이 있으니 안전하게 피해 가라"
python ch11/agent_v2.py --route 0 --scenario pedestrian --vlm --video
"""
import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ch07"))
from agent_v1 import (A_LAT, D_STOP, DT, FPS, ROUTES, STEP_M, SpeedPI, dist_to_polyline,  # noqa: E402
                      lead_distance, to_vehicle)
from collect_lights import light_rois, my_light  # noqa: E402
from command import DEFAULT, parse_command  # noqa: E402
from lights import LightFilter, read_light  # noqa: E402
from route import plan  # noqa: E402
from scenarios import Scenario  # noqa: E402

OUT = Path("outputs/ch11")
W, H = 1280, 720
K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])
WEATHERS = {"clear": carla.WeatherParameters.ClearNoon, "night": carla.WeatherParameters.ClearNight,
            "rain": carla.WeatherParameters.HardRainNoon}
STUCK_S, MRC_S = 15.0, 40.0                     # 이만큼 서 있으면 우회 시도 / 최소 위험 상태


def people_on_camera(result, pts_cam):
    """YOLO 사람 상자 + LiDAR 점 → 사람마다 차량 기준 위치 (앞, 오른쪽). LiDAR 점이 3개 이상 닿은 사람만."""
    if len(pts_cam) == 0:
        return []
    u = K[0, 2] + K[0, 0] * pts_cam[:, 1] / pts_cam[:, 0]
    v = K[1, 2] - K[1, 1] * pts_cam[:, 2] / pts_cam[:, 0]
    found = []
    for c, b in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = b
        m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        if int(c) == 0 and m.sum() >= 3:
            p = np.median(pts_cam[m], axis=0)
            found.append((p[0] + 1.5, p[1]))                     # 카메라 → 차량 기준 (앞으로 1.5m)
    return found


class PedestrianTracker:
    """보행자를 프레임 사이에 이어 속도를 구하고, 1.5초 뒤 위치를 예측한다 (06-1의 등속 예측).
    LiDAR로 잰 위치는 프레임마다 흔들리므로, 0.5초 동안 움직인 거리로 속도를 구한다."""

    def __init__(self, horizon=1.5, span=5):
        self.horizon, self.span, self.tracks = horizon, span, []   # tracks = [최근 위치 목록]

    def update(self, people_vehicle, tf, dt):
        yaw = math.radians(tf.rotation.yaw)
        new_tracks, out = [], []
        for fwd, right in people_vehicle:
            p = np.array([tf.location.x + fwd * math.cos(yaw) - right * math.sin(yaw),
                          tf.location.y + fwd * math.sin(yaw) + right * math.cos(yaw)])
            hist = [p]
            if self.tracks:
                d = [np.hypot(*(p - t[-1])) for t in self.tracks]
                j = int(np.argmin(d))
                if d[j] < 1.0:                                   # 0.1초에 1m 안이면 같은 사람
                    hist = (self.tracks[j] + [p])[-(self.span + 1):]
            vel = (hist[-1] - hist[0]) / (dt * (len(hist) - 1)) if len(hist) > 2 else np.zeros(2)
            speed = np.hypot(*vel)
            if speed > 3.0:                                      # 사람이 낼 수 없는 속도면 잡음
                vel = vel * 3.0 / speed
            new_tracks.append(hist)
            out.append((p, p + vel * self.horizon))
        self.tracks = new_tracks
        return out


def nearest_person(tracked, route_ahead_xy, tf):
    """예측까지 고려해 경로에 가장 가까운 사람의 (앞 거리, 지금 옆 거리, 예측 옆 거리)."""
    best = (np.inf, np.inf, np.inf)
    if len(route_ahead_xy) < 2:
        return best
    for p, pred in tracked:
        gap_now = dist_to_polyline(p, route_ahead_xy)
        gap_pred = dist_to_polyline(pred, route_ahead_xy)
        fwd = float(to_vehicle(p[None], tf)[0, 0])
        if fwd > 0 and min(gap_now, gap_pred) < min(best[1], best[2]):
            best = (fwd, gap_now, gap_pred)
    return best


def bypass_route(cmap, route, idx, obstacle_end_m):
    """idx부터 장애물이 끝나는 곳 8m 뒤까지 경로를 왼쪽 같은 방향 차로로 옮긴다. 불가능하면 None."""
    n = int((obstacle_end_m + 8) / STEP_M) + 3
    if idx + n >= len(route):
        return None
    new = route.copy()
    for i, k in enumerate(range(idx + 1, idx + 1 + n)):
        x, y, z = map(float, route[k])
        wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=z))
        left = wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id < 0:
            return None
        a = min(1.0, (i + 1) / 3, (n - i) / 3)                  # 처음과 끝 3점(6m)은 서서히 옮긴다
        lx, ly = left.transform.location.x, left.transform.location.y
        new[k, :2] = (1 - a) * route[k, :2] + a * np.array([lx, ly])
    return new


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    params = dict(DEFAULT)
    v1 = args.mode == "v1"                                       # 비교용: v2의 새 기능을 모두 끈 v1 설정
    if v1:
        args.light, params["allow_bypass"] = "gt", False
    if args.command:
        sys.path.insert(0, str(HERE.parent / "ch09"))
        from vlm import make_client
        client_llm, model = make_client(args.llm_backend)
        params, notes, reply, _ = parse_command(args.command, client_llm, model,
                                                "responses" if args.llm_backend == "openai" else "chat")
        print(f"명령: {args.command}\n  → {params}" + (f"\n  → 규칙이 고침: {notes}" if notes else "") + f"\n  → {reply}")
    v_set = params["cruise_kmh"] / 3.6
    advisor = None
    if args.vlm:
        from advisor import VLMAdvisor
        advisor = VLMAdvisor(args.vlm_backend)

    det = YOLO("yolo26n.pt")
    client = carla.Client(args.host, 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town03")
    world.set_weather(WEATHERS[args.weather])
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
    route, _, _ = plan(cmap, spawns[a].location, spawns[b].location)
    route_len = float(np.sum(np.hypot(*np.diff(route[:, :2], axis=0).T)))
    base_route = route.copy()

    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[a])
    phys = ego.get_physics_control()
    max_steer = math.radians(phys.wheels[0].max_steer_angle)
    wheelbase = phys.wheels[0].position.distance(phys.wheels[2].position) / 100.0
    cars = [x for x in bp.filter("vehicle.*") if int(x.get_attribute("number_of_wheels")) == 4]
    free = [sp for i, sp in enumerate(spawns) if i not in (a, b) and sp.location.distance(spawns[a].location) > 15]
    npcs = [v for v in (world.try_spawn_actor(cars[int(rng.integers(len(cars)))], free[int(i)])
                        for i in rng.choice(len(free), args.npc, replace=False)) if v]
    for v in npcs:
        v.set_autopilot(True, tm.get_port())
    walkers, ctrls = [], []
    for _ in range(args.walkers):
        wk = world.try_spawn_actor(bp.filter("walker.pedestrian.*")[int(rng.integers(10))],
                                   carla.Transform(world.get_random_location_from_navigation()))
        if wk:
            walkers.append(wk)
    world.tick()
    for wk in walkers:
        ctrls.append(world.spawn_actor(bp.find("controller.ai.walker"), carla.Transform(), attach_to=wk))
    world.tick()
    for c in ctrls:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
    scen = Scenario(args.scenario, world, cmap, route) if args.scenario != "none" else None

    def sensor(kind, tf, attrs):
        sb = bp.find(kind)
        for k, v in attrs.items():
            sb.set_attribute(k, str(v))
        return world.spawn_actor(sb, tf, attach_to=ego)

    sensors = {"rgb": sensor("sensor.camera.rgb", carla.Transform(carla.Location(x=1.5, z=1.6)),
                             {"image_size_x": W, "image_size_y": H}),
               "lidar": sensor("sensor.lidar.ray_cast", carla.Transform(carla.Location(z=2.2)),
                               {"channels": 32, "range": 50, "rotation_frequency": FPS, "points_per_second": 600000,
                                "upper_fov": 5, "lower_fov": -25})}
    if args.video:
        sensors["chase"] = sensor("sensor.camera.rgb", carla.Transform(carla.Location(x=-6.5, z=3.0),
                                                                       carla.Rotation(pitch=-12)),
                                  {"image_size_x": 960, "image_size_y": 540})
    qs = {k: queue.Queue() for k in sensors}
    for k, sen in sensors.items():
        sen.listen(qs[k].put)
    col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    inv = world.spawn_actor(bp.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
    collisions, invasions = [], []
    col.listen(lambda e: collisions.append((e.frame, e.other_actor.type_id)))
    inv.listen(lambda e: invasions.append((e.frame, [str(m.type) for m in e.crossed_lane_markings])))

    tag = f"{args.mode}_r{args.route}_{args.weather}_{args.scenario}" + (f"_s{args.seed}" if args.seed else "") + args.tag
    writer = cv2.VideoWriter(str(OUT / f"agent_v2_{tag}_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (960, 540)) if args.video else None
    pi, lf = SpeedPI(), LightFilter()
    idx, step, state, result = 0, 0, "CRUISE", "시간 초과"
    max_steps = int(route_len / 2.0 / DT)
    states, log, line_pos, red_run = {}, [], {}, set()
    lat_err, speeds, ttcs, ped_min, lead_min = [], [], [], np.inf, np.inf
    light_hits, light_total, light_unknown = 0, 0, 0
    stuck_t, bypasses, people, mrc = 0.0, 0, (np.inf, np.inf, np.inf), False
    tracker, dist_hist = PedestrianTracker(), []
    try:
        while step < max_steps:
            fid = world.tick()
            if step == 0:
                first_frame = fid
            data = {}
            for k, q in qs.items():
                d = q.get(timeout=10)
                while d.frame < fid:
                    d = q.get(timeout=10)
                data[k] = d
            step += 1
            sim_t = step * DT
            tf, vel = ego.get_transform(), ego.get_velocity()
            v = math.hypot(vel.x, vel.y)
            pos = np.array([tf.location.x, tf.location.y])
            win = route[idx:idx + 30, :2]
            idx += int(np.argmin(np.hypot(*(win - pos).T)))
            lat_err.append(dist_to_polyline(pos, base_route[max(0, idx - 3):idx + 4, :2]))
            if np.hypot(*(route[-1, :2] - pos)) < 5.0:
                result = "도착"
                break
            if scen is not None:
                scen.step(tf.location)
            img = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(H, W, 4)[:, :, :3]

            # 인지 1: LiDAR — 경로 통로 안 장애물 (v1과 같음), 충돌까지 남은 시간
            ahead = to_vehicle(route[idx:idx + 25, :2], tf)
            pts = np.frombuffer(data["lidar"].raw_data, np.float32).reshape(-1, 4)[:, :3]
            pts = pts[(np.abs(pts[:, 0]) > 2.6) | (np.abs(pts[:, 1]) > 1.2)]
            M = np.array(sensors["lidar"].get_transform().get_matrix())
            pts_w = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
            dist, hit_w = lead_distance(pts_w, route[idx:idx + 25])
            dist_hist = (dist_hist + [dist])[-11:]                   # 장애물 거리는 경로 점 간격(2m) 단위라
            old = dist_hist[0]                                       # 0.5초 동안의 변화로 접근 속도를 구한다
            if np.isfinite(dist) and np.isfinite(old) and v > 0.5 and len(dist_hist) == 11 and old - dist < 10:
                closing = (old - dist) / (10 * DT)
                if closing > 0.5:
                    ttcs.append(dist / closing)
            if v > 1.0:
                lead_min = min(lead_min, dist)

            # 인지 2: 카메라 — 보행자 (YOLO + LiDAR 거리 + 움직임 예측), 0.1초마다
            if step % 2 == 0 and not v1:
                r = det(img, classes=[0], conf=0.3, verbose=False)[0]
                cam_pts = pts.copy()
                cam_pts[:, 0] -= 1.5
                cam_pts[:, 2] += 2.2 - 1.6
                cam_pts = cam_pts[cam_pts[:, 0] > 0.5]                 # LiDAR → 카메라 위치 (07-3과 같은 계산)
                tracked = tracker.update(people_on_camera(r, cam_pts), tf, 2 * DT)
                people = nearest_person(tracked, route[idx:idx + 25, :2], tf)
            ped_gap = min(people[1], people[2])                      # 지금 또는 1.5초 뒤 경로에 가까운 쪽
            if people[1] < params["pedestrian_margin_m"]:
                ped_min = min(ped_min, people[0])

            # 인지 3: 카메라 — 신호 (지도로 위치 투영 + 켜진 등의 위치)
            tl, d_line = my_light(world, cmap, tf)
            light = None
            if tl is not None:
                reading = read_light(img, light_rois(tl, sensors["rgb"])) if args.light == "camera" else \
                    str(tl.get_state()).split(".")[-1].upper()
                light = lf.update(reading)
                gt = str(tl.get_state()).split(".")[-1].upper()
                light_total += 1
                light_hits += light == gt
                light_unknown += light is None
                for wp in tl.get_stop_waypoints():                   # 빨간불 위반은 정답으로 센다 (v1과 같음)
                    sl = wp.transform.location
                    fwd = tf.get_forward_vector()
                    past = (tf.location.x - sl.x) * fwd.x + (tf.location.y - sl.y) * fwd.y
                    before = line_pos.get((tl.id, wp.id))
                    if gt == "RED" and before is not None and before <= 0 < past:
                        red_run.add(tl.id)
                    line_pos[(tl.id, wp.id)] = past
            else:
                lf.update(None)

            # 행동: 조건마다 목표 속도를 내고 가장 낮은 것을 따른다.
            # v1은 정해진 순서에서 '먼저 맞는 조건 하나'만 봤다 → 빨간불 감속 중에는 앞에 줄 선 차를 무시했다 (11-4)
            must_stop = light == "RED" or (light == "YELLOW" and d_line > v * v / (2 * 4.0) + 1)
            gap = params["time_gap_s"]
            cand = {"CRUISE": v_set}
            if dist < D_STOP:
                cand["EMERGENCY"] = 0.0
            if must_stop and d_line < 40:                            # 정지선 3m 앞에 서도록 2.5m/s²로 감속
                cand["STOP_LIGHT"] = float(min(v_set, math.sqrt(max(0.0, 2 * 2.5 * (d_line - 3.0)))))
            if tl is not None and light is None and d_line < 30:   # 신호를 못 읽으면 천천히, 정지선 앞에서는 선다
                cand["LIGHT_UNKNOWN"] = 0.0 if d_line < 4 else min(v_set, 15 / 3.6)
            if people[0] < 30 and ped_gap < params["pedestrian_margin_m"]:   # 경로 가까이 사람: 사람 앞 5m에 서도록
                stop_at = people[0] - 5.0 if ped_gap < 2.0 else 999
                cand["PEDESTRIAN"] = float(min(15 / 3.6, math.sqrt(max(0.0, 2 * 2.5 * max(stop_at, 0)))))
            if dist < v_set * gap + D_STOP + 5:
                cand["FOLLOW"] = float(np.clip((dist - D_STOP) / gap, 0, v_set))
            order = ["EMERGENCY", "STOP_LIGHT", "LIGHT_UNKNOWN", "PEDESTRIAN", "FOLLOW", "CRUISE"]
            if v1:
                state = next(k for k in order if k in cand)
            else:
                state = min(cand, key=lambda k: (cand[k], order.index(k)))
            target = cand[state]
            if len(ahead) > 8:                                       # 커브 속도 제한 (v1과 같음)
                hh = np.unwrap(np.arctan2(*np.diff(ahead[:9], axis=0)[:, ::-1].T))
                kappa = np.max(np.abs(np.diff(hh))) / STEP_M
                if kappa > 1e-3:
                    target = min(target, math.sqrt(A_LAT / kappa))
            if advisor is not None:                                  # VLM 조언: 속도를 낮추기만
                advisor.maybe_submit(img, sim_t)
                cap = advisor.speed_cap(sim_t)
                if cap is not None and cap < target:
                    target = cap
                    state += "+VLM"

            # 막힘 대처: 신호 때문이 아닌데 오래 서 있으면 우회 → 안 되면 최소 위험 상태
            # 신호를 끝내 못 읽어 서 있는 것(LIGHT_UNKNOWN)도 막힘으로 센다
            blocked = v < 0.3 and state.split("+")[0] in ("EMERGENCY", "FOLLOW", "LIGHT_UNKNOWN") and \
                light not in ("RED", "YELLOW")
            stuck_t = stuck_t + DT if blocked else 0.0
            if stuck_t > STUCK_S and params["allow_bypass"] and np.isfinite(dist):
                along = to_vehicle(hit_w[:, :2], tf)[:, 0] if len(hit_w) else np.array([dist])
                end = float(np.max(along[along < dist + 20]))           # 장애물이 끝나는 곳 (앞으로 잰 거리)
                new = bypass_route(cmap, route, idx, end)
                if new is not None:
                    side_dist, _ = lead_distance(pts_w, new[idx + 3:idx + 25])
                    if side_dist > dist + 10:                        # 옆 차로가 장애물 너머까지 비어 있으면
                        route, bypasses, stuck_t = new, bypasses + 1, 0.0
                        print(f"  {sim_t:.1f}초: 옆 차로로 우회 (앞 장애물 {dist:.1f}m)")
            if stuck_t > MRC_S and not v1:
                ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LeftBlinker
                                                            | carla.VehicleLightState.RightBlinker))
                result, mrc = "최소 위험 상태", True
                print(f"  {sim_t:.1f}초: {MRC_S:.0f}초 넘게 막혀 비상등을 켜고 정지 (최소 위험 상태)")
                break
            states[state] = states.get(state, 0) + 1

            # 제어: Pure Pursuit + PI (v1과 같음)
            ld = float(np.clip(0.6 * v + 3.0, 4.0, 12.0))
            arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ahead, axis=0).T))])
            tx, ty = ahead[min(int(np.searchsorted(arc, ld)), len(ahead) - 1)]
            delta = math.atan2(2 * wheelbase * math.sin(math.atan2(ty, tx)), ld)
            throttle, brake = pi(target, v)
            if state.startswith("EMERGENCY"):
                throttle, brake = 0.0, 1.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake,
                                                   steer=float(np.clip(delta / max_steer, -1, 1))))
            speeds.append(v)
            log.append((round(sim_t, 2), idx, round(v * 3.6, 1), state, round(min(dist, 99), 1),
                        round(min(people[0], 99), 1), round(min(people[1], 99), 1), round(min(people[2], 99), 1),
                        light or "",
                        str(tl.get_state()).split(".")[-1].upper() if tl else "", round(min(d_line, 99), 1)))

            if writer is not None:
                vis = np.frombuffer(data["chase"].raw_data, np.uint8).reshape(540, 960, 4)[:, :, :3].copy()
                small = cv2.resize(img, (384, 216))
                if tl is not None:
                    for x1, y1, x2, y2 in light_rois(tl, sensors["rgb"]):
                        cv2.rectangle(small, (int(x1 * 0.3), int(y1 * 0.3)), (int(x2 * 0.3) + 1, int(y2 * 0.3) + 1),
                                      (0, 255, 255), 1)
                vis[8:224, 568:952] = small
                cv2.putText(vis, f"{state:<16} {v * 3.6:4.1f}km/h  target {target * 3.6:4.1f}", (12, 30), 0, 0.7,
                            (0, 255, 255), 2)
                cv2.putText(vis, f"light {light or '?':<6} lead {min(dist, 99):4.1f}m  ped {min(people[0], 99):4.1f}m"
                                 f"  route {idx * 100 // len(route)}%", (12, 60), 0, 0.7, (0, 255, 255), 2)
                if advisor is not None and advisor.latest is not None:
                    cv2.putText(vis, f"VLM: {advisor.latest[1]}", (12, 90), 0, 0.7, (255, 200, 0), 2)
                writer.write(vis)
    finally:
        if writer is not None:
            writer.release()
        for sen in list(sensors.values()) + [col, inv]:
            sen.stop()
        for c in ctrls:
            c.stop()
        extra = scen.cleanup() if scen is not None else []
        client.apply_batch([carla.command.DestroyActor(x) for x in
                            list(sensors.values()) + [col, inv] + ctrls + walkers + npcs + extra + [ego]])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    with open(OUT / f"agent_v2_{tag}_log.csv", "w") as f:
        f.write("t,route_idx,speed_kmh,state,lead_m,ped_fwd_m,ped_gap_m,ped_gap_pred_m,light,light_gt,line_m\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in log)
    events = sorted({f // FPS: t for f, t in collisions}.items())
    solid = sum(1 for _, ms in invasions if any("Solid" in m for m in ms))
    out = {"result": result, "time_s": round(step * DT, 1), "route_m": round(route_len),
           "progress": round(min(1.0, idx / (len(route) - 1)), 3), "avg_kmh": round(float(np.mean(speeds)) * 3.6, 1),
           "collisions": [t for _, t in events], "collision_t": [round(f * DT - first_frame * DT, 1) for f, _ in collisions[:3]],
           "solid": solid, "red_run": len(red_run),
           "min_ttc": round(min(ttcs), 2) if ttcs else None, "min_lead": round(float(lead_min), 1),
           "ped_min": round(float(ped_min), 1),
           "light_acc": round(light_hits / light_total, 3) if light_total else None,      # 신호를 만나지 못했으면 None
           "light_unknown": round(light_unknown / light_total, 3) if light_total else None, "bypasses": bypasses, "mrc": mrc,
           "states": {k: round(v / max(step, 1), 3) for k, v in states.items()}, "params": params,
           "vlm": advisor.log if advisor is not None else []}
    print(f"결과: {result}, {out['time_s']}초, 경로 진행 {out['progress']:.0%}, 평균 {out['avg_kmh']}km/h")
    print(f"충돌 {len(events)}회 {out['collisions'][:3]}, 실선 침범 {solid}회, 빨간불 통과 {len(red_run)}회, "
          f"우회 {bypasses}회")
    print(f"최소 TTC {out['min_ttc']}초, 앞 장애물 최소 거리(달리는 중) {out['min_lead']}m, "
          f"경로 근처 보행자 최소 거리 {out['ped_min']}m")
    if light_total:
        print(f"신호 판독: 정답 일치 {out['light_acc']:.1%}, 모름 {out['light_unknown']:.1%} "
              f"(내 차로 신호가 50m 안에 있던 {light_total}스텝)")
    print("상태 비율: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(out["states"].items(), key=lambda x: -x[1])))
    if advisor is not None:
        walls = [e["wall_s"] for e in advisor.log]
        print(f"VLM 조언 {len(advisor.log)}번, 응답 중앙값 {np.median(walls) if walls else 0:.1f}초, "
              f"위험도 {dict((h, sum(e['hazard'] == h for e in advisor.log)) for h in ['none', 'low', 'high'])}")
    with open(OUT / f"agent_v2_{tag}_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--route", type=int, default=0)
    ap.add_argument("--weather", default="clear", choices=list(WEATHERS))
    ap.add_argument("--npc", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="none", choices=["none", "pedestrian", "stalled_car", "construction"])
    ap.add_argument("--mode", default="v2", choices=["v2", "v1"], help="v1 = 정답 신호, 보행자·우회·최소 위험 상태 없음")
    ap.add_argument("--light", default="camera", choices=["camera", "gt"], help="gt = v1처럼 정답 신호 (비교용)")
    ap.add_argument("--command", default="")
    ap.add_argument("--llm-backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--vlm-backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(get_args())
