"""08장: 학습한 End-to-End 모델에 CARLA 차량의 운전대를 맡기고, 사람이 개입해야 하는 횟수를 잽니다.

모델은 조향만 정하고 속도는 25km/h로 고정(PI 제어)합니다. 다른 차는 없습니다 — 조향 능력만 봅니다.
개입: 차로 중심에서 1.5m 넘게 벗어나거나, 교차로 밖에서 차로 방향과 60도 넘게 어긋나거나(역주행), 무언가에 부딪히면
      개입 1회로 세고 차를 가장 가까운 차로 중앙에 되돌려 놓은 뒤 계속 달립니다.

python ch08/drive_eval.py --kind expert                       # 기준선: Traffic Manager 자동 주행
python ch08/drive_eval.py --kind bc --model outputs/ch08/bc_side.pt
python ch08/drive_eval.py --kind traj --model outputs/ch08/traj.pt
"""
import argparse
import math
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bc import build_model, to_tensor  # noqa: E402

OUT = Path("outputs/ch08")
FPS, DT, W, H = 20, 0.05, 320, 180


def load_policy(kind, path, device):
    """영상 → 조향(-1~1)을 돌려주는 함수를 만든다."""
    if kind == "bc":
        model = build_model().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        def policy(img, speed):
            with torch.no_grad():
                return float(model(to_tensor(img)[None].to(device))[0, 0])
        return policy
    if kind == "traj":
        from train_traj import build_traj_model, steer_from_waypoints  # 08-4
        model = build_traj_model().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        def policy(img, speed):
            with torch.no_grad():
                wps = model(to_tensor(img)[None].to(device), torch.tensor([[speed]], device=device))[0]
            return steer_from_waypoints(wps.cpu().numpy(), speed)
        return policy
    return None                                                   # expert: Traffic Manager가 운전


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.kind, args.model, device)
    client = carla.Client(args.host, 2000)
    client.set_timeout(120.0)
    world = client.load_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    cmap, bp = world.get_map(), world.get_blueprint_library()
    spawns = cmap.get_spawn_points()
    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[args.seed % len(spawns)])
    if args.kind == "expert":
        ego.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(ego, False)
        tm.ignore_lights_percentage(ego, 100)

    cb = bp.find("sensor.camera.rgb")
    for k, v in {"image_size_x": W, "image_size_y": H, "fov": 90}.items():
        cb.set_attribute(k, str(v))
    cam = world.spawn_actor(cb, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    hits = []
    col.listen(lambda e: hits.append(e.frame))
    writer = None
    if args.video:
        vb = bp.find("sensor.camera.rgb")
        vb.set_attribute("image_size_x", "960")
        vb.set_attribute("image_size_y", "540")
        chase = world.spawn_actor(vb, carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-12)),
                                  attach_to=ego)
        qv = queue.Queue()
        chase.listen(qv.put)
        OUT.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(OUT / f"eval_{args.kind}_{Path(args.model).stem if args.model else 'tm'}"
                                     f"_{args.town}_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

    target, integ, dist, lat_sum, interventions, touch = args.speed / 3.6, 0.0, 0.0, 0.0, [], 0
    last_hit, last_junction = -1, -999
    steps = int(args.minutes * 60 * FPS)
    try:
        for step in range(steps):
            fid = world.tick()
            d = q.get(timeout=10)
            while d.frame < fid:
                d = q.get(timeout=10)
            img = np.frombuffer(d.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
            tf, vel = ego.get_transform(), ego.get_velocity()
            v = math.hypot(vel.x, vel.y)
            dist += v * DT

            wp = cmap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            lat = math.hypot(tf.location.x - wp.transform.location.x, tf.location.y - wp.transform.location.y)
            dyaw = abs((tf.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
            lat_sum += lat
            if wp.is_junction:
                last_junction = step
            touch += lat > 0.8                                        # 바퀴가 차선에 닿을 정도
            crashed = len(hits) > 0 and hits[-1] != last_hit
            wrong_way = dyaw > 60 and not wp.is_junction              # 교차로 안은 차로가 겹쳐 방향 판정 생략
            if lat > 1.5 or wrong_way or crashed:
                place = "junction" if step - last_junction <= 2 * FPS else "lane"   # 교차로 안이나 지난 지 2초 이내
                interventions.append((round(step * DT, 1), "crash" if crashed else "wrong way" if wrong_way else
                                      "off lane", place))
                last_hit = hits[-1] if hits else -1
                t2 = wp.transform
                t2.location.z += 0.3
                ego.set_transform(t2)                                 # 차로 중앙에 되돌려 놓는다
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                integ = 0.0
                continue

            if policy is not None:
                steer = float(np.clip(policy(img, v), -1, 1))
                e = target - v
                integ = float(np.clip(integ + e * DT, -5, 5))
                u = 0.5 * e + 0.1 * integ
                ego.apply_control(carla.VehicleControl(throttle=float(np.clip(u, 0, 0.75)),
                                                       brake=float(np.clip(-u * 0.5, 0, 1)), steer=steer))
            else:
                steer = ego.get_control().steer
            if writer is not None:
                fv = qv.get(timeout=10)
                while fv.frame < fid:
                    fv = qv.get(timeout=10)
                vis = np.frombuffer(fv.raw_data, np.uint8).reshape(540, 960, 4)[:, :, :3].copy()
                cv2.putText(vis, f"{args.kind}  steer {steer:+.2f}  {v * 3.6:4.1f}km/h  "
                                 f"interventions {len(interventions)}  {dist / 1000:.2f}km",
                            (12, 30), 0, 0.7, (0, 255, 255), 2)
                vis[10:190, 630:950] = img
                writer.write(vis)
    finally:
        if writer is not None:
            writer.release()
        for a in [cam, col] + ([chase] if args.video else []):
            a.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in [cam, col, ego] + ([chase] if args.video else [])])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    km = dist / 1000
    print(f"[{args.kind} {Path(args.model).name if args.model else ''}] {args.town}, {args.minutes}분, {km:.2f}km")
    print(f"  개입 {len(interventions)}회 → 개입 1회당 {km / max(len(interventions), 1):.2f}km"
          f"{' (개입 없음)' if not interventions else ''}")
    print(f"  차로 중심과의 거리 평균 {lat_sum / steps:.2f}m, 차선에 닿은 시간 {touch / steps:.1%}")
    kinds, places = {}, {}
    for _, k, p in interventions:
        kinds[k] = kinds.get(k, 0) + 1
        places[p] = places.get(p, 0) + 1
    if interventions:
        print(f"  개입 원인: {kinds}, 장소: {places}, 처음 몇 번: {interventions[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--kind", default="bc", choices=["bc", "traj", "expert"])
    ap.add_argument("--model", default="")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--minutes", type=float, default=4)
    ap.add_argument("--speed", type=float, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    main(ap.parse_args())
