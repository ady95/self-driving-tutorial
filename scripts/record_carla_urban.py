"""CARLA로 책의 도심 주행 샘플 데이터(carla_urban)를 만듭니다.

03~05장은 이 스크립트로 만든 데이터를 내려받아 쓰므로, 독자가 직접 실행할 필요는 없습니다.
07장에서 CARLA를 설치한 뒤 같은 데이터를 다시 만들어 보거나 조건을 바꿔 볼 수 있습니다.

같은 시드로 날씨만 바꿔 세 번 녹화하므로, 세 영상은 같은 장면·같은 교통 흐름을 담습니다.

python scripts/record_carla_urban.py --out data/carla_urban
"""
import argparse
import csv
import json
import math
import queue
import random
import shutil
import subprocess
from pathlib import Path

import carla
import cv2
import numpy as np

SEED = 7
FPS = 20
WARMUP, FRAMES = 60, 400                 # 3초 출발 대기 후 20초 녹화
GT_EVERY, DEPTH_EVERY = 5, 10            # 정답 레이블·깊이·LiDAR 저장 간격(프레임)
W, H, FOV = 1280, 720, 90
CAM_POS = carla.Transform(carla.Location(x=1.5, z=1.6))
LIDAR_POS = carla.Transform(carla.Location(x=0.0, z=2.2))
WEATHERS = {"clear": "ClearNoon", "night": "ClearNight", "rain": "HardRainNoon"}
OBJECT_TAGS = {12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus",
               18: "motorcycle", 19: "bicycle"}       # CARLA 의미 분할 클래스 번호
MIN_PIXELS = 150                                     # 이보다 작게 보이는 객체는 정답 상자에서 뺀다


def instance_boxes(inst_bgra):
    """인스턴스 분할 영상에서 객체별 (actor id, 클래스, 보이는 영역의 상자, 픽셀 수)를 뽑는다.
    R 채널 = 클래스 번호, G·B 채널 = 객체 고유 번호."""
    tag = inst_bgra[:, :, 2]
    oid = inst_bgra[:, :, 1].astype(np.int32) * 256 + inst_bgra[:, :, 0]
    boxes = []
    mask = np.isin(tag, list(OBJECT_TAGS))
    for i in np.unique(oid[mask]):
        ys, xs = np.nonzero((oid == i) & mask)
        if len(xs) < MIN_PIXELS:
            continue
        t = int(np.bincount(tag[ys, xs]).argmax())
        boxes.append((int(i), OBJECT_TAGS.get(t, str(t)), int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max()), len(xs)))
    return boxes


def setup_world(client, tm_port):
    world = client.load_world("Town10HD_Opt")
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(s)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(SEED)
    world.set_pedestrians_seed(SEED)
    random.seed(SEED)
    world.tick()
    return world, tm


def spawn_traffic(client, world, tm, n_vehicles=40, n_walkers=60):
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng = random.Random(SEED)
    ego_bp = bp.find("vehicle.lincoln.mkz_2020")
    ego_bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(ego_bp, spawns[0])

    car_bps = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    others = rng.sample(spawns[1:], n_vehicles)
    vehicles = []
    for sp in others:
        v = world.try_spawn_actor(rng.choice(car_bps), sp)
        if v:
            vehicles.append(v)

    walker_bps = bp.filter("walker.pedestrian.*")
    walkers, controllers = [], []
    ctrl_bp = bp.find("controller.ai.walker")
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
        if w:
            walkers.append(w)
    world.tick()
    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        controllers.append(c)
    world.tick()
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.2 + rng.random() * 0.6)

    for v in [ego] + vehicles:
        v.set_autopilot(True, tm.get_port())
    return ego, vehicles, walkers, controllers


def attach_sensors(world, ego):
    bp = world.get_blueprint_library()
    sensors, queues = {}, {}
    for name, kind in [("rgb", "sensor.camera.rgb"), ("sem", "sensor.camera.semantic_segmentation"),
                       ("inst", "sensor.camera.instance_segmentation"), ("depth", "sensor.camera.depth")]:
        b = bp.find(kind)
        b.set_attribute("image_size_x", str(W))
        b.set_attribute("image_size_y", str(H))
        b.set_attribute("fov", str(FOV))
        sensors[name] = world.spawn_actor(b, CAM_POS, attach_to=ego)
    b = bp.find("sensor.lidar.ray_cast")
    for k, v in {"channels": "64", "range": "80", "points_per_second": "1300000",
                 "rotation_frequency": str(FPS), "upper_fov": "10", "lower_fov": "-25"}.items():
        b.set_attribute(k, v)
    sensors["lidar"] = world.spawn_actor(b, LIDAR_POS, attach_to=ego)
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    return sensors, queues


def set_night_lights(world, vehicles, ego, tm):
    lights = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
    for v in vehicles:
        tm.update_vehicle_lights(v, True)
    ego.set_light_state(lights)


def record(client, out_root, weather_key, tm_port):
    world, tm = setup_world(client, tm_port)
    world.set_weather(getattr(carla.WeatherParameters, WEATHERS[weather_key]))
    ego, vehicles, walkers, controllers = spawn_traffic(client, world, tm)
    if weather_key == "night":
        set_night_lights(world, vehicles, ego, tm)
    sensors, queues = attach_sensors(world, ego)

    out = out_root / weather_key
    for sub in ["semantic", "depth", "lidar"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    tmp_video = out / "rgb_raw.mp4"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []
    box_rows = []

    try:
        for i in range(WARMUP + FRAMES):
            frame_id = world.tick()
            data = {}
            for name, q in queues.items():
                d = q.get(timeout=10.0)
                while d.frame < frame_id:            # 지난 프레임 데이터는 버린다
                    d = q.get(timeout=10.0)
                data[name] = d
            if i < WARMUP:
                continue
            k = i - WARMUP

            rgb = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(H, W, 4)[:, :, :3]
            writer.write(rgb)
            inst = np.frombuffer(data["inst"].raw_data, np.uint8).reshape(H, W, 4)
            box_rows += [(k,) + b for b in instance_boxes(inst)]
            if k % GT_EVERY == 0:
                sem = np.frombuffer(data["sem"].raw_data, np.uint8).reshape(H, W, 4)[:, :, 2]
                cv2.imwrite(str(out / "semantic" / f"{k:04d}.png"), sem)   # 픽셀 값 = CARLA 클래스 번호
            if k % DEPTH_EVERY == 0:
                d = np.frombuffer(data["depth"].raw_data, np.uint8).reshape(H, W, 4).astype(np.float32)
                meters = (d[:, :, 2] + d[:, :, 1] * 256 + d[:, :, 0] * 65536) / (256 ** 3 - 1) * 1000
                cv2.imwrite(str(out / "depth" / f"{k:04d}.png"),
                            np.clip(meters * 100, 0, 65535).astype(np.uint16))  # 센티미터, 16비트
                pts = np.frombuffer(data["lidar"].raw_data, np.float32).reshape(-1, 4)
                np.save(out / "lidar" / f"{k:04d}.npy", pts.astype(np.float16))

            t = ego.get_transform()
            v = ego.get_velocity()
            c = ego.get_control()
            rows.append([k, round(data["rgb"].timestamp, 3), round(t.location.x, 3), round(t.location.y, 3),
                         round(t.location.z, 3), round(t.rotation.roll, 3), round(t.rotation.pitch, 3),
                         round(t.rotation.yaw, 3), round(math.hypot(v.x, v.y) * 3.6, 2),
                         round(c.steer, 4), round(c.throttle, 4), round(c.brake, 4)])
    finally:
        writer.release()
        for s in sensors.values():
            s.stop()
            s.destroy()
        for c in controllers:
            c.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in controllers + walkers + vehicles + [ego]])
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        tm.set_synchronous_mode(False)

    with open(out / "ego.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "x", "y", "z", "roll", "pitch", "yaw", "speed_kmh",
                    "steer", "throttle", "brake"])
        w.writerows(rows)
    with open(out / "boxes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "object_id", "class", "x1", "y1", "x2", "y2", "pixels"])
        w.writerows(box_rows)

    # mp4v → h264 (용량을 줄이고 어디서나 재생되게)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(out / "rgb.mp4")], check=True)
    tmp_video.unlink()
    return rows


def write_calib(out_root):
    f = W / (2 * math.tan(math.radians(FOV) / 2))
    calib = {
        "image_size": [W, H], "fov_deg": FOV,
        "K": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
        "camera_in_vehicle": {"x": 1.5, "y": 0.0, "z": 1.6, "roll": 0, "pitch": 0, "yaw": 0},
        "lidar_in_vehicle": {"x": 0.0, "y": 0.0, "z": 2.2, "roll": 0, "pitch": 0, "yaw": 0},
        "carla_axes": "x 앞, y 오른쪽, z 위 (왼손 좌표계). 카메라 영상은 x 오른쪽, y 아래",
        "depth_png": "16비트, 센티미터 단위 (값 / 100 = 미터)",
        "semantic_png": "픽셀 값 = CARLA 의미 분할 클래스 번호",
        "boxes_csv": f"매 프레임 객체 정답 상자 (보이는 영역 기준, {MIN_PIXELS}픽셀 이상), object_id는 프레임을 넘어 유지",
        "fps": FPS, "gt_every": GT_EVERY, "depth_lidar_every": DEPTH_EVERY,
        "weathers": WEATHERS, "seed": SEED, "map": "Town10HD_Opt",
    }
    (out_root / "calib.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/carla_urban")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--weathers", default="clear,night,rain")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print("CARLA 서버:", client.get_server_version(), flush=True)
    write_calib(out_root)

    for key in args.weathers.split(","):
        if (out_root / key).exists():
            shutil.rmtree(out_root / key)          # 다시 녹화하는 날씨만 지운다
        rows = record(client, out_root, key, args.tm_port)
        xy = np.array([[r[2], r[3]] for r in rows])
        dist = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        print(f"[{key}] {len(rows)} 프레임, 주행 거리 {dist:.1f} m, "
              f"평균 속도 {np.mean([r[8] for r in rows]):.1f} km/h", flush=True)

    # 같은 시드로 녹화한 날씨끼리 자차 궤적이 같은지 확인한다
    tracks = {}
    for key in WEATHERS:
        f = out_root / key / "ego.csv"
        if f.exists():
            tracks[key] = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(2, 3))
    keys = list(tracks)
    for k in keys[1:]:
        diff = np.max(np.linalg.norm(tracks[k] - tracks[keys[0]], axis=1))
        print(f"재현성: {keys[0]} 대비 {k} 자차 위치 최대 차이 {diff:.3f} m", flush=True)
