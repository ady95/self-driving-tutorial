"""08-2: End-to-End 학습용 주행 데이터를 CARLA에서 모읍니다.

Traffic Manager의 자동 주행을 '운전 선생님'으로 삼아, 카메라 영상과 그때의 조향값을 기록합니다.
  - 카메라 3대: 가운데, 왼쪽 0.8m, 오른쪽 0.8m (같은 높이·방향). 320x180, 화각 90도
  - 라벨: 조향·가속·제동, 속도, 자차 위치·방향 (08-4의 미래 궤적 정답용)
  - 학습 라벨을 '차로 따라가기'로 일관되게 하려고 차로 변경은 끄고, 계속 달리도록 신호는 무시합니다

python ch08/collect.py --town Town03 --minutes 10 --tm-port 8300
결과: data/e2e/Town03/{center,left,right}/NNNNN.jpg, data/e2e/Town03/labels.csv
"""
import argparse
import csv
import math
import queue
import random
from pathlib import Path

import carla
import cv2
import numpy as np

FPS, W, H = 20, 320, 180
CAMS = {"center": 0.0, "left": -0.8, "right": 0.8}          # 차량 중심 기준 좌우 위치 (m, 오른쪽 +)


def main(args):
    out = Path(args.out) / args.town
    for name in CAMS:
        (out / name).mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, 2000)
    client.set_timeout(120.0)
    world = client.load_world(args.town)
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / FPS
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    rng = random.Random(args.seed)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[0])
    cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    npcs = [v for v in (world.try_spawn_actor(rng.choice(cars), sp) for sp in spawns[1:args.npc + 1]) if v]
    for v in [ego] + npcs:
        v.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(ego, False)
    tm.ignore_lights_percentage(ego, 100)
    tm.vehicle_percentage_speed_difference(ego, -10)                 # 제한 속도보다 10% 빠르게

    sensors, queues = {}, {}
    for name, y in CAMS.items():
        b = bp.find("sensor.camera.rgb")
        b.set_attribute("image_size_x", str(W))
        b.set_attribute("image_size_y", str(H))
        b.set_attribute("fov", "90")
        sensors[name] = world.spawn_actor(b, carla.Transform(carla.Location(x=1.5, y=y, z=1.6)), attach_to=ego)
        queues[name] = queue.Queue()
        sensors[name].listen(queues[name].put)

    rows, stuck = [], 0
    n_frames = int(args.minutes * 60 * FPS)
    try:
        for i in range(60 + n_frames):
            fid = world.tick()
            imgs = {}
            for name, q in queues.items():
                d = q.get(timeout=10)
                while d.frame < fid:
                    d = q.get(timeout=10)
                imgs[name] = np.frombuffer(d.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3]
            if i < 60:
                continue
            k = i - 60
            tf, vel, c = ego.get_transform(), ego.get_velocity(), ego.get_control()
            speed = math.hypot(vel.x, vel.y)
            stuck = stuck + 1 if speed < 0.5 else 0
            if stuck > 200:                                          # 10초 넘게 갇히면 다른 곳으로 옮긴다
                ego.set_transform(rng.choice(spawns))
                stuck = 0
            for name, img in imgs.items():
                cv2.imwrite(str(out / name / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append([k, round(c.steer, 4), round(c.throttle, 3), round(c.brake, 3), round(speed, 3),
                         round(tf.location.x, 3), round(tf.location.y, 3), round(tf.rotation.yaw, 3)])
            if k % 2400 == 0:
                print(f"{args.town}: {k}/{n_frames}", flush=True)
    finally:
        for sen in sensors.values():
            sen.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in list(sensors.values()) + npcs + [ego]])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "steer", "throttle", "brake", "speed", "x", "y", "yaw"])
        w.writerows(rows)
    steer = np.array([r[1] for r in rows])
    print(f"{args.town}: {len(rows)} 프레임, 조향 |값| 평균 {np.abs(steer).mean():.3f}, "
          f"|조향|>0.1 비율 {np.mean(np.abs(steer) > 0.1):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--npc", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/e2e")
    main(ap.parse_args())
