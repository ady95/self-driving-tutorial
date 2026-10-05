"""CARLA로 책의 장거리 주행 샘플 데이터(carla_drive)를 만듭니다.

05장(위치 추정·Visual Odometry)과 08장(End-to-End)에서 씁니다. carla_urban보다 길게(2분) 달리고,
카메라·IMU·자차 정답 궤적만 남겨 가볍게 만듭니다. 계속 움직이도록 자차는 신호를 무시합니다.

python scripts/record_carla_drive.py --out data/carla_drive --tm-port 8000
"""
import argparse
import csv
import math
import queue
import random
import shutil
import subprocess
from pathlib import Path

import carla
import cv2
import numpy as np

SEED = 11
FPS = 20
WARMUP, FRAMES = 60, 2400                # 3초 출발 대기 후 120초 녹화
W, H, FOV = 1280, 720, 90
CAM_POS = carla.Transform(carla.Location(x=1.5, z=1.6))
IMU_NOISE = {"noise_accel_stddev_x": "0.05", "noise_accel_stddev_y": "0.05", "noise_accel_stddev_z": "0.05",
             "noise_gyro_stddev_x": "0.005", "noise_gyro_stddev_y": "0.005", "noise_gyro_stddev_z": "0.005",
             "noise_gyro_bias_z": "0.002", "noise_seed": str(SEED)}


def main(args):
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world("Town10HD_Opt")
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(SEED)
    rng = random.Random(SEED)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    ego_bp = bp.find("vehicle.lincoln.mkz_2020")
    ego_bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(ego_bp, spawns[3])
    cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    others = [v for v in (world.try_spawn_actor(rng.choice(cars), sp)
                          for sp in rng.sample(spawns[4:], 30)) if v]
    for v in [ego] + others:
        v.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(ego, 100)        # 신호 대기 없이 계속 달리게 한다
    tm.vehicle_percentage_speed_difference(ego, -10)

    cam_bp = bp.find("sensor.camera.rgb")
    for k, v in {"image_size_x": W, "image_size_y": H, "fov": FOV}.items():
        cam_bp.set_attribute(k, str(v))
    imu_bp = bp.find("sensor.other.imu")
    for k, v in IMU_NOISE.items():
        imu_bp.set_attribute(k, v)
    cam = world.spawn_actor(cam_bp, CAM_POS, attach_to=ego)
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
    qc, qi = queue.Queue(), queue.Queue()
    cam.listen(qc.put)
    imu.listen(qi.put)

    tmp = out / "rgb_raw.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ego_rows, imu_rows = [], []
    try:
        for i in range(WARMUP + FRAMES):
            fid = world.tick()
            img = qc.get(timeout=10.0)
            while img.frame < fid:
                img = qc.get(timeout=10.0)
            m = qi.get(timeout=10.0)
            while m.frame < fid:
                m = qi.get(timeout=10.0)
            if i < WARMUP:
                continue
            k = i - WARMUP
            writer.write(np.frombuffer(img.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3])
            t, v, c = ego.get_transform(), ego.get_velocity(), ego.get_control()
            ego_rows.append([k, round(img.timestamp, 3), round(t.location.x, 3), round(t.location.y, 3),
                             round(t.location.z, 3), round(t.rotation.roll, 3), round(t.rotation.pitch, 3),
                             round(t.rotation.yaw, 3), round(math.hypot(v.x, v.y) * 3.6, 2),
                             round(c.steer, 4), round(c.throttle, 4), round(c.brake, 4)])
            imu_rows.append([k, round(m.timestamp, 3), round(m.accelerometer.x, 4), round(m.accelerometer.y, 4),
                             round(m.accelerometer.z, 4), round(m.gyroscope.x, 5), round(m.gyroscope.y, 5),
                             round(m.gyroscope.z, 5), round(m.compass, 5)])
            if k % 400 == 0:
                print(f"{k}/{FRAMES} 프레임", flush=True)
    finally:
        writer.release()
        for sensor in (cam, imu):
            sensor.stop()
            sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(a) for a in others + [ego]])
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        tm.set_synchronous_mode(False)

    with open(out / "ego.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "x", "y", "z", "roll", "pitch", "yaw", "speed_kmh",
                    "steer", "throttle", "brake"])
        w.writerows(ego_rows)
    with open(out / "imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "compass"])
        w.writerows(imu_rows)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(out / "rgb.mp4")], check=True)
    tmp.unlink()

    xy = np.array([[r[2], r[3]] for r in ego_rows])
    dist = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    yaw = np.unwrap(np.radians([r[7] for r in ego_rows]))
    print(f"{len(ego_rows)} 프레임, 주행 거리 {dist:.1f} m, 평균 속도 {np.mean([r[8] for r in ego_rows]):.1f} km/h, "
          f"누적 회전 {np.degrees(np.sum(np.abs(np.diff(yaw)))):.0f}도")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/carla_drive")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    main(ap.parse_args())
