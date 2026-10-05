"""07-2: 가상 자동차를 만들고 센서를 달아, 직접 조작해 보고 자동 주행으로 바꿔 봅니다.

python ch07/drive_basic.py                    # 결과: outputs/ch07/drive_basic.mp4, 속도 기록
python ch07/drive_basic.py --tm-port 8300     # Traffic Manager 기본 포트(8000)가 이미 쓰이고 있을 때
"""
import argparse
import math
import queue
from pathlib import Path

import carla
import cv2
import numpy as np

OUT = Path("outputs/ch07")
OUT.mkdir(parents=True, exist_ok=True)
FPS = 20

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--tm-port", type=int, default=8000)
args = ap.parse_args()

client = carla.Client(args.host, 2000)
client.set_timeout(60.0)
world = client.load_world("Town03")
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / FPS
world.apply_settings(settings)
tm = client.get_trafficmanager(args.tm_port)
tm.set_synchronous_mode(True)

bp = world.get_blueprint_library()
spawn = world.get_map().get_spawn_points()[10]
actors = []

# 1. 차량 만들기
ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawn)
actors.append(ego)

# 2. 센서 달기: 3인칭 추적 카메라, LiDAR, 충돌 센서
cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "960")
cam_bp.set_attribute("image_size_y", "540")
chase = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-6.5, z=3.0), carla.Rotation(pitch=-12)),
                          attach_to=ego)
lidar_bp = bp.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("range", "50")
lidar_bp.set_attribute("rotation_frequency", str(FPS))
lidar_bp.set_attribute("points_per_second", "300000")
lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.2)), attach_to=ego)
collision = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
actors += [chase, lidar, collision]

q_cam, q_lidar = queue.Queue(), queue.Queue()
chase.listen(q_cam.put)
lidar.listen(q_lidar.put)
hits = []
collision.listen(lambda e: hits.append(e.other_actor.type_id))


def tick():
    frame = world.tick()
    img = q_cam.get(timeout=10)
    pts = q_lidar.get(timeout=10)
    assert img.frame == frame and pts.frame == frame          # 같은 순간의 데이터인지 확인
    return np.frombuffer(img.raw_data, np.uint8).reshape(540, 960, 4)[:, :, :3].copy(), len(pts)


def speed_kmh():
    v = ego.get_velocity()
    return math.hypot(v.x, v.y) * 3.6


writer = cv2.VideoWriter(str(OUT / "drive_basic_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))

# 3. 직접 조작: 4초 가속 → 2초 왼쪽으로 조향 → 2초 직진 → 3초 급제동
plan = [(4.0, carla.VehicleControl(throttle=0.6)),
        (2.0, carla.VehicleControl(throttle=0.4, steer=-0.25)),
        (2.0, carla.VehicleControl(throttle=0.4)),
        (3.0, carla.VehicleControl(brake=1.0))]
print(f"{'시간':>5}{'제어':>30}{'속도':>11}{'LiDAR 점':>10}")
step = 0
brake = None                                                   # (시작 위치, 시작 속도, 시작 스텝)
for duration, ctrl in plan:
    for _ in range(int(duration * FPS)):
        if ctrl.brake > 0 and brake is None:
            brake = (ego.get_location(), speed_kmh(), step)
        ego.apply_control(ctrl)
        img, n = tick()
        step += 1
        if brake is not None and len(brake) == 3 and speed_kmh() < 0.1:
            brake += (step,)                                   # 정지한 스텝
        cv2.putText(img, f"t={step / FPS:4.1f}s  {speed_kmh():5.1f} km/h  manual", (12, 30), 0, 0.8,
                    (255, 255, 255), 2)
        writer.write(img)
        if step % FPS == 0:                                    # 1초마다 출력
            label = f"thr={ctrl.throttle:.1f} steer={ctrl.steer:+.2f} brk={ctrl.brake:.1f}"
            print(f"{step / FPS:5.1f}{label:>30}{speed_kmh():8.1f}km/h{n:>10}")
loc0, v0, s0, s1 = brake
t_stop = (s1 - s0) / FPS
print(f"급제동: {v0:.1f}km/h에서 {t_stop:.2f}초 만에 정지, 제동 거리 {ego.get_location().distance(loc0):.1f}m, "
      f"평균 감속도 {v0 / 3.6 / t_stop / 9.81:.2f}g")

# 4. 자동 주행으로 전환: Traffic Manager에 맡긴다
ego.set_autopilot(True, tm.get_port())
moved, waited = 0.0, 0
prev = ego.get_location()
for _ in range(20 * FPS):
    img, _ = tick()
    step += 1
    moved += ego.get_location().distance(prev)
    prev = ego.get_location()
    red = ego.is_at_traffic_light() and ego.get_traffic_light_state() == carla.TrafficLightState.Red
    waited += red
    cv2.putText(img, f"t={step / FPS:4.1f}s  {speed_kmh():5.1f} km/h  autopilot{'  RED' if red else ''}",
                (12, 30), 0, 0.8, (0, 255, 255), 2)
    writer.write(img)
print(f"자동 주행 20초: {moved:.1f}m 이동, 빨간불 대기 {waited / FPS:.1f}초, 충돌 {len(hits)}회")

writer.release()
for a in actors:
    if a.type_id.startswith("sensor"):
        a.stop()
client.apply_batch([carla.command.DestroyActor(a) for a in actors])
settings.synchronous_mode = False
world.apply_settings(settings)
tm.set_synchronous_mode(False)
print(f"영상: {OUT / 'drive_basic_raw.mp4'}")
