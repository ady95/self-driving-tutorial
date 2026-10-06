"""11-2: 카메라 신호 판독기를 채점할 데이터를 모읍니다.

Traffic Manager 자동 주행으로 Town03을 달리며, 내 차로 정지선이 50m 안에 있을 때마다 저장합니다.
  - 카메라 영상 (1280x720, 화각 90도)
  - 정답 신호 상태 (CARLA) — 채점용으로만 쓴다
  - 내 차로 신호등 등(head) 상자를 영상에 투영한 영역 (지도 정보)
정지해 있을 때는 같은 장면이 반복되므로 1초에 한 장만 저장합니다.

python ch11/collect_lights.py --tm-port 8300
결과: data/lights/<날씨>_NNNNN.jpg, data/lights/labels.csv
"""
import argparse
import csv
import json
import math
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lights import project_box  # noqa: E402

OUT = Path("data/lights")
WEATHERS = {"clear": carla.WeatherParameters.ClearNoon, "night": carla.WeatherParameters.ClearNight,
            "rain": carla.WeatherParameters.HardRainNoon}


def my_light(world, cmap, tf, max_dist=50.0):
    """내 차로 앞의 신호등과 정지선까지 거리 (지도 정보). 없으면 (None, inf)."""
    fwd, best = tf.get_forward_vector(), (None, math.inf)
    for tl in world.get_traffic_lights_from_waypoint(cmap.get_waypoint(tf.location), max_dist):
        for wp in tl.get_stop_waypoints():
            sl = wp.transform.location
            past = (tf.location.x - sl.x) * fwd.x + (tf.location.y - sl.y) * fwd.y       # 정지선 지나면 +
            side = abs(-(sl.x - tf.location.x) * fwd.y + (sl.y - tf.location.y) * fwd.x)
            if side <= 2.0 and past < 0 and -past < best[1]:
                best = (tl, -past)
    return best


def light_rois(tl, cam):
    """신호등의 등 상자들을 카메라 영상에 투영한다."""
    inv = np.array(cam.get_transform().get_inverse_matrix())
    rois = []
    for box in tl.get_light_boxes():
        corners = np.array([[c.x, c.y, c.z] for c in box.get_world_vertices(carla.Transform())])
        r = project_box(corners, inv)
        if r is not None:
            rois.append(r)
    return rois


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--minutes", type=float, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town03")
    cmap, bp = world.get_map(), world.get_blueprint_library()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(0)
    rows = []
    try:
        for weather, preset in WEATHERS.items():
            world.set_weather(preset)
            ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), cmap.get_spawn_points()[14])
            ego.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(ego, False)
            cb = bp.find("sensor.camera.rgb")
            cb.set_attribute("image_size_x", "1280")
            cb.set_attribute("image_size_y", "720")
            cam = world.spawn_actor(cb, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=ego)
            q = queue.Queue()
            cam.listen(q.put)
            last_saved = -999
            for step in range(int(args.minutes * 60 * 20)):
                fid = world.tick()
                d = q.get(timeout=10)
                while d.frame < fid:
                    d = q.get(timeout=10)
                tl, dist = my_light(world, cmap, ego.get_transform())
                v = ego.get_velocity()
                speed = math.hypot(v.x, v.y)
                gap = 10 if speed > 0.5 else 20                       # 달릴 때 0.5초, 서 있을 때 1초마다
                if tl is None or step - last_saved < gap:
                    continue
                last_saved = step
                name = f"{weather}_{step:05d}.jpg"
                cv2.imwrite(str(OUT / name), np.frombuffer(d.raw_data, np.uint8).reshape(720, 1280, 4)[:, :, :3])
                rows.append([name, weather, str(tl.get_state()).split(".")[-1].upper(), round(dist, 1),
                             round(speed * 3.6, 1), json.dumps(light_rois(tl, cam))])
            cam.stop()
            client.apply_batch([carla.command.DestroyActor(a) for a in [cam, ego]])
            world.tick()
    finally:
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "weather", "state", "dist_m", "speed_kmh", "rois"])
        w.writerows(rows)
    print(f"{len(rows)}장 저장: {OUT}")
