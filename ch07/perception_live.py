"""07-3: CARLA 주행 루프에 03~04장의 인지 모듈을 연결합니다.

매 스텝(0.05초): CARLA 카메라·LiDAR → YOLO 검출(03-2) → 도로·차선 분할(03-3에서 학습한 모델)
                → 차량 상자 + LiDAR 거리(04-4) → 결과 그림
차량은 자동 주행(Traffic Manager)에 맡기고, 인지만 붙여 봅니다. 의미 분할 정답 카메라로
03-3의 차선 모델이 처음 보는 도시(Town03)에서도 통하는지 채점합니다.

먼저 03-3의 python ch03/train_lane.py --aug 를 실행해 outputs/ch03/lane_lraspp_aug.pt 를 만들어 두세요.
python ch07/perception_live.py --tm-port 8300
"""
import argparse
import math
import queue
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
from train_lane import SIZE, build_model, predict  # noqa: E402

OUT = Path("outputs/ch07")
OUT.mkdir(parents=True, exist_ok=True)
FPS, W, H = 20, 1280, 720
K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])    # 화각 90도, 03~05장과 같은 카메라
CAM = carla.Transform(carla.Location(x=1.5, z=1.6))
LIDAR = carla.Transform(carla.Location(z=2.2))
VEHICLES = [2, 5, 7]


def lidar_uvd(raw):
    """LiDAR 점 → 화면 (u, v)와 앞 거리 d (04-4와 같은 계산)."""
    p = np.frombuffer(raw.raw_data, np.float32).reshape(-1, 4)[:, :3].copy()
    p[:, 0] -= 1.5                                  # LiDAR → 카메라 위치로 이동
    p[:, 2] += 2.2 - 1.6
    p = p[p[:, 0] > 0.5]
    u = K[0, 0] * p[:, 1] / p[:, 0] + K[0, 2]
    v = K[1, 1] * -p[:, 2] / p[:, 0] + K[1, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok], v[ok], p[ok, 0]


def box_distance(u, v, d, box):
    x1, y1, x2, y2 = box
    m = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return float(np.percentile(d[m], 30)) if m.sum() >= 3 else None


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO("yolo26n.pt")
    lane = build_model().to(device)
    lane.load_state_dict(torch.load("outputs/ch03/lane_lraspp_aug.pt", map_location=device))
    lane.eval()

    client = carla.Client(args.host, 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town03")
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / FPS
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(0)
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[10])
    rng = np.random.default_rng(0)
    others = [v for v in (world.try_spawn_actor(bp.filter("vehicle.*")[int(rng.integers(0, 30))], sp)
                          for sp in [spawns[int(i)] for i in rng.choice(np.arange(20, len(spawns)), 40, False)]) if v]
    for v in [ego] + others:
        v.set_autopilot(True, tm.get_port())

    sensors, queues = {}, {}
    for name, kind, tf in [("rgb", "sensor.camera.rgb", CAM), ("sem", "sensor.camera.semantic_segmentation", CAM),
                           ("lidar", "sensor.lidar.ray_cast", LIDAR)]:
        b = bp.find(kind)
        if name == "lidar":
            for k, val in {"channels": "64", "range": "80", "rotation_frequency": str(FPS),
                           "points_per_second": "1300000", "upper_fov": "10", "lower_fov": "-25"}.items():
                b.set_attribute(k, val)
        else:
            b.set_attribute("image_size_x", str(W))
            b.set_attribute("image_size_y", str(H))
        sensors[name] = world.spawn_actor(b, tf, attach_to=ego)
        queues[name] = queue.Queue()
        sensors[name].listen(queues[name].put)

    writer = cv2.VideoWriter(str(OUT / "perception_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    times = {"carla": [], "yolo": [], "lane": [], "fusion": [], "draw": []}
    inter, union = np.zeros(3), np.zeros(3)
    n_cars = n_with_dist = 0
    try:
        for step in range(args.seconds * FPS):
            t0 = time.perf_counter()
            fid = world.tick()
            data = {}
            for name, q in queues.items():
                d = q.get(timeout=10)
                while d.frame < fid:
                    d = q.get(timeout=10)
                data[name] = d
            frame = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
            t1 = time.perf_counter()
            r = det(frame, classes=VEHICLES + [0, 9], conf=0.25, verbose=False)[0]
            t2 = time.perf_counter()
            small = cv2.resize(frame, SIZE)
            seg = predict(lane, small, device)                         # 0 배경, 1 도로, 2 차선
            t3 = time.perf_counter()
            u, v, dist = lidar_uvd(data["lidar"])                     # 퓨전: 차량 상자 + LiDAR 거리
            dets = []
            for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
                box = tuple(map(int, b))
                dd = box_distance(u, v, dist, box) if int(c) in VEHICLES else None
                if int(c) in VEHICLES:
                    n_cars += 1
                    n_with_dist += dd is not None
                dets.append((det.names[int(c)] + (f" {dd:.1f}m" if dd is not None else ""), box))
            t4 = time.perf_counter()
            vis = frame.copy()                                          # 그리기 (결과 영상용)
            seg_full = cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            vis[seg_full == 1] = (0.6 * vis[seg_full == 1] + [0, 80, 0]).astype(np.uint8)
            vis[seg_full == 2] = (0, 0, 255)
            for label, (x1, y1, x2, y2) in dets:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 6), 0, 0.6, (255, 200, 0), 2)
            t5 = time.perf_counter()
            for key, val in zip(times, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]):
                times[key].append(val * 1000)
            if step % 5 == 0:                                           # 정답으로 차선 모델 채점
                tag = np.frombuffer(data["sem"].raw_data, np.uint8).reshape(H, W, 4)[:, :, 2]
                gt = np.zeros((H, W), np.uint8)
                gt[tag == 1], gt[tag == 24] = 1, 2
                gt = cv2.resize(gt, SIZE, interpolation=cv2.INTER_NEAREST)
                for c in range(3):
                    inter[c] += np.sum((seg == c) & (gt == c))
                    union[c] += np.sum((seg == c) | (gt == c))
            sp = ego.get_velocity()
            cv2.putText(vis, f"t={step / FPS:4.1f}s {math.hypot(sp.x, sp.y) * 3.6:4.1f}km/h", (12, 30), 0, 0.8,
                        (255, 255, 255), 2)
            writer.write(vis)
    finally:
        writer.release()
        for sensor in sensors.values():
            sensor.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in list(sensors.values()) + others + [ego]])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    print(f"{args.seconds}초 주행, 장치 {device}")
    total = np.zeros(len(times["carla"]))
    for key, val in times.items():
        total += np.array(val)
        print(f"  {key:<7}{np.median(val):6.1f} ms")
    print(f"  합계   {np.median(total):6.1f} ms (예산 {1000 / FPS:.0f} ms), 50ms 초과 스텝 {np.mean(total > 50):.0%}")
    print(f"차량 검출 {n_cars}개 중 LiDAR 거리를 붙인 것 {n_with_dist / max(n_cars, 1):.1%}")
    print(f"03-3 차선 모델 (Town10HD에서 학습) → Town03 채점: 도로 IoU {inter[1] / union[1]:.3f}, "
          f"차선 IoU {inter[2] / union[2]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--seconds", type=int, default=40)
    main(ap.parse_args())
