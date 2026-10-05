"""07-1: CARLA 서버에 접속해 기본 정보를 확인하고, 동기 모드에서 카메라 한 대의 속도를 잽니다.

python ch07/hello_carla.py                    # 같은 PC의 CARLA 서버 (localhost:2000)
python ch07/hello_carla.py <서버 주소>         # 다른 PC의 CARLA 서버
"""
import queue
import sys
import time

import carla

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
client = carla.Client(host, 2000)
client.set_timeout(60.0)
print(f"클라이언트 {client.get_client_version()}, 서버 {client.get_server_version()}")
maps = sorted(m.split("/")[-1] for m in client.get_available_maps())
print(f"맵 {len(maps)}개: {', '.join(maps)}")

t = time.perf_counter()
world = client.load_world("Town10HD_Opt")
print(f"Town10HD_Opt 불러오기: {time.perf_counter() - t:.1f}초")

bp = world.get_blueprint_library()
print(f"블루프린트: 차량 {len(bp.filter('vehicle.*'))}종, 보행자 {len(bp.filter('walker.pedestrian.*'))}종, "
      f"센서 {len(bp.filter('sensor.*'))}종")
print(f"차량 스폰 지점: {len(world.get_map().get_spawn_points())}개")

# 동기 모드: 클라이언트가 tick()을 부를 때만 시뮬레이션이 1스텝(0.05초) 진행된다
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), world.get_map().get_spawn_points()[0])
for w, h in [(640, 360), (1280, 720)]:
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(w))
    cam_bp.set_attribute("image_size_y", str(h))
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    for _ in range(20):                     # 준비 운동
        world.tick()
        q.get(timeout=30)
    t = time.perf_counter()
    for _ in range(100):
        world.tick()
        q.get(timeout=30)
    print(f"동기 모드 + 카메라 {w}x{h}: {100 / (time.perf_counter() - t):.1f} 스텝/초")
    cam.stop()
    cam.destroy()

ego.destroy()
settings.synchronous_mode = False                # 끝낼 때는 비동기로 되돌린다
world.apply_settings(settings)
