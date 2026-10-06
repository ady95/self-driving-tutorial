"""11-4: 시험 상황 만들기 — 경로 위 정해진 지점에서 보행자·고장 차·공사 구간을 등장시킵니다.

  pedestrian   : 자차가 35m 앞까지 오면 보행자가 오른쪽 인도에서 길을 건너기 시작한다
  stalled_car  : 자차가 25m 앞까지 오면 내 차로에 차가 나타나 비상등을 켜고 멈춘다 (고장 차)
  construction : 처음부터 내 차로에 공사 구간(라바콘·차단대)이 있다
모든 상황은 경로 점 번호(2m 간격)로 위치를 정하므로 같은 경로에서 매번 같은 곳에 나타납니다.
"""
import math

import carla

SPOTS = {"pedestrian": (1, 95), "stalled_car": (1, 350), "construction": (1, 205)}   # 상황: (경로, 경로 점 번호)
TRIGGER = {"pedestrian": 35.0, "stalled_car": 25.0}           # 자차가 이 거리(m)까지 오면 시작


def lane_point(cmap, route, k, side=0.0, z=0.0):
    """경로 점 k에서 오른쪽(+)으로 side m 옮긴 위치와 그 지점의 차로 방향(yaw)."""
    x, y, h = map(float, route[k])
    wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=h))
    yaw = math.radians(wp.transform.rotation.yaw)
    loc = carla.Location(x=x - math.sin(yaw) * side, y=y + math.cos(yaw) * side, z=h + z)
    return loc, wp.transform.rotation.yaw, wp


class Scenario:
    def __init__(self, name, world, cmap, route):
        self.name, self.world, self.cmap, self.route = name, world, cmap, route
        self.actors, self.triggered, self.walker, self.walk_steps = [], False, None, 0
        self.spot = SPOTS[name][1] if name in SPOTS else None
        if name == "construction":
            self._build_construction()

    def _spawn(self, bp_name, loc, yaw):
        lib = self.world.get_blueprint_library()
        found = lib.filter(bp_name)
        if not found:
            return None
        a = self.world.try_spawn_actor(found[0], carla.Transform(loc, carla.Rotation(yaw=yaw)))
        if a is not None:
            self.actors.append(a)
        return a

    def _build_construction(self):
        k = self.spot
        loc, yaw, wp = lane_point(self.cmap, self.route, k, z=0.1)
        self._spawn("static.prop.streetbarrier", loc, yaw + 90)                  # 차로를 가로막는 차단대
        for i in range(1, 7):                                                    # 12m에 걸친 라바콘 두 줄
            for side in (-0.9, 0.9):
                c, cy, _ = lane_point(self.cmap, self.route, k + i, side=side, z=0.1)
                self._spawn("static.prop.constructioncone", c, cy)

    def step(self, ego_loc):
        """매 스텝 호출. 자차가 지점에 다가오면 상황을 시작한다."""
        if self.spot is None:
            return
        x, y = float(self.route[self.spot][0]), float(self.route[self.spot][1])
        near = math.hypot(ego_loc.x - x, ego_loc.y - y) <= TRIGGER.get(self.name, 0)
        if self.name == "pedestrian":
            if not self.triggered and near:
                loc, yaw, wp = lane_point(self.cmap, self.route, self.spot, side=wp_half(self.cmap, self.route,
                                                                                     self.spot) + 1.5, z=1.0)
                self.walker = self._spawn("walker.pedestrian.0001", loc, yaw - 90)
                self.cross_dir = carla.Vector3D(math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)), 0)
                self.triggered = True
            if self.walker is not None and self.walk_steps < 20 * 9:            # 9초 동안 1.4m/s로 건넌다
                self.walker.apply_control(carla.WalkerControl(direction=self.cross_dir, speed=1.4))
                self.walk_steps += 1
            elif self.walker is not None:
                self.walker.apply_control(carla.WalkerControl(speed=0.0))
        elif self.name == "stalled_car" and not self.triggered and near:
            loc, yaw, _ = lane_point(self.cmap, self.route, self.spot, z=0.3)
            car = self._spawn("vehicle.audi.a2", loc, yaw)
            if car is not None:
                car.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LeftBlinker
                                                            | carla.VehicleLightState.RightBlinker))
                car.apply_control(carla.VehicleControl(hand_brake=True))
            self.triggered = True

    def cleanup(self):
        return list(self.actors)


def wp_half(cmap, route, k):
    """경로 점 k의 차로 폭의 절반 + 오른쪽에 같은 방향 차로가 있으면 그 폭까지 (인도까지의 거리)."""
    x, y, h = map(float, route[k])
    wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=h))
    half, r = wp.lane_width / 2, wp.get_right_lane()
    while r is not None and r.lane_type == carla.LaneType.Driving:
        half += r.lane_width
        r = r.get_right_lane()
    return half
