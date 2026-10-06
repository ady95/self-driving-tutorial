"""11-3: VLM 조언자 — 별도 스레드에서 몇 초마다 카메라 영상을 보고 위험도를 말합니다.

09-2에서 VLM은 답 하나에 약 3초가 걸리고, 가끔 없는 것을 봤습니다. 그래서 두 가지 규칙을 둡니다.
  1. 주행 루프는 VLM을 기다리지 않는다 (별도 스레드). 답이 오면 그때 반영한다
  2. VLM은 속도를 '낮추기만' 할 수 있다. 출발·가속·차선 변경은 결정하지 못한다
"""
import json
import re
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch09"))
from vlm import ask, make_client  # noqa: E402

PROMPT = """너는 자율주행차의 안전 보조 AI다. 전방 카메라 영상을 보고 아래 JSON 하나로만 답하라.
{"hazard": "none" | "low" | "high", "reason": 한 문장}
- high: 자차가 가려는 차로 위나 바로 옆에 사람, 멈춘 차, 공사 구간, 떨어진 물체가 있다
- low : 주의할 것이 근처에 있지만 자차의 길을 막지는 않는다
- none: 특별히 주의할 것이 없다
신호등은 판단하지 마라."""
CAP = {"high": 15 / 3.6, "low": 25 / 3.6}           # 위험도별 속도 상한 (m/s)
VALID_S = 4.0                                        # 답의 유효 시간 (시뮬레이션 초)


class VLMAdvisor:
    def __init__(self, backend="ollama", model=None, every_s=2.0):
        self.client, self.model = make_client(backend, model)
        self.api = "responses" if backend == "openai" else "chat"
        self.every_s, self.busy, self.last_submit = every_s, False, -1e9
        self.latest, self.pending, self.log = None, None, []  # latest = (답이 도착한 시뮬레이션 시각, 위험도, 이유)

    def maybe_submit(self, img, sim_t):
        if self.busy or sim_t - self.last_submit < self.every_s:
            return
        self.busy, self.last_submit = True, sim_t
        small = cv2.resize(img, (640, 360))
        threading.Thread(target=self._run, args=(small, sim_t), daemon=True).start()

    def _run(self, img, sim_t):
        start = time.perf_counter()
        try:
            text, _, _ = ask(self.client, self.model, img, PROMPT, api=self.api)
            m = re.search(r"\{.*\}", text, re.S)
            ans = json.loads(m.group(0)) if m else {}
        except Exception as e:                            # VLM이 실패해도 주행은 계속된다
            ans = {"hazard": "none", "reason": f"오류: {type(e).__name__}"}
        self.pending = (sim_t, ans.get("hazard", "none"), ans.get("reason", ""), time.perf_counter() - start)
        self.busy = False

    def speed_cap(self, sim_t):
        """지금 적용할 속도 상한 (없으면 None). 도착한 답을 받아 오고, 오래된 답은 버린다."""
        p = self.pending
        if p is not None:
            self.pending = None
            asked_t, hazard, reason, wall = p
            self.latest = (sim_t, hazard, reason)
            self.log.append({"asked_t": round(asked_t, 1), "answered_t": round(sim_t, 1), "wall_s": round(wall, 2),
                             "hazard": hazard, "reason": reason})
        if self.latest is None or sim_t - self.latest[0] > VALID_S:
            return None
        return CAP.get(self.latest[1])
