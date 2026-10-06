# self-driving-tutorial

위키독스 책 『자율주행 따라하기 — Computer Vision에서 VLM·VLA 기반 Physical AI까지』의 예제 코드입니다.

- 책: https://wikidocs.net/book/21521

## 빠른 시작

```bash
git clone https://github.com/ady95/self-driving-tutorial.git
cd self-driving-tutorial
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_samples.py           # 03장부터는 --carla 를 붙여 CARLA 데이터도 받기
python ch01/check_env.py
```

PyTorch는 GPU 환경에 맞춰 따로 설치합니다. 자세한 방법은 책의 01-3을 참고하세요.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `ch01/` | 01장 자율주행 시작하기 — 환경 점검, 첫 주행 영상 분석 |
| `ch02/` | 02장 Classical Vision — 영상 처리 기초, OpenCV 차선 인식, 실패시켜 보기 |
| `ch03/` | 03장 딥러닝 인지 — YOLO 검출, 주행 가능 영역·차선 분할, 객체 추적 |
| `ch04/` | 04장 거리와 공간 — 핀홀 카메라, 단안 깊이 추정, LiDAR, 카메라-LiDAR 퓨전 |
| `ch05/` | 05장 지도와 위치 — IPM·BEV, 점유 격자, 추측 항법·GPS·EKF, Visual Odometry |
| `ch06/` | 06장 판단과 제어 — 궤적 예측, Dijkstra·A*·RRT, PID·Pure Pursuit·MPC, 차선 유지 |
| `ch07/` | 07장 CARLA — 접속·속도 측정, 차량·센서 다루기, 실시간 인지 연결, Modular Driving Agent v1 |
| `ch08/` | 08장 End-to-End — CARLA 주행 데이터 수집(카메라 3대), Behavioral Cloning(조향), Transformer 궤적 예측, 개입 횟수 주행 평가 |
| `ch09/` | 09장 VLM·VLA — Ollama·OpenAI API로 주행 장면 질의응답, CARLA 정답으로 VLM 채점(환각·지연), 행동 토큰, 자전거 모델 World Model |
| `ch10/` | 10장 평가 — Driving Score로 Agent v1을 날씨·교통량·seed별 반복 평가 |
| `scripts/` | 샘플 데이터 다운로드 등 공통 도구 |
| `data/` | 내려받은 샘플 데이터 (git에 포함하지 않음) |
| `outputs/` | 실습 결과물 (git에 포함하지 않음) |

모든 명령은 저장소 루트에서 실행합니다.

## 데이터 출처

- 샘플 주행 영상: [Udacity CarND-LaneLines-P1](https://github.com/udacity/CarND-LaneLines-P1) (MIT License)
- CARLA 데이터: `scripts/record_carla_urban.py`(도심 장면), `scripts/record_carla_drive.py`(장거리 주행)로 생성, [CARLA Simulator](https://carla.org) 에셋 CC-BY 4.0
