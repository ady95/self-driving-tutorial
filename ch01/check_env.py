"""01-3: 실습 환경 점검 — 버전과 GPU 사용 가능 여부를 출력합니다."""
import platform
import sys
import warnings

import cv2
import numpy as np

print(f"Python      : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
print(f"NumPy       : {np.__version__}")
print(f"OpenCV      : {cv2.__version__}")

try:
    import torch
except ImportError:
    print("PyTorch     : 설치되지 않음 (03장부터 필요)")
    sys.exit(0)

print(f"PyTorch     : {torch.__version__}")
warnings.filterwarnings("ignore", module="torch.cuda")   # 아래에서 직접 안내한다

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA        : 사용 가능 (PyTorch 빌드 CUDA {torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}       : {props.name}, {props.total_memory / 1024**3:.1f} GB, "
              f"compute capability {props.major}.{props.minor}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS         : 사용 가능 (Apple Silicon)")
else:
    device = torch.device("cpu")
    if "+cpu" in torch.__version__ and platform.system() == "Windows":
        print("GPU         : CPU 빌드가 설치됨 — NVIDIA GPU가 있다면 01-3의 Windows 설치 명령 참고")
    else:
        print("GPU         : 없음 — CPU로 실행합니다 (01~06장은 CPU로도 충분)")

try:
    x = torch.randn(1000, 1000, device=device)
    y = x @ x
    print(f"행렬 곱 테스트: {device} 에서 OK, 결과 크기 {tuple(y.shape)}")
except Exception as e:                 # 지원하지 않는 GPU면 여기서 실패한다
    print(f"행렬 곱 테스트: {device} 에서 실패 — {str(e).splitlines()[0]}")
    print(f"  이 PyTorch 빌드가 지원하는 GPU: {' '.join(torch.cuda.get_arch_list())}")
    print("  GPU가 오래된 세대라면 CUDA 12.6 빌드(cu126)를 설치하세요 (01-3 참고)")
    sys.exit(1)
