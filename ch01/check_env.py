"""01-3: 실습 환경 점검 — 버전과 GPU 사용 가능 여부를 출력합니다."""
import platform
import sys

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
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA        : 사용 가능 (PyTorch 빌드 CUDA {torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}       : {props.name}, {props.total_memory / 1024**3:.1f} GB")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS         : 사용 가능 (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("GPU         : 없음 — CPU로 실행합니다 (01~06장은 CPU로도 충분)")

x = torch.randn(1000, 1000, device=device)
y = x @ x
print(f"행렬 곱 테스트: {device} 에서 OK, 결과 크기 {tuple(y.shape)}")
