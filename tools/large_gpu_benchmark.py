import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from gpu_launcher import build_wsl_command, get_default_wsl_distro
from timing_utils import parse_gpu_timing_output

source_path = ROOT / 'data' / 'BrokenBone1.jpg'
benchmark_path = ROOT / 'temp_large_benchmark.png'
source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
scale = 2
large = cv2.resize(source, (source.shape[1] * scale, source.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(str(benchmark_path), large)

print(f'Input: {source_path.name} -> benchmark: {large.shape[1]}x{large.shape[0]}')
print('Both binaries use:', benchmark_path.name)

for label, executable in [('GPU1', ROOT / 'gpu1'), ('GPU Enhanced', ROOT / 'clearscan')]:
    samples = []
    for run_index in range(4):
        command = build_wsl_command(str(ROOT), str(benchmark_path), executable_path=str(executable), distro_name=get_default_wsl_distro())
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'{label} failed on run {run_index}:\n{result.stdout}\n{result.stderr}')
        parsed = parse_gpu_timing_output(result.stdout + '\n' + result.stderr)
        if run_index > 0:
            samples.append(parsed)
    averages = {key: float(np.mean([sample[key] for sample in samples])) for key in ('gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms')}
    averages['total_ms'] = sum(averages.values())
    print(f'{label}:')
    for key, value in averages.items():
        print(f'  {key}: {value:.4f}')
