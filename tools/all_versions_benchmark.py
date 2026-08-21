import subprocess
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from gpu_launcher import build_wsl_command, get_default_wsl_distro
from timing_utils import parse_gpu_timing_output

STAGES = ('gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms')
images = sorted(p for p in (ROOT / 'data').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0, help='Only validate the first N images')
args = parser.parse_args()
if args.limit > 0:
    images = images[:args.limit]

def cpu_once(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    stages = {}
    start = time.perf_counter()
    gaussian = cv2.GaussianBlur(image, (3, 3), 0.85)
    stages['gaussian_ms'] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    histogram = cv2.equalizeHist(gaussian)
    stages['histogram_ms'] = (time.perf_counter() - start) * 1000.0
    kernel = np.array([[-0.25, -0.5, -0.25], [-0.5, 4.0, -0.5], [-0.25, -0.5, -0.25]], np.float32)
    start = time.perf_counter()
    sharpen = cv2.filter2D(histogram, cv2.CV_8U, kernel, borderType=cv2.BORDER_REPLICATE)
    stages['sharpen_ms'] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    cv2.Sobel(sharpen, cv2.CV_8U, 1, 1, ksize=3)
    stages['edge_ms'] = (time.perf_counter() - start) * 1000.0
    return stages

def gpu_median(path, executable, repeats=1):
    command = build_wsl_command(str(ROOT), str(path), executable_path=str(executable), distro_name=get_default_wsl_distro())
    warmup = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    if warmup.returncode != 0:
        raise RuntimeError(f'{executable.name} warm-up failed for {path.name}: {warmup.stdout}\n{warmup.stderr}')
    values = []
    for _ in range(repeats):
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f'{executable.name} failed for {path.name}: {result.stdout}\n{result.stderr}')
        parsed = parse_gpu_timing_output(result.stdout + '\n' + result.stderr)
        values.append({key: float(parsed[key]) for key in STAGES})
    return {key: float(np.median([sample[key] for sample in values])) for key in STAGES}

rows = []
for path in images:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    print(f'\n{path.name}: {image.shape[1]}x{image.shape[0]}')
    print('  order: CPU -> GPU1 -> Enhanced')
    cpu_samples = [cpu_once(path) for _ in range(2)]
    cpu = {key: float(np.median([sample[key] for sample in cpu_samples[1:]])) for key in STAGES}
    print('  CPU complete')
    gpu1 = gpu_median(path, ROOT / 'gpu1')
    print('  GPU1 complete')
    enhanced = gpu_median(path, ROOT / 'clearscan')
    print('  Enhanced complete')
    rows.append({'image': path.name, 'width': image.shape[1], 'height': image.shape[0], 'cpu': cpu, 'gpu1': gpu1, 'enhanced': enhanced})
    for key in STAGES:
        print(f'  {key}: CPU={cpu[key]:.4f} GPU1={gpu1[key]:.4f} Enhanced={enhanced[key]:.4f}')

print('\nAVERAGES')
averages = {}
for label in ('cpu', 'gpu1', 'enhanced'):
    averages[label] = {key: float(np.mean([row[label][key] for row in rows])) for key in STAGES}
    averages[label]['total_ms'] = sum(averages[label].values())
    print(f'{label}: {averages[label]}')

with open(ROOT / 'sequential_speed_report.json', 'w') as report_file:
    json.dump({'execution_order': ['CPU', 'GPU1', 'Enhanced'], 'rows': rows, 'averages': averages}, report_file, indent=2)
print('Saved sequential_speed_report.json')
