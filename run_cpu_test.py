import cv2
import numpy as np
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
RUNTIME_DIR = ROOT / 'temp_for_gpu'  # reuse an existing temp dir
CPU_OUTPUT_DIR = ROOT / 'temp_cpu_outputs'
CPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_path = DATA_DIR / 'BrokenBone1.jpg'
if not input_path.exists():
    raise SystemExit('Sample image not found: ' + str(input_path))

print('Running CPU pipeline on', input_path)
img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit('Failed to load image')

# Gaussian
t0 = time.perf_counter()
g = cv2.GaussianBlur(img, (3,3), 0.85)
t1 = time.perf_counter()
cv2.imwrite(str(CPU_OUTPUT_DIR / 'gaussian_cpu.png'), g)
print('Gaussian time (ms):', (t1-t0)*1000.0)

# Histogram
t0 = time.perf_counter()
eq = cv2.equalizeHist(g)
t1 = time.perf_counter()
cv2.imwrite(str(CPU_OUTPUT_DIR / 'histogram_cpu.png'), eq)
print('Histogram time (ms):', (t1-t0)*1000.0)

# Sharpen
sharpen_kernel = np.array([
    [-0.25, -0.5, -0.25],
    [-0.5, 4.0, -0.5],
    [-0.25, -0.5, -0.25]
], dtype=np.float32)

t0 = time.perf_counter()
sharpen = cv2.filter2D(eq, cv2.CV_8U, sharpen_kernel, borderType=cv2.BORDER_REPLICATE)
t1 = time.perf_counter()
cv2.imwrite(str(CPU_OUTPUT_DIR / 'sharpen_cpu.png'), sharpen)
print('Sharpen time (ms):', (t1-t0)*1000.0)

# Edge
t0 = time.perf_counter()
sob = cv2.Sobel(sharpen, cv2.CV_8U, 1, 1, ksize=3)
t1 = time.perf_counter()
cv2.imwrite(str(CPU_OUTPUT_DIR / 'edge_cpu.png'), sob)
print('Edge time (ms):', (t1-t0)*1000.0)

print('Saved CPU outputs to', CPU_OUTPUT_DIR)
