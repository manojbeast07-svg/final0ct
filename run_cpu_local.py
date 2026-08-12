import cv2
import numpy as np
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

img_path = DATA_DIR / 'BrokenBone1.jpg'
if not img_path.exists():
    imgs = list(DATA_DIR.rglob('*.jpg'))
    if not imgs:
        raise SystemExit('No sample images found')
    img_path = imgs[0]

print('Using image:', img_path)
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
size = img.shape
print('Image size:', size)

times = {}

t0 = time.perf_counter()
g = cv2.GaussianBlur(img, (3, 3), 0.85)
t1 = time.perf_counter()
times['gaussian_ms'] = (t1 - t0) * 1000.0
cv2.imwrite(str(OUT / 'gaussian_cpu_run.png'), g)

t0 = time.perf_counter()
eq = cv2.equalizeHist(g)
t1 = time.perf_counter()
times['histogram_ms'] = (t1 - t0) * 1000.0
cv2.imwrite(str(OUT / 'histogram_cpu_run.png'), eq)

t0 = time.perf_counter()
blurred = cv2.GaussianBlur(eq, (3, 3), 0.85)
sharpen_kernel = np.array([
    [-0.25, -0.5, -0.25],
    [-0.5, 4.0, -0.5],
    [-0.25, -0.5, -0.25]
], dtype=np.float32)
sharpen = cv2.filter2D(eq, cv2.CV_8U, sharpen_kernel, borderType=cv2.BORDER_REPLICATE)
t1 = time.perf_counter()
times['sharpen_ms'] = (t1 - t0) * 1000.0
cv2.imwrite(str(OUT / 'sharpen_cpu_run.png'), sharpen)

t0 = time.perf_counter()
sob = cv2.Sobel(sharpen, cv2.CV_8U, 1, 1, ksize=3)
t1 = time.perf_counter()
times['edge_ms'] = (t1 - t0) * 1000.0
cv2.imwrite(str(OUT / 'edge_cpu_run.png'), sob)

combined = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
combined[:img.shape[0], :img.shape[1]] = g
combined[:img.shape[0], img.shape[1]:] = eq
combined[img.shape[0]:, :img.shape[1]] = sharpen
combined[img.shape[0]:, img.shape[1]:] = sob
cv2.imwrite(str(OUT / 'preview_cpu.png'), combined)

times['total_ms'] = sum(times.values())

print('Timings (ms):')
for k, v in times.items():
    print(f"  {k}: {v:.3f}")

print('\nSaved outputs:')
for p in OUT.glob('*cpu_run.png'):
    print(' -', p)
