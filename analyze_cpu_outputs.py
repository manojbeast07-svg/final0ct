import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'

# choose input image used by run_cpu_local.py
img_path = DATA / 'BrokenBone1.jpg'
if not img_path.exists():
    imgs = list(DATA.rglob('*.jpg'))
    if not imgs:
        raise SystemExit('No input image')
    img_path = imgs[0]

def load_gray(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)

orig = load_gray(img_path)
stages = [
    ('gaussian_cpu_run.png', 'Gaussian Blur'),
    ('histogram_cpu_run.png', 'Histogram EQ'),
    ('sharpen_cpu_run.png', 'Sharpen'),
    ('edge_cpu_run.png', 'Edge Detect')
]

def mae(a,b):
    return float(np.mean(np.abs(a-b)))

def psnr(a,b):
    return float(cv2.PSNR(a, b))

report = []
for fname, label in stages:
    p = OUT / fname
    if not p.exists():
        print('Missing', p)
        continue
    img = load_gray(p)
    if img.shape != orig.shape:
        img = cv2.resize(img, (orig.shape[1], orig.shape[0]))

    m = mae(orig, img)
    pnr = psnr(orig, img)
    s = ssim(orig.astype('uint8'), img.astype('uint8'))
    report.append((label, fname, m, pnr, s))

print('Comparison of original -> stage outputs')
print('Stage, file, MAE, PSNR (dB), SSIM')
for r in report:
    print(f'{r[0]}, {r[1]}, {r[2]:.3f}, {r[3]:.3f}, {r[4]:.4f}')

csv = ROOT / 'cpu_comparison.csv'
with open(csv, 'w') as f:
    f.write('stage,file,mae,psnr,ssim\n')
    for r in report:
        f.write(f'{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n')

print('\nSaved CSV:', csv)
