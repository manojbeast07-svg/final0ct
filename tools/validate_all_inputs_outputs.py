import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from gpu_launcher import build_wsl_command, get_default_wsl_distro

STAGES = ('gaussian', 'histogram', 'sharpen', 'edge')
IMAGES = sorted(p for p in (ROOT / 'data').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cpu_reference(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.GaussianBlur(image, (3, 3), 0.85)
    histogram = cv2.equalizeHist(gaussian)
    kernel = np.array([[-0.25, -0.5, -0.25], [-0.5, 4.0, -0.5], [-0.25, -0.5, -0.25]], dtype=np.float32)
    sharpen = cv2.filter2D(histogram, cv2.CV_8U, kernel, borderType=cv2.BORDER_REPLICATE)
    edge = cv2.Sobel(sharpen, cv2.CV_8U, 1, 1, ksize=3)
    return dict(zip(STAGES, (gaussian, histogram, sharpen, edge)))

def compare(reference, actual):
    mae = float(np.abs(reference.astype(np.float32) - actual.astype(np.float32)).mean())
    return mae, float(ssim(reference, actual))

failures = []
print(f'Checking {len(IMAGES)} input images...')
for path in IMAGES:
    source_hash = sha256(path)
    source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    reference = cpu_reference(path)
    print(f'\n{path.relative_to(ROOT)}: {source.shape[1]}x{source.shape[0]}, sha256={source_hash[:16]}...')
    for label, executable in [('GPU1', ROOT / 'gpu1'), ('Enhanced', ROOT / 'clearscan')]:
        command = build_wsl_command(str(ROOT), str(path), executable_path=str(executable), distro_name=get_default_wsl_distro())
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(f'{path.name}/{label}: return code {result.returncode}')
            print(f'  {label}: FAILED return code {result.returncode}')
            continue
        metrics = []
        for stage in STAGES:
            filename = f'gpu1_{stage}.png' if label == 'GPU1' else f'{stage}.png'
            output_path = ROOT / 'output' / filename
            output = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            if output is None:
                failures.append(f'{path.name}/{label}/{stage}: missing output')
                metrics.append(f'{stage}=MISSING')
                continue
            if output.shape != source.shape:
                failures.append(f'{path.name}/{label}/{stage}: shape {output.shape} != {source.shape}')
            mae, score = compare(reference[stage], output) if output.shape == source.shape else (None, None)
            metrics.append(f'{stage} {output.shape[1]}x{output.shape[0]} MAE={mae:.3f} SSIM={score:.4f}')
        print(f'  {label}: OK; {len(metrics)}/{len(STAGES)} stages checked')

print('\nVALIDATION RESULT:', 'PASS' if not failures else 'FAIL')
if failures:
    for failure in failures:
        print(' -', failure)
