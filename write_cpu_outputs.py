import cv2
import numpy as np
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
INPUT = DATA / 'BrokenBone1.jpg'
RUNTIME_DIR = Path(tempfile.gettempdir()) / 'clearscan'
CPU_OUTPUT_DIR = RUNTIME_DIR / 'cpu'
CPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Input not found:', INPUT)
    sys.exit(1)

# Gaussian
g = cv2.GaussianBlur(img, (3,3), 0.85)
cv2.imwrite(str(CPU_OUTPUT_DIR / 'gaussian_cpu.png'), g)
# Histogram
eq = cv2.equalizeHist(g)
cv2.imwrite(str(CPU_OUTPUT_DIR / 'histogram_cpu.png'), eq)
# Sharpen
kernel = np.array([
    [-0.25, -0.5, -0.25],
    [-0.5, 4.0, -0.5],
    [-0.25, -0.5, -0.25]
], dtype=np.float32)
sharpen = cv2.filter2D(eq, cv2.CV_8U, kernel, borderType=cv2.BORDER_REPLICATE)
cv2.imwrite(str(CPU_OUTPUT_DIR / 'sharpen_cpu.png'), sharpen)
# Edge
sob = cv2.Sobel(sharpen, cv2.CV_16S, 1, 1, ksize=3)
sob_abs = cv2.convertScaleAbs(sob)
cv2.imwrite(str(CPU_OUTPUT_DIR / 'edge_cpu.png'), sob_abs)
# Preview
combined_width = img.shape[1]*2
combined_height = img.shape[0]*2
combined = np.zeros((combined_height, combined_width), dtype=np.uint8)
combined[:img.shape[0], :img.shape[1]] = g
combined[:img.shape[0], img.shape[1]:] = eq
combined[img.shape[0]:, :img.shape[1]] = sharpen
combined[img.shape[0]:, img.shape[1]:] = sob_abs
cv2.imwrite(str(CPU_OUTPUT_DIR / 'preview.png'), combined)
print('Wrote CPU images to', CPU_OUTPUT_DIR)
