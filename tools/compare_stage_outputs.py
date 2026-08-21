import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

def metrics(a,b):
    if not a.exists() or not b.exists():
        return None
    A=cv2.imread(str(a),cv2.IMREAD_GRAYSCALE)
    B=cv2.imread(str(b),cv2.IMREAD_GRAYSCALE)
    if A is None or B is None:
        return None
    if A.shape!=B.shape:
        B=cv2.resize(B,(A.shape[1],A.shape[0]))
    mae=np.abs(A.astype(float)-B.astype(float)).mean()
    psnr=cv2.PSNR(A,B)
    s=ssim(A,B)
    return mae,psnr,s

ROOT=Path(__file__).parent.parent
cpu_dir=ROOT / 'temp_cpu_outputs'
out_dir=ROOT / 'output'
stages=[('gaussian','gaussian_cpu.png','gpu1_gaussian.png','gaussian.png'),
        ('histogram','histogram_cpu.png','gpu1_histogram.png','histogram.png'),
        ('sharpen','sharpen_cpu.png','gpu1_sharpen.png','sharpen.png'),
        ('edge','edge_cpu.png','gpu1_edge.png','edge.png')]

print('Comparing stages: CPU vs GPU1 and CPU vs GPU Enhanced')
for stage, cpu_name, gpu1_name, enh_name in stages:
    cpu_p=cpu_dir/cpu_name
    g1_p=out_dir/gpu1_name
    en_p=out_dir/enh_name
    m1=metrics(cpu_p,g1_p)
    m2=metrics(cpu_p,en_p)
    print('\nStage:', stage)
    print(' CPU vs GPU1:', m1)
    print(' CPU vs Enhanced:', m2)

print('\nFiles present in output/:')
for p in sorted(out_dir.glob('*')):
    print(' -', p.name)
