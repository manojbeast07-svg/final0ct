import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

ROOT=Path(__file__).parent.parent
out_dir=ROOT / 'output'

stages=[('gaussian','gpu1_gaussian.png','gaussian.png'),
        ('histogram','gpu1_histogram.png','histogram.png'),
        ('sharpen','gpu1_sharpen.png','sharpen.png'),
        ('edge','gpu1_edge.png','edge.png')]

print('Comparing GPU1 vs GPU Enhanced (MAE, PSNR, SSIM):')
for stage, g1_name, enh_name in stages:
    g1_p=out_dir/g1_name
    en_p=out_dir/enh_name
    if not g1_p.exists() or not en_p.exists():
        print(stage, 'MISSING FILES:', g1_p.exists(), en_p.exists())
        continue
    A=cv2.imread(str(g1_p),cv2.IMREAD_GRAYSCALE)
    B=cv2.imread(str(en_p),cv2.IMREAD_GRAYSCALE)
    if A is None or B is None:
        print(stage, 'could not read')
        continue
    if A.shape!=B.shape:
        B=cv2.resize(B,(A.shape[1],A.shape[0]))
    mae=np.abs(A.astype(float)-B.astype(float)).mean()
    psnr=cv2.PSNR(A,B)
    s=ssim(A,B)
    print(f'{stage}: MAE={mae:.6f}, PSNR={psnr:.6f}, SSIM={s:.6f}')
