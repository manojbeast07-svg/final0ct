import subprocess
import time
import shutil
from pathlib import Path
import cv2
import numpy as np
from timing_utils import parse_gpu_timing_output
from gpu_launcher import build_wsl_command, get_default_wsl_distro, resolve_gpu_executable

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
INPUT = DATA / 'BrokenBone1.jpg'
CPU_OUT = ROOT / 'temp_cpu_outputs'
GPU1_OUT = ROOT / 'temp_gpu1_outputs'
ENH_OUT = ROOT / 'temp_enhanced_outputs'
for d in (CPU_OUT, GPU1_OUT, ENH_OUT):
    d.mkdir(parents=True, exist_ok=True)

REPEATS = 3

# CPU average
print('Running CPU pipeline', REPEATS, 'times...')
def cpu_pipeline_once(inp, out_dir):
    img = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
    g = cv2.GaussianBlur(img, (3,3), 0.85)
    cv2.imwrite(str(out_dir / 'gaussian.png'), g)
    eq = cv2.equalizeHist(g)
    cv2.imwrite(str(out_dir / 'histogram.png'), eq)
    kernel = np.array([[-0.25,-0.5,-0.25],[-0.5,4.0,-0.5],[-0.25,-0.5,-0.25]], dtype=np.float32)
    sharpen = cv2.filter2D(eq, cv2.CV_8U, kernel, borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(out_dir / 'sharpen.png'), sharpen)
    sob = cv2.Sobel(sharpen, cv2.CV_8U, 1,1,ksize=3)
    cv2.imwrite(str(out_dir / 'edge.png'), sob)

cpu_times = []
for i in range(REPEATS):
    t0 = time.perf_counter()
    cpu_pipeline_once(INPUT, CPU_OUT)
    t1 = time.perf_counter()
    cpu_times.append((t1-t0)*1000.0)
print('CPU total mean (ms):', np.mean(cpu_times))

# Helper to run WSL binary and capture parsed timings and copy outputs

def run_wsl_binary(exe_path, dest_dir, repeats=1):
    stats = []
    last_out = ''
    for i in range(repeats):
        cmd = build_wsl_command(str(ROOT), str(INPUT), executable_path=str(exe_path), distro_name=get_default_wsl_distro())
        print('Running:', ' '.join(cmd))
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        out = proc.stdout + '\n' + proc.stderr
        last_out = out
        parsed = parse_gpu_timing_output(out)
        stats.append(parsed)
        # copy generated outputs to dest_dir with iteration suffix
        for name in ('gaussian.png','histogram.png','sharpen.png','edge.png'):
            src = ROOT / 'output' / name
            if src.exists():
                shutil.copy(src, dest_dir / f"{name.split('.')[0]}_{i}.png")
    # average stats
    averaged = {}
    keys = set(k for s in stats for k in s.keys())
    for k in keys:
        vals = [s[k] for s in stats if k in s]
        if vals:
            averaged[k] = float(np.mean(vals))
    averaged['sample_count'] = len(stats)
    return averaged, last_out

# Run GPU1
gpu1_exe = ROOT / 'gpu1'
if not gpu1_exe.exists():
    print('gpu1 not found, skipping')
    gpu1_stats = None
else:
    print('Running GPU1...')
    gpu1_stats, gpu1_out = run_wsl_binary(gpu1_exe, GPU1_OUT, repeats=REPEATS)
    print('GPU1 stats:', gpu1_stats)

# Run enhanced clearscan
enh_exe = resolve_gpu_executable(ROOT)
if enh_exe is None:
    print('clearscan exe not found, skipping')
    enh_stats = None
else:
    print('Running GPU Enhanced (clearscan)...')
    enh_stats, enh_out = run_wsl_binary(enh_exe, ENH_OUT, repeats=REPEATS)
    print('Enhanced stats:', enh_stats)

# Compute MAE/SSIM between CPU out and each GPU version (first repeat)
from skimage.metrics import structural_similarity as ssim

def load_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img

report = {}
# compare GPU1
if GPU1_OUT.exists() and any(GPU1_OUT.glob('gaussian_*.png')):
    report['gpu1'] = {}
    for stage in ['gaussian','histogram','sharpen','edge']:
        cpu_img = load_gray(CPU_OUT / f'{stage}.png')
        gpu_img = load_gray(GPU1_OUT / f'{stage}_0.png')
        if cpu_img is None or gpu_img is None:
            continue
        if cpu_img.shape != gpu_img.shape:
            gpu_img = cv2.resize(gpu_img, (cpu_img.shape[1], cpu_img.shape[0]))
        mae = float(np.abs(cpu_img.astype(float)-gpu_img.astype(float)).mean())
        psnr = float(cv2.PSNR(cpu_img, gpu_img))
        s = float(ssim(cpu_img, gpu_img))
        report['gpu1'][stage] = {'mae':mae,'psnr':psnr,'ssim':s}

if ENH_OUT.exists() and any(ENH_OUT.glob('gaussian_*.png')):
    report['enhanced'] = {}
    for stage in ['gaussian','histogram','sharpen','edge']:
        cpu_img = load_gray(CPU_OUT / f'{stage}.png')
        gpu_img = load_gray(ENH_OUT / f'{stage}_0.png')
        if cpu_img is None or gpu_img is None:
            continue
        if cpu_img.shape != gpu_img.shape:
            gpu_img = cv2.resize(gpu_img, (cpu_img.shape[1], cpu_img.shape[0]))
        mae = float(np.abs(cpu_img.astype(float)-gpu_img.astype(float)).mean())
        psnr = float(cv2.PSNR(cpu_img, gpu_img))
        s = float(ssim(cpu_img, gpu_img))
        report['enhanced'][stage] = {'mae':mae,'psnr':psnr,'ssim':s}

print('\nComparison report:')
print(report)

# Save report to file
import json
with open(ROOT / 'pipeline_report.json','w') as f:
    json.dump({'cpu_mean_ms':float(np.mean(cpu_times)),'gpu1':gpu1_stats,'enhanced':enh_stats,'metrics':report}, f, indent=2)
print('Saved pipeline_report.json')
