import random
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_launcher import build_wsl_command, get_default_wsl_distro
from timing_utils import parse_gpu_timing_output

ROOT = Path(__file__).parent.parent
images = sorted(p for p in (ROOT / 'data').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
rng = random.Random(21)
selected = images
all_timings = {'GPU1': {key: [] for key in ('gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms')},
               'ENHANCED': {key: [] for key in ('gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms')}}

for image in selected:
    print(f'\nIMAGE: {image.relative_to(ROOT)}')
    local = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    print(f'INPUT_SIZE: {local.shape[1]}x{local.shape[0]}')
    for label, executable in [('GPU1', ROOT / 'gpu1'), ('ENHANCED', ROOT / 'clearscan')]:
        command = build_wsl_command(str(ROOT), str(image), executable_path=str(executable), distro_name=get_default_wsl_distro())
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        output = result.stdout + '\n' + result.stderr
        parsed = parse_gpu_timing_output(output)
        for key in all_timings[label]:
            if key in parsed:
                all_timings[label][key].append(parsed[key])
        print(f'{label}: return_code={result.returncode} timings={parsed}')
        for stage in ('gaussian', 'histogram', 'sharpen', 'edge'):
            name = f'gpu1_{stage}.png' if label == 'GPU1' else f'{stage}.png'
            path = ROOT / 'output' / name
            image_out = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
            shape = None if image_out is None else f'{image_out.shape[1]}x{image_out.shape[0]}'
            print(f'  {name}: {shape}')

print('\nAVERAGE TIMINGS ACROSS ALL IMAGES:')
for key in ('gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms'):
    gpu1_values = all_timings['GPU1'][key]
    enhanced_values = all_timings['ENHANCED'][key]
    gpu1_average = sum(gpu1_values) / len(gpu1_values)
    enhanced_average = sum(enhanced_values) / len(enhanced_values)
    faster = 'GPU Enhanced' if enhanced_average < gpu1_average else 'GPU1'
    print(f'{key}: GPU1={gpu1_average:.4f} ms, Enhanced={enhanced_average:.4f} ms, faster={faster}')
