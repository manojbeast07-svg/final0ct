from pathlib import Path
import cv2

ROOT=Path(__file__).parent.parent
out=ROOT/'output'
temp_cpu=ROOT/'temp_cpu_outputs'
temp_g1=ROOT/'temp_gpu1_outputs'
temp_en=ROOT/'temp_enhanced_outputs'

pairs=[
    ('output','gaussian.png','gpu1_gaussian.png','gaussian.png'),
    ('temp','temp_cpu_outputs/gaussian.png','temp_gpu1_outputs/gaussian_0.png','temp_enhanced_outputs/gaussian_0.png'),
]

print('Checking output folder images:')
for name in ['gaussian.png','gpu1_gaussian.png','gaussian.png','histogram.png','gpu1_histogram.png','histogram.png','sharpen.png','gpu1_sharpen.png','sharpen.png','edge.png','gpu1_edge.png','edge.png']:
    p=ROOT/'output'/name
    if p.exists():
        img=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if img is None:
            print(name, 'cannot read')
        else:
            print(name, img.shape)
    else:
        print(name, 'MISSING')

print('\nChecking temp folders (first files where available):')
for folder in [temp_cpu,temp_g1,temp_en]:
    print('Folder', folder.name)
    files=sorted([p for p in folder.glob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg']])
    if not files:
        print('  (no images)')
    else:
        for f in files[:5]:
            img=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)
            print(' ', f.name, img.shape if img is not None else 'cannot read')
