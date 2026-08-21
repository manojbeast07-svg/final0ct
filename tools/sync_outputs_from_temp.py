from pathlib import Path
import shutil
ROOT=Path(__file__).parent.parent
out=ROOT/'output'
temp_g1=ROOT/'temp_gpu1_outputs'
temp_en=ROOT/'temp_enhanced_outputs'

stages=['gaussian','histogram','sharpen','edge']
for s in stages:
    g1=temp_g1/f"{s}_0.png"
    en=temp_en/f"{s}_0.png"
    if en.exists():
        shutil.copy(en, out/f"{s}.png")
        print('Copied', en, '->', out/f"{s}.png")
    if g1.exists():
        shutil.copy(g1, out/f"gpu1_{s}.png")
        print('Copied', g1, '->', out/f"gpu1_{s}.png")
print('Sync complete')
