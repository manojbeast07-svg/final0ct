import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_launcher import build_wsl_command, get_default_wsl_distro

root = Path(__file__).parent.parent
command = build_wsl_command(str(root), str(root / 'data' / 'BrokenBone1.jpg'), executable_path=str(root / 'clearscan'), distro_name=get_default_wsl_distro())
try:
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, timeout=30)
    print('RETURN_CODE:', result.returncode)
    print(result.stdout)
    print(result.stderr)
except subprocess.TimeoutExpired as error:
    print('TIMEOUT after 30 seconds')
    print(error.stdout or '')
    print(error.stderr or '')
