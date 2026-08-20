import subprocess
from pathlib import Path
from gpu_launcher import build_wsl_command, get_default_wsl_distro, resolve_gpu_executable

ROOT = Path(__file__).parent
input_path = ROOT / 'data' / 'BrokenBone1.jpg'
exe = resolve_gpu_executable(ROOT)
print('Resolved executable:', exe)
if exe is None:
    raise SystemExit('No GPU executable found')

cmd = build_wsl_command(str(ROOT), str(input_path), executable_path=str(exe), distro_name=get_default_wsl_distro())
print('Running WSL command:')
print(' '.join(cmd))
proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
print('Return code:', proc.returncode)
print('STDOUT:')
print(proc.stdout)
print('STDERR:')
print(proc.stderr)
