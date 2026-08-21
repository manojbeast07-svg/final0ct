import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent.parent
script = root / 'tools' / 'validate_all_inputs_outputs.py'
result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
lines = result.stdout.splitlines()
for line in lines:
    if line.startswith('VALIDATION RESULT:'):
        print(line)
if result.returncode != 0:
    print(result.stderr)
raise SystemExit(result.returncode)
