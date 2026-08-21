import subprocess

command = [
    r'C:\Windows\System32\wsl.exe', '-d', 'Ubuntu', '-e', 'bash', '-lc',
    "cd '/mnt/c/Users/manoj/Downloads/ClearScan-main/ClearScan-main' && make clean && make -j2 && make gpu1",
]
result = subprocess.run(command, text=True)
raise SystemExit(result.returncode)
