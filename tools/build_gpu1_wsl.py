import subprocess

command = [
    r'C:\Windows\System32\wsl.exe', '-d', 'Ubuntu', '-e', 'bash', '-lc',
    "cd '/mnt/c/Users/manoj/Downloads/ClearScan-main/ClearScan-main' && make gpu1",
]
raise SystemExit(subprocess.run(command).returncode)
