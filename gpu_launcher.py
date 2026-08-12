import os
import platform
import subprocess
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_windows_incompatible_error(exc):
    if exc is None:
        return False
    winerror = getattr(exc, "winerror", None)
    if winerror == 193:
        return True
    msg = str(exc).lower()
    return "not a valid win32 application" in msg or "win32 application" in msg


def build_wsl_command(project_dir, input_path, executable_path="clearscan", distro_name=None):
    def normalize_path(path):
        value = str(path).replace('\\', '/')
        if value.startswith('/'):
            return value
        return str(Path(value).resolve()).replace('\\', '/')

    project_dir = normalize_path(project_dir)
    input_path = normalize_path(input_path)
    executable_path = str(executable_path).replace('\\', '/')

    def to_wsl_path(path):
        if len(path) >= 3 and path[1:3] == ':/':
            return f"/mnt/{path[0].lower()}{path[2:]}"
        return path

    linux_project = to_wsl_path(project_dir)
    linux_input = to_wsl_path(input_path)

    linux_executable = executable_path
    if len(linux_executable) >= 3 and linux_executable[1:3] == ':/':
        linux_executable = to_wsl_path(linux_executable)
    elif not linux_executable.startswith('/'):
        linux_executable = f"{linux_project}/{linux_executable}"

    distro = distro_name or get_default_wsl_distro()
    command = [
        "wsl.exe",
        "-d",
        distro,
        "-e",
        "bash",
        "-lc",
        f"cd '{linux_project}' && '{linux_executable}' '{linux_input}'",
    ]
    return command


def resolve_gpu_executable(root: Path):
    linux_exe = root / "clearscan"
    if linux_exe.exists():
        return linux_exe
    if os.name == "nt":
        win_exe = root / "clearscan.exe"
        if win_exe.exists():
            return win_exe
    return None


def get_default_wsl_distro() -> str:
    try:
        result = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, text=True, check=True)
        lines = [d for d in result.stdout.splitlines() if d.strip()]
        distros = [d.replace("\x00", "").strip() for d in lines if d.strip()]
        if not distros:
            return "Ubuntu"
        for distro in distros:
            if distro.lower().startswith("ubuntu"):
                return distro
        return distros[0]
    except Exception:
        return "Ubuntu"


def should_use_wsl(root: Path):
    if os.name != "nt":
        return False
    return (root / "clearscan").exists()


def looks_like_image_path(path):
    return Path(str(path)).suffix.lower() in IMAGE_EXTENSIONS


def looks_like_image_upload(uploaded_file):
    name = getattr(uploaded_file, "name", "") or ""
    if looks_like_image_path(name):
        return True

    mime_type = (getattr(uploaded_file, "type", "") or "").lower()
    if mime_type.startswith("image/"):
        return True

    try:
        uploaded_file.seek(0)
    except Exception:
        return False

    header = uploaded_file.read(16)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if not header:
        return False

    return (
        header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith(b"\xff\xd8\xff")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or header.startswith(b"BM")
        or header.startswith((b"II*\x00", b"MM\x00*"))
        or (header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP")
    )
