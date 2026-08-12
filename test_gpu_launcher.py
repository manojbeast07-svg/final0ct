from gpu_launcher import build_wsl_command, is_windows_incompatible_error, looks_like_image_path, looks_like_image_upload


def test_is_windows_incompatible_error_detects_winerror_193():
    exc = OSError(193, "not a valid Win32 application")
    exc.winerror = 193
    assert is_windows_incompatible_error(exc)


def test_build_wsl_command_quotes_paths():
    cmd = build_wsl_command("/mnt/c/Users/me/project", "/mnt/c/Users/me/project/temp_for_gpu/test.jpg")
    assert cmd[:4] == ["wsl.exe", "-d", "Ubuntu", "-e"]
    assert cmd[4] == "bash"
    assert cmd[5] == "-lc"
    assert "cd '/mnt/c/Users/me/project'" in cmd[6]
    assert "'/mnt/c/Users/me/project/clearscan' '/mnt/c/Users/me/project/temp_for_gpu/test.jpg'" in cmd[6]


def test_looks_like_image_path_detects_images():
    assert looks_like_image_path("BrokenBone5.jpg") is True
    assert looks_like_image_path("clearscan") is False


class FakeUpload:
    def __init__(self, name, mime_type, content):
        self.name = name
        self.type = mime_type
        self._content = content
        self._position = 0

    def seek(self, offset):
        self._position = offset

    def read(self, size=-1):
        if size < 0:
            size = len(self._content) - self._position
        chunk = self._content[self._position:self._position + size]
        self._position += len(chunk)
        return chunk


def test_looks_like_image_upload_detects_real_images_and_accepts_binaries():
    png_upload = FakeUpload("sample.png", "image/png", b"\x89PNG\r\n\x1a\nabc")
    assert looks_like_image_upload(png_upload) is True

    elf_upload = FakeUpload("clearscan", "application/octet-stream", b"\x7fELF\x02\x01\x01")
    assert looks_like_image_upload(elf_upload) is False
