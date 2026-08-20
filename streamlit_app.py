import streamlit as st
import cv2
import numpy as np
import subprocess
import time
import os
import tempfile
import zipfile
import io
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from timing_utils import STAGE_KEYS, parse_gpu_timing_output, build_gpu_stats_from_cpu, summarize_gpu_comparison, summarize_previous_gpu_vs_cpu
from gpu_launcher import build_wsl_command, get_default_wsl_distro, is_windows_incompatible_error, looks_like_image_path, looks_like_image_upload, resolve_gpu_executable, should_use_wsl

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"
RUNTIME_DIR = Path(tempfile.gettempdir()) / "clearscan"
CPU_OUTPUT_DIR = RUNTIME_DIR / "cpu"
GPU1_OUTPUT_DIR = RUNTIME_DIR / "gpu1"

if "cpu_times" not in st.session_state:
    st.session_state.cpu_times = None
if "gpu_stats" not in st.session_state:
    st.session_state.gpu_stats = None
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = {}

STAGE_FILES = [
    ("gaussian", "Gaussian Blur", "gaussian_cpu.png", "gaussian.png", "gaussian_ms"),
    ("histogram", "Histogram Equalization", "histogram_cpu.png", "histogram.png", "histogram_ms"),
    ("sharpen", "Sharpen", "sharpen_cpu.png", "sharpen.png", "sharpen_ms"),
    ("edge", "Edge Detection", "edge_cpu.png", "edge.png", "edge_ms"),
]

DEFAULT_PREVIOUS_GPU_BASELINE = {
    "gaussian_ms": 2.0326,
    "histogram_ms": 0.3052,
    "sharpen_ms": 0.0512,
    "edge_ms": 0.0451,
}

if "previous_gpu_baseline" not in st.session_state:
    st.session_state.previous_gpu_baseline = DEFAULT_PREVIOUS_GPU_BASELINE.copy()
if "compare_previous_gpu_only" not in st.session_state:
    st.session_state.compare_previous_gpu_only = False

STAGE_LABELS = {
    "gaussian_ms": "Gaussian Blur",
    "histogram_ms": "Histogram Equalization",
    "sharpen_ms": "Sharpen",
    "edge_ms": "Edge Detection",
}

st.title("ClearScan: CPU vs GPU pipeline comparison")

# Sidebar controls for interactivity
st.sidebar.header("Run controls")
repeats = st.sidebar.slider("CUDA samples (repeats)", min_value=1, max_value=10, value=3)
run_cpu_checkbox = st.sidebar.checkbox("Run CPU baseline", value=True)
run_gpu1_checkbox = st.sidebar.checkbox("Run GPU1 (simple CUDA)", value=True)
run_enhanced_checkbox = st.sidebar.checkbox("Run GPU Enhanced (optimized CUDA)", value=True)
auto_run = st.sidebar.checkbox("Auto-run on image select", value=False)
show_logs = st.sidebar.checkbox("Show verbose logs", value=False)
uploaded_bin_sidebar = st.sidebar.file_uploader("Upload GPU binary (optional)", type=None)
binary_path_input = st.sidebar.text_input("GPU binary path (optional)", value=str(ROOT / "clearscan"))
run_button = st.sidebar.button("Run Selected Pipelines")
live_preview = st.sidebar.button("Live CPU Preview")

uploaded = st.file_uploader("Upload an X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])
samples = [p.name for p in DATA_DIR.rglob('*.png')] + [p.name for p in DATA_DIR.rglob('*.jpg')]
if samples:
    samples = sorted(list(set(samples)))
    choice = st.selectbox("Or pick a sample image", ["-- none --"] + samples)
else:
    choice = "-- none --"

if uploaded is None and choice == "-- none --":
    st.info("Upload an image or select a sample to begin.")
    st.stop()

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if uploaded:
    input_path = RUNTIME_DIR / "inputs" / Path(uploaded.name).name
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded.read())
else:
    input_path = next(DATA_DIR.rglob(choice))

st.write("Input:", input_path.name)
st.image(str(input_path), width="stretch")

# area for logs and progress
log_area = st.empty()
progress_bar = st.progress(0)

def time_cpu_pipeline(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    results = {}
    CPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    g = cv2.GaussianBlur(img, (3, 3), 0.85)
    t1 = time.perf_counter()
    results['gaussian_ms'] = (t1 - t0) * 1000.0
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'gaussian.png'), g)

    t0 = time.perf_counter()
    eq = cv2.equalizeHist(g)
    t1 = time.perf_counter()
    results['histogram_ms'] = (t1 - t0) * 1000.0
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'histogram.png'), eq)

    t0 = time.perf_counter()
    blurred = cv2.GaussianBlur(eq, (3, 3), 0.85)
    sharpen_kernel = np.array([
        [-0.25, -0.5, -0.25],
        [-0.5, 4.0, -0.5],
        [-0.25, -0.5, -0.25]
    ], dtype=np.float32)
    sharpen = cv2.filter2D(eq, cv2.CV_8U, sharpen_kernel, borderType=cv2.BORDER_REPLICATE)
    t1 = time.perf_counter()
    results['sharpen_ms'] = (t1 - t0) * 1000.0
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'sharpen.png'), sharpen)

    t0 = time.perf_counter()
    sob = cv2.Sobel(sharpen, cv2.CV_8U, 1, 1, ksize=3)
    t1 = time.perf_counter()
    results['edge_ms'] = (t1 - t0) * 1000.0
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'edge.png'), sob)

    combined_width = img.shape[1] * 2
    combined_height = img.shape[0] * 2
    combined = np.zeros((combined_height, combined_width), dtype=np.uint8)
    combined[:img.shape[0], :img.shape[1]] = g
    combined[:img.shape[0], img.shape[1]:] = eq
    combined[img.shape[0]:, :img.shape[1]] = sharpen
    combined[img.shape[0]:, img.shape[1]:] = sob
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'preview.png'), combined)

    results['total_ms'] = sum([results[k] for k in results])
    return results


def time_cpu_pipeline_average(path, repeats=5):
    samples = [time_cpu_pipeline(path) for _ in range(repeats)]
    averaged = {}
    for key in samples[0]:
        averaged[key] = float(np.mean([sample[key] for sample in samples]))
    averaged["sample_count"] = repeats
    return averaged


def time_cpu_preview(path, sigma=0.85, sharpen_strength=1.0, edge_threshold=0):
    """Quick CPU preview that uses adjustable parameters and writes CPU_OUTPUT_DIR images."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    CPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gaussian with adjustable sigma
    t0 = time.perf_counter()
    g = cv2.GaussianBlur(img, (3, 3), sigma)
    t1 = time.perf_counter()
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'gaussian_cpu.png'), g)

    # Histogram equalization
    eq = cv2.equalizeHist(g)
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'histogram_cpu.png'), eq)

    # Sharpen: scale the kernel center by sharpen_strength
    kernel = np.array([
        [-0.25 * sharpen_strength, -0.5 * sharpen_strength, -0.25 * sharpen_strength],
        [-0.5 * sharpen_strength, 4.0 * sharpen_strength, -0.5 * sharpen_strength],
        [-0.25 * sharpen_strength, -0.5 * sharpen_strength, -0.25 * sharpen_strength]
    ], dtype=np.float32)
    sharpen = cv2.filter2D(eq, cv2.CV_8U, kernel, borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'sharpen_cpu.png'), sharpen)

    # Edge detection (Sobel) and optional thresholding
    sob = cv2.Sobel(sharpen, cv2.CV_16S, 1, 1, ksize=3)
    sob_abs = cv2.convertScaleAbs(sob)
    if edge_threshold > 0:
        _, sob_thr = cv2.threshold(sob_abs, int(edge_threshold), 255, cv2.THRESH_BINARY)
        sob_out = sob_thr
    else:
        sob_out = sob_abs
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'edge_cpu.png'), sob_out)

    # Combined preview
    combined_width = img.shape[1] * 2
    combined_height = img.shape[0] * 2
    combined = np.zeros((combined_height, combined_width), dtype=np.uint8)
    combined[:img.shape[0], :img.shape[1]] = g
    combined[:img.shape[0], img.shape[1]:] = eq
    combined[img.shape[0]:, :img.shape[1]] = sharpen
    combined[img.shape[0]:, img.shape[1]:] = sob_out
    cv2.imwrite(str(CPU_OUTPUT_DIR / 'preview.png'), combined)
    return {
        'gaussian_ms': (t1 - t0) * 1000.0,
    }


def format_bytes(value: float) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1024**2:.3f} MB"


def format_bytes_per_ms(bytes_value: float, ms_value: float) -> str:
    if bytes_value is None or ms_value is None or ms_value == 0:
        return "N/A"
    return f"{bytes_value / 1024**2 / ms_value:.4f} MB/ms"


def run_cuda_binary(binary_path, input_path, output_prefix=""):
    if os.name == "nt" and Path(binary_path).suffix.lower() != ".exe":
        command = build_wsl_command(str(ROOT), str(input_path), executable_path=binary_path, distro_name=get_default_wsl_distro())
    else:
        command = [str(binary_path), str(input_path)]
    process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    output = process.stdout + "\n" + process.stderr
    parsed = parse_gpu_timing_output(output)
    if len(parsed) < 4:
        raise RuntimeError(output.strip() or "CUDA binary produced no timing output")
    return parsed, output


def run_cuda_binary_average(binary_path, input_path, repeats=5):
    samples = []
    last_output = ""
    for _ in range(repeats):
        parsed, last_output = run_cuda_binary(binary_path, input_path)
        samples.append(parsed)
    averaged = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            averaged[key] = float(np.mean(values))
    averaged["sample_count"] = repeats
    return averaged, last_output


def output_metrics(cpu_path, gpu_path):
    cpu_image = cv2.imread(str(cpu_path), cv2.IMREAD_GRAYSCALE)
    gpu_image = cv2.imread(str(gpu_path), cv2.IMREAD_GRAYSCALE)
    if cpu_image is None or gpu_image is None:
        return {"mae": None, "psnr": None, "ssim": None}
    if cpu_image.shape != gpu_image.shape:
        gpu_image = cv2.resize(gpu_image, (cpu_image.shape[1], cpu_image.shape[0]))
    return {
        "mae": float(np.abs(cpu_image.astype(np.float32) - gpu_image.astype(np.float32)).mean()),
        "psnr": float(cv2.PSNR(cpu_image, gpu_image)),
        "ssim": float(ssim(cpu_image, gpu_image)),
    }


st.header("Automatic CPU vs GPU demonstration")
st.caption("Selecting one image runs the CPU baseline, simple CUDA GPU1, and optimized CUDA GPU Enhanced pipelines.")
image_id = f"{input_path.resolve()}:{input_path.stat().st_mtime_ns}"

# Decide whether to run automatically or on user request
should_run = run_button or (auto_run and (
    st.session_state.get("last_image_id") != image_id
    or st.session_state.get("gpu1_stats") is None
    or st.session_state.get("gpu_enhanced_stats") is None
))

if should_run:
    with st.spinner("Running selected pipelines..."):
        progress_bar.progress(5)
        # CPU
        if run_cpu_checkbox:
            cpu_times = time_cpu_pipeline_average(input_path, repeats=repeats)
            st.session_state.cpu_times = cpu_times
        else:
            cpu_times = st.session_state.get("cpu_times")
        progress_bar.progress(30)

        # Prepare run path
        run_path = RUNTIME_DIR / "gpu" / input_path.name
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_bytes(input_path.read_bytes())

        # Determine executable paths (allow sidebar override/upload)
        gpu1_path = ROOT / ("gpu1.exe" if os.name == "nt" and (ROOT / "gpu1.exe").exists() else "gpu1")
        enhanced_path = ROOT / ("clearscan.exe" if os.name == "nt" and (ROOT / "clearscan.exe").exists() else "clearscan")
        # Uploaded binary overrides
        if uploaded_bin_sidebar is not None:
            bin_path = ROOT / uploaded_bin_sidebar.name
            with open(bin_path, 'wb') as f:
                f.write(uploaded_bin_sidebar.read())
            os.chmod(bin_path, 0o755)
            enhanced_path = bin_path
            gpu1_path = bin_path
        # Text input override
        if binary_path_input:
            candidate = Path(binary_path_input)
            if candidate.exists():
                enhanced_path = candidate
                gpu1_path = candidate

        # Run GPU1 and Enhanced as requested
        GPU1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gpu1_stats = None
        enhanced_stats = None
        st.session_state.gpu1_output = None
        st.session_state.gpu_enhanced_output = None

        # Run GPU1
        if run_gpu1_checkbox and gpu1_path.exists():
            progress_bar.progress(50)
            gpu1_stats, gpu1_output = run_cuda_binary_average(gpu1_path, run_path, repeats=repeats)
            st.session_state.gpu1_stats = gpu1_stats
            st.session_state.gpu1_output = gpu1_output
            # copy stage images
            for stage_name in ("gaussian", "histogram", "sharpen", "edge"):
                source = OUTPUT_DIR / f"gpu1_{stage_name}.png"
                target = GPU1_OUTPUT_DIR / f"{stage_name}.png"
                if source.exists():
                    target.write_bytes(source.read_bytes())
        else:
            st.session_state.gpu1_stats = None

        progress_bar.progress(75)

        # Run enhanced GPU
        if run_enhanced_checkbox and enhanced_path.exists():
            enhanced_stats, enhanced_output = run_cuda_binary_average(enhanced_path, run_path, repeats=repeats)
            st.session_state.gpu_enhanced_stats = enhanced_stats
            st.session_state.gpu_enhanced_output = enhanced_output
        else:
            st.session_state.gpu_enhanced_stats = None

        progress_bar.progress(95)

        # Store comparison history
        st.session_state.comparison_history[image_id] = {
            "image": input_path.name,
            "cpu_ms": cpu_times,
            "gpu1": st.session_state.get("gpu1_stats"),
            "gpu_enhanced": st.session_state.get("gpu_enhanced_stats"),
        }
        progress_bar.progress(100)
    st.session_state.last_image_id = image_id

# Live CPU preview action
if live_preview:
    # show controls for preview
    st.sidebar.markdown("---")
    st.sidebar.subheader("Preview parameters")
    sigma = st.sidebar.slider("Gaussian sigma", 0.1, 3.0, 0.85, 0.05)
    sharpen_strength = st.sidebar.slider("Sharpen strength", 0.1, 3.0, 1.0, 0.1)
    edge_threshold = st.sidebar.slider("Edge threshold (0 = none)", 0, 255, 0)
    with st.spinner("Running CPU preview..."):
        time_cpu_preview(input_path, sigma=sigma, sharpen_strength=sharpen_strength, edge_threshold=edge_threshold)
    st.success("Preview updated")
    cols = st.columns(4)
    cols[0].image(str(CPU_OUTPUT_DIR / 'gaussian_cpu.png'), caption='Gaussian (CPU preview)')
    cols[1].image(str(CPU_OUTPUT_DIR / 'histogram_cpu.png'), caption='Histogram EQ (CPU preview)')
    cols[2].image(str(CPU_OUTPUT_DIR / 'sharpen_cpu.png'), caption='Sharpen (CPU preview)')
    cols[3].image(str(CPU_OUTPUT_DIR / 'edge_cpu.png'), caption='Edge (CPU preview)')

cpu_times = st.session_state.get("cpu_times")
gpu_versions = {
    "CPU": cpu_times,
    "GPU1 (simple CUDA)": st.session_state.get("gpu1_stats"),
    "GPU Enhanced (optimized CUDA)": st.session_state.get("gpu_enhanced_stats"),
}


def stage_time(stats, key):
    value = stats.get(key, 0.0) if stats else 0.0
    if isinstance(value, dict):
        value = value.get("mean_ms", 0.0)
    return float(value or 0.0)


def performance_gap(cpu_ms, gpu_ms):
    if not cpu_ms or not gpu_ms:
        return "-"
    if gpu_ms < cpu_ms:
        return f"{cpu_ms / gpu_ms:.2f}x faster"
    return f"{gpu_ms / cpu_ms:.2f}x slower"


def optimization_gap(gpu1_ms, enhanced_ms):
    if not gpu1_ms or not enhanced_ms:
        return "-"
    if enhanced_ms < gpu1_ms:
        return f"{gpu1_ms / enhanced_ms:.2f}x faster"
    return f"{enhanced_ms / gpu1_ms:.2f}x slower"


def memory_mb(stats):
    value = stats.get("device_bytes") if stats else None
    if value is None:
        return "N/A"
    return f"{float(value) / 1024**2:.3f} MB"


def pinned_memory_mb(stats):
    value = stats.get("pinned_host_bytes") if stats else None
    if value is None:
        return "N/A"
    return f"{float(value) / 1024**2:.3f} MB"


if all(gpu_versions.values()):
    st.subheader("Runtime comparison")
    timing_rows = []
    total_cpu = 0.0
    total_gpu1 = 0.0
    total_enhanced = 0.0
    for key, label in zip(STAGE_KEYS, ["Gaussian", "Histogram", "Sharpen", "Edge"]):
        cpu_ms = float(cpu_times.get(key, 0.0))
        gpu1_ms = stage_time(gpu_versions["GPU1 (simple CUDA)"], key)
        enhanced_ms = stage_time(gpu_versions["GPU Enhanced (optimized CUDA)"], key)
        total_cpu += cpu_ms
        total_gpu1 += gpu1_ms
        total_enhanced += enhanced_ms
        row = {
            "stage": label,
            "CPU ms": round(cpu_ms, 4),
            "GPU1 ms": round(gpu1_ms, 4),
            "GPU Enhanced ms": round(enhanced_ms, 4),
            "CPU -> GPU1": performance_gap(cpu_ms, gpu1_ms),
            "GPU1 -> Enhanced": optimization_gap(gpu1_ms, enhanced_ms),
        }
        timing_rows.append(row)
    timing_rows.append({
        "stage": "TOTAL PIPELINE",
        "CPU ms": round(total_cpu, 4),
        "GPU1 ms": round(total_gpu1, 4),
        "GPU Enhanced ms": round(total_enhanced, 4),
        "CPU -> GPU1": performance_gap(total_cpu, total_gpu1),
        "GPU1 -> Enhanced": optimization_gap(total_gpu1, total_enhanced),
    })
    st.table(timing_rows)
    st.info(
        "GPU Enhanced is faster than GPU1 when its gap is marked faster because it reuses neighboring pixels in shared-memory tiles. "
        "GPU1 reads each neighbor directly from global memory, while GPU Enhanced reduces repeated global-memory traffic with coalesced tile loads and cached filter coefficients."
    )
    if total_cpu > total_gpu1 > total_enhanced:
        st.success("Performance order achieved: CPU > GPU1 > GPU Enhanced.")
    else:
        st.warning("This workload does not achieve CPU > GPU1 > GPU Enhanced for the total pipeline. CUDA launch overhead and image size affect the order.")

    st.subheader("Memory comparison")
    st.table([
        {"version": "CPU", "device memory": "N/A", "host image memory": f"{input_path.stat().st_size / 1024**2:.3f} MB"},
        {"version": "GPU1 (simple CUDA)", "device memory": memory_mb(gpu_versions["GPU1 (simple CUDA)"]), "host image memory": "pinned buffers not used"},
        {"version": "GPU Enhanced (optimized CUDA)", "device memory": memory_mb(gpu_versions["GPU Enhanced (optimized CUDA)"]), "host image memory": pinned_memory_mb(gpu_versions["GPU Enhanced (optimized CUDA)"])},
    ])

    st.subheader("Output accuracy against CPU")
    accuracy_rows = []
    stage_images = [
        ("Gaussian", "gaussian.png"),
        ("Histogram", "histogram.png"),
        ("Sharpen", "sharpen.png"),
        ("Edge", "edge.png"),
    ]
    for stage, image_name in stage_images:
        cpu_image = CPU_OUTPUT_DIR / image_name
        gpu1_image = GPU1_OUTPUT_DIR / image_name
        enhanced_image = OUTPUT_DIR / image_name
        gpu1_metrics = output_metrics(cpu_image, gpu1_image)
        enhanced_metrics = output_metrics(cpu_image, enhanced_image)
        accuracy_rows.append({"stage": stage, "GPU1 MAE (0 ideal)": round(gpu1_metrics["mae"], 3), "GPU1 SSIM (1 ideal)": round(gpu1_metrics["ssim"], 4), "Enhanced MAE (0 ideal)": round(enhanced_metrics["mae"], 3), "Enhanced SSIM (1 ideal)": round(enhanced_metrics["ssim"], 4)})
    st.table(accuracy_rows)
    st.subheader("Images after each filter")
    for stage, image_name in stage_images:
        st.markdown(f"**{stage}**")
        image_columns = st.columns(3)
        image_columns[0].image(str(CPU_OUTPUT_DIR / image_name), caption="CPU")
        image_columns[1].image(str(GPU1_OUTPUT_DIR / image_name), caption="GPU1")
        image_columns[2].image(str(OUTPUT_DIR / image_name), caption="GPU Enhanced")
    sample_count = gpu_versions["GPU1 (simple CUDA)"].get("sample_count", 1)
    st.caption(f"Stored result for {input_path.name} using {sample_count} CUDA samples per version. Gaussian can be slower when launch overhead exceeds the CPU blur time; the gap column shows the measured multiple.")
    st.success("All three pipelines completed for the selected image.")
    st.stop()

st.header("Run CPU pipeline (OpenCV)")
if st.button("Run CPU" ):
    with st.spinner("Running CPU pipeline..."):
        cpu_times = time_cpu_pipeline(input_path)
    st.session_state.cpu_times = cpu_times
    st.success("CPU pipeline completed")
    st.write(cpu_times)
    cols = st.columns(4)
    cols[0].image(str(CPU_OUTPUT_DIR / 'gaussian_cpu.png'), caption='Gaussian (CPU)')
    cols[1].image(str(CPU_OUTPUT_DIR / 'histogram_cpu.png'), caption='Histogram EQ (CPU)')
    cols[2].image(str(CPU_OUTPUT_DIR / 'sharpen_cpu.png'), caption='Sharpen (CPU)')
    cols[3].image(str(CPU_OUTPUT_DIR / 'edge_cpu.png'), caption='Edge (CPU)')

cpu_times = st.session_state.cpu_times

# Optional verbose logs
if show_logs:
    st.subheader("GPU logs")
    gpu1_out = st.session_state.get("gpu1_output") or ""
    enhanced_out = st.session_state.get("gpu_enhanced_output") or ""
    st.text_area("GPU1 output", value=gpu1_out, height=200)
    st.text_area("GPU Enhanced output", value=enhanced_out, height=200)

# Download outputs as ZIP
if st.button("Prepare outputs ZIP"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for f in OUTPUT_DIR.glob('*.png'):
            z.writestr(f.name, f.read_bytes())
        cpu_preview = CPU_OUTPUT_DIR / 'preview.png'
        if cpu_preview.exists():
            z.writestr('cpu_preview.png', cpu_preview.read_bytes())
    buf.seek(0)
    st.download_button("Download outputs as ZIP", data=buf.getvalue(), file_name=f"{input_path.stem}_outputs.zip", mime="application/zip")

st.header("Run GPU pipeline (clearscan binary)")
exe_name = 'clearscan.exe' if os.name == 'nt' else 'clearscan'
exe_path = resolve_gpu_executable(ROOT)
use_wsl = False
if os.name == 'nt':
    linux_exe = ROOT / 'clearscan'
    if linux_exe.exists():
        use_wsl = True
        exe_path = linux_exe

if exe_path is None:
    exe_path = ROOT / exe_name

# Allow uploading a pre-built clearscan binary (optional)
uploaded_bin = st.file_uploader("Or upload a pre-built clearscan binary (optional)", type=None)
if uploaded_bin is not None:
    data = uploaded_bin.read()
    uploaded_bin.seek(0)
    if looks_like_image_upload(uploaded_bin):
        st.error("That looks like an image file, not the GPU binary. Please upload the compiled clearscan executable instead.")
    else:
        bin_path = ROOT / uploaded_bin.name
        with open(bin_path, 'wb') as f:
            f.write(data)
        if not os.access(bin_path, os.X_OK):
            os.chmod(bin_path, 0o755)
        exe_path = bin_path
        use_wsl = os.name == "nt" and bin_path.suffix.lower() != ".exe"

if not exe_path.exists():
    st.warning(f"GPU binary not found at {exe_path}. The app will fall back to CPU timing data for now.")
    if st.button("Attempt to build (make)"):
        with st.spinner("Running make..."):
            proc = subprocess.run(["make"], cwd=str(ROOT), capture_output=True, text=True)
        st.text(proc.stdout + "\n" + proc.stderr)
        if (ROOT / exe_name).exists():
            st.success("Build succeeded")
            exe_path = ROOT / exe_name
        elif (ROOT / 'clearscan').exists():
            st.success("Build succeeded")
            use_wsl = True
            exe_path = ROOT / 'clearscan'
        else:
            st.error("Build failed or not available on this platform")

if exe_path.exists() and os.name == "nt" and not use_wsl:
    st.info("This Windows host cannot run the CUDA GPU binary directly. The app will use CPU timing as a fallback unless you provide a compatible GPU executable.")
elif exe_path.exists() and os.name == "nt" and use_wsl:
    st.info("A Linux GPU binary exists and will be executed through WSL to collect real GPU timings.")

runs = st.number_input("GPU runs (for averaging)", min_value=1, max_value=50, value=3)
if st.button("Run GPU now"):
    # copy input to a path the binary expects (it takes arg1)
    run_path = RUNTIME_DIR / "gpu" / input_path.name
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'rb') as r, open(run_path, 'wb') as w:
        w.write(r.read())

    names = ['gaussian_ms', 'histogram_ms', 'sharpen_ms', 'edge_ms']
    memory_keys = ['device_bytes', 'pinned_host_bytes']
    all_runs = {n: [] for n in names}
    memory_runs = {n: [] for n in memory_keys}
    wall_times = []
    last_output = ""

    for i in range(int(runs)):
        st.write(f"Run {i+1}/{int(runs)}")
        if exe_path.exists() or should_use_wsl(ROOT):
            if use_wsl or should_use_wsl(ROOT):
                cmd = build_wsl_command(str(ROOT), str(run_path), executable_path=exe_path, distro_name=get_default_wsl_distro())
            else:
                cmd = [str(exe_path), str(run_path)]
            try:
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
                t1 = time.perf_counter()
                wall_ms = (t1 - t0) * 1000.0
                wall_times.append(wall_ms)
                last_output = proc.stdout + "\n" + proc.stderr
                parsed = parse_gpu_timing_output(last_output)
                if parsed:
                    for name in names:
                        if name in parsed:
                            all_runs[name].append(parsed[name])
                    for name in memory_keys:
                        if name in parsed:
                            memory_runs[name].append(parsed[name])
                else:
                    detail = "The binary did not emit parseable timing values."
                    if should_use_wsl(ROOT):
                        detail = "The Linux GPU binary could not be executed successfully in WSL, so no timing values were produced."
                    st.warning(f"GPU binary produced no parseable timing values; using the CPU timing fallback for this run. {detail}")
            except (OSError, ValueError) as exc:
                wall_ms = 0.0
                wall_times.append(wall_ms)
                last_output = f"Could not launch GPU binary: {exc}"
                if is_windows_incompatible_error(exc):
                    last_output = "Could not launch GPU binary: the available binary is not compatible with this host. The app will use CPU timing fallback."
                st.warning(last_output)
                st.text(last_output)
                continue
        else:
            wall_ms = 0.0
            wall_times.append(wall_ms)
            last_output = "GPU binary unavailable"
            st.warning("GPU binary unavailable; using CPU timing fallback.")

        st.text(last_output)

    import numpy as _np
    stats = {}
    for name in names:
        arr = _np.array(all_runs[name]) if all_runs[name] else _np.array([])
        if arr.size:
            stats[name] = {
                'runs': len(arr.tolist()),
                'mean_ms': float(arr.mean()),
                'std_ms': float(arr.std()),
                'raw': arr.tolist()
            }
        else:
            stats[name] = {
                'runs': 0,
                'mean_ms': None,
                'std_ms': None,
                'raw': []
            }

    if not any(stats[name]['runs'] for name in names):
        stats = build_gpu_stats_from_cpu(st.session_state.cpu_times if 'cpu_times' in st.session_state else None, wall_ms=float(_np.mean(wall_times)) if wall_times else 0.0, source='fallback')
    else:
        stats['wall'] = {'mean_ms': float(_np.mean(wall_times)), 'std_ms': float(_np.std(wall_times)), 'raw': wall_times}
        stats['memory'] = {
            'device_bytes': float(_np.mean(memory_runs['device_bytes'])) if memory_runs['device_bytes'] else None,
            'pinned_host_bytes': float(_np.mean(memory_runs['pinned_host_bytes'])) if memory_runs['pinned_host_bytes'] else None,
            'device_samples': memory_runs['device_bytes'],
            'pinned_host_samples': memory_runs['pinned_host_bytes'],
        }

    st.session_state.gpu_stats = stats
    st.success("GPU runs completed")
    st.write(stats)

    # display GPU-generated images from last run if present
    gp_imgs = ['gaussian.png', 'histogram.png', 'sharpen.png', 'edge.png', 'preview.png']
    found = []
    for name in gp_imgs:
        p = OUTPUT_DIR / name
        if p.exists():
            found.append(p)
    if found:
        cols = st.columns(len(found))
        for i, p in enumerate(found):
            cols[i].image(str(p), caption=p.name)

    def load_gray(p):
        import numpy as _np
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return img.astype(_np.float32)

    mismatch_detected = False
    if cpu_times and found:
        # map stage filenames to cpu/gpu names
        mapping = [
            ('gaussian_cpu.png', 'gaussian.png', 'gaussian_ms'),
            ('histogram_cpu.png', 'histogram.png', 'histogram_ms'),
            ('sharpen_cpu.png', 'sharpen.png', 'sharpen_ms'),
            ('edge_cpu.png', 'edge.png', 'edge_ms')
        ]
        report = []
        for cpu_name, gpu_name, key in mapping:
            cpu_p = CPU_OUTPUT_DIR / cpu_name
            gpu_p = OUTPUT_DIR / gpu_name
            if cpu_p.exists() and gpu_p.exists():
                a = load_gray(cpu_p)
                b = load_gray(gpu_p)
                # resize if necessary
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]))

                # MAE
                mae = float((abs(a - b)).mean())
                # PSNR
                psnr = float(cv2.PSNR(a, b))
                # SSIM (0-1)
                s = float(ssim(a.astype('uint8'), b.astype('uint8')))
                cpu_ms = cpu_times.get(key)
                gpu_val = stats.get(key, {}) if 'stats' in locals() else None
                gpu_mean = gpu_val['mean_ms'] if gpu_val else None
                speedup = (cpu_ms / gpu_mean) if (cpu_ms and gpu_mean) else None
                report.append({'stage': key, 'cpu_ms': cpu_ms, 'gpu_mean_ms': gpu_mean, 'speedup': (f"{speedup:.2f}x" if speedup else '-'), 'mae': mae, 'psnr': psnr, 'ssim': s})

        st.header('Accuracy & Runtime Comparison')
        st.table(report)

        # Inference based on metrics
        def infer(rep):
            lines = []
            for r in rep:
                perf = 'similar' if (r['mae'] < 2 and r['psnr'] > 30 and r['ssim'] > 0.95) else ('noticeable differences' if r['mae'] < 10 else 'significant differences')
                speed = r['speedup']
                lines.append(f"Stage {r['stage']}: {perf}; CPU {r['cpu_ms']:.2f}ms vs GPU {r['gpu_mean_ms']:.2f}ms ({speed})")
            return '\n'.join(lines)

        st.subheader('Inference')
        st.text(infer(report))

        mismatch_detected = any(
            (r['mae'] >= 10 or r['psnr'] < 30 or r['ssim'] < 0.95)
            for r in report
        )
        if mismatch_detected:
            st.warning(
                "Output mismatch detected. CPU-based accuracy comparison may not be meaningful, "
                "so the app will compare the current GPU only against the previous GPU baseline."
            )

    if cpu_times and not mismatch_detected:
        st.header("Detailed timing comparison")
        rows = []
        for key in names:
            c = cpu_times.get(key, None)
            g_mean = stats[key]['mean_ms']
            g_std = stats[key]['std_ms']
            speedup = (c / g_mean) if (c is not None and g_mean) else None
            rows.append({
                'stage': key,
                'cpu_ms': c,
                'gpu_mean_ms': g_mean,
                'gpu_std_ms': g_std,
                'speedup_mean': (f"{speedup:.2f}x" if speedup else "-"),
                'gpu_raw_ms': stats[key]['raw']
            })

        st.table(rows)
        st.write("Wall-clock (binary) mean ms:", stats['wall']['mean_ms'], "std:", stats['wall']['std_ms'])

    cpu_times = st.session_state.cpu_times
    gpu_stats = st.session_state.gpu_stats
    compare_previous_gpu_only = st.session_state.compare_previous_gpu_only
    if mismatch_detected:
        compare_previous_gpu_only = True

    if gpu_stats:
        if compare_previous_gpu_only and cpu_times:
            summary = summarize_previous_gpu_vs_cpu(cpu_times, st.session_state.previous_gpu_baseline)
            st.subheader("Previous GPU baseline vs CPU summary")
            st.write(summary['conclusion'])
            st.json(summary['details'])

            comparison_rows = []
            for metric_key, label in STAGE_LABELS.items():
                cpu_ms = cpu_times.get(metric_key)
                previous_mean = st.session_state.previous_gpu_baseline.get(metric_key)
                speedup = (cpu_ms / previous_mean) if (cpu_ms and previous_mean) else None
                comparison_rows.append({
                    'stage': label,
                    'cpu_ms': cpu_ms,
                    'previous_gpu_ms': previous_mean,
                    'difference_ms': (cpu_ms - previous_mean) if (cpu_ms is not None and previous_mean is not None) else None,
                    'speedup': (f"{speedup:.2f}x" if speedup else '-')
                })

            st.subheader('CPU vs Previous GPU Baseline')
            st.table(comparison_rows)
        else:
            summary = summarize_gpu_comparison(
                gpu_stats,
                None if mismatch_detected else cpu_times,
                st.session_state.previous_gpu_baseline,
            )
            st.subheader("Summary")
            st.write(summary['conclusion'])

            if gpu_stats.get('memory'):
                mem = gpu_stats['memory']
                device_bytes = mem.get('device_bytes')
                pinned_host_bytes = mem.get('pinned_host_bytes')
                wall_ms = gpu_stats.get('wall', {}).get('mean_ms')
                st.markdown("**Memory usage and efficiency (GPU)**")
                st.write(f"- Device memory: {format_bytes(device_bytes)}")
                st.write(f"- Pinned host memory: {format_bytes(pinned_host_bytes)}")
                st.write(f"- Device memory per wall-clock ms: {format_bytes_per_ms(device_bytes, wall_ms)}")
                if not mismatch_detected and cpu_times:
                    total_cpu_ms = float(sum(cpu_times.get(k, 0.0) for k in names))
                    total_gpu_ms = float(sum((gpu_stats.get(k, {}) or {}).get('mean_ms') or 0.0 for k in names))
                    if total_gpu_ms > 0 and device_bytes is not None:
                        st.write(f"- Efficiency: {format_bytes_per_ms(device_bytes, total_gpu_ms)} of device memory per pipeline ms.")
            else:
                st.markdown("**Memory usage (GPU)**")
                st.write("Memory usage data not available.")

            st.json(summary['details'])

            if not mismatch_detected and cpu_times and gpu_stats:
                st.header("Total Runtime Summary")
                total_cpu_ms = float(sum(cpu_times.get(k, 0.0) for k in names))
                total_gpu_ms = float(sum((gpu_stats.get(k, {}) or {}).get('mean_ms') or 0.0 for k in names))
                total_diff_ms = total_cpu_ms - total_gpu_ms
                total_speedup = (total_cpu_ms / total_gpu_ms) if total_gpu_ms else None
                summary_cols = st.columns(4)
                summary_cols[0].metric("CPU total (ms)", f"{total_cpu_ms:.3f}")
                summary_cols[1].metric("GPU total (ms)", f"{total_gpu_ms:.3f}")
                summary_cols[2].metric("Difference (ms)", f"{total_diff_ms:.3f}")
                summary_cols[3].metric("Overall speedup", f"{total_speedup:.2f}x" if total_speedup else "-")

            comparison_rows = []
            for metric_key, label in STAGE_LABELS.items():
                current_mean = gpu_stats.get(metric_key, {}).get('mean_ms')
                previous_mean = st.session_state.previous_gpu_baseline.get(metric_key)
                if current_mean is None or previous_mean is None:
                    improvement = None
                else:
                    improvement = previous_mean / current_mean if current_mean else None
                comparison_rows.append({
                    'stage': label,
                    'previous_gpu_ms': previous_mean,
                    'current_gpu_ms': current_mean,
                    'improvement': f"{improvement:.2f}x" if improvement else '-' 
                })

            st.subheader('Previous GPU vs Current GPU')
            st.table(comparison_rows)

            if not mismatch_detected and cpu_times and gpu_stats:
                stage_rows = []
                for stage_key, stage_label, cpu_name, gpu_name, metric_key in STAGE_FILES:
                    cpu_path = OUTPUT_DIR / cpu_name
                    gpu_path = OUTPUT_DIR / gpu_name
                    cpu_ms = cpu_times.get(metric_key) if cpu_times else None
                    gpu_mean = gpu_stats.get(metric_key, {}).get('mean_ms') if gpu_stats else None
                    previous_gpu_mean = st.session_state.previous_gpu_baseline.get(metric_key)
                    improvement = (previous_gpu_mean / gpu_mean) if (previous_gpu_mean and gpu_mean) else None
                    stage_rows.append({
                        'stage': stage_label,
                        'cpu_ms': cpu_ms,
                        'previous_gpu_ms': previous_gpu_mean,
                        'current_gpu_ms': gpu_mean,
                        'improvement': (f"{improvement:.2f}x" if improvement else '-'),
                        'difference_ms': (cpu_ms - gpu_mean) if (cpu_ms is not None and gpu_mean is not None) else None,
                        'speedup': (f"{(cpu_ms / gpu_mean):.2f}x" if (cpu_ms is not None and gpu_mean) else '-'),
                        'cpu_output': str(cpu_path) if cpu_path.exists() else '-',
                        'gpu_output': str(gpu_path) if gpu_path.exists() else '-'
                    })

                if stage_rows:
                    st.header("Per-Filter Runtime and Output Comparison")
                    st.table(stage_rows)

                    for stage_key, stage_label, cpu_name, gpu_name, metric_key in STAGE_FILES:
                        cpu_path = OUTPUT_DIR / cpu_name
                        gpu_path = OUTPUT_DIR / gpu_name
                        st.subheader(stage_label)
                        img_cols = st.columns(2)
                        if cpu_path.exists():
                            img_cols[0].image(str(cpu_path), caption=f"{stage_label} (CPU)")
                        else:
                            img_cols[0].write("CPU output missing")
                        if gpu_path.exists():
                            img_cols[1].image(str(gpu_path), caption=f"{stage_label} (GPU)")
                        else:
                            img_cols[1].write("GPU output missing")

                    final_cols = st.columns(2)
                    final_cpu = OUTPUT_DIR / 'preview_cpu.png'
                    final_gpu = OUTPUT_DIR / 'preview.png'
                    if final_cpu.exists():
                        final_cols[0].image(str(final_cpu), caption='Final output (CPU preview)')
                    else:
                        final_cols[0].write('CPU final preview not available')
                    if final_gpu.exists():
                        final_cols[1].image(str(final_gpu), caption='Final output (GPU preview)')
                    else:
                        final_cols[1].write('GPU final preview not available')
