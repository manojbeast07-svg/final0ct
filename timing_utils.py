import json
import re
from typing import Dict, Optional

STAGE_KEYS = ("gaussian_ms", "histogram_ms", "sharpen_ms", "edge_ms")
MEMORY_KEYS = ("device_bytes", "pinned_host_bytes")


def parse_gpu_timing_output(output_text: str) -> Dict[str, float]:
    """Extract stage timings from either JSON or legacy GPU: N ms output."""
    parsed: Dict[str, float] = {}

    if output_text:
        text = output_text or ""
        decoder = json.JSONDecoder()
        json_payloads = []
        # Decode each balanced JSON object independently; repeated binary runs
        # produce multiple reports in one captured stdout/stderr string.
        offset = 0
        while offset < len(text):
            start = text.find("{", offset)
            if start == -1:
                break
            try:
                candidate, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                offset = start + 1
                continue
            if isinstance(candidate, dict):
                json_payloads.append(candidate)
            offset = start + end

        for payload in reversed(json_payloads):
            try:
                gpu_payload = payload.get("gpu", {})
                if isinstance(gpu_payload, dict):
                    for key in STAGE_KEYS + MEMORY_KEYS:
                        value = gpu_payload.get(key)
                        if value is not None:
                            parsed[key] = float(value)
            except (TypeError, ValueError):
                continue
            if all(key in parsed for key in STAGE_KEYS):
                break
        if all(key in parsed for key in STAGE_KEYS):
            memory_device = re.search(r"Device allocation:\s*([0-9]+)\s*bytes", text)
            memory_pinned = re.search(r"Pinned host allocation:\s*([0-9]+)\s*bytes", text)
            if memory_device:
                parsed["device_bytes"] = float(memory_device.group(1))
            if memory_pinned:
                parsed["pinned_host_bytes"] = float(memory_pinned.group(1))
            return parsed

        values = []
        for line in text.splitlines():
            match = re.search(r"GPU:\s*([0-9]+\.?[0-9]*)\s*ms", line, re.IGNORECASE)
            if match:
                values.append(float(match.group(1)))
            else:
                match = re.search(r"([0-9]+\.?[0-9]*)\s*ms", line)
                if line.strip().startswith("[") and match:
                    values.append(float(match.group(1)))

        if len(values) >= len(STAGE_KEYS):
            for idx, key in enumerate(STAGE_KEYS):
                parsed[key] = values[idx]
            memory_device = re.search(r"Device allocation:\s*([0-9]+)\s*bytes", text)
            memory_pinned = re.search(r"Pinned host allocation:\s*([0-9]+)\s*bytes", text)
            if memory_device:
                parsed["device_bytes"] = float(memory_device.group(1))
            if memory_pinned:
                parsed["pinned_host_bytes"] = float(memory_pinned.group(1))

    return parsed


def build_gpu_stats_from_cpu(cpu_times: Optional[Dict[str, float]], wall_ms: Optional[float] = None, source: str = "fallback") -> Dict[str, object]:
    """Create a stats dictionary usable by the Streamlit UI when GPU timings are unavailable."""
    stats: Dict[str, object] = {"source": source}
    if cpu_times is None:
        return stats

    for key in STAGE_KEYS:
        value = cpu_times.get(key)
        stats[key] = {
            "runs": 1,
            "mean_ms": float(value) if value is not None else None,
            "std_ms": 0.0,
            "raw": [float(value)] if value is not None else [],
        }

    if wall_ms is None:
        wall_ms = 0.0
    stats["wall"] = {
        "mean_ms": float(wall_ms),
        "std_ms": 0.0,
        "raw": [float(wall_ms)],
    }
    return stats


def summarize_gpu_comparison(gpu_stats: Dict[str, object], cpu_times: Optional[Dict[str, float]] = None, previous_baseline: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    report: Dict[str, object] = {
        "better_than_cpu": True,
        "better_than_previous": True,
        "details": {},
        "conclusion": "",
    }

    for key in STAGE_KEYS:
        gpu_val = gpu_stats.get(key, {}) if gpu_stats else {}
        gpu_ms = gpu_val.get("mean_ms") if isinstance(gpu_val, dict) else None
        cpu_ms = cpu_times.get(key) if cpu_times else None
        prev_ms = previous_baseline.get(key) if previous_baseline else None

        better_than_cpu = True
        better_than_previous = True
        if cpu_ms is not None and gpu_ms is not None:
            better_than_cpu = gpu_ms < cpu_ms
        if prev_ms is not None and gpu_ms is not None:
            better_than_previous = gpu_ms < prev_ms

        if cpu_ms is not None and gpu_ms is not None and not better_than_cpu:
            report["better_than_cpu"] = False
        if prev_ms is not None and gpu_ms is not None and not better_than_previous:
            report["better_than_previous"] = False

        report["details"][key] = {
            "gpu_ms": gpu_ms,
            "cpu_ms": cpu_ms,
            "previous_ms": prev_ms,
            "better_than_cpu": better_than_cpu,
            "better_than_previous": better_than_previous,
        }

    conclusions = []
    if report["better_than_cpu"]:
        conclusions.append("This GPU pipeline is faster than the CPU baseline on all measured stages.")
    else:
        conclusions.append("Some stages still need improvement compared to the CPU baseline.")

    if report["better_than_previous"]:
        conclusions.append("The current GPU version improves upon the previous GPU baseline for all measured stages.")
    else:
        conclusions.append("The current GPU version needs tuning to beat the previous GPU baseline in all stages.")

    if cpu_times is None:
        conclusions.append("CPU baseline was not fully available for this comparison.")

    memory_info = gpu_stats.get("memory") if isinstance(gpu_stats, dict) else None
    device_bytes = None
    pinned_host_bytes = None
    if isinstance(memory_info, dict):
        device_bytes = memory_info.get("device_bytes")
        pinned_host_bytes = memory_info.get("pinned_host_bytes")

    if device_bytes is not None or pinned_host_bytes is not None:
        parts = []
        if device_bytes is not None:
            parts.append(f"{device_bytes / 1024**2:.2f} MB device")
        if pinned_host_bytes is not None:
            parts.append(f"{pinned_host_bytes / 1024**2:.2f} MB pinned host")
        conclusions.append("Memory footprint: " + " and ".join(parts) + ".")

    report["conclusion"] = " ".join(conclusions)
    return report


def summarize_previous_gpu_vs_cpu(cpu_times: Optional[Dict[str, float]], previous_baseline: Optional[Dict[str, float]]) -> Dict[str, object]:
    report: Dict[str, object] = {
        "better_than_cpu": True,
        "details": {},
        "conclusion": "",
    }

    for key in STAGE_KEYS:
        cpu_ms = cpu_times.get(key) if cpu_times else None
        prev_ms = previous_baseline.get(key) if previous_baseline else None

        better_than_cpu = True
        if cpu_ms is not None and prev_ms is not None:
            better_than_cpu = prev_ms < cpu_ms
        if cpu_ms is not None and prev_ms is not None and not better_than_cpu:
            report["better_than_cpu"] = False

        report["details"][key] = {
            "cpu_ms": cpu_ms,
            "previous_gpu_ms": prev_ms,
            "better_than_cpu": better_than_cpu,
        }

    if report["better_than_cpu"]:
        report["conclusion"] = "The previous GPU baseline is faster than the CPU baseline on all measured stages."
    else:
        report["conclusion"] = "The previous GPU baseline needs tuning to beat the CPU baseline in all stages."

    if cpu_times is None:
        report["conclusion"] += " CPU baseline was not fully available for this comparison."

    return report
