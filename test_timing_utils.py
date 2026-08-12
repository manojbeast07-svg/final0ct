from timing_utils import parse_gpu_timing_output, summarize_gpu_comparison


def test_parse_gpu_timing_output_from_json():
    sample = '''
    some output
    {"gpu": {"gaussian_ms": 1.2345, "histogram_ms": 2.3456, "sharpen_ms": 3.4567, "edge_ms": 4.5678, "device_bytes": 12345678, "pinned_host_bytes": 8765432}, "cpu": {"gaussian_ms": 5.0, "histogram_ms": 6.0, "sharpen_ms": 7.0, "edge_ms": 8.0}}
    '''
    parsed = parse_gpu_timing_output(sample)
    assert parsed["gaussian_ms"] == 1.2345
    assert parsed["histogram_ms"] == 2.3456
    assert parsed["sharpen_ms"] == 3.4567
    assert parsed["edge_ms"] == 4.5678
    assert parsed["device_bytes"] == 12345678.0
    assert parsed["pinned_host_bytes"] == 8765432.0


def test_parse_gpu_timing_output_from_legacy_format():
    sample = '''
    GPU: 1.2 ms
    GPU: 2.3 ms
    GPU: 3.4 ms
    GPU: 4.5 ms
    Device allocation: 12345678 bytes
    Pinned host allocation: 8765432 bytes
    '''
    parsed = parse_gpu_timing_output(sample)
    assert parsed["gaussian_ms"] == 1.2
    assert parsed["histogram_ms"] == 2.3
    assert parsed["sharpen_ms"] == 3.4
    assert parsed["edge_ms"] == 4.5
    assert parsed["device_bytes"] == 12345678.0
    assert parsed["pinned_host_bytes"] == 8765432.0


def test_summarize_gpu_comparison_includes_memory_info():
    gpu_stats = {
        'gaussian_ms': {'mean_ms': 1.0},
        'histogram_ms': {'mean_ms': 2.0},
        'sharpen_ms': {'mean_ms': 3.0},
        'edge_ms': {'mean_ms': 4.0},
        'memory': {'device_bytes': 1024 * 1024 * 16, 'pinned_host_bytes': 1024 * 1024 * 4},
        'wall': {'mean_ms': 10.0}
    }
    cpu_times = {'gaussian_ms': 5.0, 'histogram_ms': 6.0, 'sharpen_ms': 7.0, 'edge_ms': 8.0}
    summary = summarize_gpu_comparison(gpu_stats, cpu_times, {'gaussian_ms': 1.5, 'histogram_ms': 2.5, 'sharpen_ms': 3.5, 'edge_ms': 4.5})
    assert "Memory footprint" in summary['conclusion']
    assert summary['better_than_cpu'] is True
    assert summary['better_than_previous'] is True
