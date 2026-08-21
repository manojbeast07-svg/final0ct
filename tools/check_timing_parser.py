import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from timing_utils import parse_gpu_timing_output

sample = (
    '{"gpu": {"gaussian_ms": 9.0, "histogram_ms": 8.0, "sharpen_ms": 7.0, "edge_ms": 6.0}}\n'
    'logs between runs\n'
    '{"gpu": {"gaussian_ms": 1.0, "histogram_ms": 2.0, "sharpen_ms": 3.0, "edge_ms": 4.0}}'
)
expected = {'gaussian_ms': 1.0, 'histogram_ms': 2.0, 'sharpen_ms': 3.0, 'edge_ms': 4.0}
actual = parse_gpu_timing_output(sample)
assert all(actual[key] == value for key, value in expected.items()), actual
print('timing parser check: PASS')
