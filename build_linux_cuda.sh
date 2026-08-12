#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install CUDA Toolkit first." >&2
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config not found. Install it first." >&2
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "OpenCV 4 development headers not found. Install libopencv-dev first." >&2
  exit 1
fi

mkdir -p output
make clean >/dev/null 2>&1 || true
make
make gpu1

echo "Build complete. Run: ./gpu1 data/BrokenBone1.jpg and ./clearscan data/BrokenBone1.jpg"
