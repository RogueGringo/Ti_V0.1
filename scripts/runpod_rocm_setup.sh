#!/bin/bash
# RunPod MI300X Setup Script for Ti V0.1 (ROCm / AMD)
# =====================================================
# One-liner deploy on a fresh RunPod AMD MI300X instance:
#
#   curl -sL https://raw.githubusercontent.com/RogueGringo/Ti_V0.1/master/scripts/runpod_rocm_setup.sh | bash
#
# Or manually:
#   1. Start an AMD MI300X pod on RunPod, select the ROCm PyTorch Docker template
#   2. SSH in:  ssh root@<POD_IP>
#   3. Run:     bash <(curl -sL https://raw.githubusercontent.com/RogueGringo/Ti_V0.1/master/scripts/runpod_rocm_setup.sh)
#
# Expected setup time: ~5 minutes (PyTorch ROCm wheel is large, ~2.5 GB)
#
# MI300X specs: 192 GB VRAM (HBM3)
#   K=100 N=9877: dim=987,700  -- ~40 GB  (fits in 192 GB with room to spare)
#   K=200 N=9877: dim=1,975,400 -- ~160 GB (fits in 192 GB, tight but doable)
#
# Note: CuPy is CUDA-only and is NOT installed here. The PyTorch backend
# (TorchSheafLaplacian) uses the same torch.cuda API that PyTorch exposes
# over ROCm via HIP — no code changes required.

set -euo pipefail

echo "============================================"
echo "  Ti V0.1 -- RunPod MI300X Setup (ROCm)"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. GPU check
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Check (rocm-smi) ---"
if ! command -v rocm-smi &>/dev/null; then
    echo "ERROR: rocm-smi not found. Is this a ROCm image?" >&2
    echo "       Expected template: ROCm PyTorch Docker (RunPod catalog)" >&2
    exit 1
fi
rocm-smi
echo ""

# ---------------------------------------------------------------------------
# 2. Clone repo
# ---------------------------------------------------------------------------
echo "--- Cloning Ti V0.1 ---"
cd /workspace
if [ -d "Ti_V0.1" ]; then
    echo "  Repo already present, pulling latest..."
    cd Ti_V0.1
    git pull
else
    git clone https://github.com/RogueGringo/Ti_V0.1.git
    cd Ti_V0.1
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Install project dependencies
# ---------------------------------------------------------------------------
echo "--- Installing project dependencies ---"
pip install -e ".[dev]" 2>&1 | tail -10
echo ""

# ---------------------------------------------------------------------------
# 4. Install PyTorch for ROCm 6.2
#    PyTorch ships a ROCm-enabled wheel under the whl/rocm6.2 index.
#    torch.cuda.* calls transparently dispatch to HIP/ROCm on AMD hardware.
#    CuPy is deliberately NOT installed — it is CUDA-only.
# ---------------------------------------------------------------------------
echo "--- Installing PyTorch ROCm 6.2 (this is ~2.5 GB, ~3-4 minutes) ---"
pip install torch \
    --index-url https://download.pytorch.org/whl/rocm6.2 \
    2>&1 | tail -5
echo ""

# ---------------------------------------------------------------------------
# 5. Verify PyTorch sees the MI300X
#    PyTorch uses the torch.cuda API even on ROCm — HIP provides the same
#    interface. torch.cuda.is_available() returns True on MI300X.
# ---------------------------------------------------------------------------
echo "--- PyTorch / ROCm Verification ---"
python -c "
import torch
import sys

print(f'  PyTorch version : {torch.__version__}')
print(f'  ROCm/CUDA avail : {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('ERROR: torch.cuda.is_available() returned False.', file=sys.stderr)
    print('       Check that the ROCm PyTorch wheel installed correctly.', file=sys.stderr)
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f'  Device name     : {device_name}')

free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gb  = free_bytes  / 1e9
total_gb = total_bytes / 1e9
print(f'  VRAM            : {free_gb:.1f} GB free / {total_gb:.1f} GB total')

if total_gb < 100:
    print(f'  WARNING: expected ~192 GB on MI300X, got {total_gb:.1f} GB', file=sys.stderr)
else:
    print(f'  MI300X 192 GB confirmed.')

# Quick tensor round-trip on GPU
a = torch.randn(1024, 1024, dtype=torch.float64, device='cuda')
b = a @ a.T
print(f'  Matmul test     : {float(b[0,0]):.4f} (nonzero = working)')
"
echo ""

# ---------------------------------------------------------------------------
# 6. Verify data
# ---------------------------------------------------------------------------
echo "--- Data Verification ---"
python -c "
from atft.sources.zeta_zeros import ZetaZerosSource
source = ZetaZerosSource('data/odlyzko_zeros.txt')
cloud = source.generate(9877)
print(f'  Zeros loaded: N={len(cloud.points)}')
print(f'  Range: [{cloud.points[0,0]:.2f}, {cloud.points[-1,0]:.2f}]')
"
echo ""

# ---------------------------------------------------------------------------
# 7. Smoke test: TorchSheafLaplacian at K=20, N=100
#    TorchSheafLaplacian is the new PyTorch-native backend that replaces
#    the CuPy GPUSheafLaplacian on ROCm hardware. Same math, torch sparse API.
# ---------------------------------------------------------------------------
echo "--- Smoke Test (TorchSheafLaplacian, K=20, N=100) ---"
python -c "
import numpy as np
import torch
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian

device = 'cuda'
print(f'  Using device: {torch.cuda.get_device_name(0)}')

source = ZetaZerosSource('data/odlyzko_zeros.txt')
cloud = source.generate(100)
zeros = SpectralUnfolding(method='zeta').transform(cloud).points[:, 0]

builder = TransportMapBuilder(K=20, sigma=0.50)
lap = TorchSheafLaplacian(builder, zeros, transport_mode='superposition', device=device)
eigs = lap.smallest_eigenvalues(3.0, k=5)

print(f'  Eigenvalues: {eigs[:5]}')
print(f'  Spectral sum: {np.sum(eigs):.6f}')
print()
print('  SMOKE TEST PASSED')
"
echo ""

# ---------------------------------------------------------------------------
# 8. Create output directory
# ---------------------------------------------------------------------------
mkdir -p output
echo "  Output directory: /workspace/Ti_V0.1/output"
echo ""

# ---------------------------------------------------------------------------
# 9. Print run commands
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Setup complete.  MI300X ready."
echo ""
echo "  VRAM budget reference:"
echo "    K=100 N=9877: dim=987,700   -- ~40 GB  (fits easily in 192 GB)"
echo "    K=200 N=9877: dim=1,975,400 -- ~160 GB (fits in 192 GB, tight)"
echo ""
echo "  ---- Run Commands ----"
echo ""
echo "  # 1. K=100 full sweep with controls (~39h estimate, definitive):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 --backend torch-gpu \\"
echo "      2>&1 | tee output/k100_sweep_mi300x.log"
echo ""
echo "  # 2. K=200 scout (zeta only, ~10h estimate):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k200 --backend torch-gpu \\"
echo "      2>&1 | tee output/k200_scout_mi300x.log"
echo ""
echo "  # 3. K=100 at N=9877 fine sigma grid -- THE DEFINITIVE EXPERIMENT:"
echo "  #    Fine grid: sigma in [0.44 ... 0.56] step 0.005, eps in [3.0, 5.0]"
echo "  #    Expected runtime: ~6h on MI300X (192 GB means no VRAM pressure)"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 --backend torch-gpu --fine-sigma-grid \\"
echo "      2>&1 | tee output/k100_N9877_fine_sigma_mi300x.log"
echo ""
echo "  # When done, download results:"
echo "  # scp root@<POD_IP>:/workspace/Ti_V0.1/output/*.json ."
echo "============================================"
