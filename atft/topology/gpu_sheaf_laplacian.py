"""GPU-Accelerated Sheaf Laplacian for Massive Scale ATFT.

Hybrid CPU/GPU architecture:
  - CPU: edge discovery (inherited from BaseSheafLaplacian), batched
    matrix exponentials (transport via builder)
  - GPU: sparse CSR assembly in VRAM, eigensolver (eigsh or LOBPCG)

CuPy's COO constructor automatically sums duplicate entries during tocsr(),
so overlapping diagonal blocks merge natively in hardware.

Requires: cupy, cupyx, nvidia-cuda-runtime-cu12, nvidia-cusolver-cu12, etc.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cp_splinalg
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from atft.topology.base_sheaf_laplacian import BaseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


class GPUSheafLaplacian(BaseSheafLaplacian):
    """GPU-accelerated sparse sheaf Laplacian using CuPy.

    Uses CPU for the batched non-Hermitian matrix exponentials (transport),
    and GPU for the massive sparse CSR assembly and LOBPCG eigensolver.

    Args:
        builder: TransportMapBuilder providing K, sigma, transport methods.
        zeros: 1D array of (possibly unsorted) unfolded zeta zeros.
        transport_mode: "superposition" (default), "fe", or "resonant".
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64] | np.ndarray,
        transport_mode: str = "superposition",
    ) -> None:
        if not GPU_AVAILABLE:
            raise ImportError(
                "CuPy is required for GPUSheafLaplacian. "
                "Run: pip install cupy-cuda12x"
            )
        super().__init__(builder, zeros, transport_mode)

    def build_matrix(self, epsilon: float):
        """Assembles the N*K x N*K Laplacian directly in VRAM."""
        i_idx_cpu, j_idx_cpu, gaps = self.build_edge_list(epsilon)
        M = len(gaps)
        K = self._K

        if M == 0:
            return csp.csr_matrix(
                (self._N * K, self._N * K), dtype=cp.complex128,
            )

        # 1. CPU computes the transport matrices
        U_all_cpu = self._compute_transport(gaps)

        # 2. Move to GPU
        i_idx = cp.array(i_idx_cpu)
        j_idx = cp.array(j_idx_cpu)
        U_all = cp.array(U_all_cpu)
        U_dagger = cp.conj(cp.transpose(U_all, axes=(0, 2, 1)))

        # 3. Block Expansion (Broadcasting)
        row_base_i = i_idx * K
        col_base_j = j_idx * K
        row_base_j = j_idx * K
        col_base_i = i_idx * K

        k_range = cp.arange(K)
        r_offset, c_offset = cp.meshgrid(k_range, k_range, indexing='ij')
        r_off_broad = r_offset[None, :, :]
        c_off_broad = c_offset[None, :, :]

        # (i, j) Off-diagonal blocks: -U_dagger
        row_ij = row_base_i[:, None, None] + r_off_broad
        col_ij = col_base_j[:, None, None] + c_off_broad
        data_ij = -U_dagger

        # (j, i) Off-diagonal blocks: -U
        row_ji = row_base_j[:, None, None] + r_off_broad
        col_ji = col_base_i[:, None, None] + c_off_broad
        data_ji = -U_all

        # (i, i) Diagonal blocks: U_dagger @ U
        row_ii = row_base_i[:, None, None] + r_off_broad
        col_ii = col_base_i[:, None, None] + c_off_broad
        data_ii = cp.matmul(U_dagger, U_all)

        # (j, j) Diagonal blocks: Identity
        row_jj = row_base_j[:, None, None] + r_off_broad
        col_jj = col_base_j[:, None, None] + c_off_broad
        I_K = cp.eye(K, dtype=cp.complex128)
        data_jj = cp.broadcast_to(
            I_K[None, :, :], (M, K, K),
        ).copy()

        # Flatten everything
        all_rows = cp.concatenate([
            row_ij.ravel(), row_ji.ravel(),
            row_ii.ravel(), row_jj.ravel(),
        ])
        all_cols = cp.concatenate([
            col_ij.ravel(), col_ji.ravel(),
            col_ii.ravel(), col_jj.ravel(),
        ])
        all_data = cp.concatenate([
            data_ij.ravel(), data_ji.ravel(),
            data_ii.ravel(), data_jj.ravel(),
        ])

        dim = self._N * K
        L_csr = csp.coo_matrix(
            (all_data, (all_rows, all_cols)), shape=(dim, dim),
        ).tocsr()

        return L_csr

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100,
    ) -> NDArray[np.float64]:
        """Compute k smallest eigenvalues on GPU.

        Uses the spectral flip trick: finding k smallest eigenvalues of L is
        equivalent to finding k largest eigenvalues of (lambda_max*I - L) and
        subtracting from lambda_max. Lanczos converges MUCH faster for largest
        eigenvalues than smallest.

        Solver chain:
          1. Spectral flip: eigsh(lambda_max*I - L, which='LM')
          2. Direct eigsh(which='SA') -- fallback
          3. CPU dense -- last resort
        """
        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]

        if dim == 0 or L_csr.nnz == 0:
            return np.zeros(k)

        k_actual = min(k, dim - 2) if dim > 2 else dim
        if k_actual <= 0:
            return np.zeros(k)

        # Small matrices: dense on CPU
        if dim <= 500:
            L_np = cp.asnumpy(L_csr.toarray())
            return self._postprocess_eigenvalues(
                np.linalg.eigvalsh(L_np), k,
            )

        eigs = None

        # Strategy 1: Spectral flip
        try:
            lam_max_arr, _ = cp_splinalg.eigsh(
                L_csr, k=1, which='LM', tol=1e-3,
            )
            lam_max = float(cp.asnumpy(lam_max_arr)[0]) * 1.05

            I_sparse = csp.eye(dim, dtype=cp.complex128, format='csr')
            M = lam_max * I_sparse - L_csr

            mu_arr, _ = cp_splinalg.eigsh(
                M, k=k_actual, which='LM', tol=1e-4,
            )
            mu = cp.asnumpy(mu_arr).real
            eigs = lam_max - mu
        except Exception as e:
            print(f"GPU spectral flip failed: {e}")

        # Strategy 2: Direct eigsh(SA)
        if eigs is None:
            try:
                eigs_raw, _ = cp_splinalg.eigsh(
                    L_csr, k=k_actual, which='SA', tol=1e-5,
                )
                eigs = cp.asnumpy(eigs_raw).real
            except Exception as e:
                print(f"GPU eigsh(SA) failed: {e}")

        # Strategy 3: CPU dense fallback
        if eigs is None:
            L_np = cp.asnumpy(L_csr.toarray())
            eigs = np.linalg.eigvalsh(L_np)

        return self._postprocess_eigenvalues(eigs, k)
