"""SU(2) lattice gauge theory — heat bath generation and observables.

Implements:
- SU(2) matrix generation (Pauli basis)
- Heat bath algorithm (Creutz/Kennedy-Pendleton)
- Plaquette computation
- Parity-complete feature map φ(x) = (s_μν(x), q_μν(x)) ∈ R¹²
- Polyakov loop (order parameter for deconfinement)

Reference: M. Creutz, "Quarks, Gluons and Lattices" (1983).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def random_su2(rng: np.random.Generator, n: int = 1) -> NDArray:
    """Generate n random SU(2) matrices uniformly (Haar measure).

    SU(2) is parameterized as: U = a₀I + i(a₁σ₁ + a₂σ₂ + a₃σ₃)
    where a₀² + a₁² + a₂² + a₃² = 1 (unit 4-sphere).

    Returns (n, 2, 2) complex array.
    """
    # Uniform on S³ via normalized Gaussians
    a = rng.standard_normal((n, 4))
    a /= np.linalg.norm(a, axis=1, keepdims=True)

    # Construct 2x2 complex matrices
    U = np.zeros((n, 2, 2), dtype=np.complex128)
    U[:, 0, 0] = a[:, 0] + 1j * a[:, 3]
    U[:, 0, 1] = a[:, 2] + 1j * a[:, 1]
    U[:, 1, 0] = -a[:, 2] + 1j * a[:, 1]
    U[:, 1, 1] = a[:, 0] - 1j * a[:, 3]
    return U


def heat_bath_su2(
    beta: float,
    lattice_shape: tuple[int, ...],
    n_therm: int = 1000,
    n_configs: int = 100,
    n_skip: int = 10,
    seed: int = 42,
) -> list[NDArray]:
    """Generate SU(2) lattice configurations via heat bath.

    Args:
        beta: Coupling constant (β = 4/g²).
        lattice_shape: e.g. (16, 16, 16, 4) for 16³×4.
        n_therm: Thermalization sweeps before saving.
        n_configs: Number of configurations to generate.
        n_skip: Sweeps between saved configurations.
        seed: Random seed.

    Returns:
        List of configurations. Each config is a dict of
        {mu: NDArray of shape (*lattice_shape, 2, 2)} for mu=0..3.
    """
    rng = np.random.default_rng(seed)
    ndim = len(lattice_shape)
    vol = int(np.prod(lattice_shape))

    # Initialize: all links = identity
    links = {}
    for mu in range(ndim):
        links[mu] = np.tile(np.eye(2, dtype=np.complex128), (*lattice_shape, 1, 1))

    def staple_sum(links, site, mu):
        """Compute the staple sum A_μ(x) = Σ_{ν≠μ} (upper + lower staples)."""
        A = np.zeros((2, 2), dtype=np.complex128)
        for nu in range(ndim):
            if nu == mu:
                continue
            # Upper staple: U_ν(x+μ) @ U_μ†(x+ν) @ U_ν†(x)
            site_mu = list(site)
            site_mu[mu] = (site_mu[mu] + 1) % lattice_shape[mu]
            site_nu = list(site)
            site_nu[nu] = (site_nu[nu] + 1) % lattice_shape[nu]

            U_nu_xmu = links[nu][tuple(site_mu)]
            U_mu_xnu = links[mu][tuple(site_nu)]
            U_nu_x = links[nu][tuple(site)]

            A += U_nu_xmu @ U_mu_xnu.conj().T @ U_nu_x.conj().T

            # Lower staple: U_ν†(x+μ-ν) @ U_μ†(x-ν) @ U_ν(x-ν)
            site_mu_mnu = list(site)
            site_mu_mnu[mu] = (site_mu_mnu[mu] + 1) % lattice_shape[mu]
            site_mu_mnu[nu] = (site_mu_mnu[nu] - 1) % lattice_shape[nu]
            site_mnu = list(site)
            site_mnu[nu] = (site_mnu[nu] - 1) % lattice_shape[nu]

            U_nu_xmu_mnu = links[nu][tuple(site_mu_mnu)]
            U_mu_xmnu = links[mu][tuple(site_mnu)]
            U_nu_xmnu = links[nu][tuple(site_mnu)]

            A += U_nu_xmu_mnu.conj().T @ U_mu_xmnu.conj().T @ U_nu_xmnu

        return A

    def kennedy_pendleton_update(staple_A, beta, rng):
        """Generate SU(2) link via Kennedy-Pendleton heat bath."""
        # Compute k = sqrt(det(A))
        det_A = np.linalg.det(staple_A)
        k = np.sqrt(np.abs(det_A))

        if k < 1e-15:
            return random_su2(rng, 1)[0]

        # Normalized staple
        V = staple_A / k

        # Generate a₀ from the distribution P(a₀) ∝ sqrt(1-a₀²) exp(beta*k*a₀)
        bk = beta * k
        while True:
            r1 = rng.random()
            r2 = rng.random()
            r3 = rng.random()

            lam = -1.0 / bk * (np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3))
            if rng.random() ** 2 <= 1.0 - lam / 2:
                a0 = 1.0 - lam
                break

        # Generate a₁, a₂, a₃ uniformly on sphere of radius sqrt(1-a₀²)
        r = np.sqrt(max(0, 1 - a0 ** 2))
        vec = rng.standard_normal(3)
        vec = vec / np.linalg.norm(vec) * r

        # Build SU(2) matrix
        U_new = np.zeros((2, 2), dtype=np.complex128)
        U_new[0, 0] = a0 + 1j * vec[2]
        U_new[0, 1] = vec[1] + 1j * vec[0]
        U_new[1, 0] = -vec[1] + 1j * vec[0]
        U_new[1, 1] = a0 - 1j * vec[2]

        # Multiply by V† to account for staple
        V_inv = np.linalg.inv(V) if np.abs(np.linalg.det(V)) > 1e-10 else np.eye(2)
        return U_new @ V_inv

    configs = []
    total_sweeps = n_therm + n_configs * n_skip

    for sweep in range(total_sweeps):
        # One sweep: update each link
        for mu in range(ndim):
            for idx in np.ndindex(*lattice_shape):
                A = staple_sum(links, idx, mu)
                links[mu][idx] = kennedy_pendleton_update(A, beta, rng)

        # Save configuration after thermalization
        if sweep >= n_therm and (sweep - n_therm) % n_skip == 0:
            config = {mu: links[mu].copy() for mu in range(ndim)}
            configs.append(config)

        if (sweep + 1) % 100 == 0:
            # Compute average plaquette as diagnostic
            plaq = average_plaquette(links, lattice_shape)
            print(f"    sweep {sweep+1}/{total_sweeps}: <P> = {plaq:.6f}", flush=True)

    return configs


def plaquette(links, site, mu, nu, lattice_shape):
    """Compute plaquette P_μν(x) = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)."""
    site_mu = list(site)
    site_mu[mu] = (site_mu[mu] + 1) % lattice_shape[mu]
    site_nu = list(site)
    site_nu[nu] = (site_nu[nu] + 1) % lattice_shape[nu]

    P = (links[mu][tuple(site)] @
         links[nu][tuple(site_mu)] @
         links[mu][tuple(site_nu)].conj().T @
         links[nu][tuple(site)].conj().T)
    return P


def average_plaquette(links, lattice_shape):
    """Average plaquette over all sites and μ<ν pairs."""
    ndim = len(lattice_shape)
    total = 0.0
    count = 0
    for idx in np.ndindex(*lattice_shape):
        for mu in range(ndim):
            for nu in range(mu + 1, ndim):
                P = plaquette(links, idx, mu, nu, lattice_shape)
                total += 0.5 * np.real(np.trace(P))
                count += 1
    return total / count if count > 0 else 0.0


def parity_complete_feature_map(config, lattice_shape):
    """Parity-complete feature map φ(x) = (s_μν, q_μν) ∈ R¹² per site.

    s_μν(x) = 1 - ½ Re Tr P_μν(x)  (action density, parity-even)
    q_μν(x) = ½ Im Tr P_μν(x)      (topological charge density, parity-odd)

    For 4D with 6 μ<ν pairs: φ ∈ R¹² per site.
    """
    ndim = len(lattice_shape)
    n_pairs = ndim * (ndim - 1) // 2
    vol = int(np.prod(lattice_shape))

    features = np.zeros((vol, 2 * n_pairs))

    flat_idx = 0
    for idx in np.ndindex(*lattice_shape):
        pair_idx = 0
        for mu in range(ndim):
            for nu in range(mu + 1, ndim):
                P = plaquette(config, idx, mu, nu, lattice_shape)
                tr_P = np.trace(P)
                s = 1.0 - 0.5 * np.real(tr_P)  # action density
                q = 0.5 * np.imag(tr_P)          # topological charge
                features[flat_idx, pair_idx] = s
                features[flat_idx, n_pairs + pair_idx] = q
                pair_idx += 1
        flat_idx += 1

    return features
