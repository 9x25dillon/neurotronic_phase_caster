"""
Order, Absence and Records
==========================

Operational estimators for the theory in docs/ORDER_RECORDS_THEORY.md. Every
term the theory defines has a function here that turns it into a number with
an uncertainty or a pass/fail criterion.

  geometric absence (reciprocal space)  ->  systematic_absences, ice_ih_absence_rule
  geometric absence (real space)        ->  order_deficit, hexatic_defects (topological charge)
  local order of water / ice            ->  tetrahedral_order (Errington-Debenedetti q)
  morphic field (coarse order field)    ->  coarse_grain, phase-field models A and B
  geometric clock                       ->  coarsening_length, CoarseningClock (L ~ t^(1/z))
  mathematical impression (record)      ->  record_fidelity, record_lifetime, kramers_lifetime,
                                            landauer_cost_j, pauling_residual_entropy
  approximate fractal scaling           ->  box_counting_dimension (scaling-window diagnostics)

Evidence levels follow the repository convention: L0 standard physics and
published values, L1 model calculations, L2 parameters to measure, L3 speculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import Delaunay, cKDTree
except ImportError as exc:  # pragma: no cover
    raise ImportError("order_records needs scipy.spatial") from exc

try:
    from .water_structure_assay import label_permutations, mutual_information_bits, nearest_centroid_cv
except ImportError:  # executed as a script or with src/ on sys.path
    from water_structure_assay import label_permutations, mutual_information_bits, nearest_centroid_cv

K_B = 1.380649e-23  # L0 J/K
R_GAS = 8.314462618  # L0 J/(mol K)


# =============================================================================
# 1. Crystallography of ice Ih and geometric absence in reciprocal space
# =============================================================================

ICE_IH_SPACE_GROUP = "P6_3/mmc (No. 194), oxygen on Wyckoff 4f"
ICE_IH_IDEAL_Z = 1.0 / 16.0  # L0 ideal tetrahedral value; real ice is close
ICE_IH_IDEAL_C_OVER_A = math.sqrt(8.0 / 3.0)


def ice_ih_oxygen_fractional(z: float = ICE_IH_IDEAL_Z) -> np.ndarray:
    """Four oxygens of the hexagonal cell, Wyckoff 4f: (1/3,2/3,z), (2/3,1/3,z+1/2), (2/3,1/3,-z), (1/3,2/3,1/2-z)."""
    return np.array([
        [1 / 3, 2 / 3, z],
        [2 / 3, 1 / 3, z + 0.5],
        [2 / 3, 1 / 3, -z],
        [1 / 3, 2 / 3, 0.5 - z],
    ]) % 1.0


def structure_factor(hkl: Sequence[int], fractional: np.ndarray, form_factors=None) -> complex:
    """L0 F(hkl) = sum_j f_j exp(2 pi i (h x_j + k y_j + l z_j))."""
    h = np.asarray(hkl, dtype=float)
    f = np.ones(len(fractional)) if form_factors is None else np.asarray(form_factors, dtype=float)
    return complex(np.sum(f * np.exp(2j * np.pi * (fractional @ h))))


def systematic_absences(hkl_max: int = 4, n_trials: int = 7, tol: float = 1e-9, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Reflections with F = 0 for every tested value of the free coordinate z.

    Zeros that survive all z are enforced by symmetry (screw axes, glide planes,
    special positions); accidental zeros at one z are excluded. These are the
    'geometric absences' of the crystal in reciprocal space.
    """
    rng = np.random.default_rng(seed)
    zs = np.concatenate([[ICE_IH_IDEAL_Z], rng.uniform(0.01, 0.24, n_trials - 1)])
    absent = []
    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == k == l == 0:
                    continue
                if all(abs(structure_factor((h, k, l), ice_ih_oxygen_fractional(z))) < tol for z in zs):
                    absent.append((h, k, l))
    return absent


def ice_ih_absence_rule(h: int, k: int, l: int) -> bool:
    """L0 for the 4f oxygen sublattice: absent iff l is odd and h - k = 0 (mod 3).

    Contains the general P6_3/mmc conditions (000l and hh-2h-l with l odd: the
    6_3 screw axis and the c-glide) plus the extra 4f special-position condition.
    """
    return (l % 2 == 1) and ((h - k) % 3 == 0)


def _wrap(x: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Map into [0, box) exactly; np.mod can return box itself for tiny negative inputs."""
    y = np.mod(x, box)
    return np.where(y >= box, 0.0, y)


@dataclass(frozen=True)
class HexagonalCell:
    a_m: float = 4.497e-10  # L0 ice Ih near 100 K (approx.)
    c_m: float = 7.322e-10

    @property
    def vectors(self) -> np.ndarray:
        a, c = self.a_m, self.c_m
        return np.array([[a, 0.0, 0.0], [-a / 2, a * math.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])


def ice_ih_supercell(nx: int = 3, ny: int = 2, nz: int = 2, cell: HexagonalCell = HexagonalCell(),
                     z: float = ICE_IH_IDEAL_Z) -> Tuple[np.ndarray, np.ndarray]:
    """Orthorhombic periodic supercell (a, sqrt(3) a, c per repeat). Returns (positions m, box m)."""
    box = np.array([nx * cell.a_m, ny * math.sqrt(3) * cell.a_m, nz * cell.c_m])
    frac = ice_ih_oxygen_fractional(z)
    span = range(-2 * max(nx, ny) - 2, 2 * max(nx, ny) + 3)
    pts = []
    for i in span:
        for j in span:
            for kz in range(nz):
                pts.append((frac + np.array([i, j, kz])) @ cell.vectors)
    pts = np.vstack(pts)
    eps = 1e-6 * cell.a_m
    inside = np.all((pts >= -eps) & (pts < box - eps), axis=1)
    pts = _wrap(pts[inside], box)
    unique = np.unique(np.round(pts / eps).astype(np.int64), axis=0, return_index=True)[1]
    return pts[np.sort(unique)], box


PAULING_RESIDUAL_ENTROPY_J_PER_MOL_K = R_GAS * math.log(1.5)  # L0 3.37; calorimetry gives ~3.4


def pauling_residual_entropy(n_mol: float = 1.0) -> float:
    """L0 S0 = N k ln(3/2): proton disorder frozen in by the ice rules, a record of non-equilibrium freezing."""
    return n_mol * PAULING_RESIDUAL_ENTROPY_J_PER_MOL_K


# =============================================================================
# 2. Local order: tetrahedral order parameter
# =============================================================================


def tetrahedral_order(positions: np.ndarray, box: Optional[np.ndarray] = None) -> np.ndarray:
    """L0 Errington-Debenedetti q_i = 1 - (3/8) sum_{j<k} (cos psi_jk + 1/3)^2 over the four nearest neighbours.

    q = 1 for a perfect tetrahedron; <q> = 0 for an ideal gas. Periodic when ``box`` is given.
    """
    x = np.asarray(positions, dtype=float)
    if box is not None:
        box = np.asarray(box, dtype=float)
        x = _wrap(x, box)
    tree = cKDTree(x, boxsize=box) if box is not None else cKDTree(x)
    _, idx = tree.query(x, k=5)
    vec = x[idx[:, 1:]] - x[:, None, :]
    if box is not None:
        vec -= box * np.round(vec / box)
    unit = vec / np.linalg.norm(vec, axis=2, keepdims=True)
    cosines = np.einsum("nif,njf->nij", unit, unit)
    iu = np.triu_indices(4, k=1)
    return 1.0 - 0.375 * np.sum((cosines[:, iu[0], iu[1]] + 1.0 / 3.0) ** 2, axis=1)


# =============================================================================
# 3. Coarse-graining: order field and order deficit (geometric absence, real space)
# =============================================================================


def coarse_grain(positions: np.ndarray, values: np.ndarray, box: np.ndarray, grid: int, ell: float) -> np.ndarray:
    """Gaussian-weighted local average m(r) = sum_i v_i W(r - r_i) / sum_i W(r - r_i) on a periodic grid.

    Works in 2D or 3D (dimension from ``positions``). ``ell`` is the coarse-graining length.
    """
    x = np.asarray(positions, dtype=float)
    d = x.shape[1]
    shape = (grid,) * d
    idx = np.floor(np.mod(x, box) / box * grid).astype(int) % grid
    num = np.zeros(shape)
    den = np.zeros(shape)
    np.add.at(num, tuple(idx.T), values)
    np.add.at(den, tuple(idx.T), 1.0)
    kernel_hat = np.ones(shape)
    for axis in range(d):
        k = 2 * np.pi * np.fft.fftfreq(grid, d=box[axis] / grid)
        kshape = [1] * d
        kshape[axis] = grid
        kernel_hat = kernel_hat * np.exp(-0.5 * (k * ell) ** 2).reshape(kshape)
    smooth_num = np.real(np.fft.ifftn(np.fft.fftn(num) * kernel_hat))
    smooth_den = np.real(np.fft.ifftn(np.fft.fftn(den) * kernel_hat))
    return smooth_num / np.maximum(smooth_den, 1e-12)


def order_deficit(field: np.ndarray, reference: float) -> np.ndarray:
    """a(r) = 1 - |m(r)| / m_ref, clipped to [0, 1]: where order is absent relative to the reference phase."""
    return np.clip(1.0 - np.abs(field) / reference, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class HexaticDefects:
    coordination: np.ndarray  # Delaunay neighbour count per point
    charge: np.ndarray  # 6 - z (disclination charge in units of pi/3)
    psi6: np.ndarray  # complex bond-orientational order per point
    interior: np.ndarray  # points far enough from the boundary to be classified
    defect_indices: np.ndarray

    @property
    def net_charge(self) -> int:
        return int(self.charge[self.interior].sum())

    @property
    def defect_density(self) -> float:
        return self.defect_indices.size / max(1, int(self.interior.sum()))


def hexatic_defects(points: np.ndarray, boundary_margin: float) -> HexaticDefects:
    """2D topological defects of a triangular (hexatic) order: sites with coordination z != 6.

    Charge 6 - z is the disclination winding in units of pi/3. A vacancy is a neutral
    cluster (net charge 0); a free disclination carries net charge. psi6_i = <exp(6 i theta_ij)>.
    """
    p = np.asarray(points, dtype=float)
    tri = Delaunay(p)
    indptr, indices = tri.vertex_neighbor_vertices
    z = np.diff(indptr)
    psi = np.empty(p.shape[0], dtype=complex)
    for i in range(p.shape[0]):
        nb = indices[indptr[i]:indptr[i + 1]]
        d = p[nb] - p[i]
        psi[i] = np.mean(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])))
    lo, hi = p.min(axis=0) + boundary_margin, p.max(axis=0) - boundary_margin
    interior = np.all((p > lo) & (p < hi), axis=1)
    charge = 6 - z
    defects = np.nonzero(interior & (charge != 0))[0]
    return HexaticDefects(z, charge, psi, interior, defects)


def triangular_lattice(n: int, spacing: float = 1.0) -> np.ndarray:
    pts = [(i + 0.5 * (j % 2), j * math.sqrt(3) / 2) for j in range(n) for i in range(n)]
    return np.array(pts) * spacing


# =============================================================================
# 4. Phase-field order-parameter dynamics: models A and B, and the geometric clock
# =============================================================================


def _k2(n: int) -> np.ndarray:
    k = 2 * np.pi * np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx**2 + ky**2


def simulate_phase_ordering(model: str, times: Sequence[float], n: int = 128, dt: Optional[float] = None,
                            noise: float = 0.1, seed: int = 0, stabilizer: float = 2.0) -> Dict[float, np.ndarray]:
    """Quench from disorder and return snapshots of the order field phi at the requested times.

    model "A" (non-conserved, Allen-Cahn):   d phi/dt = lap phi + phi - phi^3
    model "B" (conserved,   Cahn-Hilliard):  d phi/dt = lap(phi^3 - phi - lap phi)
    Linearly stabilised semi-implicit spectral scheme on a periodic n x n grid, unit
    interface width and mobility (L0 equations, L1 numerics). Deterministic for a seed.
    """
    if model not in ("A", "B"):
        raise ValueError("model must be 'A' or 'B'")
    dt = dt if dt is not None else (0.5 if model == "A" else 1.0)
    rng = np.random.default_rng(seed)
    phi = noise * rng.standard_normal((n, n))
    phi -= phi.mean()
    k2 = _k2(n)
    s = stabilizer
    if model == "A":
        denom = 1.0 + dt * (k2 + s)
    else:
        denom = 1.0 + dt * (k2**2 + s * k2)
    targets = sorted(float(t) for t in times)
    out: Dict[float, np.ndarray] = {}
    t = 0.0
    for target in targets:
        while t < target - 1e-9:
            ph = np.fft.fft2(phi)
            if model == "A":
                ph = (ph * (1.0 + dt * s) + dt * np.fft.fft2(phi - phi**3)) / denom
            else:
                ph = (ph * (1.0 + dt * s * k2) - dt * k2 * np.fft.fft2(phi**3 - phi)) / denom
            phi = np.real(np.fft.ifft2(ph))
            t += dt
        out[target] = phi.copy()
    return out


def coarsening_length(phi: np.ndarray) -> float:
    """Domain size L = 2 N / (number of nearest-neighbour bonds crossing phi = 0).

    Interface density is used rather than the first moment of S(k): Porod's
    k^-(d+1) tail makes that moment diverge logarithmically in 2D and biases L.
    """
    s = np.sign(phi)
    bonds = np.sum(s != np.roll(s, 1, 0)) + np.sum(s != np.roll(s, 1, 1))
    return 2.0 * phi.size / max(int(bonds), 1)


def growth_exponent(times: Sequence[float], lengths: Sequence[float]) -> Tuple[float, float]:
    """Least-squares slope of log L vs log t and its standard error."""
    x, y = np.log(np.asarray(times, float)), np.log(np.asarray(lengths, float))
    coef, cov = np.polyfit(x, y, 1, cov=True)
    return float(coef[0]), float(math.sqrt(cov[0, 0]))


@dataclass(frozen=True)
class ClockReading:
    low: float
    estimate: float
    high: float
    domains: float  # approximate number of domains the snapshot averages over
    reliable: bool


@dataclass(frozen=True, eq=False)
class CoarseningClock:
    """Calibrated L(t) for one dynamical class; inverts a frozen snapshot into elapsed time since the quench.

    Dynamical scaling (L0): S(k,t) = L^d g(kL), L ~ t^(1/z), z = 2 (model A), 3 (model B).
    Finite windows give effective exponents below the asymptote; the clock uses the
    measured calibration curve, so no asymptotic assumption enters the estimate.
    """

    model: str
    times: np.ndarray
    lengths: np.ndarray  # mean over calibration seeds
    spread: np.ndarray  # sample std (ddof=1) over calibration seeds

    @classmethod
    def calibrate(cls, model: str, times: Sequence[float], seeds: Sequence[int] = tuple(range(6)),
                  n: int = 128) -> "CoarseningClock":
        """Measured single-snapshot precision on 128^2 (L1): about 15 % (model A) and 12 % (model B);
        model B improves to about 5 % on 256^2, since the length scatter falls with sample area."""
        if len(seeds) < 3:
            raise ValueError("need >= 3 calibration seeds to estimate the ensemble spread")
        rows = []
        for seed in seeds:
            snaps = simulate_phase_ordering(model, times, n=n, seed=seed)
            rows.append([coarsening_length(snaps[float(t)]) for t in sorted(times)])
        arr = np.array(rows)
        return cls(model, np.array(sorted(times), float), arr.mean(axis=0), arr.std(axis=0, ddof=1))

    @property
    def exponent(self) -> Tuple[float, float]:
        return growth_exponent(self.times, self.lengths)

    def _invert(self, L: float, curve: np.ndarray) -> float:
        if not np.all(np.diff(curve) > 0):
            raise ValueError("calibration curve is not monotone; extend the time range or add seeds")
        return float(np.exp(np.interp(math.log(L), np.log(curve), np.log(self.times))))

    def elapsed_time(self, phi: np.ndarray) -> float:
        """Log-log interpolation of the calibration curve at the snapshot's L (clamped to the calibrated range)."""
        return self._invert(coarsening_length(phi), self.lengths)

    def read(self, phi: np.ndarray, n_sigma: float = 2.0, min_domains: float = 25.0) -> "ClockReading":
        """Elapsed-time reading with its interval and a sampling check.

        Precision depends on how many domains the snapshot averages over, N ~ (n/L)^2.
        Below ``min_domains`` (L1 heuristic) single snapshots show heavy-tailed scatter
        (3-sigma readings were observed at N ~ 13), so the reading is marked unreliable.
        """
        lo, est, hi = self.elapsed_time_interval(phi, n_sigma)
        domains = (phi.shape[0] / coarsening_length(phi)) ** 2
        return ClockReading(lo, est, hi, float(domains), domains >= min_domains)

    def elapsed_time_interval(self, phi: np.ndarray, n_sigma: float = 2.0) -> Tuple[float, float, float]:
        """(low, estimate, high): the snapshot read against the mean curve +/- n_sigma ensemble std.

        A single frozen configuration is one draw from the ensemble, so its reading
        inherits the ensemble spread; this interval states that spread in time units
        (n_sigma = 2 gives roughly 95 % coverage).
        """
        L = coarsening_length(phi)
        upper_curve = self.lengths + n_sigma * self.spread
        lower_curve = np.maximum(self.lengths - n_sigma * self.spread, 1e-9)
        return self._invert(L, upper_curve), self._invert(L, self.lengths), self._invert(L, lower_curve)


# =============================================================================
# 5. Fractal scaling with explicit scaling-window diagnostics
# =============================================================================


@dataclass(frozen=True, eq=False)
class BoxCountResult:
    box_sizes: np.ndarray
    counts: np.ndarray
    dimension: float
    standard_error: float
    local_slopes: np.ndarray
    decades: float  # decades of box size used in the fit
    approximately_fractal: bool  # non-integer D with a slope plateau over >= 1 decade

    def summary(self) -> str:
        return (f"D = {self.dimension:.3f} +/- {self.standard_error:.3f} over {self.decades:.2f} decades; "
                f"local slopes {np.round(self.local_slopes, 3).tolist()}; "
                f"{'approximately fractal' if self.approximately_fractal else 'not established as fractal'}")


def box_counting_dimension(mask: np.ndarray, box_sizes: Optional[Sequence[int]] = None, min_box: int = 2,
                           plateau_tol: float = 0.1, integer_tol: float = 0.05) -> BoxCountResult:
    """Box-counting dimension of a 2D boolean set with the evidence needed to call it fractal.

    A set is reported 'approximately fractal' only if (a) the fit spans >= 1 decade,
    (b) every local slope lies within ``plateau_tol`` of the fitted D, and (c) D is not
    within ``integer_tol`` of an integer. Anything else is a scaling claim without support.
    Default sizes double from ``min_box`` up to n/8, keeping >= 8 boxes per side; pass
    ``box_sizes`` commensurate with a known self-similarity ratio when there is one.
    """
    m = np.asarray(mask, dtype=bool)
    n = min(m.shape)
    if box_sizes is None:
        sizes = []
        size = min_box
        while size <= n // 8:
            sizes.append(size)
            size *= 2
    else:
        sizes = sorted(int(s) for s in box_sizes)
    if len(sizes) < 3:
        raise ValueError("need at least three box sizes")
    counts = []
    for s in sizes:
        h, w = (m.shape[0] // s) * s, (m.shape[1] // s) * s
        blocks = m[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        counts.append(int(blocks.sum()))
    eps = np.array(sizes, float)
    cnt = np.array(counts, float)
    if np.any(cnt == 0):
        raise ValueError("empty set at some scale")
    x, y = np.log(1.0 / eps), np.log(cnt)
    coef, cov = np.polyfit(x, y, 1, cov=True)
    d = float(coef[0])
    local = np.diff(y) / np.diff(x)
    decades = float(math.log10(eps[-1] / eps[0]))
    fractal = bool(decades >= 1.0 and np.all(np.abs(local - d) <= plateau_tol)
                   and abs(d - round(d)) > integer_tol)
    return BoxCountResult(eps, cnt, d, float(math.sqrt(cov[0, 0])), local, decades, fractal)


def sierpinski_carpet(level: int) -> np.ndarray:
    """L0 test set, D = ln 8 / ln 3 = 1.8928."""
    m = np.ones((1, 1), dtype=bool)
    for _ in range(level):
        z = np.zeros_like(m)
        m = np.block([[m, m, m], [m, z, m], [m, m, m]])
    return m


# =============================================================================
# 6. Records ("mathematical impressions"): fidelity, lifetime, cost
# =============================================================================


@dataclass(frozen=True)
class RecordFidelity:
    accuracy: float
    chance: float
    excess_bits: float  # plug-in MI minus its permutation-null mean
    p_value: float
    n: int


def record_fidelity(past: np.ndarray, present_features: np.ndarray, groups: Optional[np.ndarray] = None,
                    n_permutations: int = 1000, seed: int = 0, standardize: bool = True) -> RecordFidelity:
    """How much the present configuration says about a past variable.

    Cross-validated nearest-centroid decoding of ``past`` (integer classes) from
    ``present_features`` (n, F), permutation p-value, and bias-corrected bits.
    """
    y = np.asarray(past, dtype=int)
    x = np.atleast_2d(np.asarray(present_features, dtype=float))
    if x.shape[0] != y.size:
        x = x.T
    classes = int(y.max()) + 1
    g = np.arange(y.size) % 5 if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    perms = label_permutations(y, n_permutations, rng)
    preds = nearest_centroid_cv(x, perms, g, classes, standardize)
    acc = (preds == perms).mean(axis=1)
    mi = mutual_information_bits(perms, preds, classes)
    p = float((1 + np.sum(acc[1:] >= acc[0] - 1e-12)) / (n_permutations + 1))
    return RecordFidelity(float(acc[0]), 1.0 / classes, float(mi[0] - mi[1:].mean()), p, int(y.size))


def record_lifetime(times: Sequence[float], fidelity_bits: Sequence[float]) -> Tuple[float, float]:
    """Fit I(t) = I0 exp(-t / tau) to positive fidelities; returns (tau, I0)."""
    t = np.asarray(times, float)
    i = np.asarray(fidelity_bits, float)
    keep = i > 0
    if keep.sum() < 2:
        raise ValueError("need at least two positive fidelity values")
    slope, intercept = np.polyfit(t[keep], np.log(i[keep]), 1)
    if slope >= 0:
        return math.inf, float(math.exp(intercept))
    return float(-1.0 / slope), float(math.exp(intercept))


def kramers_lifetime(barrier_j: float, temperature_k: float, attempt_time_s: float = 1e-12) -> float:
    """L0 (overdamped, high-barrier) record lifetime tau = tau0 exp(dF / kT): what makes a record keep."""
    return attempt_time_s * math.exp(barrier_j / (K_B * temperature_k))


def barrier_for_lifetime(lifetime_s: float, temperature_k: float, attempt_time_s: float = 1e-12) -> float:
    """Inverse of kramers_lifetime, in joules: the barrier a record needs to last ``lifetime_s``."""
    return K_B * temperature_k * math.log(lifetime_s / attempt_time_s)


def landauer_cost_j(bits: float, temperature_k: float) -> float:
    """L0 minimum heat to erase ``bits`` (k T ln 2 per bit). Resetting a record medium for reuse costs at least this."""
    return bits * K_B * temperature_k * math.log(2.0)


# =============================================================================
# Demo
# =============================================================================


def demo() -> str:
    rows: List[str] = ["ORDER, ABSENCE AND RECORDS"]
    absent = systematic_absences(3)
    rule_ok = all(ice_ih_absence_rule(*hkl) for hkl in absent)
    rows.append(f"  ice Ih ({ICE_IH_SPACE_GROUP}): {len(absent)} symmetry-enforced absences with |h|,|k|,|l| <= 3; "
                f"all match 'l odd and h-k = 0 mod 3': {rule_ok}  [L0]")
    rows.append(f"    F(0001) = {abs(structure_factor((0, 0, 1), ice_ih_oxygen_fractional())):.2e}, "
                f"F(0002) = {abs(structure_factor((0, 0, 2), ice_ih_oxygen_fractional())):.3f}")
    pos, box = ice_ih_supercell(4, 3, 3, HexagonalCell(4.497e-10, 4.497e-10 * ICE_IH_IDEAL_C_OVER_A))
    rng = np.random.default_rng(0)
    gas = rng.uniform(0, 1, (4000, 3)) * box
    rows.append(f"  tetrahedral order q: ideal ice {tetrahedral_order(pos, box).mean():.4f}, "
                f"ideal gas {tetrahedral_order(gas, box).mean():+.3f}  [L0]")
    rows.append(f"  Pauling residual entropy of ice: {PAULING_RESIDUAL_ENTROPY_J_PER_MOL_K:.2f} J/(mol K)  [L0]")
    lat = triangular_lattice(20)
    vac = np.delete(lat, 210, axis=0)
    dv = hexatic_defects(vac, 2.5)
    rows.append(f"  vacancy in a triangular crystal: {dv.defect_indices.size} defect sites, net charge {dv.net_charge}  [L0]")
    for model in ("A", "B"):
        times = (25, 50, 100, 200) if model == "A" else (200, 400, 800, 1600)
        grid = 256 if model == "A" else 128  # model A coarsens faster; keep >= 25 domains in the box
        clock = CoarseningClock.calibrate(model, times, n=grid)
        e, se = clock.exponent
        t_test = times[2]
        test = simulate_phase_ordering(model, [t_test], n=grid, seed=9)[float(t_test)]
        r = clock.read(test)
        rows.append(f"  model {model}: effective growth exponent {e:.3f} +/- {se:.3f} "
                    f"(asymptote {'1/2' if model == 'A' else '1/3'}); snapshot at t={t_test} read as "
                    f"t={r.estimate:.0f} (2-sigma range {r.low:.0f}-{r.high:.0f}, ~{r.domains:.0f} domains)  "
                    f"[L0 law, L1 numerics]")
    rows.append("  box counting:")
    rows.append("    Sierpinski carpet  "
                + box_counting_dimension(sierpinski_carpet(5), box_sizes=(1, 3, 9, 27)).summary())
    phi = simulate_phase_ordering("A", [200.0], n=256, seed=3)[200.0]
    s = np.sign(phi)
    interface = (s != np.roll(s, 1, 0)) | (s != np.roll(s, 1, 1))
    rows.append("    model A interfaces " + box_counting_dimension(interface).summary())
    rows.append(f"  record lifetime 1 h at 273 K needs a barrier of "
                f"{barrier_for_lifetime(3600, 273.15) / (K_B * 273.15):.1f} kT; "
                f"Landauer cost of 1 kbit at 273 K = {landauer_cost_j(1000, 273.15):.2e} J  [L0]")
    return "\n".join(rows)


if __name__ == "__main__":
    print(demo())
