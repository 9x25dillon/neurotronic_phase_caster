"""
Crystal Writer
==============

Writes a symbol stream into a growing crystal as growth-rate modulation and reads
it back from the dopant striation profile. Growth zoning is the crystal's own
record of its history (L0); this module makes that record deliberate. It is
prediction P3 of docs/ORDER_RECORDS_THEORY.md, generalised from ice to crystals.

Precedent (L0): growth striations in melt-grown semiconductors record interface
temperature fluctuations of millikelvin size, and Peltier interface demarcation
(current pulses through the melt-crystal interface) writes time-stamp striations
on purpose.

Physics chain
-------------
  text   -> bits -> Hamming(7,4) -> interleave -> Gray-mapped symbols   ChannelCode (optional)
  symbol -> (drive level, layline charge)                       WriterCodebook
  drive  -> interface temperature rise dT(t, x)                 ThermalActuator, first-order lag, lateral spreading
  dT     -> microscopic growth velocity v(t, x)
              melt growth     (heat-transport limited): v = v_pull - (1/G) d(dT)/dt     [edge-sensitive]
              solution growth (interface-kinetics):     v = v0 (sigma(T)/sigma0)^g,
                                                        sigma(T) = (1 + sigma0) exp(-(dlnC_eq/dT) dT) - 1   [level-sensitive]
              slow drift of the growth rate:            v -> v (1 + a f(t))          GrowthDrift (optional)
  v      -> dopant uptake, Burton-Prim-Slichter:  k_eff = k0 / (k0 + (1 - k0) exp(-v delta / D_l))
  k_eff  -> concentration laid down at depth z = integral of v dt     -> striation record c(z, x)
  record -> solid-state diffusion during cool-down and storage: D_s = D0 exp(-Ea / kT)
  read   -> PSF blur, noise, bands located by clock-track markers, features referenced to the clock
            track's own baseline, nearest-centroid decoding, then de-interleave and Hamming correction

Integrity controls in every report: symbol k decoded from band k-1 (retro: must sit at chance, a leak
otherwise), from band k+1 (forward: carry-over between neighbours), and from a shuffled band order
(calibration of the estimator itself).

Evidence levels follow the repository convention (L0 laws, L1 models, L2 parameters, L3 speculation).
Every preset parameter below is L2: replace it with measured values for the actual growth system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import lfilter
    from scipy.ndimage import gaussian_filter1d
except ImportError as exc:  # pragma: no cover
    raise ImportError("crystal_writer needs scipy") from exc

try:
    from .order_records import RecordFidelity, record_fidelity
    from .photonic_thz_pulse_engine import PhotonicTHzEmitterArray
    from .water_structure_assay import (
        RingIllumination,
        SampleCell,
        SymbolCodebook,
        _binomial_sf,
        nearest_centroid_cv,
    )
except ImportError:  # executed as a script or with src/ on sys.path
    from order_records import RecordFidelity, record_fidelity
    from photonic_thz_pulse_engine import PhotonicTHzEmitterArray
    from water_structure_assay import (
        RingIllumination,
        SampleCell,
        SymbolCodebook,
        _binomial_sf,
        nearest_centroid_cv,
    )

K_B_EV = 8.617333262e-5  # L0 eV/K


# =============================================================================
# 1. Growth systems
# =============================================================================


@dataclass(frozen=True)
class GrowthSystem:
    name: str
    mode: str  # "melt" or "solution"
    growth_temperature_k: float
    k0: float  # equilibrium distribution coefficient of the dopant
    liquid_diffusivity_m2_s: float
    boundary_layer_m: float
    nominal_velocity_m_s: float  # pull rate (melt) or velocity at sigma0 (solution)
    solid_d0_m2_s: float
    solid_ea_ev: float
    interface_thermal_time_s: float  # lag of interface temperature behind the heater
    lateral_smoothing_m: float  # heat spreading along the interface
    thermal_gradient_k_m: float = 0.0  # melt only
    supersaturation: float = 0.0  # solution only
    solubility_dlnc_dt: float = 0.0  # solution only, d ln C_eq / dT
    kinetic_order: float = 1.0  # solution only

    def __post_init__(self) -> None:
        if self.mode not in ("melt", "solution"):
            raise ValueError("mode must be 'melt' or 'solution'")
        if not 0.0 < self.k0 < 1.0:
            raise ValueError("this model assumes 0 < k0 < 1 (dopant rejected at the interface)")
        if self.mode == "melt" and self.thermal_gradient_k_m <= 0:
            raise ValueError("melt growth needs a positive thermal gradient")
        if self.mode == "solution" and (self.supersaturation <= 0 or self.solubility_dlnc_dt <= 0):
            raise ValueError("solution growth needs supersaturation > 0 and dlnC_eq/dT > 0")

    # --- incorporation ---------------------------------------------------------------
    def k_eff(self, v_m_s) -> np.ndarray:
        """L0 form (BPS): k_eff = k0 / (k0 + (1 - k0) exp(-v delta / D_l))."""
        u = np.asarray(v_m_s, dtype=float) * self.boundary_layer_m / self.liquid_diffusivity_m2_s
        return self.k0 / (self.k0 + (1.0 - self.k0) * np.exp(-u))

    @property
    def peclet(self) -> float:
        return self.nominal_velocity_m_s * self.boundary_layer_m / self.liquid_diffusivity_m2_s

    @property
    def dlnk_dlnv(self) -> float:
        """Sensitivity of uptake to growth rate at the nominal velocity."""
        u = self.peclet
        e = (1.0 - self.k0) * math.exp(-u)
        return u * e / (self.k0 + e)

    # --- growth velocity ---------------------------------------------------------------
    def velocity(self, delta_t_k: np.ndarray, dt_s: float) -> np.ndarray:
        """Microscopic growth velocity v(t, x) from the interface temperature excursion dT(t, x) (axis 0 = time)."""
        d = np.asarray(delta_t_k, dtype=float)
        if self.mode == "melt":
            # causal backward difference: v(t) must not see dT(t + dt) (a central difference would)
            rate = np.diff(d, axis=0, prepend=d[:1]) / dt_s
            return self.nominal_velocity_m_s - rate / self.thermal_gradient_k_m
        s0 = self.supersaturation
        sigma = (1.0 + s0) * np.exp(-self.solubility_dlnc_dt * d) - 1.0
        return self.nominal_velocity_m_s * np.clip(sigma / s0, 0.0, None) ** self.kinetic_order

    def contrast_per_kelvin(self) -> float:
        """Small-signal |d ln c / d dT| for a step of dT (melt: at the step edge, rise time = interface lag)."""
        if self.mode == "melt":
            peak_rate_per_k = 1.0 / self.interface_thermal_time_s
            dlnv = peak_rate_per_k / (self.thermal_gradient_k_m * self.nominal_velocity_m_s)
        else:
            s0 = self.supersaturation
            dlnv = self.kinetic_order * (1.0 + s0) * self.solubility_dlnc_dt / s0
        return self.dlnk_dlnv * dlnv

    def stall_limit_k(self) -> float:
        """Peak dT at which the interface stops and the record is overwritten (L1).

        Melt: meltback at a step edge when the isotherm retreats faster than the pull, G v_pull tau.
        Solution: dissolution when the supersaturation reaches zero, ln(1 + sigma0) / (dlnC_eq/dT).
        """
        if self.mode == "melt":
            return self.thermal_gradient_k_m * self.nominal_velocity_m_s * self.interface_thermal_time_s
        return math.log(1.0 + self.supersaturation) / self.solubility_dlnc_dt

    # --- record stability ---------------------------------------------------------------
    def solid_diffusivity(self, temperature_k: float) -> float:
        return self.solid_d0_m2_s * math.exp(-self.solid_ea_ev / (K_B_EV * temperature_k))

    def record_lifetime_s(self, band_period_m: float, temperature_k: float) -> float:
        """L0 e-folding time of a sinusoidal striation of period lambda: lambda^2 / (4 pi^2 D_s)."""
        return band_period_m**2 / (4.0 * math.pi**2 * self.solid_diffusivity(temperature_k))


GROWTH_SYSTEMS = {
    # Ga-doped Ge by Czochralski: the classic striation and Peltier-demarcation system.
    "ge_ga_czochralski": GrowthSystem(
        name="ge_ga_czochralski", mode="melt", growth_temperature_k=1211.0,
        k0=0.087,  # L0-ish published equilibrium value for Ga in Ge
        liquid_diffusivity_m2_s=1.9e-8, boundary_layer_m=3e-4, nominal_velocity_m_s=2e-5,  # L2
        solid_d0_m2_s=3.4e-3, solid_ea_ev=3.1,  # L2 Ga in solid Ge (order of magnitude)
        interface_thermal_time_s=0.2, lateral_smoothing_m=0.3e-3, thermal_gradient_k_m=3000.0,  # L2
    ),
    # KDP grown from aqueous solution; any dopant with k0 < 1 (dye, Fe3+).
    "kdp_aqueous_solution": GrowthSystem(
        name="kdp_aqueous_solution", mode="solution", growth_temperature_k=313.0,
        k0=0.1, liquid_diffusivity_m2_s=1e-9, boundary_layer_m=1e-4, nominal_velocity_m_s=5e-7,  # L2
        solid_d0_m2_s=1e-6, solid_ea_ev=1.0,  # L2 rough
        interface_thermal_time_s=2.0, lateral_smoothing_m=0.3e-3,  # L2
        supersaturation=0.05, solubility_dlnc_dt=0.02, kinetic_order=1.5,  # L2
    ),
}


@dataclass(frozen=True)
class GrowthDrift:
    """Slow multiplicative drift of the growth rate, v -> v (1 + a f(t)) (L1).

    ``period_s = 0``: linear ramp from 1 - a to 1 + a across the run (melt level falling, crucible
    depletion, solute depletion in a closed solution). ``period_s > 0``: sinusoid slower than a slot
    (convective or rotational oscillation). Applied to the data and clock tracks alike.
    """

    amplitude: float
    period_s: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.amplitude < 1.0:
            raise ValueError("drift amplitude must lie in [0, 1)")
        if self.period_s < 0:
            raise ValueError("drift period must be >= 0 (0 means a linear ramp)")

    def factor(self, t_s: np.ndarray, duration_s: float) -> np.ndarray:
        t = np.asarray(t_s, dtype=float)
        if self.period_s > 0:
            return 1.0 + self.amplitude * np.sin(2.0 * math.pi * t / self.period_s + self.phase_rad)
        return 1.0 + self.amplitude * (2.0 * t / duration_s - 1.0)

    @property
    def label(self) -> str:
        shape = "ramp" if self.period_s == 0 else f"sine, period {self.period_s:g} s"
        return f"+/-{self.amplitude * 100:g} % {shape}"


# =============================================================================
# 2. Codebook and actuators
# =============================================================================


@dataclass(frozen=True)
class WriterCodebook:
    """Symbol s <-> (drive level index, layline charge). Levels are fractions of the actuator's full dT."""

    levels: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    layline_charges: Tuple[int, ...] = (0, 1, 2, 3)

    @property
    def size(self) -> int:
        return len(self.levels) * len(self.layline_charges)

    def decompose(self, symbol: int) -> Tuple[float, int]:
        if not 0 <= int(symbol) < self.size:
            raise ValueError(f"symbol {symbol} outside codebook of size {self.size}")
        i, j = divmod(int(symbol), len(self.layline_charges))
        return self.levels[i], self.layline_charges[j]

    @classmethod
    def from_frequency_nodes(cls, nodes: SymbolCodebook, array: PhotonicTHzEmitterArray,
                             illumination: RingIllumination, cell: SampleCell) -> "WriterCodebook":
        """Frequency nodes become drive levels through their absorbed power (theory P2: the node reaches
        matter only through absorption). Levels are normalised to the strongest node at charge 0."""
        powers = []
        for f in nodes.frequency_nodes_hz:
            plan = array.plan({"frequency_thz": f / 1e12, "amplitude_normalized": 1.0, "duration_ms": 1000.0,
                               "pattern": "continuous", "emitters": list(range(array.N_CHANNELS))})
            powers.append(illumination.absorbed_power_from_plan(cell, plan))
        p = np.array(powers)
        return cls(tuple(float(v) for v in p / p.max()), nodes.layline_charges)

    @classmethod
    def spot_ring(cls) -> "WriterCodebook":
        """Two drive levels x {central spot (l = 0), ring (l = 3)}: 2 bits per band.

        Lateral heat spreading at the interface blurs the ring charges l = 1, 2, 3 into one another but
        leaves spot against ring distinct, so this is the codebook the crystal actually resolves.
        """
        return cls(levels=(0.5, 1.0), layline_charges=(0, 3))


CODEBOOKS = {
    "full_16": WriterCodebook(),  # 4 levels x 4 charges, 4 bits/band: degradation benchmark
    "spot_ring_4": WriterCodebook.spot_ring(),  # 2 levels x spot/ring, 2 bits/band: operating mode
}


def layline_profile(charge: int, x_m: np.ndarray, illumination: RingIllumination, cell: SampleCell,
                    smoothing_m: float, frequency_hz: float = 1e12) -> np.ndarray:
    """Heating profile across the interface for one layline: centre row of the ring-array intensity,
    spread by lateral heat flow, normalised to peak 1 (so full scale means peak interface dT)."""
    n = illumination.n_emitters
    phases = SymbolCodebook.layline_phases(charge, n)
    intensity, _ = illumination.intensity_map(cell, frequency_hz, np.full(n, 1.0), phases)
    row = intensity[intensity.shape[0] // 2]
    axis = np.linspace(-cell.aperture_radius_m, cell.aperture_radius_m, row.size)
    prof = np.interp(x_m, axis, row)
    dx = float(x_m[1] - x_m[0])
    prof = gaussian_filter1d(prof, smoothing_m / dx, mode="nearest") if smoothing_m > 0 else prof
    return prof / prof.max()


@dataclass(frozen=True, eq=False)
class ThermalActuator:
    """Maps a symbol to an interface temperature-rise profile dT_s(x) at full drive."""

    name: str
    full_scale_k: float
    x_m: np.ndarray
    profiles: np.ndarray  # (n_charges, n_x), peak 1
    charges: Tuple[int, ...]

    def delta_t(self, codebook: WriterCodebook, symbol: int) -> np.ndarray:
        level, charge = codebook.decompose(symbol)
        return level * self.full_scale_k * self.profiles[self.charges.index(charge)]


def make_actuator(name: str, full_scale_k: float, system: GrowthSystem, codebook: WriterCodebook,
                  illumination: RingIllumination = RingIllumination(), aperture_radius_m: float = 2.5e-3,
                  n_x: int = 41) -> ThermalActuator:
    cell = SampleCell(aperture_radius_m=aperture_radius_m)
    x = np.linspace(-0.9 * aperture_radius_m, 0.9 * aperture_radius_m, n_x)
    profiles = np.array([layline_profile(c, x, illumination, cell, system.lateral_smoothing_m)
                         for c in codebook.layline_charges])
    return ThermalActuator(name, float(full_scale_k), x, profiles, codebook.layline_charges)


SPOT_THERMAL_RESISTANCE_K_PER_W = 1.0 / (4.0 * math.pi * 0.6 * 1e-3)  # L1 ~133 K/W: 1 mm source in water


def photonic_full_scale_k(array: Optional[PhotonicTHzEmitterArray] = None,
                          illumination: RingIllumination = RingIllumination(),
                          thermal_resistance_k_per_w: float = SPOT_THERMAL_RESISTANCE_K_PER_W) -> float:
    """Interface dT from the 12-channel THz photomixer array at full drive, 1 THz, charge 0 (L1)."""
    array = array or PhotonicTHzEmitterArray()
    plan = array.plan({"frequency_thz": 1.0, "amplitude_normalized": 1.0, "duration_ms": 1000.0,
                       "pattern": "continuous", "emitters": list(range(array.N_CHANNELS))})
    return illumination.absorbed_power_from_plan(SampleCell(), plan) * thermal_resistance_k_per_w


def ir_pump_full_scale_k(absorbed_power_w: float = 10e-3,
                         thermal_resistance_k_per_w: float = SPOT_THERMAL_RESISTANCE_K_PER_W) -> float:
    """Interface dT from 1550 nm pump light absorbed in an aqueous solution (water absorbs ~10 /cm there)."""
    return absorbed_power_w * thermal_resistance_k_per_w


# =============================================================================
# 3. Channel coding: Gray mapping, Hamming(7,4), block interleaving
# =============================================================================


def _bits_per_axis(n: int, axis: str) -> int:
    if n < 1 or n & (n - 1):
        raise ValueError(f"{axis} count {n} must be a power of two for a bit mapping")
    return n.bit_length() - 1


def gray_code(i):
    """Reflected binary Gray code: neighbouring integers differ in exactly one bit."""
    i = np.asarray(i, dtype=np.int64)
    return i ^ (i >> 1)


def symbol_words(codebook: WriterCodebook, gray: bool = True) -> np.ndarray:
    """Bit word carried by each symbol: (G(level index) << charge bits) | G(charge index).

    The misread structure of the crystal is per axis and one step at a time (a neighbouring level or a
    neighbouring layline). With ``gray`` each such error flips exactly one bit, which a single-error-
    correcting code can repair; a diagonal error (both axes) flips two.
    """
    _bits_per_axis(len(codebook.levels), "level")
    charge_bits = _bits_per_axis(len(codebook.layline_charges), "layline")
    lev = np.arange(len(codebook.levels))
    chg = np.arange(len(codebook.layline_charges))
    if gray:
        lev, chg = gray_code(lev), gray_code(chg)
    return ((lev[:, None] << charge_bits) | chg[None, :]).ravel()


def symbols_to_bits(symbols: Sequence[int], codebook: WriterCodebook, gray: bool = True) -> np.ndarray:
    bps = int(math.log2(codebook.size))
    words = symbol_words(codebook, gray)[np.asarray(symbols, dtype=int)]
    return ((words[:, None] >> np.arange(bps - 1, -1, -1)) & 1).ravel().astype(np.int8)


def bits_to_symbols(bits: Sequence[int], codebook: WriterCodebook, gray: bool = True) -> np.ndarray:
    bps = int(math.log2(codebook.size))
    b = np.asarray(bits, dtype=np.int64)
    if b.size % bps:
        raise ValueError(f"bit count {b.size} is not a multiple of {bps} bits per symbol")
    words = b.reshape(-1, bps) @ (1 << np.arange(bps - 1, -1, -1))
    return np.argsort(symbol_words(codebook, gray))[words]


# Hamming(7,4), positions 1..7 = p1 p2 d1 p3 d2 d3 d4; the syndrome is the binary index of a flipped position.
_H74_DATA_POS = np.array([2, 4, 5, 6])
_H74_CHECK = np.array([[(j + 1) >> r & 1 for j in range(7)] for r in range(3)])


def hamming74_encode(data_bits: Sequence[int]) -> np.ndarray:
    d = np.asarray(data_bits, dtype=np.int8)
    if d.size % 4:
        raise ValueError("Hamming(7,4) needs a multiple of 4 data bits")
    d = d.reshape(-1, 4)
    w = np.zeros((d.shape[0], 7), dtype=np.int8)
    w[:, _H74_DATA_POS] = d
    for r in range(3):
        cover = [j for j in range(7) if (j + 1) >> r & 1 and j not in (0, 1, 3)]
        w[:, (1 << r) - 1] = w[:, cover].sum(axis=1) % 2
    return w.ravel()


def hamming74_decode(code_bits: Sequence[int]) -> Tuple[np.ndarray, int]:
    """Single-error correction per codeword. Returns (data bits, number of codewords corrected).

    Two errors in one codeword are miscorrected (the code's distance is 3); interleaving and Gray mapping
    exist to keep a single misread band from putting two errors into one codeword.
    """
    w = np.asarray(code_bits, dtype=np.int8).reshape(-1, 7).copy()
    syndrome = (w @ _H74_CHECK.T % 2) @ np.array([1, 2, 4])
    hit = np.nonzero(syndrome)[0]
    w[hit, syndrome[hit] - 1] ^= 1
    return w[:, _H74_DATA_POS].ravel(), int(hit.size)


def interleave(bits: Sequence[int], n: int = 7, depth: Optional[int] = None) -> np.ndarray:
    """Block interleaver: ``depth`` codewords of length n per block, sent column by column.

    Consecutive transmitted bits then come from different codewords, so the bits of one band (2 or 4 of
    them) and a run of drift-damaged bands are spread over many codewords. ``depth=None`` interleaves the
    whole message; ``depth=1`` is the identity.
    """
    return _block_permute(bits, n, depth, inverse=False)


def deinterleave(bits: Sequence[int], n: int = 7, depth: Optional[int] = None) -> np.ndarray:
    return _block_permute(bits, n, depth, inverse=True)


def _block_permute(bits: Sequence[int], n: int, depth: Optional[int], inverse: bool) -> np.ndarray:
    b = np.asarray(bits)
    if b.size % n:
        raise ValueError("interleaver input must be whole codewords")
    n_cw = b.size // n
    d = n_cw if depth is None else max(1, int(depth))
    out = b.copy()
    for start in range(0, n_cw, d):
        m = min(d, n_cw - start)
        block = b[start * n:(start + m) * n]
        # forward: rows are codewords, read by column; inverse: undo that transpose
        out[start * n:(start + m) * n] = (block.reshape(n, m).T if inverse else block.reshape(m, n).T).ravel()
    return out


def text_to_bits(text: str) -> np.ndarray:
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(data).astype(np.int8)


def bits_to_text(bits: Sequence[int]) -> str:
    b = np.asarray(bits, dtype=np.uint8)
    return np.packbits(b[: b.size - b.size % 8]).tobytes().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ChannelCode:
    """Bits -> optional Hamming(7,4) -> optional block interleave -> Gray-mapped symbols.

    Hamming(7,4) corrects one bit per 7-bit codeword at rate 4/7. It earns its overhead only when the raw
    symbol error rate is low (a few per cent): at the 16-symbol codebook's ~35 % error the message is lost
    with or without it, and that mode stays uncoded as the degradation benchmark.
    """

    fec: str = "hamming74"  # "hamming74" or "none"
    gray: bool = True
    interleave_depth: Optional[int] = None  # codewords per interleaver block; None = whole message, 1 = off

    def __post_init__(self) -> None:
        if self.fec not in ("hamming74", "none"):
            raise ValueError("fec must be 'hamming74' or 'none'")

    @property
    def rate(self) -> float:
        return 4.0 / 7.0 if self.fec == "hamming74" else 1.0

    @property
    def label(self) -> str:
        parts = ["Hamming(7,4)" if self.fec == "hamming74" else "uncoded", "Gray" if self.gray else "natural"]
        if self.fec == "hamming74":
            parts.append("no interleave" if self.interleave_depth == 1 else
                         f"interleave depth {self.interleave_depth or 'full'}")
        return ", ".join(parts)

    def n_code_bits(self, n_data_bits: int) -> int:
        return 7 * math.ceil(n_data_bits / 4) if self.fec == "hamming74" else n_data_bits

    def n_symbols(self, n_data_bits: int, codebook: WriterCodebook) -> int:
        return math.ceil(self.n_code_bits(n_data_bits) / math.log2(codebook.size))

    def encode(self, data_bits: Sequence[int], codebook: WriterCodebook) -> np.ndarray:
        bits = np.asarray(data_bits, dtype=np.int8)
        if self.fec == "hamming74":
            bits = np.concatenate([bits, np.zeros((-bits.size) % 4, np.int8)])
            bits = interleave(hamming74_encode(bits), 7, self.interleave_depth)
        bps = int(math.log2(codebook.size))
        bits = np.concatenate([bits, np.zeros((-bits.size) % bps, np.int8)])
        return bits_to_symbols(bits, codebook, self.gray)

    def decode(self, symbols: Sequence[int], codebook: WriterCodebook, n_data_bits: int) -> Tuple[np.ndarray, int]:
        """Hard-decision decode. Returns (data bits, codewords corrected)."""
        bits = symbols_to_bits(symbols, codebook, self.gray)[: self.n_code_bits(n_data_bits)]
        if self.fec == "none":
            return bits[:n_data_bits], 0
        data, corrected = hamming74_decode(deinterleave(bits, 7, self.interleave_depth))
        return data[:n_data_bits], corrected

    def encode_text(self, text: str, codebook: WriterCodebook) -> np.ndarray:
        return self.encode(text_to_bits(text), codebook)

    def decode_text(self, symbols: Sequence[int], codebook: WriterCodebook, n_bytes: int) -> Tuple[str, int]:
        data, corrected = self.decode(symbols, codebook, 8 * n_bytes)
        return bits_to_text(data), corrected


# =============================================================================
# 4. Writing: growth simulation
# =============================================================================


@dataclass(frozen=True)
class WriteSchedule:
    symbols: Tuple[int, ...]
    is_training: Tuple[bool, ...]
    train_group: Tuple[int, ...]  # repeat index for training slots, -1 for payload
    slot_on_s: float
    slot_off_s: float

    @property
    def n_slots(self) -> int:
        return len(self.symbols)

    @property
    def slot_s(self) -> float:
        return self.slot_on_s + self.slot_off_s

    @property
    def duration_s(self) -> float:
        return self.n_slots * self.slot_s


def make_schedule(codebook: WriterCodebook, payload: Sequence[int], n_train: int = 6, slot_on_s: float = 1.0,
                  slot_off_s: float = 1.0, seed: int = 0) -> WriteSchedule:
    """Preamble of n_train x codebook.size i.i.d. uniform symbols (n_train contiguous CV groups), then the payload.

    I.i.d. draws are required for the retro check: a preamble of per-repeat permutations makes each symbol
    partly predictable from its predecessors, which shows up as a false backwards leak.
    """
    if n_train < 2:
        raise ValueError("need >= 2 training groups for cross-validated fidelity")
    rng = np.random.default_rng(seed)
    syms: List[int] = []
    train: List[bool] = []
    group: List[int] = []
    for r in range(n_train):
        for s in rng.integers(0, codebook.size, codebook.size):
            syms.append(int(s))
            train.append(True)
            group.append(r)
    for s in payload:
        if not 0 <= int(s) < codebook.size:
            raise ValueError("payload symbol outside codebook")
        syms.append(int(s))
        train.append(False)
        group.append(-1)
    return WriteSchedule(tuple(syms), tuple(train), tuple(group), slot_on_s, slot_off_s)


@dataclass(frozen=True, eq=False)
class CrystalRecord:
    """Flat-interface, local-uptake record (L1).

    The interface advances as one front (lateral coupling through step flow on facets, or the
    isotherm in melt growth), at the lateral mean of the local kinetic velocities; dopant uptake
    at each lateral position follows the local velocity. A clock track at the crystal edge
    carries one marker pulse per slot, the demarcation-line technique used to time-stamp growth.
    """

    system: GrowthSystem
    schedule: WriteSchedule
    t_s: np.ndarray
    x_m: np.ndarray
    delta_t_k: np.ndarray  # (t, x) interface temperature excursion
    velocity_m_s: np.ndarray  # (t, x) local kinetic velocity (sets uptake)
    interface_velocity_m_s: np.ndarray  # (t,) advance of the flat front
    concentration_rel: np.ndarray  # (t, x), c / c_nominal
    clock_concentration_rel: np.ndarray  # (t,), clock track
    depth_m: np.ndarray  # (t,) depth of material laid down at t
    stalled: bool  # interface stopped or reversed (meltback / dissolution): record locally overwritten
    marker_full_scale_k: float
    marker_duration_s: float
    drift: Optional[GrowthDrift] = None

    @property
    def length_m(self) -> float:
        return float(self.depth_m[-1])

    def section(self, dz_m: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z, c(z, x), c_clock(z)) on a uniform depth grid."""
        dz = dz_m or self.system.nominal_velocity_m_s * float(self.t_s[1] - self.t_s[0])
        z = np.arange(0.0, self.length_m, dz)
        c = np.empty((z.size, self.x_m.size))
        for j in range(self.x_m.size):
            c[:, j] = np.interp(z, self.depth_m, self.concentration_rel[:, j])
        return z, c, np.interp(z, self.depth_m, self.clock_concentration_rel)


def write_crystal(system: GrowthSystem, actuator: ThermalActuator, codebook: WriterCodebook,
                  schedule: WriteSchedule, marker_full_scale_k: Optional[float] = None,
                  marker_duration_s: Optional[float] = None, dt_s: Optional[float] = None,
                  drift: Optional[GrowthDrift] = None) -> CrystalRecord:
    """Grow the crystal under the drive schedule. Deterministic; noise enters only at readout.

    Markers default to half the stall limit for two interface time constants at every slot start.
    ``drift`` scales the growth rate of data and clock tracks by the same slow factor.
    """
    tau = system.interface_thermal_time_s
    dt = dt_s or min(tau / 10.0, schedule.slot_on_s / 20.0)
    marker_k = marker_full_scale_k if marker_full_scale_k is not None else 0.5 * system.stall_limit_k()
    marker_s = marker_duration_s if marker_duration_s is not None else 2.0 * tau
    steps_on = int(round(schedule.slot_on_s / dt))
    steps_slot = int(round(schedule.slot_s / dt))
    steps_marker = max(1, int(round(marker_s / dt)))
    n_t = steps_slot * schedule.n_slots
    drive = np.zeros((n_t, actuator.x_m.size))
    clock = np.zeros(n_t)
    for k, s in enumerate(schedule.symbols):
        drive[k * steps_slot:k * steps_slot + steps_on] = actuator.delta_t(codebook, s)
        clock[k * steps_slot:k * steps_slot + steps_marker] = marker_k
    decay = math.exp(-dt / tau)
    delta_t = lfilter([1.0 - decay], [1.0, -decay], drive, axis=0)
    delta_clock = lfilter([1.0 - decay], [1.0, -decay], clock)
    t = np.arange(n_t) * dt
    v = system.velocity(delta_t, dt)
    v_clock = system.velocity(delta_clock[:, None], dt)[:, 0]
    if drift is not None:
        f = drift.factor(t, schedule.duration_s)
        v, v_clock = v * f[:, None], v_clock * f
    v_front = v.mean(axis=1)
    floor = 1e-3 * system.nominal_velocity_m_s
    stalled = bool(np.any(v <= floor) or np.any(v_clock <= floor) or np.any(v_front <= floor))
    k_nom = system.k_eff(system.nominal_velocity_m_s)
    c = system.k_eff(np.maximum(v, floor)) / k_nom
    c_clock = system.k_eff(np.maximum(v_clock, floor)) / k_nom
    z = np.cumsum(np.maximum(v_front, floor)) * dt
    return CrystalRecord(system, schedule, t, actuator.x_m, delta_t, np.maximum(v, floor), v_front, c, c_clock, z,
                         stalled, float(marker_k), float(steps_marker * dt), drift)


# =============================================================================
# 5. Storage and readout
# =============================================================================


@dataclass(frozen=True)
class Readout:
    """Profiling instrument (spreading resistance, SIMS, cathodoluminescence; L2).

    Data features are bin averages, ``depth_samples_per_slot`` bins per band at ``lateral_samples``
    positions. The clock track is scanned densely. Every reported value carries ``relative_noise``.

    ``clock_reference`` subtracts a baseline from the log features: the clock track's mean log uptake over
    the last ``reference_fraction`` of each band's window, interpolated to every depth bin. The clock track
    drifts with the data track but carries no symbol, so the subtraction removes a slowly varying growth
    rate (and its uptake baseline), which the markers alone cannot: markers fix where a band is, the
    reference fixes what zero means in it. Cost without drift: under 1 point (Ge:Ga), 2 points (KDP).
    """

    psf_sigma_m: float = 1.0e-6  # L2 depth resolution
    relative_noise: float = 0.01  # L2 per reported value
    lateral_samples: int = 15
    depth_samples_per_slot: int = 8
    trailing_guard: float = 0.05  # margin below the thinnest band (5th-percentile width)
    clock_reference: bool = True
    reference_fraction: float = 0.5  # tail of the window, after the marker has relaxed


def anneal(z_m: np.ndarray, c: np.ndarray, system: GrowthSystem, temperature_k: float, time_s: float) -> np.ndarray:
    """Solid-state diffusion along depth: Gaussian smoothing with sigma = sqrt(2 D_s t)."""
    sigma = math.sqrt(2.0 * system.solid_diffusivity(temperature_k) * time_s)
    dz = float(z_m[1] - z_m[0])
    if sigma / dz < 1e-3:
        return c.copy()
    return gaussian_filter1d(c, sigma / dz, axis=0, mode="nearest")


def locate_markers(z_m: np.ndarray, clock_measured: np.ndarray, n_slots: int, smoothing_m: float) -> Tuple[np.ndarray, bool]:
    """Slot boundaries from the clock track: the n-1 strongest uptake dips after the first slot.

    Returns (boundaries of length n_slots + 1, sync_ok). Slot 0 starts at z = 0 by construction.
    """
    from scipy.signal import find_peaks

    dz = float(z_m[1] - z_m[0])
    signal = gaussian_filter1d(-np.log(np.clip(clock_measured, 1e-12, None)), max(smoothing_m / dz, 1.0))
    mean_slot = z_m[-1] / n_slots
    peaks, props = find_peaks(signal, distance=max(1, int(0.5 * mean_slot / dz)), height=-np.inf)
    peaks = peaks[z_m[peaks] > 0.5 * mean_slot]
    if peaks.size < n_slots - 1:
        uniform = np.linspace(0.0, z_m[-1], n_slots + 1)
        return uniform, False
    strongest = np.sort(peaks[np.argsort(signal[peaks])[::-1][: n_slots - 1]])
    return np.concatenate([[0.0], z_m[strongest], [z_m[-1]]]), True


def read_crystal(record: CrystalRecord, readout: Readout = Readout(), cooldown_s: float = 600.0,
                 storage: Sequence[Tuple[float, float]] = ((300.0, 3.15e7),),
                 seed: int = 0) -> Tuple[np.ndarray, bool]:
    """Section, anneal (cool-down at growth temperature, then storage), blur, measure, segment, bin.

    Returns (features (n_slots, lateral x depth bins) as log relative concentration, referenced to the
    clock-track baseline when ``readout.clock_reference``, and sync_ok).
    """
    rng = np.random.default_rng(seed)
    z, c, c_clock = record.section()
    sysm = record.system
    for temperature, duration in ((sysm.growth_temperature_k, cooldown_s),) + tuple(storage):
        c = anneal(z, c, sysm, temperature, duration)
        c_clock = anneal(z, c_clock[:, None], sysm, temperature, duration)[:, 0]
    dz = float(z[1] - z[0])
    c = gaussian_filter1d(c, readout.psf_sigma_m / dz, axis=0, mode="nearest")
    c_clock = gaussian_filter1d(c_clock, readout.psf_sigma_m / dz, mode="nearest")
    clock_meas = c_clock * (1.0 + readout.relative_noise * rng.standard_normal(c_clock.shape))
    marker_scale = sysm.nominal_velocity_m_s * sysm.interface_thermal_time_s
    bounds, sync_ok = locate_markers(z, clock_meas, record.schedule.n_slots, max(readout.psf_sigma_m, 0.5 * marker_scale))
    # Marker lag from the known protocol: melt uptake dips at the pulse edge (max d(dT)/dt), solution uptake
    # dips at the end of the pulse (max dT). Shift each boundary back by that fraction of the local band.
    lag_frac = 0.0 if sysm.mode == "melt" else record.marker_duration_s / record.schedule.slot_s
    # The correction uses the global median band width: a local width would include band k's own thickness,
    # which depends on symbol k, and leak it into band k-1's start.
    if sync_ok and lag_frac > 0:
        w_med = float(np.median(np.diff(bounds)))
        bounds = np.concatenate([[0.0], np.maximum(bounds[1:-1] - lag_frac * w_med, 0.0), [bounds[-1]]])

    lat = np.linspace(0, record.x_m.size - 1, readout.lateral_samples).round().astype(int)
    cl = c[:, lat]
    nb = readout.depth_samples_per_slot
    feats = np.empty((record.schedule.n_slots, readout.lateral_samples * nb))
    # Fixed-length windows from each band's own start. A window ending at the next marker would let the next
    # symbol (which shifts that marker) leak backwards; a window longer than the thinnest band would overrun
    # into the next symbol's onset (solution growth thins heated bands). So: fit the thinnest bands.
    width = float(np.quantile(np.diff(bounds), 0.05)) * (1.0 - readout.trailing_guard)
    log_clock = np.log(np.clip(clock_meas, 1e-12, None))
    ref_z = np.empty(record.schedule.n_slots)
    ref_v = np.empty(record.schedule.n_slots)
    bin_z = np.empty((record.schedule.n_slots, nb))
    for k in range(record.schedule.n_slots):
        edges = np.linspace(bounds[k], min(bounds[k] + width, z[-1]), nb + 1)
        idx = np.clip(np.searchsorted(z, edges), 0, z.size - 1)
        block = np.empty((readout.lateral_samples, nb))
        for b in range(nb):
            lo, hi = idx[b], max(idx[b + 1], idx[b] + 1)
            block[:, b] = cl[lo:hi].mean(axis=0)
        block = block * (1.0 + readout.relative_noise * rng.standard_normal(block.shape))
        feats[k] = np.log(np.clip(block, 1e-12, None)).ravel()
        bin_z[k] = 0.5 * (edges[:-1] + edges[1:])
        lo = int(np.searchsorted(z, edges[0] + (1.0 - readout.reference_fraction) * (edges[-1] - edges[0])))
        hi = max(int(idx[-1]), lo + 1)
        ref_v[k] = log_clock[lo:hi].mean()
        ref_z[k] = 0.5 * (z[lo] + z[min(hi, z.size - 1)])
    if readout.clock_reference:
        # Baseline at each bin's own depth, interpolated between band tails: follows a drift that changes
        # within a band, which one constant per band cannot (+2 points in KDP, equal in Ge:Ga). Checked for a
        # backward leak (band k's tail entering band k-1's late bins): retro rates match the per-band version.
        per_bin = np.interp(bin_z, ref_z, ref_v)  # (slots, nb)
        feats -= np.repeat(per_bin[:, None, :], readout.lateral_samples, axis=1).reshape(feats.shape)
    return feats, sync_ok


# =============================================================================
# 6. Decoding and report
# =============================================================================


@dataclass(frozen=True)
class TemporalControls:
    """Symbol k decoded from a band other than its own. All three sit at chance in a clean record.

    retro    band k-1 -> symbol k   a reader leak: the reader let a later symbol shape an earlier band
    forward  band k+1 -> symbol k   carry-over: heat tail, solute boundary layer or diffusion pushing
                                     symbol k into the next band (physical and causal, but interference)
    shuffled band order randomised, then retro   calibration: the estimator and its permutation test
    """

    retro: RecordFidelity
    forward: RecordFidelity
    shuffled: RecordFidelity


def temporal_controls(symbols: Sequence[int], features: np.ndarray, groups: Sequence[int],
                      n_permutations: int = 500, seed: int = 0) -> TemporalControls:
    """Neighbour-band decoding for a contiguous run of slots (the i.i.d. preamble)."""
    y = np.asarray(symbols, dtype=int)
    f = np.asarray(features, dtype=float)
    g = np.asarray(groups)
    retro = record_fidelity(y[1:], f[:-1], g[1:], n_permutations, seed + 7, standardize=False)
    forward = record_fidelity(y[:-1], f[1:], g[:-1], n_permutations, seed + 11, standardize=False)
    perm = np.random.default_rng(seed + 13).permutation(y.size)
    ys, fs, gs = y[perm], f[perm], g[perm]
    shuffled = record_fidelity(ys[1:], fs[:-1], gs[1:], n_permutations, seed + 17, standardize=False)
    return TemporalControls(retro, forward, shuffled)


@dataclass(frozen=True)
class WriterReport:
    system: str
    actuator: str
    full_scale_k: float
    peclet: float
    contrast_per_kelvin: float
    stalled: bool
    sync_ok: bool  # clock-track markers found for every slot
    drift: Optional[str]
    clock_reference: bool
    training_accuracy: float
    training_excess_bits: float
    training_p: float
    level_accuracy: float  # drive level alone, from the cross-validated preamble
    layline_accuracy: float  # layline charge alone
    retro_accuracy: float  # symbol k decoded from slot k-1: must sit at chance
    retro_p: float
    forward_accuracy: float  # symbol k decoded from slot k+1: carry-over between neighbours
    forward_p: float
    shuffled_accuracy: float  # retro after randomising band order: estimator calibration
    shuffled_p: float
    payload_accuracy: float  # raw symbol accuracy, before any error correction
    payload_p: float
    chance: float
    code: Optional[str]
    code_rate: float
    payload_symbols: int
    raw_bit_errors: Optional[int]  # channel bits wrong before correction
    codewords_corrected: Optional[int]
    data_bit_errors: Optional[int]  # message bits wrong after correction
    message_exact: Optional[bool]
    sent_text: Optional[str]
    received_text: Optional[str]
    band_period_m: float
    lifetime_300k_s: float
    required_full_scale_k: float

    def summary(self) -> str:
        yrs = self.lifetime_300k_s / 3.15e7
        c = self.chance
        lines = [
            f"{self.system} written by {self.actuator}: full-scale dT {self.full_scale_k * 1e3:.2f} mK, "
            f"Peclet {self.peclet:.3f}, contrast {self.contrast_per_kelvin * 100:.2f} %/K"
            + (f", growth-rate drift {self.drift}" if self.drift else "")
            + ("  INTERFACE STALLED (meltback or dissolution): record overwritten" if self.stalled else "")
            + ("" if self.sync_ok else "  CLOCK TRACK LOST: bands located by mean growth rate"),
            f"  preamble decode {self.training_accuracy:.3f} (chance {c:.3f}), "
            f"{self.training_excess_bits:.2f} bits/symbol, p {self.training_p:.3g}; "
            f"level {self.level_accuracy:.2f}, layline {self.layline_accuracy:.2f}"
            + ("" if self.clock_reference else "; no clock-track reference"),
            f"  neighbour controls (chance {c:.3f}): retro k-1 {self.retro_accuracy:.3f} p {self.retro_p:.2g}; "
            f"forward k+1 {self.forward_accuracy:.3f} p {self.forward_p:.2g}; "
            f"shuffled {self.shuffled_accuracy:.3f} p {self.shuffled_p:.2g}",
        ]
        if self.payload_symbols:
            lines.append(f"  payload {self.payload_symbols} bands, raw symbol accuracy {self.payload_accuracy:.3f}, "
                         f"binomial p {self.payload_p:.3g}")
        if self.code is not None and self.data_bit_errors is not None:
            corr = "" if self.codewords_corrected is None else f", {self.codewords_corrected} codewords corrected"
            lines.append(f"  code {self.code} (rate {self.code_rate:.3g}): {self.raw_bit_errors} channel bit errors"
                         f"{corr}, {self.data_bit_errors} message bit errors after decoding")
        if self.sent_text is not None:
            lines.append(f"  sent {self.sent_text!r} -> read {self.received_text!r}"
                         + ("  exact" if self.message_exact else ""))
        lines.append(f"  band period {self.band_period_m * 1e6:.1f} um; record e-folding time at 300 K "
                     f"{yrs:.3g} years; full scale needed for SNR 5: {self.required_full_scale_k * 1e3:.2f} mK")
        return "\n".join(lines)


def required_full_scale_k(system: GrowthSystem, codebook: WriterCodebook, readout: Readout, snr: float = 5.0) -> float:
    """dT full scale at which the closest drive levels differ by ``snr`` noise units in one band average (L1)."""
    levels = np.sort(np.array(codebook.levels))
    step = float(np.min(np.diff(levels))) if levels.size > 1 else float(levels[0])
    noise = readout.relative_noise / math.sqrt(readout.lateral_samples * readout.depth_samples_per_slot)
    return snr * noise / (system.contrast_per_kelvin() * step)


UNCODED = ChannelCode(fec="none", gray=False)  # byte bits straight onto symbols: the original mapping
HAMMING_GRAY = ChannelCode()  # Hamming(7,4), Gray, whole-message interleave


def run_writer(system: GrowthSystem, actuator: ThermalActuator, codebook: WriterCodebook,
               message: Optional[str] = None, payload: Optional[Sequence[int]] = None,
               readout: Readout = Readout(), n_train: int = 6, slot_on_s: Optional[float] = None,
               slot_off_s: Optional[float] = None, n_permutations: int = 500, seed: int = 0,
               storage: Sequence[Tuple[float, float]] = ((300.0, 3.15e7),),
               code: Optional[ChannelCode] = None, drift: Optional[GrowthDrift] = None,
               ) -> Tuple[WriterReport, CrystalRecord]:
    """Write a preamble plus a message, store, read back, decode.

    ``message`` is encoded with ``code`` (default ``UNCODED``: byte bits straight onto symbols; pass
    ``HAMMING_GRAY`` for error correction). ``payload`` sends raw symbols instead. The preamble is always
    uncoded i.i.d. symbols: it trains the decoder and carries the neighbour controls.
    """
    if message is not None:
        code = code or UNCODED
        payload = code.encode_text(message, codebook).tolist()
    payload = list(payload or [])
    on = slot_on_s if slot_on_s is not None else 5.0 * system.interface_thermal_time_s
    off = slot_off_s if slot_off_s is not None else 5.0 * system.interface_thermal_time_s
    sched = make_schedule(codebook, payload, n_train, on, off, seed)
    rec = write_crystal(system, actuator, codebook, sched, drift=drift)
    feats, sync_ok = read_crystal(rec, readout, storage=storage, seed=seed + 1)
    y = np.array(sched.symbols)
    train = np.array(sched.is_training)
    groups = np.array(sched.train_group)

    # One instrument, one noise level per value: no z-scoring (see nearest_centroid_cv)
    fid = record_fidelity(y[train], feats[train], groups[train], n_permutations, seed, standardize=False)
    cv_pred = nearest_centroid_cv(feats[train], y[train][None, :], groups[train], codebook.size, False)[0]
    n_charges = len(codebook.layline_charges)
    level_acc = float(np.mean(cv_pred // n_charges == y[train] // n_charges))
    layline_acc = float(np.mean(cv_pred % n_charges == y[train] % n_charges))
    ctrl = temporal_controls(y[train], feats[train], groups[train], n_permutations, seed)

    pay_acc = pay_p = math.nan
    received = None
    raw_err = corrected = data_err = exact = None
    if payload:
        test = ~train
        labels = np.concatenate([y[train], np.zeros(int(test.sum()), int)])
        g = np.concatenate([np.zeros(int(train.sum()), int), np.ones(int(test.sum()), int)])
        x_all = np.vstack([feats[train], feats[test]])
        pred = nearest_centroid_cv(x_all, labels[None, :], g, codebook.size, False)[0][int(train.sum()):]
        k = int(np.sum(pred == y[test]))
        pay_acc = k / pred.size
        pay_p = _binomial_sf(k, pred.size, 1.0 / codebook.size)
        if message is not None:
            n_bytes = len(message.encode("utf-8"))
            raw_err = int(np.sum(symbols_to_bits(pred, codebook, code.gray) != symbols_to_bits(y[test], codebook, code.gray)))
            data, corrected = code.decode(pred, codebook, 8 * n_bytes)
            data_err = int(np.sum(data != text_to_bits(message)))
            received = bits_to_text(data)
            exact = data_err == 0
            corrected = corrected if code.fec != "none" else None

    period = system.nominal_velocity_m_s * sched.slot_s
    report = WriterReport(
        system=system.name, actuator=actuator.name, full_scale_k=actuator.full_scale_k, peclet=system.peclet,
        contrast_per_kelvin=system.contrast_per_kelvin(), stalled=rec.stalled, sync_ok=sync_ok,
        drift=drift.label if drift is not None else None, clock_reference=readout.clock_reference,
        training_accuracy=fid.accuracy, training_excess_bits=fid.excess_bits, training_p=fid.p_value,
        level_accuracy=level_acc, layline_accuracy=layline_acc,
        retro_accuracy=ctrl.retro.accuracy, retro_p=ctrl.retro.p_value,
        forward_accuracy=ctrl.forward.accuracy, forward_p=ctrl.forward.p_value,
        shuffled_accuracy=ctrl.shuffled.accuracy, shuffled_p=ctrl.shuffled.p_value,
        payload_accuracy=pay_acc, payload_p=pay_p, chance=1.0 / codebook.size,
        code=code.label if (message is not None and code is not None) else None,
        code_rate=code.rate if (message is not None and code is not None) else 1.0,
        payload_symbols=len(payload), raw_bit_errors=raw_err, codewords_corrected=corrected,
        data_bit_errors=data_err, message_exact=exact,
        sent_text=message, received_text=received, band_period_m=period,
        lifetime_300k_s=system.record_lifetime_s(period, 300.0),
        required_full_scale_k=required_full_scale_k(system, codebook, readout),
    )
    return report, rec


# =============================================================================
# 7. Demo
# =============================================================================


def demo(message: str = "crystal") -> str:
    full = CODEBOOKS["full_16"]
    spot_ring = CODEBOOKS["spot_ring_4"]
    ge = GROWTH_SYSTEMS["ge_ga_czochralski"]
    kdp = GROWTH_SYSTEMS["kdp_aqueous_solution"]
    ir = ir_pump_full_scale_k(12e-3)
    rows = ["CRYSTAL WRITER"]
    cases = [
        # (system, codebook, actuator label, full scale, code, drift)
        (ge, full, "Peltier interface pulses", 8e-3, UNCODED, None),  # degradation benchmark
        (ge, spot_ring, "Peltier interface pulses", 8e-3, HAMMING_GRAY, None),  # operating mode
        (ge, spot_ring, "Peltier interface pulses", 8e-3, HAMMING_GRAY, GrowthDrift(0.1)),
        (kdp, full, "THz photomixer array", photonic_full_scale_k(), UNCODED, None),
        (kdp, full, "1550 nm pump light, 12 mW absorbed", ir, UNCODED, None),
        (kdp, spot_ring, "1550 nm pump light, 12 mW absorbed", ir, HAMMING_GRAY, None),
    ]
    for system, cb, label, scale, code, drift in cases:
        act = make_actuator(label, scale, system, cb)
        rep, _ = run_writer(system, act, cb, message=message, code=code, drift=drift, seed=3)
        rows.append(f"[{cb.size} symbols: levels {cb.levels}, laylines {cb.layline_charges}]")
        rows.extend("  " + line for line in rep.summary().splitlines())
        rows.append(f"  stall limit (peak dT before {'meltback' if system.mode == 'melt' else 'dissolution'}): "
                    f"{system.stall_limit_k() * 1e3:.1f} mK")
    return "\n".join(rows)


if __name__ == "__main__":
    print(demo())
