"""
Water Structure Assay
=====================

A falsification harness, built on the photonic THz pulse engine, for two claims:

  (P) Persistence  - a THz drive leaves liquid water in a state that differs
                     from a matched control after every known classical
                     memory (heat) has relaxed.
  (C) Channel      - a symbol stream (text, EEG-derived substrate states, or
                     random draws) written into the water as a frequency node
                     plus a geometric phase layline can be read back, and the
                     temporal channel map shows when that information exists:
                     before the symbol was chosen, during the drive, after
                     shutdown, and in later trials.

Physics baseline (L0). Liquid water's hydrogen-bond network rearranges on a
~1 ps timescale; the slowest known memory of a driven sample cell is thermal
(seconds for the cell, ~minutes for lateral diffusion). Anything that survives
``SampleCell.classical_memory_s`` with no counterpart in the sham and
heater-matched arms has no L0 explanation. The assay is designed so that such
a result, if it appears, is hard to dismiss, and so that its absence is an
informative null rather than a shrug.

Morphic-field and temporal-communication interpretations are L3. The assay
makes them testable; it does not presuppose them. The simulator can inject a
hypothetical persistent effect (``InjectedEffect``) solely as a positive
control that proves the pipeline would detect one.

Pipeline
--------
    stream (TextStream / EEGSubstrateStream / RandomStream)
        -> symbol -> SymbolCodebook (frequency node x layline charge)
        -> equalised drive command -> PhotonicTHzEmitterArray.plan()
    Preregistration (SHA-256 digest)  +  blinded schedule (hash-committed key)
    ThermalCalibration (heater steps, before the main run)
    run_assay(Instrument) -> AssayDataset (windows: pre, during, post@delays)
    analyze(dataset) -> BlindedAnalysis      (codes only, no arm names)
    unblind(analysis, key) -> AssayReport    (verdicts, channel map, message)

Evidence levels follow the repository convention (rtd_300ghz/CHANGELOG.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .photonic_thz_pulse_engine import (
        C0,
        TAU,
        PhotonicDrivePlan,
        PhotonicTHzEmitterArray,
        water_absorption_per_m,
    )
except ImportError:  # executed as a script or with src/ on sys.path
    from photonic_thz_pulse_engine import (
        C0,
        TAU,
        PhotonicDrivePlan,
        PhotonicTHzEmitterArray,
        water_absorption_per_m,
    )

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None


# =============================================================================
# 1. Sample cell: optics and heat
# =============================================================================

WATER_THERMAL_DIFFUSIVITY = 1.43e-7  # L0 m^2/s near 25 C
WATER_VOLUMETRIC_HEAT_CAPACITY = 4.18e6  # L0 J/(m^3 K)
T_REF_K = 298.15


@dataclass(frozen=True)
class SampleCell:
    """Thin water layer between THz-transparent windows on a TEC-stabilised holder.

    alpha(f, T) = alpha_table(f) (1 + kappa (T - T_ref)). Water's THz absorption
    rises with temperature (L0); kappa is L2 and is measured in situ by
    ``thermal_calibration`` so the analysis never relies on this default.
    """

    path_length_m: float = 100e-6  # L2
    aperture_radius_m: float = 2.5e-3  # L2
    thermal_resistance_k_per_w: float = 200.0  # L2 cell to holder
    holder_temperature_k: float = T_REF_K
    kappa_per_k: float = 0.005  # L2 nominal relative d(alpha)/dT

    @property
    def heat_capacity_j_per_k(self) -> float:
        return WATER_VOLUMETRIC_HEAT_CAPACITY * math.pi * self.aperture_radius_m**2 * self.path_length_m

    @property
    def thermal_time_constant_s(self) -> float:
        return self.thermal_resistance_k_per_w * self.heat_capacity_j_per_k

    @property
    def lateral_diffusion_time_s(self) -> float:
        """L1 a^2 / D: time for a spatially non-uniform heating pattern to even out."""
        return self.aperture_radius_m**2 / WATER_THERMAL_DIFFUSIVITY

    @property
    def classical_memory_s(self) -> float:
        """L1 horizon after which heat leaves no trace (5 x slowest thermal time)."""
        return 5.0 * max(self.thermal_time_constant_s, self.lateral_diffusion_time_s)

    def alpha_per_m(self, frequency_hz, temperature_k: float) -> np.ndarray:
        return water_absorption_per_m(frequency_hz) * (1.0 + self.kappa_per_k * (temperature_k - T_REF_K))

    def absorbed_fraction(self, frequency_hz, temperature_k: float = T_REF_K) -> np.ndarray:
        return 1.0 - np.exp(-self.alpha_per_m(frequency_hz, temperature_k) * self.path_length_m)


# =============================================================================
# 2. Illumination geometry and the symbol codebook (frequency node x layline)
# =============================================================================


@dataclass(frozen=True)
class RingIllumination:
    """N mutually coherent beams on a cone converging on the cell (bench fixture).

    Beam n arrives at polar angle theta from azimuth 2 pi n / N with transverse
    wavevector k sin(theta). A helical phase pattern phi_n = 2 pi l n / N (the
    layline of charge l) builds a Bessel-like field J_l: l = 0 peaks on axis,
    l != 0 has an on-axis null and a ring. L1 scalar, paraxial-spot model.
    """

    n_emitters: int = 12
    cone_half_angle_rad: float = math.radians(20.0)  # L2
    spot_radius_m: float = 2.0e-3  # L2 1/e^2 radius of each focused beam at the cell
    grid_points: int = 41

    def intensity_map(self, cell: SampleCell, frequency_hz: float, powers_w, phases_rad) -> Tuple[np.ndarray, float]:
        """Return (intensity W/m^2 on the cell-face grid, grid cell area m^2); points outside the aperture are 0."""
        p = np.asarray(powers_w, dtype=float).reshape(-1)
        phi = np.asarray(phases_rad, dtype=float).reshape(-1)
        if p.size != self.n_emitters or phi.size != self.n_emitters:
            raise ValueError("powers and phases need one entry per emitter")
        a = cell.aperture_radius_m
        axis = np.linspace(-a, a, self.grid_points)
        x, y = np.meshgrid(axis, axis, indexing="xy")
        inside = x**2 + y**2 <= a**2
        k_t = TAU * frequency_hz / C0 * math.sin(self.cone_half_angle_rad)
        az = TAU * np.arange(self.n_emitters) / self.n_emitters
        w = self.spot_radius_m
        envelope = np.exp(-(x**2 + y**2) / w**2)
        amp = np.sqrt(2.0 * p / (math.pi * w**2))
        phase = phi[:, None, None] + k_t * (x[None] * np.cos(az)[:, None, None] + y[None] * np.sin(az)[:, None, None])
        field = np.sum(amp[:, None, None] * np.exp(1j * phase), axis=0) * envelope
        intensity = np.where(inside, np.abs(field) ** 2, 0.0)
        return intensity, (axis[1] - axis[0]) ** 2

    def absorbed_power_w(self, cell: SampleCell, frequency_hz: float, powers_w, phases_rad,
                         temperature_k: float = T_REF_K) -> float:
        intensity, d_area = self.intensity_map(cell, frequency_hz, powers_w, phases_rad)
        return float(np.sum(intensity) * d_area * cell.absorbed_fraction(frequency_hz, temperature_k))

    def absorbed_power_from_plan(self, cell: SampleCell, plan: PhotonicDrivePlan, temperature_k: float = T_REF_K) -> float:
        powers = np.array([c.thz_power_w if c.enabled else 0.0 for c in plan.channels])
        phases = np.array([c.thz_phase_rad for c in plan.channels])
        return self.absorbed_power_w(cell, plan.carrier.frequency_hz, powers, phases, temperature_k)


@dataclass(frozen=True)
class SymbolCodebook:
    """Symbol s <-> (frequency node i, layline charge l), s = i * n_charges + l_index.

    Frequency node: the THz beat frequency. Layline: the helical phase
    pattern phi_n = 2 pi l n / N across the ring. Both are physical, measurable
    properties of the drive (L0); any further meaning is L3.
    """

    frequency_nodes_hz: Tuple[float, ...] = (0.85e12, 0.95e12, 1.05e12, 1.15e12)
    layline_charges: Tuple[int, ...] = (0, 1, 2, 3)

    @property
    def size(self) -> int:
        return len(self.frequency_nodes_hz) * len(self.layline_charges)

    @property
    def bits_per_symbol(self) -> float:
        return math.log2(self.size)

    def decompose(self, symbol: int) -> Tuple[int, int]:
        if not 0 <= int(symbol) < self.size:
            raise ValueError(f"symbol {symbol} outside codebook of size {self.size}")
        node, charge_index = divmod(int(symbol), len(self.layline_charges))
        return node, self.layline_charges[charge_index]

    def compose(self, node_index: int, charge: int) -> int:
        return int(node_index) * len(self.layline_charges) + self.layline_charges.index(int(charge))

    @staticmethod
    def layline_phases(charge: int, n_emitters: int) -> np.ndarray:
        return np.mod(TAU * int(charge) * np.arange(n_emitters) / n_emitters, TAU)

    def command(self, symbol: int, amplitude: float, duration_ms: float, n_emitters: int) -> Dict[str, object]:
        node, charge = self.decompose(symbol)
        return {
            "frequency_thz": self.frequency_nodes_hz[node] / 1e12,
            "amplitude_normalized": float(np.clip(amplitude, 0.0, 1.0)),
            "duration_ms": float(duration_ms),
            "pattern": "continuous",
            "emitters": list(range(n_emitters)),
            "phase_offset_rad": 0.0,
            "phase_pattern_rad": self.layline_phases(charge, n_emitters).tolist(),
        }


@dataclass(frozen=True)
class CodebookCalibration:
    """Per-symbol drive amplitude that equalises absorbed THz power in the cell (L1 model)."""

    amplitudes: Tuple[float, ...]
    absorbed_power_w: Tuple[float, ...]  # after equalisation
    unequalised_absorbed_power_w: Tuple[float, ...]  # at amplitude 1


def calibrate_codebook(codebook: SymbolCodebook, array: PhotonicTHzEmitterArray, cell: SampleCell,
                       illumination: RingIllumination, amplitude: float = 1.0,
                       equalize: bool = True) -> CodebookCalibration:
    """Scale amplitudes so every symbol deposits the same power, removing the thermal side channel by design.

    THz power is linear in ``amplitude_normalized`` (photocurrent sqrt(a) I_max),
    so absorbed power scales linearly too.
    """
    raw = []
    for s in range(codebook.size):
        plan = array.plan(codebook.command(s, 1.0, 1000.0, array.N_CHANNELS))
        raw.append(illumination.absorbed_power_from_plan(cell, plan))
    raw_arr = np.array(raw)
    if equalize:
        target = amplitude * float(raw_arr.min())
        amps = target / raw_arr
    else:
        amps = np.full(codebook.size, amplitude)
    return CodebookCalibration(tuple(float(a) for a in amps), tuple(float(v) for v in amps * raw_arr),
                               tuple(float(v) for v in raw_arr))


# =============================================================================
# 3. Streams: what gets written into the water
# =============================================================================


class SymbolStream:
    """Source of symbols. ``retro_valid`` is True only when the next symbol is unpredictable before it is drawn."""

    kind = "abstract"
    retro_valid = False

    def next_symbol(self, alphabet_size: int) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class RandomStream(SymbolStream):
    """Uniform symbols. ``seed=None`` uses the OS CSPRNG (secrets), required for a real retro test."""

    kind = "random"
    retro_valid = True

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def next_symbol(self, alphabet_size: int) -> int:
        if self._rng is None:
            return secrets.randbelow(int(alphabet_size))
        return int(self._rng.integers(alphabet_size))

    def choice(self, options: Sequence[int]) -> int:
        idx = secrets.randbelow(len(options)) if self._rng is None else int(self._rng.integers(len(options)))
        return int(options[idx])


class TextStream(SymbolStream):
    """UTF-8 text -> base-M digits (most significant first). M must be a power of two."""

    kind = "text"
    retro_valid = False

    def __init__(self, text: str, alphabet_size: int = 16):
        if alphabet_size < 2 or alphabet_size & (alphabet_size - 1):
            raise ValueError("alphabet_size must be a power of two >= 2")
        self.text = text
        self.alphabet_size = alphabet_size
        self.symbols = text_to_symbols(text, alphabet_size)
        self._i = 0

    def __len__(self) -> int:
        return len(self.symbols)

    def next_symbol(self, alphabet_size: int) -> int:
        if alphabet_size != self.alphabet_size:
            raise ValueError("stream alphabet does not match codebook")
        if self._i >= len(self.symbols):
            raise StopIteration("text stream exhausted")
        s = self.symbols[self._i]
        self._i += 1
        return s


def text_to_symbols(text: str, alphabet_size: int = 16) -> List[int]:
    bits = int(math.log2(alphabet_size))
    if 8 % bits:
        raise ValueError("bits per symbol must divide 8")
    out: List[int] = []
    for byte in text.encode("utf-8"):
        for shift in range(8 - bits, -1, -bits):
            out.append((byte >> shift) & (alphabet_size - 1))
    return out


def symbols_to_text(symbols: Sequence[int], alphabet_size: int = 16) -> str:
    bits = int(math.log2(alphabet_size))
    per_byte = 8 // bits
    data = bytearray()
    for i in range(0, len(symbols) - per_byte + 1, per_byte):
        value = 0
        for s in symbols[i:i + per_byte]:
            value = (value << bits) | (int(s) & (alphabet_size - 1))
        data.append(value)
    return data.decode("utf-8", errors="replace")


EEG_BANDS_HZ: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 100.0),
)
SUBSTRATES = ("physical", "emotional", "cognitive", "social", "divine_unity")  # README band -> substrate convention


def eeg_band_powers(segment: np.ndarray, fs_hz: float) -> np.ndarray:
    """L0 Hann-windowed periodogram band powers, averaged over channels. segment: (channels, samples)."""
    x = np.atleast_2d(np.asarray(segment, dtype=float))
    x = x - x.mean(axis=1, keepdims=True)
    win = np.hanning(x.shape[1])
    spec = np.abs(np.fft.rfft(x * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs_hz)
    return np.array([spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1).mean() for _, lo, hi in EEG_BANDS_HZ])


class EEGSubstrateStream(SymbolStream):
    """EEG windows -> dominant relative band -> substrate index 0..4 (the project's ABCR convention).

    Band powers are L0 signal processing; reading them as consciousness
    substrates is the project's L3 model, carried here as a label only.
    """

    kind = "eeg_substrate"
    retro_valid = False

    def __init__(self, eeg: np.ndarray, fs_hz: float, window_s: float = 2.0):
        self.eeg = np.atleast_2d(np.asarray(eeg, dtype=float))
        self.fs_hz = float(fs_hz)
        self.window = int(round(window_s * fs_hz))
        if self.window < 8 or self.window > self.eeg.shape[1]:
            raise ValueError("window must fit inside the recording")
        self._start = 0

    def substrate_sequence(self) -> List[str]:
        out = []
        for start in range(0, self.eeg.shape[1] - self.window + 1, self.window):
            p = eeg_band_powers(self.eeg[:, start:start + self.window], self.fs_hz)
            widths = np.array([hi - lo for _, lo, hi in EEG_BANDS_HZ])
            out.append(SUBSTRATES[int(np.argmax(p / widths))])
        return out

    def next_symbol(self, alphabet_size: int) -> int:
        if alphabet_size < len(SUBSTRATES):
            raise ValueError("codebook too small for five substrates")
        if self._start + self.window > self.eeg.shape[1]:
            raise StopIteration("EEG stream exhausted")
        seg = self.eeg[:, self._start:self._start + self.window]
        self._start += self.window
        p = eeg_band_powers(seg, self.fs_hz)
        widths = np.array([hi - lo for _, lo, hi in EEG_BANDS_HZ])
        return int(np.argmax(p / widths))  # power density per Hz, so wide bands do not win by width


# =============================================================================
# 4. Instrument interface and physics simulator
# =============================================================================


@dataclass(frozen=True, eq=False)
class DriveAction:
    kind: str  # "thz", "sham", "heater", "off"
    plan: Optional[PhotonicDrivePlan] = None
    heater_power_w: float = 0.0


@dataclass(frozen=True, eq=False)
class Reading:
    time_s: float
    temperature_k: float
    transmission: np.ndarray


class Instrument:
    """Hardware abstraction. A bench implementation drives the emitter MCU, a heater film,
    a THz receiver (second photomixer on the same laser pair) and a cell thermistor."""

    def apply(self, action: DriveAction) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def wait(self, seconds: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self, frequencies_hz: np.ndarray) -> Reading:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class InjectedEffect:
    """L3 HYPOTHETICAL effect, for positive controls only: a symbol-specific, slowly
    decaying change in absorption that heat cannot produce. Real hardware has no
    such knob; the simulator uses it to prove the analysis would see one."""

    codebook: SymbolCodebook
    magnitude_rel: float = 0.05
    rise_time_s: float = 5.0
    memory_time_s: float = 3600.0
    linewidth_hwhm_hz: float = 0.15e12

    def signature(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """(symbols, F): Lorentzian at the node, scaled by layline charge."""
        f = np.asarray(frequencies_hz, dtype=float)
        n_ch = len(self.codebook.layline_charges)
        rows = []
        for s in range(self.codebook.size):
            node, charge = self.codebook.decompose(s)
            f0 = self.codebook.frequency_nodes_hz[node]
            lor = 1.0 / (1.0 + ((f - f0) / self.linewidth_hwhm_hz) ** 2)
            rows.append(lor * (self.codebook.layline_charges.index(charge) + 1) / n_ch)
        return np.array(rows)


@dataclass(frozen=True)
class ReceiverModel:
    relative_noise: float = 1e-3  # L2 per-point transmission noise (~60 dB dynamic range)
    gain_drift_per_sqrt_s: float = 2e-5  # L2 log-gain random walk
    thermistor_noise_k: float = 2e-3  # L2


class SimulatedInstrument(Instrument):
    """Exact-update physics model: first-order cell heating, temperature-dependent water
    absorption, receiver noise and drift, optional injected L3 effect."""

    def __init__(self, cell: SampleCell, illumination: RingIllumination, receiver: ReceiverModel = ReceiverModel(),
                 injected: Optional[InjectedEffect] = None, ir_leak_heating_w: float = 0.0, seed: int = 0):
        self.cell = cell
        self.illumination = illumination
        self.receiver = receiver
        self.injected = injected
        self.ir_leak_heating_w = ir_leak_heating_w
        self.rng = np.random.default_rng(seed)
        self.time_s = 0.0
        self.temperature_k = cell.holder_temperature_k
        self.log_gain = 0.0
        self._heat_in_w = 0.0
        self._active_symbol: Optional[int] = None
        n = injected.codebook.size if injected else 0
        self._effect_state = np.zeros(n)

    # --- interface ---------------------------------------------------------------
    def apply(self, action: DriveAction) -> None:
        self._active_symbol = None
        if action.kind == "thz":
            if action.plan is None or not action.plan.safe:
                raise ValueError("refusing to drive without a safe plan")
            self._heat_in_w = self.illumination.absorbed_power_from_plan(self.cell, action.plan, self.temperature_k)
            if self.injected is not None:
                self._active_symbol = self._symbol_from_plan(action.plan)
        elif action.kind == "sham":
            self._heat_in_w = self.ir_leak_heating_w
        elif action.kind == "heater":
            self._heat_in_w = float(action.heater_power_w)
        elif action.kind == "off":
            self._heat_in_w = 0.0
        else:
            raise ValueError(f"unknown action {action.kind}")

    def wait(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot wait a negative time")
        if seconds == 0:
            return
        c = self.cell
        t_inf = c.holder_temperature_k + c.thermal_resistance_k_per_w * self._heat_in_w
        self.temperature_k = t_inf + (self.temperature_k - t_inf) * math.exp(-seconds / c.thermal_time_constant_s)
        self.log_gain += self.receiver.gain_drift_per_sqrt_s * math.sqrt(seconds) * self.rng.standard_normal()
        if self.injected is not None:
            decay = math.exp(-seconds / self.injected.memory_time_s)
            if self._active_symbol is not None:
                s = self._active_symbol
                others = np.arange(self._effect_state.size) != s
                self._effect_state[others] *= decay
                self._effect_state[s] = 1.0 + (self._effect_state[s] - 1.0) * math.exp(-seconds / self.injected.rise_time_s)
            else:
                self._effect_state *= decay
        self.time_s += seconds

    def read(self, frequencies_hz: np.ndarray) -> Reading:
        f = np.asarray(frequencies_hz, dtype=float)
        alpha = self.cell.alpha_per_m(f, self.temperature_k)
        if self.injected is not None:
            alpha = alpha * (1.0 + self.injected.magnitude_rel * self._effect_state @ self.injected.signature(f))
        trans = np.exp(-alpha * self.cell.path_length_m + self.log_gain)
        trans = trans * (1.0 + self.receiver.relative_noise * self.rng.standard_normal(f.size))
        temp = self.temperature_k + self.receiver.thermistor_noise_k * self.rng.standard_normal()
        return Reading(self.time_s, temp, trans)

    # --- helpers ---------------------------------------------------------------
    def _symbol_from_plan(self, plan: PhotonicDrivePlan) -> int:
        cb = self.injected.codebook
        node = int(np.argmin(np.abs(np.array(cb.frequency_nodes_hz) - plan.carrier.frequency_hz)))
        phases = np.array([c.thz_phase_rad for c in plan.channels])
        n = phases.size
        step = np.angle(np.mean(np.exp(1j * (np.roll(phases, -1)[:-1] - phases[:-1]))))
        charge = int(round(step * n / TAU)) % n
        if charge not in cb.layline_charges:
            charge = min(cb.layline_charges, key=lambda c: abs(c - charge))
        return cb.compose(node, charge)


def alpha_from_transmission(transmission: np.ndarray, path_length_m: float) -> np.ndarray:
    t = np.clip(np.asarray(transmission, dtype=float), 1e-300, None)
    return -np.log(t) / path_length_m


# =============================================================================
# 5. Preregistration and blinding
# =============================================================================

ARMS = ("active", "sham", "thermal")


@dataclass(frozen=True)
class Preregistration:
    """Everything the analysis is allowed to depend on, fixed and hashed before data exist."""

    title: str = "THz drive persistence and symbol channel in liquid water"
    hypothesis: str = (
        "H1 (persistence): after a THz drive, band-mean residual absorption differs between active and sham "
        "by at least sesoi_rel_alpha at post delays beyond the classical memory horizon. "
        "H2 (channel): symbols are decodable above chance from active-arm readouts beyond the classical memory "
        "horizon and not from sham or heater-matched readouts. H0 for both: no difference beyond chance."
    )
    n_blocks: int = 4
    slots_per_arm_per_block: int = 0  # 0: one slot per codebook symbol
    drive_s: float = 30.0
    pre_window_s: float = 5.0
    post_delays_s: Tuple[float, ...] = (1.0, 10.0, 60.0, 300.0)
    intertrial_s: float = 600.0
    readout_frequencies_hz: Tuple[float, ...] = tuple(float(f) for f in np.linspace(0.3e12, 2.0e12, 16))
    sesoi_rel_alpha: float = 2e-3
    alpha: float = 0.05
    n_permutations: int = 2000  # min permutation p = 1/2001; Holm over ~45 tests still reaches 0.05
    n_bootstrap: int = 2000
    lags: Tuple[int, ...] = (-2, -1, 1, 2)
    drive_amplitude: float = 1.0
    equalize_absorbed_power: bool = True
    message_window: str = "post@300s"
    analysis_seed: int = 20261127

    @property
    def windows(self) -> Tuple[str, ...]:
        return ("pre", "during") + tuple(f"post@{d:g}s" for d in self.post_delays_s)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def trial_duration_s(self) -> float:
        return self.intertrial_s + self.pre_window_s + self.drive_s + max(self.post_delays_s)

    def validate(self, cell: SampleCell) -> List[str]:
        issues = []
        if self.intertrial_s < cell.classical_memory_s:
            issues.append(f"intertrial_s {self.intertrial_s:.0f} s is shorter than the classical memory "
                          f"{cell.classical_memory_s:.0f} s: heat from one trial reaches the next")
        if not any(d >= cell.classical_memory_s for d in self.post_delays_s):
            issues.append("no post delay exceeds the classical memory horizon; H1/H2 cannot be tested")
        if self.message_window not in self.windows:
            issues.append("message_window is not one of the recorded windows")
        if self.n_blocks < 2:
            issues.append("need >= 2 blocks for leave-one-block-out decoding")
        return issues


@dataclass(frozen=True)
class AllocationKey:
    """Maps opaque arm codes to arms. Only its commitment hash is visible before unblinding."""

    codes: Tuple[Tuple[str, str], ...]  # (code, arm)
    salt: str

    def arm_of(self, code: str) -> str:
        return dict(self.codes)[code]

    def code_of(self, arm: str) -> str:
        return {a: c for c, a in self.codes}[arm]

    def commitment(self) -> str:
        payload = json.dumps({"codes": sorted(self.codes), "salt": self.salt}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ScheduledTrial:
    index: int
    block: int  # -1 for payload trials
    arm_code: str
    phase: str  # "calibration" or "payload"
    payload_index: int = -1


def make_allocation_key(seed: Optional[int] = None) -> AllocationKey:
    rng = np.random.default_rng(seed) if seed is not None else None
    letters = "ABCDEFGHJKMNPQRSTUVWXYZ"
    codes: List[str] = []
    while len(codes) < len(ARMS):
        if rng is None:
            c = secrets.choice(letters) + str(secrets.randbelow(10))
        else:
            c = letters[int(rng.integers(len(letters)))] + str(int(rng.integers(10)))
        if c not in codes:
            codes.append(c)
    salt = secrets.token_hex(16) if rng is None else f"{int(rng.integers(2**62)):016x}"
    return AllocationKey(tuple(zip(codes, ARMS)), salt)


def make_schedule(prereg: Preregistration, codebook_size: int, key: AllocationKey,
                  payload_length: int = 0, seed: Optional[int] = None) -> Tuple[ScheduledTrial, ...]:
    """Calibration blocks: codebook_size slots per arm per block, arm order shuffled. Symbols are
    drawn i.i.d. at drive time, after the pre-window is recorded. Payload: each stream symbol is
    sent once under active and once under sham, in random order."""
    rng = np.random.default_rng(seed)
    per_arm = prereg.slots_per_arm_per_block or codebook_size
    trials: List[ScheduledTrial] = []
    for block in range(prereg.n_blocks):
        slots = [key.code_of(arm) for arm in ARMS for _ in range(per_arm)]
        order = rng.permutation(len(slots))
        trials.extend(ScheduledTrial(0, block, slots[j], "calibration") for j in order)
    for p in range(payload_length):
        pair = [key.code_of("active"), key.code_of("sham")]
        if rng.integers(2):
            pair.reverse()
        trials.extend(ScheduledTrial(0, -1, c, "payload", p) for c in pair)
    return tuple(ScheduledTrial(i, t.block, t.arm_code, t.phase, t.payload_index) for i, t in enumerate(trials))


# =============================================================================
# 6. Thermal calibration (unblinded, before the main run)
# =============================================================================


@dataclass(frozen=True, eq=False)
class ThermalCalibration:
    frequencies_hz: np.ndarray
    alpha_ref_per_m: np.ndarray  # absorption at holder temperature
    slope_rel_per_k: np.ndarray  # d(alpha/alpha_ref)/dT per frequency
    r_squared: np.ndarray

    def residual(self, alpha_per_m: np.ndarray, temperature_k, holder_temperature_k: float) -> np.ndarray:
        """Absorption deviation from the reference with the measured-temperature contribution removed."""
        dt = np.asarray(temperature_k, dtype=float)[..., None] - holder_temperature_k
        return alpha_per_m / self.alpha_ref_per_m - 1.0 - self.slope_rel_per_k * dt


def thermal_calibration(instrument: Instrument, cell: SampleCell, frequencies_hz: Sequence[float],
                        heater_steps_w: Sequence[float] = (0.0, 2.5e-3, 5.0e-3, 7.5e-3, 10.0e-3),
                        settle_s: Optional[float] = None, reads_per_step: int = 16) -> ThermalCalibration:
    """Step a heater film, wait for equilibrium, regress alpha on measured temperature per frequency."""
    f = np.asarray(frequencies_hz, dtype=float)
    settle = settle_s if settle_s is not None else 10.0 * cell.thermal_time_constant_s
    temps, alphas = [], []
    for p in heater_steps_w:
        instrument.apply(DriveAction("heater", heater_power_w=float(p)))
        instrument.wait(settle)
        for _ in range(reads_per_step):
            r = instrument.read(f)
            temps.append(r.temperature_k)
            alphas.append(alpha_from_transmission(r.transmission, cell.path_length_m))
            instrument.wait(1.0)
    instrument.apply(DriveAction("off"))
    instrument.wait(settle)
    dt = np.array(temps) - cell.holder_temperature_k
    a = np.array(alphas)
    design = np.column_stack([np.ones_like(dt), dt])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    fit = design @ coef
    ss_res = np.sum((a - fit) ** 2, axis=0)
    ss_tot = np.sum((a - a.mean(axis=0)) ** 2, axis=0)
    alpha_ref = coef[0]
    return ThermalCalibration(f, alpha_ref, coef[1] / alpha_ref, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0))


# =============================================================================
# 7. Running the assay
# =============================================================================


@dataclass(frozen=True, eq=False)
class AssayDataset:
    prereg_digest: str
    key_commitment: str
    codebook: SymbolCodebook
    windows: Tuple[str, ...]
    frequencies_hz: np.ndarray
    trial_index: np.ndarray
    block: np.ndarray
    arm_code: np.ndarray
    phase: np.ndarray
    payload_index: np.ndarray
    symbol: np.ndarray
    retro_valid: np.ndarray
    alpha_per_m: np.ndarray  # (trials, windows, F)
    temperature_k: np.ndarray  # (trials, windows)
    time_s: np.ndarray  # (trials, windows)
    holder_temperature_k: float
    classical_memory_s: float
    stream_kind: str


def run_assay(instrument: Instrument, array: PhotonicTHzEmitterArray, cell: SampleCell,
              illumination: RingIllumination, codebook: SymbolCodebook, prereg: Preregistration,
              schedule: Sequence[ScheduledTrial], key: AllocationKey,
              calibration_stream: Optional[RandomStream] = None,
              payload_stream: Optional[SymbolStream] = None,
              cal: Optional[CodebookCalibration] = None) -> AssayDataset:
    """Execute the schedule. The run controller holds the key; the dataset stores codes only.

    Calibration symbols are drawn i.i.d. uniform at drive time, after the
    pre-window is recorded. Drawing without replacement would make later
    symbols in a block predictable from earlier ones, and any slow memory in
    the sample would then leak into the retro (pre-window) test.
    """
    issues = prereg.validate(cell)
    if issues:
        raise ValueError("preregistration invalid for this cell: " + "; ".join(issues))
    if any(t.phase == "payload" for t in schedule) and payload_stream is None:
        raise ValueError("schedule has payload trials but no payload stream")
    cal = cal or calibrate_codebook(codebook, array, cell, illumination, prereg.drive_amplitude,
                                    prereg.equalize_absorbed_power)
    draws = calibration_stream or RandomStream()
    payload_symbols: Dict[int, int] = {}
    f = np.array(prereg.readout_frequencies_hz)
    n, w = len(schedule), len(prereg.windows)
    alpha = np.zeros((n, w, f.size))
    temp = np.zeros((n, w))
    times = np.zeros((n, w))
    symbols = np.zeros(n, dtype=int)
    retro = np.zeros(n, dtype=bool)

    def record(i: int, j: int) -> None:
        r = instrument.read(f)
        alpha[i, j] = alpha_from_transmission(r.transmission, cell.path_length_m)
        temp[i, j] = r.temperature_k
        times[i, j] = r.time_s

    for i, trial in enumerate(schedule):
        arm = key.arm_of(trial.arm_code)
        instrument.apply(DriveAction("off"))
        instrument.wait(prereg.intertrial_s)
        instrument.wait(prereg.pre_window_s)
        record(i, 0)  # the symbol does not exist yet
        if trial.phase == "calibration":
            s = draws.next_symbol(codebook.size)
            retro[i] = draws.retro_valid
        else:
            if trial.payload_index not in payload_symbols:
                payload_symbols[trial.payload_index] = payload_stream.next_symbol(codebook.size)
            s = payload_symbols[trial.payload_index]
        symbols[i] = s
        command = codebook.command(s, cal.amplitudes[s], prereg.drive_s * 1e3, array.N_CHANNELS)
        plan = array.plan(command)
        if not plan.safe:
            raise RuntimeError(f"trial {i}: drive plan failed the interlock: {plan.warnings}")
        if arm == "active":
            instrument.apply(DriveAction("thz", plan))
        elif arm == "sham":
            instrument.apply(DriveAction("sham", plan))
        else:
            instrument.apply(DriveAction("heater", heater_power_w=cal.absorbed_power_w[s]))
        instrument.wait(prereg.drive_s)
        record(i, 1)
        instrument.apply(DriveAction("off"))
        elapsed = 0.0
        for j, d in enumerate(prereg.post_delays_s, start=2):
            instrument.wait(d - elapsed)
            elapsed = d
            record(i, j)

    return AssayDataset(
        prereg_digest=prereg.digest(),
        key_commitment=key.commitment(),
        codebook=codebook,
        windows=prereg.windows,
        frequencies_hz=f,
        trial_index=np.array([t.index for t in schedule]),
        block=np.array([t.block for t in schedule]),
        arm_code=np.array([t.arm_code for t in schedule]),
        phase=np.array([t.phase for t in schedule]),
        payload_index=np.array([t.payload_index for t in schedule]),
        symbol=symbols,
        retro_valid=retro,
        alpha_per_m=alpha,
        temperature_k=temp,
        time_s=times,
        holder_temperature_k=cell.holder_temperature_k,
        classical_memory_s=cell.classical_memory_s,
        stream_kind=payload_stream.kind if payload_stream is not None else "none",
    )


# =============================================================================
# 8. Statistics (numpy, vectorised, seeded)
# =============================================================================


def holm(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (p.size - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp * (1.0 - 3.0 / (4.0 * (na + nb) - 9.0))


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    perms = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)
    diffs = np.abs(perms[:, :a.size].mean(axis=1) - perms[:, a.size:].mean(axis=1))
    return float((1 + np.sum(diffs >= obs - 1e-15)) / (n_perm + 1))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator,
                 level: float = 0.95) -> Tuple[float, float]:
    ia = rng.integers(0, a.size, (n_boot, a.size))
    ib = rng.integers(0, b.size, (n_boot, b.size))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def tost_p(a: np.ndarray, b: np.ndarray, bound: float) -> float:
    """Two one-sided Welch tests: small p means the difference lies within +/- bound."""
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    diff = a.mean() - b.mean()
    if se == 0:
        return 0.0 if abs(diff) < bound else 1.0
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    t_lo, t_hi = (diff + bound) / se, (diff - bound) / se
    if _stats is not None:
        p_lo, p_hi = _stats.t.sf(t_lo, df), _stats.t.cdf(t_hi, df)
    else:  # pragma: no cover
        p_lo = 0.5 * math.erfc(t_lo / math.sqrt(2))
        p_hi = 0.5 * math.erfc(-t_hi / math.sqrt(2))
    return float(max(p_lo, p_hi))


def label_permutations(y: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Row 0: observed labels; rows 1..n_perm: full permutations. Valid because calibration
    symbols are drawn i.i.d. at drive time, so labels are exchangeable under H0."""
    out = np.tile(y, (n_perm + 1, 1))
    out[1:] = rng.permuted(out[1:], axis=1)
    return out


def nearest_centroid_cv(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_classes: int,
                        standardize: bool = True) -> np.ndarray:
    """Leave-one-group-out nearest-centroid predictions for each row of ``labels`` (P, n). Vectorised over P.

    ``standardize`` z-scores features on the training fold. Turn it off when every feature carries the
    same measurement noise (e.g. log concentrations from one instrument): z-scoring then up-weights
    signal-free features to the same weight as informative ones.
    """
    labels = np.atleast_2d(labels)
    p_count, n = labels.shape
    preds = np.full((p_count, n), -1, dtype=int)
    classes = np.arange(n_classes)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() == 0:
            continue
        mu = x[tr].mean(axis=0)
        sd = (x[tr].std(axis=0) + 1e-12) if standardize else np.ones(x.shape[1])
        z_tr, z_te = (x[tr] - mu) / sd, (x[te] - mu) / sd
        onehot = (labels[:, tr][..., None] == classes).astype(float)  # (P, ntr, M)
        counts = onehot.sum(axis=1)  # (P, M)
        cent = np.einsum("pnm,nf->pmf", onehot, z_tr) / np.maximum(counts, 1.0)[..., None]
        dist = ((z_te**2).sum(axis=1)[None, :, None] - 2.0 * np.einsum("tf,pmf->ptm", z_te, cent)
                + (cent**2).sum(axis=2)[:, None, :])
        dist = np.where(counts[:, None, :] > 0, dist, np.inf)
        preds[:, te] = np.argmin(dist, axis=2)
    return preds


def mutual_information_bits(true: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Plug-in MI (bits) of the confusion matrix, vectorised over rows of (P, n) label/prediction arrays.

    Plug-in MI is strongly biased upward when n is small next to n_classes^2;
    callers subtract the mean over permutation rows (see ``analyze``), which
    removes the bias exactly for the null.
    """
    true, pred = np.atleast_2d(true), np.atleast_2d(pred)
    p_rows, n = true.shape
    if n == 0:
        return np.zeros(p_rows)
    flat = (np.arange(p_rows)[:, None] * n_classes + true) * n_classes + pred
    joint = np.bincount(flat.ravel(), minlength=p_rows * n_classes * n_classes).reshape(p_rows, n_classes, n_classes) / n
    px, py = joint.sum(axis=2), joint.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px[:, :, None] * py[:, None, :])
        terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
    return terms.sum(axis=(1, 2))


# =============================================================================
# 9. Blinded analysis
# =============================================================================


@dataclass(frozen=True)
class ContrastResult:
    window: str
    code_a: str
    code_b: str
    n_a: int
    n_b: int
    mean_diff: float
    ci_low: float
    ci_high: float
    hedges_g: float
    p_perm: float
    p_holm: float
    tost_p: float
    beyond_classical_memory: bool
    family: str


@dataclass(frozen=True)
class DecodeResult:
    code: str
    window: str
    lag: int
    n: int
    accuracy: float
    chance: float
    mi_bits: float  # excess over the permutation-null mean; ~0 under H0, can be slightly negative
    p_perm: float
    p_holm: float
    beyond_classical_memory: bool
    family: str


# Holm families are fixed by test type, never by arm, so they are defined while blind.
# A leak detector sharing a family with the claims it protects would be diluted by them.
FAMILIES = {
    "beyond_horizon": "windows after the classical memory horizon (primary claims)",
    "within_horizon": "during drive and post windows inside the horizon",
    "integrity": "pre-window and negative lags: readable before the symbol existed",
    "carryover": "positive lags: readable in later trials",
}


def _holm_by_family(pvalues: Sequence[float], families: Sequence[str]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    fam = np.asarray(families)
    out = np.empty_like(p)
    for f in np.unique(fam):
        idx = np.nonzero(fam == f)[0]
        out[idx] = holm(p[idx])
    return out


@dataclass(frozen=True, eq=False)
class BlindedAnalysis:
    prereg_digest: str
    key_commitment: str
    contrasts: Tuple[ContrastResult, ...]
    decoding: Tuple[DecodeResult, ...]
    features: Dict[str, np.ndarray]  # window -> (trials, F) residual features
    calibration_mask: np.ndarray
    dataset: AssayDataset


def _window_delay(name: str) -> Optional[float]:
    return float(name[len("post@"):-1]) if name.startswith("post@") else None


def build_features(ds: AssayDataset, thermal: ThermalCalibration) -> Dict[str, np.ndarray]:
    """Per window: thermally corrected residual relative to the trial's own pre-window;
    'pre' itself: absolute residual (deviation from the reference water spectrum)."""
    if not np.allclose(thermal.frequencies_hz, ds.frequencies_hz):
        raise ValueError("thermal calibration was taken at other frequencies")
    resid = thermal.residual(ds.alpha_per_m, ds.temperature_k, ds.holder_temperature_k)  # (n, W, F)
    feats = {"pre": resid[:, 0, :]}
    for j, name in enumerate(ds.windows[1:], start=1):
        feats[name] = resid[:, j, :] - resid[:, 0, :]
    return feats


def analyze(ds: AssayDataset, prereg: Preregistration, thermal: ThermalCalibration) -> BlindedAnalysis:
    """Run the preregistered analysis on coded arms. Refuses a preregistration that changed."""
    if prereg.digest() != ds.prereg_digest:
        raise ValueError("preregistration digest mismatch: the analysis plan changed after data collection")
    rng = np.random.default_rng(prereg.analysis_seed)
    feats = build_features(ds, thermal)
    cal = ds.phase == "calibration"
    codes = sorted(set(ds.arm_code[cal].tolist()))
    m = ds.codebook.size

    # --- H1: band-mean residual contrasts between every pair of codes ---------------
    raw_c = []
    for name in ds.windows[1:]:
        scalar = feats[name].mean(axis=1)
        delay = _window_delay(name)
        beyond = delay is not None and delay >= ds.classical_memory_s
        for i, ca in enumerate(codes):
            for cb in codes[i + 1:]:
                a, b = scalar[cal & (ds.arm_code == ca)], scalar[cal & (ds.arm_code == cb)]
                lo, hi = bootstrap_ci(a, b, prereg.n_bootstrap, rng)
                raw_c.append([name, ca, cb, a.size, b.size, float(a.mean() - b.mean()), lo, hi, hedges_g(a, b),
                              permutation_p(a, b, prereg.n_permutations, rng), tost_p(a, b, prereg.sesoi_rel_alpha),
                              beyond, "beyond_horizon" if beyond else "within_horizon"])
    adj = _holm_by_family([r[9] for r in raw_c], [r[12] for r in raw_c])
    contrasts = tuple(ContrastResult(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], float(p), r[10], r[11],
                                     r[12]) for r, p in zip(raw_c, adj))

    # --- H2: temporal channel map ----------------------------------------------------
    order = np.argsort(ds.trial_index)
    position = np.empty_like(order)
    position[order] = np.arange(order.size)
    raw_d = []
    for code in codes:
        sel = np.nonzero(cal & (ds.arm_code == code))[0]
        y, groups = ds.symbol[sel], ds.block[sel]
        tests: List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for name in ds.windows:
            tests.append((name, 0, feats[name][sel], y, groups))
        for lag in prereg.lags:
            tgt = position[sel] + lag
            ok = (tgt >= 0) & (tgt < order.size)
            tests.append(("pre", lag, feats["pre"][order[tgt[ok]]], y[ok], groups[ok]))
        for name, lag, x, yy, gg in tests:
            perms = label_permutations(yy, prereg.n_permutations, rng)
            preds = nearest_centroid_cv(x, perms, gg, m)
            acc = (preds == perms).mean(axis=1)
            p = float((1 + np.sum(acc[1:] >= acc[0] - 1e-12)) / (prereg.n_permutations + 1))
            mi = mutual_information_bits(perms, preds, m)
            excess_bits = float(mi[0] - mi[1:].mean())  # bias-corrected against the label-shuffled null
            delay = _window_delay(name)
            beyond = bool(delay is not None and delay >= ds.classical_memory_s)
            if name == "pre":
                family = "carryover" if lag > 0 else "integrity"
            else:
                family = "beyond_horizon" if beyond else "within_horizon"
            raw_d.append([code, name, lag, yy.size, float(acc[0]), 1.0 / m, excess_bits, p, beyond, family])
    adj_d = _holm_by_family([r[7] for r in raw_d], [r[9] for r in raw_d])
    decoding = tuple(DecodeResult(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], float(p), r[8], r[9])
                     for r, p in zip(raw_d, adj_d))
    return BlindedAnalysis(ds.prereg_digest, ds.key_commitment, contrasts, decoding, feats, cal, ds)


# =============================================================================
# 10. Unblinding, verdicts, message recovery
# =============================================================================


@dataclass(frozen=True)
class MessageRecovery:
    arm: str
    window: str
    sent: Tuple[int, ...]
    received: Tuple[int, ...]
    symbol_accuracy: float
    chance: float
    p_binomial: float
    sent_text: Optional[str]
    received_text: Optional[str]


@dataclass(frozen=True, eq=False)
class AssayReport:
    prereg_digest: str
    persistence_verdict: str
    channel_verdict: str
    integrity_flags: Tuple[str, ...]
    contrasts: Tuple[ContrastResult, ...]
    decoding: Tuple[DecodeResult, ...]
    messages: Tuple[MessageRecovery, ...]
    arm_codes: Dict[str, str]

    def channel_map(self, arm: str) -> List[Tuple[str, int, float, float, float]]:
        """(window, lag, accuracy, bits, Holm p) for one arm in temporal order."""
        rows = [d for d in self.decoding if d.code == self.arm_codes[arm]]
        def key(d: DecodeResult) -> Tuple[float, float]:
            if d.window == "pre":
                return (float(d.lag), -1.0)
            if d.window == "during":
                return (0.0, 0.0)
            return (0.0, _window_delay(d.window) or 0.0)
        return [(d.window, d.lag, d.accuracy, d.mi_bits, d.p_holm) for d in sorted(rows, key=key)]

    def summary(self) -> str:
        out = [f"preregistration {self.prereg_digest[:16]}...",
               f"H1 persistence : {self.persistence_verdict}",
               f"H2 channel     : {self.channel_verdict}"]
        out += [f"integrity      : {flag}" for flag in self.integrity_flags] or ["integrity      : no flags"]
        for arm in ARMS:
            out.append(f"channel map [{arm}] (window, lag, accuracy, bits/symbol, Holm p):")
            for w, lag, acc, bits, p in self.channel_map(arm):
                label = w if lag == 0 else f"pre of trial {lag:+d}"
                out.append(f"    {label:<18s} acc {acc:5.3f}  {bits:5.3f} b  p {p:.3g}")
        for msg in self.messages:
            out.append(f"message [{msg.arm} @ {msg.window}] accuracy {msg.symbol_accuracy:.3f} "
                       f"(chance {msg.chance:.3f}, p {msg.p_binomial:.3g})")
            if msg.sent_text is not None:
                out.append(f"    sent     {msg.sent_text!r}")
                out.append(f"    received {msg.received_text!r}")
        return "\n".join(out)

    def to_json(self) -> str:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, float) and not math.isfinite(o):
                return str(o)
            raise TypeError(type(o))
        body = {
            "prereg_digest": self.prereg_digest,
            "persistence_verdict": self.persistence_verdict,
            "channel_verdict": self.channel_verdict,
            "integrity_flags": list(self.integrity_flags),
            "arm_codes": self.arm_codes,
            "contrasts": [asdict(c) for c in self.contrasts],
            "decoding": [asdict(d) for d in self.decoding],
            "messages": [asdict(m) for m in self.messages],
        }
        return json.dumps(body, default=conv, indent=2)


def _binomial_sf(k: int, n: int, p: float) -> float:
    if _stats is not None:
        return float(_stats.binom.sf(k - 1, n, p))
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))  # pragma: no cover


def unblind(analysis: BlindedAnalysis, key: AllocationKey, prereg: Preregistration,
            payload_text: Optional[str] = None) -> AssayReport:
    if key.commitment() != analysis.key_commitment:
        raise ValueError("allocation key does not match the committed hash")
    alpha = prereg.alpha
    code = {arm: key.code_of(arm) for arm in ARMS}

    # --- H1 -------------------------------------------------------------------------
    def contrast(window: str, a: str, b: str) -> ContrastResult:
        for c in analysis.contrasts:
            if c.window == window and {c.code_a, c.code_b} == {code[a], code[b]}:
                if c.code_a == code[a]:
                    return c
                return ContrastResult(c.window, c.code_b, c.code_a, c.n_b, c.n_a, -c.mean_diff, -c.ci_high,
                                      -c.ci_low, -c.hedges_g, c.p_perm, c.p_holm, c.tost_p, c.beyond_classical_memory,
                                      c.family)
        raise KeyError(window)

    beyond = [w for w in prereg.windows if (_window_delay(w) or 0.0) >= analysis.dataset.classical_memory_s]
    primary = [contrast(w, "active", "sham") for w in beyond]
    detected = [c for c in primary if c.p_holm < alpha and abs(c.mean_diff) >= prereg.sesoi_rel_alpha
                and (c.ci_low > 0 or c.ci_high < 0)]
    if detected:
        thermal_also = [contrast(c.window, "thermal", "sham") for c in detected]
        if any(t.p_holm < alpha for t in thermal_also):
            persistence = "difference beyond classical memory, but the heater-matched arm shows it too: classical origin"
        else:
            c = detected[0]
            persistence = (f"DETECTED: active - sham = {c.mean_diff:+.2e} at {c.window} "
                           f"(95% CI {c.ci_low:+.2e} to {c.ci_high:+.2e}, Holm p {c.p_holm:.3g}); replicate before any interpretation")
    elif primary and all(c.tost_p < alpha for c in primary):
        persistence = (f"NULL: active and sham are equivalent within +/-{prereg.sesoi_rel_alpha:.1e} "
                       f"relative absorption at every delay beyond {analysis.dataset.classical_memory_s:.0f} s")
    else:
        persistence = "INCONCLUSIVE: neither a difference nor equivalence at the preregistered bounds"

    # --- H2 -------------------------------------------------------------------------
    def sig(arm: str, window: str, lag: int = 0) -> bool:
        return any(d.code == code[arm] and d.window == window and d.lag == lag and d.p_holm < alpha
                   for d in analysis.decoding)

    flags: List[str] = []
    ds = analysis.dataset
    retro_ok = bool(ds.retro_valid[analysis.calibration_mask].all())
    neg = [lag for lag in prereg.lags if lag < 0]
    if sig("active", "pre") or any(sig("active", "pre", lag) for lag in neg):
        msg = "symbols decodable before they were drawn"
        flags.append(("RETRO: " + msg + "; audit for pipeline leakage before any interpretation") if retro_ok
                     else ("LEAK: " + msg + " with a predictable stream; the pipeline leaks labels"))
    for arm in ("sham", "thermal"):
        if any(d.code == code[arm] and d.p_holm < alpha and d.window != "during" for d in analysis.decoding):
            flags.append(f"CONTROL: information in the {arm} arm after shutdown; check IR, electrical and acoustic paths")
    if sig("sham", "during"):
        flags.append("CONTROL: sham decodable during drive; THz is not the carrier")

    beyond_active = [w for w in beyond if sig("active", w)]
    beyond_ctrl = [w for w in beyond if sig("sham", w) or sig("thermal", w)]
    pos_lags = [lag for lag in prereg.lags if lag > 0 and sig("active", "pre", lag)]
    if flags:  # a leaking pipeline invalidates every channel claim, so this outranks all of them
        channel = "INTEGRITY FLAGS RAISED: see flags; no channel claim is valid until resolved"
    elif beyond_active and not beyond_ctrl:
        channel = (f"ANOMALOUS CHANNEL: symbols decodable in the active arm only at {', '.join(beyond_active)}"
                   + (f" and in later trials (lags {pos_lags})" if pos_lags else "")
                   + "; replicate on independent hardware before any interpretation")
    elif sig("active", "during") and sig("thermal", "during"):
        channel = "classical: symbols readable during drive in both active and heater-matched arms (thermal channel)"
    elif sig("active", "during") or any(sig("active", w) for w in prereg.windows if w.startswith("post@")):
        channel = "information present only within the classical memory horizon; consistent with heat, not persistence"
    else:
        channel = "no information beyond chance in any window or lag"

    # --- message recovery -------------------------------------------------------------
    messages: List[MessageRecovery] = []
    pay = ds.phase == "payload"
    if pay.any():
        win = prereg.message_window
        x = analysis.features[win]
        m = ds.codebook.size
        for arm in ("active", "sham"):
            tr = analysis.calibration_mask & (ds.arm_code == code[arm])
            te = np.nonzero(pay & (ds.arm_code == code[arm]))[0]
            te = te[np.argsort(ds.payload_index[te])]
            x_all = np.vstack([x[tr], x[te]])
            labels = np.concatenate([ds.symbol[tr], ds.symbol[te]])
            groups = np.concatenate([np.zeros(tr.sum(), int), np.ones(te.size, int)])
            # Train on calibration (group 0), predict payload (group 1): two-fold, keep the payload fold
            preds = nearest_centroid_cv(x_all, labels[None, :], groups, m)[0][tr.sum():]
            sent = ds.symbol[te]
            k = int(np.sum(preds == sent))
            text_sent = text_recv = None
            if payload_text is not None:
                text_sent = payload_text
                text_recv = symbols_to_text(preds.tolist(), m)
            messages.append(MessageRecovery(arm, win, tuple(int(s) for s in sent), tuple(int(s) for s in preds),
                                            k / max(1, te.size), 1.0 / m, _binomial_sf(k, te.size, 1.0 / m),
                                            text_sent, text_recv))

    return AssayReport(analysis.prereg_digest, persistence, channel, tuple(flags), analysis.contrasts,
                       analysis.decoding, tuple(messages), code)


# =============================================================================
# 11. End-to-end simulation helper, power analysis, demo
# =============================================================================


@dataclass(frozen=True, eq=False)
class SimulationResult:
    prereg: Preregistration
    key: AllocationKey
    thermal: ThermalCalibration
    dataset: AssayDataset
    analysis: BlindedAnalysis
    report: AssayReport
    codebook_calibration: CodebookCalibration


def simulate_assay(prereg: Preregistration = Preregistration(), codebook: SymbolCodebook = SymbolCodebook(),
                   cell: SampleCell = SampleCell(), illumination: RingIllumination = RingIllumination(),
                   injected_magnitude: float = 0.0, injected_memory_s: float = 3600.0,
                   payload_text: Optional[str] = None, receiver: ReceiverModel = ReceiverModel(),
                   seed: int = 0) -> SimulationResult:
    """Full pipeline on the simulator: thermal calibration, blinded run, analysis, unblinding."""
    injected = (InjectedEffect(codebook, injected_magnitude, memory_time_s=injected_memory_s)
                if injected_magnitude > 0 else None)
    inst = SimulatedInstrument(cell, illumination, receiver, injected, seed=seed)
    array = PhotonicTHzEmitterArray()
    thermal = thermal_calibration(inst, cell, prereg.readout_frequencies_hz)
    key = make_allocation_key(seed)
    stream = TextStream(payload_text, codebook.size) if payload_text else None
    schedule = make_schedule(prereg, codebook.size, key, len(stream) if stream else 0, seed)
    cal = calibrate_codebook(codebook, array, cell, illumination, prereg.drive_amplitude, prereg.equalize_absorbed_power)
    ds = run_assay(inst, array, cell, illumination, codebook, prereg, schedule, key,
                   RandomStream(seed + 1), stream, cal)
    analysis = analyze(ds, prereg, thermal)
    report = unblind(analysis, key, prereg, payload_text)
    return SimulationResult(prereg, key, thermal, ds, analysis, report, cal)


def power_analysis(magnitudes: Sequence[float], runs: int = 10, prereg: Preregistration = Preregistration(),
                   codebook: SymbolCodebook = SymbolCodebook(), seed: int = 100) -> Dict[float, Dict[str, float]]:
    """Detection rate of H1 and H2 versus injected magnitude (0 gives the false-positive rate)."""
    out: Dict[float, Dict[str, float]] = {}
    for mag in magnitudes:
        h1 = h2 = 0
        for r in range(runs):
            rep = simulate_assay(prereg, codebook, injected_magnitude=mag, seed=seed + 1000 * r).report
            h1 += rep.persistence_verdict.startswith("DETECTED")
            h2 += rep.channel_verdict.startswith("ANOMALOUS")
        out[float(mag)] = {"h1_rate": h1 / runs, "h2_rate": h2 / runs}
    return out


def demo(message: str = "hi", injected_magnitude: float = 0.05) -> str:
    rows: List[str] = []
    cell = SampleCell()
    cb = SymbolCodebook()
    prereg = Preregistration()
    per_arm = prereg.slots_per_arm_per_block or cb.size
    n_trials = prereg.n_blocks * 3 * per_arm + 2 * len(text_to_symbols(message, cb.size))
    rows.append("WATER STRUCTURE ASSAY")
    rows.append(f"  cell: {cell.path_length_m * 1e6:.0f} um x {cell.aperture_radius_m * 2e3:.1f} mm, "
                f"tau_th {cell.thermal_time_constant_s:.2f} s, lateral diffusion {cell.lateral_diffusion_time_s:.0f} s, "
                f"classical memory horizon {cell.classical_memory_s:.0f} s  [L1]")
    rows.append(f"  codebook: {len(cb.frequency_nodes_hz)} frequency nodes x {len(cb.layline_charges)} laylines "
                f"= {cb.size} symbols ({cb.bits_per_symbol:.0f} bits)")
    rows.append(f"  run: {n_trials} trials x {prereg.trial_duration_s() / 60:.1f} min = "
                f"{n_trials * prereg.trial_duration_s() / 3600:.1f} h of bench time")
    for label, mag in (("NULL (no effect in nature)", 0.0),
                       (f"POSITIVE CONTROL (injected L3 effect, {mag_pct(injected_magnitude)})", injected_magnitude)):
        res = simulate_assay(prereg, cb, cell, injected_magnitude=mag, payload_text=message, seed=7)
        rows.append(f"\n--- {label} ---")
        rows.append(f"  thermal calibration slope {np.median(res.thermal.slope_rel_per_k) * 100:.3f} %/K "
                    f"(median R^2 {np.median(res.thermal.r_squared):.3f})")
        rows.extend("  " + line for line in res.report.summary().splitlines())
    return "\n".join(rows)


def mag_pct(m: float) -> str:
    return f"{100 * m:.0f} % peak absorption change"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Water structure assay (simulated)")
    parser.add_argument("--message", default="hi", help="text written through the channel")
    parser.add_argument("--inject", type=float, default=0.05, help="positive-control magnitude (0 disables)")
    parser.add_argument("--json", help="write the positive-control report as JSON")
    args = parser.parse_args(argv)
    print(demo(args.message, args.inject))
    if args.json:
        res = simulate_assay(injected_magnitude=args.inject, payload_text=args.message, seed=7)
        with open(args.json, "w") as fh:
            fh.write(res.report.to_json())
        print(f"json: {args.json}")


if __name__ == "__main__":
    main()
