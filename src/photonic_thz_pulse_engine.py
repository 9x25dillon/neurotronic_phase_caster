"""
Photonic THz Pulse Engine
=========================

All-optical synthesis of terahertz-rate infrared (IR) pulse trains and their
conversion into free-space THz radiation, packaged as a drop-in emitter backend
for the Neurotronic Phase Caster (``EmitterArray`` subclass + pulse planner).

Why photonic
------------
Electronic switches and electro-optic modulators roll off near 100 GHz (RC time
constants, carrier transit), so a 1 THz (1 ps period) IR envelope is built by
interfering phase-locked IR lines. With N lines spaced df around a carrier f0
the slowly varying envelope is

    A(t) = sum_k a_k exp(+i (2 pi k df t + phi_k)),        P(t) = |A(t)|^2

N = 2 gives a sinusoidal beat at df. N >> 2 with locked phases gives a train of
period T = 1/df whose FWHM approaches 0.886 T / N (flat spectrum) and whose
peak-to-average power ratio equals N.

Signal chain
------------
    CW lasers / seed comb
          |
          v
    CombState (immutable line amplitudes)  --->  rate multiplier
          |                                      (Fabry-Perot etalon, temporal Talbot)
          v
    all-optical gate (Kerr gate, saturable absorber)       [optional, time domain]
          |
          v
    THz converter (photomixer, optical rectification)
          |
          v
    PhotonicTHzEmitterArray  (12 channels, one shared laser pair -> phase coherent)
          |
          v
    skin exposure limit check  +  tissue attenuation by depth

Evidence levels (repository convention, see rtd_300ghz/CHANGELOG.md)
--------------------------------------------------------------------
L0  identities, standard formulas, published standard values, specified inputs
L1  model calculations (plane wave, lumped element, undepleted pump)
L2  device and material parameters that need measurement on the actual part
L3  speculation

Each preset parameter carries its level in a trailing comment. This module
models optics, opto-electronics and exposure only. It contains no model of a
neural or cognitive effect, and none should be inferred from its outputs.

Conventions
-----------
SI units internally. Envelope convention exp(+i omega t). A spectral phase
phi_k added to line k shifts the pulse in time by -d(phi)/d(omega). Line k of an
N-line comb sits at f0 + (k - (N - 1)/2) df, so lines are symmetric about f0.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # vectorised IIR filtering; pure-numpy fallback below keeps scipy optional
    from scipy.signal import lfilter as _lfilter
except ImportError:  # pragma: no cover - exercised only without scipy
    _lfilter = None

try:
    from .thz_coherence_wearable_spec import EmitterArray, EmitterConfig
except ImportError:  # executed as a script or with src/ on sys.path
    from thz_coherence_wearable_spec import EmitterArray, EmitterConfig


# =============================================================================
# Constants (L0)
# =============================================================================

C0 = 299_792_458.0  # L0 speed of light, m/s
H_PLANCK = 6.62607015e-34  # L0 J s
Q_E = 1.602176634e-19  # L0 C
EPS0 = 8.8541878128e-12  # L0 F/m
TAU = 2.0 * math.pi

GAUSS_TBP = 2.0 * math.log(2.0) / math.pi  # L0 0.4413, transform-limited Gaussian
SECH2_TBP = 0.3148  # L0 transform-limited sech^2
SECH2_FWHM_PER_TAU = 2.0 * math.acosh(math.sqrt(2.0))  # L0 1.7627

PS2_PER_KM = 1e-27  # 1 ps^2/km in s^2/m
PS3_PER_KM = 1e-39  # 1 ps^3/km in s^3/m


def wavelength_to_frequency(wavelength_m: float) -> float:
    """L0 f = c / lambda."""
    return C0 / wavelength_m


def frequency_to_wavelength(frequency_hz: float) -> float:
    """L0 lambda = c / f."""
    return C0 / frequency_hz


def wavelength_spacing_for_beat(beat_hz: float, center_wavelength_m: float = 1550e-9) -> float:
    """L0 exact wavelength separation of two lines placed at f0 -/+ beat/2.

    At 1550 nm a 1 THz beat needs 8.01 nm (the small-signal form
    lambda^2 df / c gives the same value to 0.01 %).
    """
    if beat_hz <= 0:
        raise ValueError("beat_hz must be positive")
    f0 = wavelength_to_frequency(center_wavelength_m)
    if beat_hz >= 2.0 * f0:
        raise ValueError("beat_hz exceeds twice the optical carrier")
    return C0 / (f0 - beat_hz / 2.0) - C0 / (f0 + beat_hz / 2.0)


# =============================================================================
# 1. Comb state and waveform synthesis
# =============================================================================


@dataclass(frozen=True, eq=False)
class CombState:
    """Immutable set of phase-coherent optical lines on a uniform grid.

    ``amplitudes`` are complex sqrt(W): |a_k|^2 is the power of line k and
    arg(a_k) its phase. Every optical operation in this module maps a
    CombState to a new CombState, so pipelines compose as pure functions.
    """

    center_hz: float
    spacing_hz: float
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        amps = np.array(self.amplitudes, dtype=complex).reshape(-1)
        if amps.size < 1:
            raise ValueError("a comb needs at least one line")
        if not np.all(np.isfinite(amps)):
            raise ValueError("non-finite line amplitude")
        if not (self.spacing_hz > 0 and math.isfinite(self.spacing_hz)):
            raise ValueError("spacing_hz must be positive and finite")
        if not (self.center_hz > 0 and math.isfinite(self.center_hz)):
            raise ValueError("center_hz must be positive and finite")
        half_span = 0.5 * (amps.size - 1) * self.spacing_hz
        if half_span >= self.center_hz:
            raise ValueError("comb extends to non-positive optical frequency")
        amps.setflags(write=False)
        object.__setattr__(self, "amplitudes", amps)

    # --- derived quantities -------------------------------------------------
    @property
    def n_lines(self) -> int:
        return int(self.amplitudes.size)

    @property
    def line_offsets(self) -> np.ndarray:
        """Line index relative to the comb center (half-integer for even N)."""
        return np.arange(self.n_lines) - 0.5 * (self.n_lines - 1)

    @property
    def line_frequencies_hz(self) -> np.ndarray:
        return self.center_hz + self.line_offsets * self.spacing_hz

    @property
    def line_wavelengths_m(self) -> np.ndarray:
        return C0 / self.line_frequencies_hz

    @property
    def line_powers_w(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    @property
    def total_power_w(self) -> float:
        return float(np.sum(self.line_powers_w))

    @property
    def period_s(self) -> float:
        """Period of the line grid (the intensity period is T / r after r-fold multiplication)."""
        return 1.0 / self.spacing_hz

    @property
    def adjacent_wavelength_spacing_m(self) -> float:
        return wavelength_spacing_for_beat(self.spacing_hz, C0 / self.center_hz)

    @property
    def optical_span_hz(self) -> float:
        return (self.n_lines - 1) * self.spacing_hz

    def with_amplitudes(self, amplitudes: np.ndarray) -> "CombState":
        return CombState(self.center_hz, self.spacing_hz, amplitudes)

    def scaled_to_power(self, total_power_w: float) -> "CombState":
        if total_power_w < 0:
            raise ValueError("total_power_w must be non-negative")
        current = self.total_power_w
        if current == 0.0:
            raise ValueError("cannot scale a comb with zero power")
        return self.with_amplitudes(self.amplitudes * math.sqrt(total_power_w / current))


def make_comb(
    spacing_hz: float,
    n_lines: int,
    center_wavelength_m: float = 1550e-9,
    total_power_w: float = 1e-3,
    envelope: str = "flat",
    envelope_fwhm_hz: Optional[float] = None,
    phases="locked",
    seed: int = 0,
) -> CombState:
    """Build a comb of ``n_lines`` lines spaced ``spacing_hz`` around ``center_wavelength_m``.

    envelope : "flat", "gaussian" or "sech2" power spectrum. ``envelope_fwhm_hz``
        is the power-spectrum FWHM (default: one third of the line span).
    phases   : "locked" (transform limited, mode locked), "random" (free-running
        lines, deterministic via ``seed``) or an explicit array of radians.
    """
    n_lines = int(n_lines)
    if n_lines < 1:
        raise ValueError("n_lines must be >= 1")
    offsets = np.arange(n_lines) - 0.5 * (n_lines - 1)
    nu = offsets * spacing_hz
    if envelope == "flat":
        power = np.ones(n_lines)
    else:
        fwhm = envelope_fwhm_hz if envelope_fwhm_hz is not None else max(spacing_hz, n_lines * spacing_hz / 3.0)
        if fwhm <= 0:
            raise ValueError("envelope_fwhm_hz must be positive")
        if envelope == "gaussian":
            power = np.exp(-4.0 * math.log(2.0) * (nu / fwhm) ** 2)
        elif envelope == "sech2":
            power = 1.0 / np.cosh(SECH2_FWHM_PER_TAU * nu / fwhm) ** 2
        else:
            raise ValueError("envelope must be 'flat', 'gaussian' or 'sech2'")

    if isinstance(phases, str):
        if phases == "locked":
            phi = np.zeros(n_lines)
        elif phases == "random":
            phi = np.random.default_rng(seed).uniform(0.0, TAU, n_lines)
        else:
            raise ValueError("phases must be 'locked', 'random' or an array")
    else:
        phi = np.asarray(phases, dtype=float).reshape(-1)
        if phi.size != n_lines:
            raise ValueError("explicit phase array must have n_lines entries")

    amps = np.sqrt(power) * np.exp(1j * phi)
    comb = CombState(wavelength_to_frequency(center_wavelength_m), spacing_hz, amps)
    return comb.scaled_to_power(total_power_w)


def default_samples_per_period(n_lines: int, multiple_of: int = 1) -> int:
    """Power-of-two grid (times ``multiple_of``) with >= 16 samples per transform-limited FWHM."""
    base = max(256, 16 * int(n_lines))
    m = max(1, int(multiple_of))
    return m * int(2 ** math.ceil(math.log2(max(1.0, base / m))))


def intensity_waveform(
    comb: CombState, n_periods: int = 1, samples_per_period: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Optical power P(t) in W over ``n_periods`` grid periods. O(M log M) via inverse FFT.

    The reference frequency only rotates the envelope by a global phase, so
    |A(t)|^2 is computed with lines placed on FFT bins 0..N-1.
    """
    m = int(samples_per_period or default_samples_per_period(comb.n_lines))
    if m < comb.n_lines:
        raise ValueError("samples_per_period must be >= n_lines")
    spectrum = np.zeros(m, dtype=complex)
    spectrum[: comb.n_lines] = comb.amplitudes
    one_period = np.abs(m * np.fft.ifft(spectrum)) ** 2
    power = np.tile(one_period, int(n_periods))
    t = np.arange(power.size) * (comb.period_s / m)
    return t, power


def line_autocorrelation(amplitudes: np.ndarray) -> np.ndarray:
    """P_q = sum_k a_{k+q} conj(a_k) for q = 0..N-1 (complex power harmonics, W).

    P(t) = sum_q P_q exp(i 2 pi q df t), so |P_q| is the optical power
    modulation at harmonic q of the line spacing. O(N log N).
    """
    a = np.asarray(amplitudes, dtype=complex)
    n = a.size
    size = int(2 ** math.ceil(math.log2(max(2, 2 * n))))
    spec = np.fft.fft(a, size)
    acf = np.fft.ifft(spec * np.conj(spec))
    return acf[:n]


@dataclass(frozen=True)
class PulseMetrics:
    repetition_rate_hz: float  # dominant intensity harmonic
    period_s: float
    fwhm_s: float  # nan when the waveform never drops below half maximum
    peak_power_w: float
    average_power_w: float
    peak_to_average: float
    duty_cycle: float  # fwhm / period
    modulation_depth: float  # (Pmax - Pmin) / (Pmax + Pmin)
    subharmonic_suppression_db: float  # dominant harmonic vs strongest non-multiple


def flat_comb_fwhm_s(n_lines: int, spacing_hz: float) -> float:
    """L0 exact FWHM of sin^2(N u)/sin^2(u), u = pi df t (flat, locked comb).

    N = 2 returns T/2; large N approaches 0.8859 / (N df).
    """
    n = int(n_lines)
    if n < 2:
        return math.inf
    lo, hi = 1e-15, math.pi / n
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = (math.sin(n * mid) / (n * math.sin(mid))) ** 2
        if value > 0.5:
            lo = mid
        else:
            hi = mid
    return 2.0 * 0.5 * (lo + hi) / (math.pi * spacing_hz)


def _fwhm_around_peak(power: np.ndarray, dt: float, max_half_window: int) -> float:
    i0 = int(np.argmax(power))
    half = 0.5 * power[i0]
    n = power.size

    def walk(direction: int) -> float:
        for step in range(1, max_half_window + 1):
            j = (i0 + direction * step) % n
            if power[j] <= half:
                jp = (i0 + direction * (step - 1)) % n
                span = power[jp] - power[j]
                frac = (power[jp] - half) / span if span > 0 else 0.0
                return (step - 1 + frac) * dt
        return math.nan

    return walk(+1) + walk(-1)


def pulse_metrics(comb: CombState, samples_per_period: Optional[int] = None) -> PulseMetrics:
    """Repetition rate, FWHM, peak/average and harmonic purity of P(t)."""
    acf = line_autocorrelation(comb.amplitudes)
    mags = np.abs(acf)
    p_avg = float(mags[0])
    if comb.n_lines == 1 or np.max(mags[1:], initial=0.0) <= 1e-15 * max(p_avg, 1e-300):
        return PulseMetrics(0.0, math.inf, math.nan, p_avg, p_avg, 1.0, math.nan, 0.0, math.inf)

    q_star = int(np.argmax(mags[1:]) + 1)
    rate = q_star * comb.spacing_hz
    q = np.arange(1, comb.n_lines)
    non_multiple = mags[1:][(q % q_star) != 0]
    leak = float(np.max(non_multiple)) if non_multiple.size else 0.0
    suppression = math.inf if leak <= 0 else 20.0 * math.log10(mags[q_star] / leak)

    m = int(samples_per_period or default_samples_per_period(comb.n_lines, multiple_of=q_star))
    _, power = intensity_waveform(comb, 1, m)
    dt = comb.period_s / m
    window = max(1, int(round(m / q_star / 2)))
    fwhm = _fwhm_around_peak(power, dt, window)
    p_max, p_min = float(power.max()), float(power.min())
    depth = (p_max - p_min) / (p_max + p_min) if (p_max + p_min) > 0 else 0.0
    return PulseMetrics(
        repetition_rate_hz=rate,
        period_s=1.0 / rate,
        fwhm_s=fwhm,
        peak_power_w=p_max,
        average_power_w=p_avg,
        peak_to_average=p_max / p_avg,
        duty_cycle=fwhm * rate if math.isfinite(fwhm) else math.nan,
        modulation_depth=depth,
        subharmonic_suppression_db=suppression,
    )


# =============================================================================
# 2. Dual-laser heterodyne (N = 2)
# =============================================================================


@dataclass(frozen=True)
class LaserLine:
    wavelength_m: float
    power_w: float
    linewidth_hz: float = 100e3  # L2 narrow-linewidth DFB / external-cavity class
    polarization_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0 or self.power_w < 0 or self.linewidth_hz < 0:
            raise ValueError("invalid laser parameters")

    @property
    def frequency_hz(self) -> float:
        return wavelength_to_frequency(self.wavelength_m)


@dataclass(frozen=True)
class HeterodynePair:
    """Two CW lasers whose intensity beats at |f_a - f_b|."""

    laser_a: LaserLine
    laser_b: LaserLine

    @classmethod
    def symmetric(
        cls,
        beat_hz: float,
        center_wavelength_m: float = 1550e-9,
        total_power_w: float = 1e-3,
        linewidth_hz: float = 100e3,
    ) -> "HeterodynePair":
        f0 = wavelength_to_frequency(center_wavelength_m)
        p = 0.5 * total_power_w
        return cls(
            LaserLine(C0 / (f0 - beat_hz / 2.0), p, linewidth_hz),
            LaserLine(C0 / (f0 + beat_hz / 2.0), p, linewidth_hz),
        )

    @property
    def beat_frequency_hz(self) -> float:
        return abs(self.laser_a.frequency_hz - self.laser_b.frequency_hz)

    @property
    def wavelength_separation_m(self) -> float:
        return abs(self.laser_a.wavelength_m - self.laser_b.wavelength_m)

    @property
    def polarization_overlap(self) -> float:
        """Fraction of laser_b power co-polarized with laser_a (cos^2 of the angle)."""
        return math.cos(self.laser_a.polarization_rad - self.laser_b.polarization_rad) ** 2

    @property
    def modulation_depth(self) -> float:
        """L0 m = 2 sqrt(Pa Pb) |cos(theta)| / (Pa + Pb)."""
        pa, pb = self.laser_a.power_w, self.laser_b.power_w
        if pa + pb == 0:
            return 0.0
        return 2.0 * math.sqrt(pa * pb * self.polarization_overlap) / (pa + pb)

    @property
    def beat_linewidth_hz(self) -> float:
        """L1 uncorrelated Lorentzian lasers: linewidths add."""
        return self.laser_a.linewidth_hz + self.laser_b.linewidth_hz

    @property
    def mean_power_w(self) -> float:
        return self.laser_a.power_w + self.laser_b.power_w

    def to_comb(self) -> Tuple[CombState, float]:
        """Return (interfering two-line comb, non-interfering DC power in W)."""
        beat = self.beat_frequency_hz
        if beat <= 0:
            raise ValueError("lasers are degenerate; no beat")
        low, high = sorted((self.laser_a, self.laser_b), key=lambda l: l.frequency_hz)
        rel = math.cos(high.polarization_rad - low.polarization_rad)
        amps = np.array([math.sqrt(low.power_w), math.sqrt(high.power_w) * abs(rel)], dtype=complex)
        center = 0.5 * (low.frequency_hz + high.frequency_hz)
        dc_only = high.power_w * (1.0 - rel**2)
        return CombState(center, beat, amps), dc_only

    def retuned(self, beat_hz: float, tune: str = "b") -> "HeterodynePair":
        """Move one laser so the beat equals ``beat_hz``; the other laser stays fixed."""
        if beat_hz <= 0:
            raise ValueError("beat_hz must be positive")
        fixed, moving = (self.laser_a, self.laser_b) if tune == "b" else (self.laser_b, self.laser_a)
        side = 1.0 if moving.frequency_hz >= fixed.frequency_hz else -1.0
        new = replace(moving, wavelength_m=C0 / (fixed.frequency_hz + side * beat_hz))
        return HeterodynePair(fixed, new) if tune == "b" else HeterodynePair(new, fixed)

    def scaled_to_power(self, total_power_w: float) -> "HeterodynePair":
        scale = total_power_w / self.mean_power_w
        return HeterodynePair(
            replace(self.laser_a, power_w=self.laser_a.power_w * scale),
            replace(self.laser_b, power_w=self.laser_b.power_w * scale),
        )


# =============================================================================
# 3. Kerr soliton microresonator
# =============================================================================


@dataclass(frozen=True)
class MicroringResonator:
    """Ring cavity whose free spectral range sets the soliton repetition rate."""

    radius_m: float
    group_index: float = 2.0  # L2 Si3N4 waveguide
    beta2_s2_per_m: float = -60.0 * PS2_PER_KM  # L2 anomalous dispersion required for bright solitons
    thermo_optic_per_k: float = 2.45e-5  # L2 Si3N4 dn/dT
    loaded_q: float = 1.0e6  # L2
    pump_wavelength_m: float = 1550e-9

    @classmethod
    def for_repetition_rate(cls, rep_rate_hz: float, group_index: float = 2.0, **kwargs) -> "MicroringResonator":
        """L0 f_rep = c / (n_g L)  ->  L = c / (n_g f_rep), R = L / 2 pi."""
        circumference = C0 / (group_index * rep_rate_hz)
        return cls(radius_m=circumference / TAU, group_index=group_index, **kwargs)

    @property
    def circumference_m(self) -> float:
        return TAU * self.radius_m

    @property
    def fsr_hz(self) -> float:
        return C0 / (self.group_index * self.circumference_m)

    @property
    def resonance_linewidth_hz(self) -> float:
        return wavelength_to_frequency(self.pump_wavelength_m) / self.loaded_q

    @property
    def fsr_thermal_coefficient_hz_per_k(self) -> float:
        """L1 dFSR/dT ~= -FSR (dn/dT) / n_g (thermal expansion neglected)."""
        return -self.fsr_hz * self.thermo_optic_per_k / self.group_index

    def soliton_tau_s(self, detuning_hz: float) -> float:
        """L1 sech parameter tau_s = sqrt(-beta2 c / (2 n_g delta_omega)) (Herr et al. 2014)."""
        if self.beta2_s2_per_m >= 0:
            raise ValueError("normal dispersion: no bright dissipative Kerr soliton")
        if detuning_hz <= 0:
            raise ValueError("soliton existence requires red detuning > 0")
        return math.sqrt(-self.beta2_s2_per_m * C0 / (2.0 * self.group_index * TAU * detuning_hz))

    def soliton_fwhm_s(self, detuning_hz: float) -> float:
        return SECH2_FWHM_PER_TAU * self.soliton_tau_s(detuning_hz)

    def soliton_comb(self, detuning_hz: float, total_power_w: float = 1e-3, span_fwhms: float = 3.0) -> CombState:
        """sech^2 comb covering +/- ``span_fwhms`` spectral FWHM (phase-locked lines)."""
        spectral_fwhm = SECH2_TBP / self.soliton_fwhm_s(detuning_hz)
        n = max(3, 2 * int(math.ceil(span_fwhms * spectral_fwhm / self.fsr_hz)) + 1)
        return make_comb(
            self.fsr_hz, n, self.pump_wavelength_m, total_power_w, envelope="sech2", envelope_fwhm_hz=spectral_fwhm
        )


# =============================================================================
# 4. Repetition-rate multiplication
# =============================================================================


def _reference_index(comb: CombState, reference_line: Optional[int]) -> int:
    if reference_line is None:
        return int(np.argmax(comb.line_powers_w))
    if not 0 <= reference_line < comb.n_lines:
        raise ValueError("reference_line out of range")
    return int(reference_line)


@dataclass(frozen=True)
class FabryPerotEtalon:
    """Airy filter that keeps every m-th comb line (FSR = m * seed spacing)."""

    fsr_hz: float
    finesse: float = 200.0  # L2
    refractive_index: float = 1.0

    @classmethod
    def for_multiplication(cls, seed_spacing_hz: float, factor: int, finesse: float = 200.0) -> "FabryPerotEtalon":
        if int(factor) < 1:
            raise ValueError("factor must be >= 1")
        return cls(fsr_hz=int(factor) * seed_spacing_hz, finesse=finesse)

    @property
    def coefficient_of_finesse(self) -> float:
        return (2.0 * self.finesse / math.pi) ** 2

    @property
    def length_m(self) -> float:
        """L0 L = c / (2 n FSR)."""
        return C0 / (2.0 * self.refractive_index * self.fsr_hz)

    @property
    def passband_fwhm_hz(self) -> float:
        return self.fsr_hz / self.finesse

    def transmission(self, offset_hz) -> np.ndarray:
        """L0 Airy power transmission of a lossless etalon at ``offset_hz`` from a resonance."""
        s = np.sin(math.pi * np.asarray(offset_hz, dtype=float) / self.fsr_hz)
        return 1.0 / (1.0 + self.coefficient_of_finesse * s**2)

    def adjacent_suppression_db(self, seed_spacing_hz: float) -> float:
        return float(-10.0 * np.log10(self.transmission(seed_spacing_hz)))

    def apply(self, comb: CombState, reference_line: Optional[int] = None) -> CombState:
        ref = _reference_index(comb, reference_line)
        offsets = (np.arange(comb.n_lines) - ref) * comb.spacing_hz
        return comb.with_amplitudes(comb.amplitudes * np.sqrt(self.transmission(offsets)))


@dataclass(frozen=True)
class TalbotMultiplier:
    """Fractional temporal Talbot effect: lossless r-fold rate multiplication by dispersion.

    L0 condition: Phi2 = (s / r) T^2 / (2 pi), gcd(s, r) = 1, T the input period.
    Each line acquires pi (s/r) k^2, the intensity becomes T/r periodic and
    every sub-pulse carries 1/r of the original peak power.

    implementation
      "shaper": line-by-line phase mask pi (s/r) k^2 mod 2 pi (programmable
                pulse shaper). Pure quadratic phase, mask period 2r lines.
      "fiber" : dispersive fiber of length Phi2/|beta2|. Third-order dispersion
                beta3 L adds cubic phase; at THz output rates with sub-ps pulses
                it dominates (hundreds of rad in SMF-28) and destroys the
                multiplication. Check ``third_order_phase_rad`` first.
    """

    factor: int
    numerator: int = 1
    implementation: str = "shaper"
    beta2_s2_per_m: float = -21.7 * PS2_PER_KM  # L2 SMF-28 at 1550 nm (D = 17 ps/nm/km)
    beta3_s3_per_m: float = 0.12 * PS3_PER_KM  # L2 SMF-28 at 1550 nm

    def __post_init__(self) -> None:
        if int(self.factor) < 1 or int(self.numerator) < 1:
            raise ValueError("factor and numerator must be positive integers")
        if math.gcd(int(self.factor), int(self.numerator)) != 1:
            raise ValueError("numerator and factor must be coprime")
        if self.implementation not in ("shaper", "fiber"):
            raise ValueError("implementation must be 'shaper' or 'fiber'")
        if self.beta2_s2_per_m == 0:
            raise ValueError("beta2 must be non-zero")

    def required_gdd_s2(self, seed_spacing_hz: float) -> float:
        period = 1.0 / seed_spacing_hz
        return (self.numerator / self.factor) * period**2 / TAU

    def fiber_length_m(self, seed_spacing_hz: float) -> float:
        return self.required_gdd_s2(seed_spacing_hz) / abs(self.beta2_s2_per_m)

    def third_order_phase_rad(self, seed_spacing_hz: float, half_bandwidth_hz: float) -> float:
        """L1 cubic phase beta3 L w^3 / 6 at ``half_bandwidth_hz`` from center (fiber case)."""
        w = TAU * half_bandwidth_hz
        return abs(self.beta3_s3_per_m * self.fiber_length_m(seed_spacing_hz) * w**3 / 6.0)

    def mask_phases_rad(self, n_lines: int) -> np.ndarray:
        k = np.arange(int(n_lines)) - (int(n_lines) - 1) // 2
        return np.mod(math.pi * self.numerator * k.astype(float) ** 2 / self.factor, TAU)

    def apply(self, comb: CombState, include_third_order: bool = True) -> CombState:
        if self.implementation == "shaper":
            phase = self.mask_phases_rad(comb.n_lines)
        else:
            length = self.fiber_length_m(comb.spacing_hz)
            phi2 = math.copysign(self.required_gdd_s2(comb.spacing_hz), self.beta2_s2_per_m)
            phi3 = self.beta3_s3_per_m * length if include_third_order else 0.0
            omega = TAU * comb.line_offsets * comb.spacing_hz
            phase = 0.5 * phi2 * omega**2 + phi3 * omega**3 / 6.0
        return comb.with_amplitudes(comb.amplitudes * np.exp(1j * phase))


def design_seed_comb(
    output_rate_hz: float,
    seed_spacing_hz: float,
    duty_cycle: float,
    center_wavelength_m: float = 1550e-9,
    total_power_w: float = 1e-3,
) -> Tuple[CombState, int]:
    """Gaussian, phase-locked seed comb for r-fold multiplication to ``output_rate_hz``.

    r = round(output_rate / seed_spacing); the seed spacing is then re-tuned to
    output_rate / r so the output lands exactly on target. The spectral FWHM is
    the transform limit of a pulse of ``duty_cycle / output_rate``; the comb
    spans +/- 3 sigma of the amplitude envelope (truncation error ~1e-3).
    """
    if not 0.0 < duty_cycle < 0.5:
        raise ValueError("duty_cycle must lie in (0, 0.5)")
    factor = max(1, int(round(output_rate_hz / seed_spacing_hz)))
    seed = output_rate_hz / factor
    spectral_fwhm = GAUSS_TBP * output_rate_hz / duty_cycle
    sigma_lines = spectral_fwhm / (2.0 * math.sqrt(math.log(2.0))) / seed
    n_lines = max(3 * factor, int(math.ceil(6.0 * sigma_lines)))
    comb = make_comb(seed, n_lines, center_wavelength_m, total_power_w, "gaussian", spectral_fwhm)
    return comb, factor


# =============================================================================
# 5. All-optical gates (time domain)
# =============================================================================


def _uniform_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("need at least two time samples")
    d = np.diff(t)
    dt = float(d.mean())
    if dt <= 0 or np.max(np.abs(d - dt)) > 1e-6 * dt:
        raise ValueError("time grid must be uniform and increasing")
    return dt


def _first_order_lowpass(x: np.ndarray, dt: float, tau: float, periodic: bool) -> np.ndarray:
    """Unit-DC-gain exponential response y' = (x - y)/tau, exact for sample-held input. O(n)."""
    x = np.asarray(x, dtype=float)
    if tau <= 0:
        return x.copy()
    decay = math.exp(-dt / tau)
    b, a = [1.0 - decay], [1.0, -decay]

    def run(y_prev: float) -> np.ndarray:
        if _lfilter is not None:
            return _lfilter(b, a, x, zi=[decay * y_prev])[0]
        out = np.empty_like(x)
        y = y_prev
        for i, xi in enumerate(x):
            y = decay * y + (1.0 - decay) * xi
            out[i] = y
        return out

    if not periodic:
        return run(0.0)
    free = run(0.0)
    # Periodic steady state: y0 = y_end(y0)  ->  y0 = free_end / (1 - decay^n)
    y0 = free[-1] / (1.0 - decay**x.size)
    return run(y0)


@dataclass(frozen=True)
class KerrMedium:
    name: str
    n2_m2_per_w: float
    length_m: float
    delayed_fraction: float = 0.0  # share of the response with time constant response_time_s
    response_time_s: float = 0.0
    # L1 isotropic electronic Kerr with the pump at 45 deg; reorientational media differ (L2)
    birefringence_factor: float = 4.0 / 3.0
    damage_intensity_w_per_m2: float = math.inf


KERR_MEDIA: Dict[str, KerrMedium] = {
    # n2, response and damage values are order-of-magnitude literature figures near 1550 nm (L2)
    "fused_silica_1mm": KerrMedium("fused_silica_1mm", 2.7e-20, 1e-3, 0.18, 32e-15, 4.0 / 3.0, 1e17),
    "cs2_1mm": KerrMedium("cs2_1mm", 3.0e-18, 1e-3, 0.85, 1.6e-12, 4.0 / 3.0, 5e15),
    "as2s3_waveguide_10mm": KerrMedium("as2s3_waveguide_10mm", 3.0e-18, 1e-2, 0.1, 50e-15, 4.0 / 3.0, 1e15),
}


@dataclass(frozen=True)
class GateResult:
    t: np.ndarray
    transmission: np.ndarray
    probe_out_w: np.ndarray
    peak_transmission: float
    window_fwhm_s: float
    lag_s: float  # time of peak transmission minus time of peak pump
    contrast_db: float
    damage_risk: bool


def _gate_result(t, transmission, probe_w, pump_w, dt, leak, damage) -> GateResult:
    transmission = np.asarray(transmission, dtype=float)
    peak = float(transmission.max())
    floor = max(float(transmission.min()), leak, 1e-300)
    window = _fwhm_around_peak(transmission - transmission.min(), dt, transmission.size // 2)
    lag = (int(np.argmax(transmission)) - int(np.argmax(pump_w))) * dt
    return GateResult(
        t=np.asarray(t, dtype=float),
        transmission=transmission,
        probe_out_w=transmission * np.asarray(probe_w, dtype=float),
        peak_transmission=peak,
        window_fwhm_s=window,
        lag_s=lag,
        contrast_db=10.0 * math.log10(peak / floor) if peak > 0 else 0.0,
        damage_risk=damage,
    )


@dataclass(frozen=True)
class KerrGate:
    """Optical Kerr gate: pump-induced birefringence between crossed polarizers.

    L0 T(t) = leak + (1 - leak) sin^2(2 psi) sin^2(dphi(t)/2),
       dphi = (2 pi / lambda) L b n2 I_eff(t),  I_eff = response-filtered pump intensity.
    """

    medium: KerrMedium
    mode_area_m2: float = 50e-12  # L2 (~8 um mode field diameter)
    probe_wavelength_m: float = 1550e-9
    pump_probe_angle_rad: float = math.pi / 4.0
    polarizer_leakage: float = 1e-5  # L2 extinction ratio of the analyzer

    @property
    def pi_intensity_w_per_m2(self) -> float:
        """L0 pump intensity for a pi phase difference (full switching)."""
        m = self.medium
        return self.probe_wavelength_m / (2.0 * m.birefringence_factor * m.n2_m2_per_w * m.length_m)

    @property
    def pi_peak_power_w(self) -> float:
        return self.pi_intensity_w_per_m2 * self.mode_area_m2

    def induced_phase(self, t: np.ndarray, pump_power_w: np.ndarray, periodic: bool = True) -> np.ndarray:
        dt = _uniform_dt(t)
        intensity = np.asarray(pump_power_w, dtype=float) / self.mode_area_m2
        m = self.medium
        slow = _first_order_lowpass(intensity, dt, m.response_time_s, periodic)
        i_eff = (1.0 - m.delayed_fraction) * intensity + m.delayed_fraction * slow
        return TAU / self.probe_wavelength_m * m.length_m * m.birefringence_factor * m.n2_m2_per_w * i_eff

    def switch(self, t: np.ndarray, pump_power_w: np.ndarray, probe_power_w=1.0, periodic: bool = True) -> GateResult:
        dt = _uniform_dt(t)
        dphi = self.induced_phase(t, pump_power_w, periodic)
        geometry = math.sin(2.0 * self.pump_probe_angle_rad) ** 2
        trans = self.polarizer_leakage + (1.0 - self.polarizer_leakage) * geometry * np.sin(dphi / 2.0) ** 2
        probe = np.broadcast_to(np.asarray(probe_power_w, dtype=float), trans.shape)
        peak_i = float(np.max(pump_power_w)) / self.mode_area_m2
        return _gate_result(t, trans, probe, pump_power_w, dt, self.polarizer_leakage,
                            peak_i >= self.medium.damage_intensity_w_per_m2)


@dataclass(frozen=True)
class SaturableAbsorber:
    name: str
    modulation_depth: float  # saturable loss q0
    nonsaturable_loss: float
    saturation_fluence_j_per_m2: float
    recovery_time_s: float


SATURABLE_ABSORBERS: Dict[str, SaturableAbsorber] = {
    # order-of-magnitude parameters (L2); measure the actual film / SESAM
    "graphene_monolayer": SaturableAbsorber("graphene_monolayer", 0.015, 0.01, 0.1, 1.5e-12),
    "sesam_inp": SaturableAbsorber("sesam_inp", 0.02, 0.005, 0.5, 2.0e-12),
}


@dataclass(frozen=True)
class SaturableAbsorberGate:
    """Pump-saturated absorber gating a probe.  dq/dt = (q0 - q)/tau - q I(t)/F_sat.

    Integrated with the exact exponential solution per sample (unconditionally
    stable), iterated to the periodic steady state. O(n) per pass.
    """

    absorber: SaturableAbsorber
    beam_area_m2: float = 50e-12  # L2

    def loss_trace(self, t: np.ndarray, pump_power_w: np.ndarray, periodic: bool = True,
                   tol: float = 1e-12, max_passes: int = 50) -> np.ndarray:
        dt = _uniform_dt(t)
        ab = self.absorber
        intensity = np.asarray(pump_power_w, dtype=float) / self.beam_area_m2
        rate = 1.0 / ab.recovery_time_s + intensity / ab.saturation_fluence_j_per_m2
        q_ss = (ab.modulation_depth / ab.recovery_time_s) / rate
        decay = np.exp(-rate * dt)
        q = np.empty_like(intensity)
        q_prev = ab.modulation_depth
        for _ in range(max_passes if periodic else 1):
            start = q_prev
            for i in range(intensity.size):
                q_prev = q_ss[i] + (q_prev - q_ss[i]) * decay[i]
                q[i] = q_prev
            if abs(q_prev - start) <= tol:
                break
        return q

    def switch(self, t: np.ndarray, pump_power_w: np.ndarray, probe_power_w=1.0, periodic: bool = True) -> GateResult:
        dt = _uniform_dt(t)
        q = self.loss_trace(t, pump_power_w, periodic)
        trans = (1.0 - q) * (1.0 - self.absorber.nonsaturable_loss)
        probe = np.broadcast_to(np.asarray(probe_power_w, dtype=float), trans.shape)
        return _gate_result(t, trans, probe, pump_power_w, dt, float(trans.min()), False)


# =============================================================================
# 6. IR-envelope -> THz converters
# =============================================================================


@dataclass(frozen=True)
class PhotomixerOutput:
    dc_photocurrent_a: float
    harmonic_frequencies_hz: np.ndarray
    harmonic_current_peak_a: np.ndarray
    harmonic_power_w: np.ndarray
    fundamental_power_w: float
    total_power_w: float
    over_current: bool


@dataclass(frozen=True)
class Photomixer:
    """Lumped photomixer (Brown et al.): P(f) = 1/2 R_A I_ac^2 / ((1 + (w tau)^2)(1 + (w R_A C)^2)).

    L1 model. Presets are L2 and must be replaced by the measured response of
    the actual device (calibrated THz power meter, IR blocked).
    """

    name: str
    responsivity_a_per_w: float
    response_time_s: float  # carrier lifetime (photoconductor) or transit time (UTC-PD)
    capacitance_f: float
    antenna_resistance_ohm: float
    max_photocurrent_a: float  # thermal / space-charge limit
    design_wavelength_m: float

    def rolloff(self, frequency_hz) -> np.ndarray:
        w = TAU * np.asarray(frequency_hz, dtype=float)
        return 1.0 / ((1.0 + (w * self.response_time_s) ** 2) *
                      (1.0 + (w * self.antenna_resistance_ohm * self.capacitance_f) ** 2))

    def thz_power_w(self, frequency_hz, ac_current_peak_a) -> np.ndarray:
        i = np.asarray(ac_current_peak_a, dtype=float)
        return 0.5 * self.antenna_resistance_ohm * i**2 * self.rolloff(frequency_hz)

    def max_power_w(self, frequency_hz, modulation_depth: float = 1.0) -> np.ndarray:
        return self.thz_power_w(frequency_hz, modulation_depth * self.max_photocurrent_a)

    def optical_power_for(self, thz_power_w: float, frequency_hz: float, modulation_depth: float = 1.0) -> float:
        """Inverse of the heterodyne response: optical power giving ``thz_power_w`` at ``frequency_hz``."""
        if thz_power_w < 0 or modulation_depth <= 0:
            raise ValueError("thz_power_w >= 0 and modulation_depth > 0 required")
        i_ac = math.sqrt(2.0 * thz_power_w / (self.antenna_resistance_ohm * float(self.rolloff(frequency_hz))))
        return i_ac / (modulation_depth * self.responsivity_a_per_w)

    def response_to_comb(self, comb: CombState, extra_dc_power_w: float = 0.0,
                         max_harmonics: Optional[int] = None) -> PhotomixerOutput:
        acf = line_autocorrelation(comb.amplitudes)
        n_h = comb.n_lines - 1 if max_harmonics is None else min(comb.n_lines - 1, int(max_harmonics))
        q = np.arange(1, n_h + 1)
        i_dc = self.responsivity_a_per_w * (float(acf[0].real) + extra_dc_power_w)
        i_q = 2.0 * self.responsivity_a_per_w * np.abs(acf[1:n_h + 1])
        freqs = q * comb.spacing_hz
        p_q = self.thz_power_w(freqs, i_q)
        fundamental = float(p_q[int(np.argmax(i_q))]) if p_q.size else 0.0
        return PhotomixerOutput(
            dc_photocurrent_a=i_dc,
            harmonic_frequencies_hz=freqs,
            harmonic_current_peak_a=i_q,
            harmonic_power_w=p_q,
            fundamental_power_w=fundamental,
            total_power_w=float(np.sum(p_q)),
            over_current=i_dc > self.max_photocurrent_a * (1.0 + 1e-9),
        )


def self_complementary_antenna_ohm(substrate_permittivity: float) -> float:
    """L1 R = 60 pi / sqrt((1 + eps_r)/2): 72 ohm on InP (12.6), 71.5 ohm on GaAs (12.9)."""
    return 60.0 * math.pi / math.sqrt((1.0 + substrate_permittivity) / 2.0)


PHOTOMIXERS: Dict[str, Photomixer] = {
    # UTC-PD with log-spiral antenna on InP. Literature: >10 uW at 1 THz.
    "utc_pd_inp_1550": Photomixer(
        name="utc_pd_inp_1550",
        responsivity_a_per_w=0.15,  # L2
        response_time_s=0.5e-12,  # L2 electron transit
        capacitance_f=10e-15,  # L2
        antenna_resistance_ohm=self_complementary_antenna_ohm(12.6),  # L1
        max_photocurrent_a=10e-3,  # L2 thermal limit
        design_wavelength_m=1550e-9,
    ),
    # LT-GaAs photoconductive mixer; needs ~780-850 nm pump. Literature: ~1 uW at 1 THz.
    "ltgaas_pcm_800": Photomixer(
        name="ltgaas_pcm_800",
        responsivity_a_per_w=0.01,  # L2
        response_time_s=0.3e-12,  # L2 carrier lifetime
        capacitance_f=4e-15,  # L2
        antenna_resistance_ohm=self_complementary_antenna_ohm(12.9),  # L1
        max_photocurrent_a=1e-3,  # L2
        design_wavelength_m=800e-9,
    ),
}


@dataclass(frozen=True)
class NonlinearCrystal:
    name: str
    d_eff_m_per_v: float
    n_optical: float
    n_group_optical: float
    n_thz: float
    alpha_thz_per_m: float  # THz power absorption coefficient
    pump_wavelength_m: float


CRYSTALS: Dict[str, NonlinearCrystal] = {
    # nominal literature values near 1 THz (L2); verify for the actual cut and doping
    "znte_800nm": NonlinearCrystal("znte_800nm", 68.5e-12, 2.85, 3.24, 3.17, 1.3e2, 800e-9),
    "gap_1550nm": NonlinearCrystal("gap_1550nm", 21e-12, 3.05, 3.13, 3.34, 0.2e2, 1550e-9),
    "linbo3_800nm": NonlinearCrystal("linbo3_800nm", 168e-12, 2.16, 2.25, 4.96, 16e2, 800e-9),
}


@dataclass(frozen=True)
class OpticalRectifier:
    """Plane-wave, undepleted optical rectification / intra-comb DFG.

    L1  P_THz(q) = 2 d^2 W_q^2 |P_q|^2 L_eff^2 T_out / (n_opt^2 n_THz eps0 c^3 A)
        L_eff^2  = |exp((i dk - alpha/2) L) - 1|^2 / (dk^2 + alpha^2/4),
        dk       = (W/c)(n_THz - n_g).
    Ignores THz diffraction (lambda_THz ~ 300 um exceeds typical pump spots),
    so treat results as upper bounds.
    """

    crystal: NonlinearCrystal
    length_m: float = 1e-3
    beam_area_m2: float = math.pi * (0.5e-3) ** 2

    def delta_k(self, frequency_hz) -> np.ndarray:
        c = self.crystal
        return TAU * np.asarray(frequency_hz, dtype=float) / C0 * (c.n_thz - c.n_group_optical)

    def coherence_length_m(self, frequency_hz) -> np.ndarray:
        dk = np.abs(self.delta_k(frequency_hz))
        with np.errstate(divide="ignore"):
            return np.where(dk > 0, math.pi / dk, np.inf)

    def effective_length_sq(self, frequency_hz) -> np.ndarray:
        dk = self.delta_k(frequency_hz)
        g = 1j * dk - 0.5 * self.crystal.alpha_thz_per_m
        num = np.abs(np.exp(g * self.length_m) - 1.0) ** 2
        den = np.abs(g) ** 2
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), self.length_m**2)

    @property
    def tilted_pulse_front_angle_rad(self) -> float:
        """L0 cos(gamma) = n_g / n_THz; zero when collinear matching is possible."""
        c = self.crystal
        return math.acos(c.n_group_optical / c.n_thz) if c.n_thz > c.n_group_optical else 0.0

    def response_to_comb(self, comb: CombState) -> Tuple[np.ndarray, np.ndarray]:
        """Return (harmonic frequencies, THz power per harmonic in W)."""
        c = self.crystal
        acf = line_autocorrelation(comb.amplitudes)[1:]
        freqs = np.arange(1, comb.n_lines) * comb.spacing_hz
        omega = TAU * freqs
        t_out = 4.0 * c.n_thz / (c.n_thz + 1.0) ** 2
        p = (2.0 * c.d_eff_m_per_v**2 * omega**2 * np.abs(acf) ** 2 * self.effective_length_sq(freqs) * t_out
             / (c.n_optical**2 * c.n_thz * EPS0 * C0**3 * self.beam_area_m2))
        return freqs, p


# =============================================================================
# 7. Tissue attenuation
# =============================================================================

# L0 liquid water power absorption coefficient (1/m). PMC9600075 Table 2.
WATER_ABSORPTION_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.5e12, 157.8e2),
    (1.0e12, 220.3e2),
    (1.5e12, 270.6e2),
    (2.0e12, 316.9e2),
)


def water_absorption_per_m(frequency_hz) -> np.ndarray:
    """Log-log interpolation of the table (L0 inside 0.5-2 THz, L1 power-law extrapolation outside)."""
    f = np.asarray(frequency_hz, dtype=float)
    if np.any(f <= 0):
        raise ValueError("frequency must be positive")
    lf = np.log([p[0] for p in WATER_ABSORPTION_TABLE])
    la = np.log([p[1] for p in WATER_ABSORPTION_TABLE])
    x = np.log(f)
    slope_lo = (la[1] - la[0]) / (lf[1] - lf[0])
    slope_hi = (la[-1] - la[-2]) / (lf[-1] - lf[-2])
    y = np.interp(x, lf, la)
    y = np.where(x < lf[0], la[0] + slope_lo * (x - lf[0]), y)
    y = np.where(x > lf[-1], la[-1] + slope_hi * (x - lf[-1]), y)
    return np.exp(y)


@dataclass(frozen=True)
class TissueLayer:
    name: str
    thickness_m: float
    water_fraction: float
    dry_absorption_per_m: float = 0.0  # 0 makes the attenuation a lower bound

    def absorption_per_m(self, frequency_hz: float) -> float:
        return float(self.water_fraction * water_absorption_per_m(frequency_hz)
                     + (1.0 - self.water_fraction) * self.dry_absorption_per_m)


# L2 nominal adult head; thicknesses and water fractions vary between people and sites
DEFAULT_HEAD_STACK: Tuple[TissueLayer, ...] = (
    TissueLayer("scalp", 5e-3, 0.70),
    TissueLayer("skull", 7e-3, 0.20),
    TissueLayer("csf", 2e-3, 0.99),
    TissueLayer("cortex", 3e-3, 0.80),
)

SKIN_THZ_REFRACTIVE_INDEX = 2.0  # L2


@dataclass(frozen=True)
class LayerTransmission:
    name: str
    depth_top_m: float
    depth_bottom_m: float
    penetration_depth_m: float  # 1/alpha within this layer
    attenuation_at_bottom_db: float  # cumulative, includes surface reflection
    fraction_at_bottom: float  # may underflow to 0.0; use the dB value


def tissue_transmission(
    frequency_hz: float,
    stack: Sequence[TissueLayer] = DEFAULT_HEAD_STACK,
    surface_index: float = SKIN_THZ_REFRACTIVE_INDEX,
) -> Tuple[LayerTransmission, ...]:
    """L1 Beer-Lambert through the stack, normal incidence, one Fresnel loss at the surface."""
    reflect = ((surface_index - 1.0) / (surface_index + 1.0)) ** 2
    nepers = -math.log(1.0 - reflect)
    depth = 0.0
    out: List[LayerTransmission] = []
    for layer in stack:
        alpha = layer.absorption_per_m(frequency_hz)
        nepers += alpha * layer.thickness_m
        out.append(LayerTransmission(
            name=layer.name,
            depth_top_m=depth,
            depth_bottom_m=depth + layer.thickness_m,
            penetration_depth_m=1.0 / alpha if alpha > 0 else math.inf,
            attenuation_at_bottom_db=10.0 * math.log10(math.e) * nepers,
            fraction_at_bottom=math.exp(-nepers),
        ))
        depth += layer.thickness_m
    return tuple(out)


def attenuation_profile_db(
    frequency_hz: float,
    depths_m,
    stack: Sequence[TissueLayer] = DEFAULT_HEAD_STACK,
    surface_index: float = SKIN_THZ_REFRACTIVE_INDEX,
) -> np.ndarray:
    """Cumulative attenuation (dB) at each depth below the surface; nan below the stack."""
    z = np.asarray(depths_m, dtype=float)
    reflect = ((surface_index - 1.0) / (surface_index + 1.0)) ** 2
    to_db = 10.0 * math.log10(math.e)
    out = np.full(z.shape, np.nan)
    acc_np = -math.log(1.0 - reflect)
    top = 0.0
    for layer in stack:
        alpha = layer.absorption_per_m(frequency_hz)
        inside = (z >= top) & (z <= top + layer.thickness_m)
        out[inside] = to_db * (acc_np + alpha * (z[inside] - top))
        acc_np += alpha * layer.thickness_m
        top += layer.thickness_m
    return out


# =============================================================================
# 8. Skin exposure limits (ICNIRP 2013 / IEC 60825-1, 1.4 um - 1 mm)
# =============================================================================

SKIN_LIMITING_APERTURE_M = 3.5e-3  # L0


def skin_exposure_limit_w_per_m2(exposure_duration_s: float, irradiated_area_m2: Optional[float] = None) -> float:
    """L0 skin exposure limit as an irradiance for 1.4 um - 1 mm (0.3 - 214 THz).

    100 ns - 10 s : H = 5600 t^0.25 J/m^2      (returned as H / t)
    >= 10 s       : E = 1000 W/m^2, reduced to 10/A_s W/m^2 for 0.01-0.1 m^2 and
                    100 W/m^2 above 0.1 m^2.
    < 100 ns      : H = 100 J/m^2.
    Verify against the current edition before any human exposure. This is an
    engineering interlock, not a clearance.
    """
    t = float(exposure_duration_s)
    if t <= 0:
        raise ValueError("exposure duration must be positive")
    if t < 1e-7:
        return 100.0 / t
    if t < 10.0:
        return 5600.0 * t**0.25 / t
    limit = 1000.0
    if irradiated_area_m2 is not None and irradiated_area_m2 > 0.01:
        limit = 10.0 / irradiated_area_m2 if irradiated_area_m2 < 0.1 else 100.0
    return limit


@dataclass(frozen=True)
class ExposureAssessment:
    power_w: float
    exposure_duration_s: float
    averaging_diameter_m: float
    irradiance_w_per_m2: float
    limit_w_per_m2: float
    fraction_of_limit: float
    compliant: bool


def assess_skin_exposure(power_w: float, beam_diameter_m: float, exposure_duration_s: float,
                         irradiated_area_m2: Optional[float] = None) -> ExposureAssessment:
    """Irradiance averaged over max(beam, 3.5 mm limiting aperture)."""
    d = max(float(beam_diameter_m), SKIN_LIMITING_APERTURE_M)
    irradiance = power_w / (math.pi * (d / 2.0) ** 2)
    limit = skin_exposure_limit_w_per_m2(exposure_duration_s, irradiated_area_m2)
    ratio = irradiance / limit
    return ExposureAssessment(power_w, exposure_duration_s, d, irradiance, limit, ratio, ratio <= 1.0)


# =============================================================================
# 9. Phase-coherent 12-channel array + pulse planner (EmitterArray backend)
# =============================================================================

PATTERNS = ("continuous", "pulsed", "modulated")
PULSED_METHODS = ("talbot_shaper", "talbot_fiber", "etalon", "microcomb")
MIN_SUBHARMONIC_SUPPRESSION_DB = 20.0
VOA_BLOCK_DB = 60.0  # L2 variable optical attenuator maximum


@dataclass(frozen=True, eq=False)
class CarrierPlan:
    method: str
    frequency_hz: float
    comb: CombState  # IR comb delivered to each photomixer (unit-free shape; scaled per channel)
    extra_dc_fraction: float
    metrics: PulseMetrics
    laser_wavelengths_m: Tuple[float, ...]
    details: Dict[str, float]
    fixed_frequency_device: bool = False


@dataclass(frozen=True)
class EnvelopePlan:
    pattern: str
    duration_s: float
    gate_mechanism: str
    gate_rise_time_s: float
    modulation_hz: Optional[float]


@dataclass(frozen=True)
class ChannelDrive:
    channel_id: int
    enabled: bool
    optical_power_w: float
    attenuation_db: float  # VOA setting relative to the full per-channel share
    thz_phase_rad: float
    optical_delay_s: float
    fiber_stretch_m: float
    thz_power_w: float  # all harmonics
    thz_fundamental_w: float


@dataclass(frozen=True, eq=False)
class PhotonicDrivePlan:
    command: Dict[str, object]
    carrier: CarrierPlan
    envelope: EnvelopePlan
    channels: Tuple[ChannelDrive, ...]
    amplifier_output_w: float
    total_thz_power_w: float
    exposure: ExposureAssessment
    tissue: Tuple[LayerTransmission, ...]
    warnings: Tuple[str, ...]

    @property
    def safe(self) -> bool:
        return self.exposure.compliant and not any(w.startswith("BLOCK") for w in self.warnings)

    def to_mcu_command(self) -> Dict[str, object]:
        """Setpoints for laser controllers, VOAs, fiber stretchers and photomixer bias gate."""
        return {
            "carrier_method": self.carrier.method,
            "thz_frequency_hz": self.carrier.frequency_hz,
            "laser_wavelengths_nm": [w * 1e9 for w in self.carrier.laser_wavelengths_m],
            "amplifier_output_mw": self.amplifier_output_w * 1e3,
            "voa_attenuation_db": [c.attenuation_db for c in self.channels],
            "fiber_stretch_um": [c.fiber_stretch_m * 1e6 for c in self.channels],
            "bias_enable": [c.enabled for c in self.channels],
            "bias_gate_ms": self.envelope.duration_s * 1e3,
            "modulation_hz": self.envelope.modulation_hz,
            "interlock_ok": self.safe,
        }

    def summary(self) -> str:
        m = self.carrier.metrics
        cortex = self.tissue[-1]
        lines = [
            f"carrier  {self.carrier.method} @ {self.carrier.frequency_hz / 1e12:.4f} THz, "
            f"FWHM {m.fwhm_s * 1e15:.0f} fs, peak/avg {m.peak_to_average:.1f}",
            f"drive    {sum(c.enabled for c in self.channels)} ch x "
            f"{max((c.optical_power_w for c in self.channels), default=0) * 1e3:.1f} mW optical, "
            f"amplifier {self.amplifier_output_w:.3f} W",
            f"THz      {self.total_thz_power_w * 1e6:.2f} uW total; skin {self.exposure.irradiance_w_per_m2:.3g} W/m^2 "
            f"({100 * self.exposure.fraction_of_limit:.3g} % of limit)",
            f"depth    {cortex.name} bottom at {cortex.depth_bottom_m * 1e3:.0f} mm: "
            f"-{cortex.attenuation_at_bottom_db:.0f} dB",
        ]
        lines += [f"warning  {w}" for w in self.warnings]
        return "\n".join(lines)


class PhotonicTHzEmitterArray(EmitterArray):
    """EmitterArray backed by one shared heterodyne pair split to N photomixers.

    Sharing the pair makes every channel phase coherent: the THz phase of a
    channel equals 2 pi f tau, where tau is its optical delay (fiber stretcher),
    so the array can be phased. All channels share one THz frequency; separate
    frequency groups need one laser pair per group.

    ``max_power_mw`` of each EmitterConfig is derived from the photomixer
    response at the current frequency, so the existing SafetySystem and
    THzCoherenceWearable work unchanged.
    """

    FREQUENCY_RANGE_HZ = (0.8e12, 1.2e12)  # L0 specified, docs/THZ_WEARABLE_HARDWARE_SPEC.md

    def __init__(
        self,
        photomixer: Optional[Photomixer] = None,
        frequency_hz: float = 1.0e12,
        center_wavelength_m: float = 1550e-9,
        laser_linewidth_hz: float = 100e3,
        n_channels: int = 12,
        beam_diameter_at_skin_m: float = 3.5e-3,  # L2 collimated spot at the scalp
        fiber_group_index: float = 1.4682,  # L2 SMF-28
        fiber_thermal_coefficient_per_k: float = 1.0e-5,  # L2 (1/L) d(n_g L)/dT, silica
        splitter_excess_loss_db: float = 1.0,  # L2
        pulsed_method: str = "talbot_shaper",
        seed_spacing_hz: float = 25e9,  # L2 electro-optic comb seed
        pulse_duty_cycle: float = 0.2,  # target FWHM x rate for 'pulsed' (200 fs at 1 THz)
        shaper_bandwidth_hz: float = 5e12,  # L2 C-band LCOS pulse shaper
        etalon_finesse: float = 400.0,  # L2
        microcomb_detuning_hz: float = 5e9,  # L2
        envelope_modulation_hz: Optional[float] = None,
        frequency_range_hz: Optional[Tuple[float, float]] = None,
        tissue_stack: Sequence[TissueLayer] = DEFAULT_HEAD_STACK,
    ):
        super().__init__()
        if pulsed_method not in PULSED_METHODS:
            raise ValueError(f"pulsed_method must be one of {PULSED_METHODS}")
        if not 0.0 < pulse_duty_cycle < 0.5:
            raise ValueError("pulse_duty_cycle must lie in (0, 0.5)")
        self.N_CHANNELS = int(n_channels)
        self.photomixer = photomixer or PHOTOMIXERS["utc_pd_inp_1550"]
        self.frequency_range_hz = tuple(frequency_range_hz or self.FREQUENCY_RANGE_HZ)
        self.beam_diameter_at_skin_m = beam_diameter_at_skin_m
        self.fiber_group_index = fiber_group_index
        self.fiber_thermal_coefficient_per_k = fiber_thermal_coefficient_per_k
        self.splitter_excess_loss_db = splitter_excess_loss_db
        self.pulsed_method = pulsed_method
        self.seed_spacing_hz = seed_spacing_hz
        self.pulse_duty_cycle = pulse_duty_cycle
        self.shaper_bandwidth_hz = shaper_bandwidth_hz
        self.etalon_finesse = etalon_finesse
        self.microcomb_detuning_hz = microcomb_detuning_hz
        self.envelope_modulation_hz = envelope_modulation_hz
        self.tissue_stack = tuple(tissue_stack)
        self.pair = HeterodynePair.symmetric(frequency_hz, center_wavelength_m, 1e-3, laser_linewidth_hz)
        self.phases_rad = np.zeros(self.N_CHANNELS)
        self.requested_power_mw = np.zeros(self.N_CHANNELS)
        self.last_plan: Optional[PhotonicDrivePlan] = None
        self._check_frequency(frequency_hz)
        max_mw = self.channel_max_power_w(frequency_hz) * 1e3
        self.emitters = [EmitterConfig(i, frequency_hz / 1e12, max_mw, 0.0, False) for i in range(self.N_CHANNELS)]

    # --- frequency and power --------------------------------------------------
    @property
    def frequency_hz(self) -> float:
        return self.pair.beat_frequency_hz

    def _check_frequency(self, frequency_hz: float) -> None:
        lo, hi = self.frequency_range_hz
        if not lo <= frequency_hz <= hi:
            raise ValueError(f"{frequency_hz / 1e12:.3f} THz outside {lo / 1e12:.2f}-{hi / 1e12:.2f} THz")

    def channel_max_power_w(self, frequency_hz: Optional[float] = None) -> float:
        f = self.frequency_hz if frequency_hz is None else frequency_hz
        return float(self.photomixer.max_power_w(f, 1.0))

    def set_frequency(self, frequency_hz: float) -> None:
        """Retune laser_b; laser_a is the wavelength reference."""
        self._check_frequency(frequency_hz)
        self.pair = self.pair.retuned(frequency_hz, tune="b")
        max_mw = self.channel_max_power_w(frequency_hz) * 1e3
        for e in self.emitters:
            e.frequency_thz = frequency_hz / 1e12
            e.max_power_mw = max_mw
            e.current_power_mw = min(e.current_power_mw, max_mw)

    def set_power(self, channel: int, power_mw: float) -> None:
        if 0 <= channel < self.N_CHANNELS:
            self.requested_power_mw[channel] = max(0.0, float(power_mw))
            e = self.emitters[channel]
            e.current_power_mw = float(np.clip(power_mw, 0.0, e.max_power_mw))

    def power_shortfall_mw(self) -> np.ndarray:
        applied = np.array([e.current_power_mw for e in self.emitters])
        return np.maximum(self.requested_power_mw - applied, 0.0)

    # --- phase ------------------------------------------------------------------
    def set_phase(self, channel: int, phase_rad: float) -> None:
        if 0 <= channel < self.N_CHANNELS:
            self.phases_rad[channel] = float(phase_rad) % TAU

    def set_phase_pattern(self, phases_rad: Iterable[float]) -> None:
        for i, p in enumerate(list(phases_rad)[: self.N_CHANNELS]):
            self.set_phase(i, p)

    def optical_delay_s(self, channel: int) -> float:
        """Envelope delay giving the requested THz phase (shortest non-negative delay)."""
        return self.phases_rad[channel] / (TAU * self.frequency_hz)

    def fiber_stretch_m(self, channel: int) -> float:
        return self.optical_delay_s(channel) * C0 / self.fiber_group_index

    def thermal_phase_drift_rad_per_k(self, fiber_length_m: float) -> float:
        """L1 THz phase drift of one channel per kelvin of fiber temperature change."""
        return TAU * self.frequency_hz * fiber_length_m * self.fiber_thermal_coefficient_per_k / C0

    # --- planning -------------------------------------------------------------------
    def _carrier(self, frequency_hz: float, pattern: str) -> CarrierPlan:
        if pattern in ("continuous", "modulated"):
            pair = self.pair.retuned(frequency_hz, tune="b")
            comb, dc = pair.to_comb()
            return CarrierPlan(
                method="heterodyne",
                frequency_hz=frequency_hz,
                comb=comb,
                extra_dc_fraction=dc / pair.mean_power_w,
                metrics=pulse_metrics(comb),
                laser_wavelengths_m=(pair.laser_a.wavelength_m, pair.laser_b.wavelength_m),
                details={
                    "wavelength_separation_nm": pair.wavelength_separation_m * 1e9,
                    "beat_linewidth_hz": pair.beat_linewidth_hz,
                    "modulation_depth": pair.modulation_depth,
                },
            )

        center = self.pair.laser_a.wavelength_m
        if self.pulsed_method == "microcomb":
            ring = MicroringResonator.for_repetition_rate(frequency_hz, pump_wavelength_m=center)
            comb = ring.soliton_comb(self.microcomb_detuning_hz)
            return CarrierPlan(
                method="kerr_microcomb",
                frequency_hz=ring.fsr_hz,
                comb=comb,
                extra_dc_fraction=0.0,
                metrics=pulse_metrics(comb),
                laser_wavelengths_m=(center,),
                details={
                    "ring_radius_um": ring.radius_m * 1e6,
                    "ring_circumference_um": ring.circumference_m * 1e6,
                    "soliton_fwhm_fs": ring.soliton_fwhm_s(self.microcomb_detuning_hz) * 1e15,
                    "fsr_thermal_mhz_per_k": ring.fsr_thermal_coefficient_hz_per_k / 1e6,
                },
                fixed_frequency_device=True,
            )

        seed_comb, factor = design_seed_comb(frequency_hz, self.seed_spacing_hz, self.pulse_duty_cycle, center)
        seed, n_lines = seed_comb.spacing_hz, seed_comb.n_lines
        spectral_fwhm = GAUSS_TBP * frequency_hz / self.pulse_duty_cycle
        if self.pulsed_method.startswith("talbot"):
            impl = self.pulsed_method.split("_", 1)[1]
            mult = TalbotMultiplier(factor, implementation=impl)
            out = mult.apply(seed_comb)
            details = {
                "seed_spacing_ghz": seed / 1e9,
                "multiplication_factor": float(factor),
                "seed_lines": float(n_lines),
                "spectral_fwhm_thz": spectral_fwhm / 1e12,
                "power_efficiency": 1.0,
            }
            if impl == "fiber":
                details["gdd_ps2"] = mult.required_gdd_s2(seed) * 1e24
                details["smf28_length_m"] = mult.fiber_length_m(seed)
                details["tod_phase_at_half_fwhm_rad"] = mult.third_order_phase_rad(seed, spectral_fwhm / 2.0)
            else:
                details["mask_period_lines"] = float(2 * factor)
            method = f"talbot_{impl}_comb"
        else:
            etalon = FabryPerotEtalon.for_multiplication(seed, factor, self.etalon_finesse)
            out = etalon.apply(seed_comb)
            details = {
                "seed_spacing_ghz": seed / 1e9,
                "multiplication_factor": float(factor),
                "etalon_length_um": etalon.length_m * 1e6,
                "adjacent_suppression_db": etalon.adjacent_suppression_db(seed),
                "seed_lines": float(n_lines),
                "power_efficiency": out.total_power_w / seed_comb.total_power_w,
            }
            method = "etalon_filtered_comb"
        samples = default_samples_per_period(n_lines, multiple_of=factor)
        return CarrierPlan(
            method=method,
            frequency_hz=frequency_hz,
            comb=out,
            extra_dc_fraction=0.0,
            metrics=pulse_metrics(out, samples),
            laser_wavelengths_m=(center,),
            details=details,
        )

    def plan(self, command: Mapping[str, object]) -> PhotonicDrivePlan:
        """Translate ``THzPulseParameters.to_hardware_command()`` into photonic setpoints.

        amplitude_normalized a in [0, 1] maps to a fraction a of the channel's
        maximum THz power (photocurrent sqrt(a) I_max), so the request never
        exceeds the photomixer limit.
        """
        f = float(command["frequency_thz"]) * 1e12
        self._check_frequency(f)
        amplitude = float(np.clip(float(command.get("amplitude_normalized", 0.0)), 0.0, 1.0))
        duration_s = float(command.get("duration_ms", 100.0)) / 1e3
        pattern = str(command.get("pattern", "continuous"))
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")
        targets = sorted(set(int(i) for i in command.get("emitters", range(self.N_CHANNELS))))
        if any(not 0 <= i < self.N_CHANNELS for i in targets):
            raise ValueError("emitter index out of range")
        phase = float(command.get("phase_offset_rad", 0.0)) % TAU
        pattern_rad = command.get("phase_pattern_rad")
        if pattern_rad is None:
            phases = np.full(self.N_CHANNELS, phase)
        else:
            phases = np.mod(np.asarray(pattern_rad, dtype=float).reshape(-1) + phase, TAU)
            if phases.size != self.N_CHANNELS:
                raise ValueError(f"phase_pattern_rad needs {self.N_CHANNELS} entries")

        carrier = self._carrier(f, pattern)
        pm = self.photomixer
        i_dc = math.sqrt(amplitude) * pm.max_photocurrent_a
        p_opt = i_dc / pm.responsivity_a_per_w
        warnings: List[str] = []
        if amplitude > 0:
            comb = carrier.comb.scaled_to_power(p_opt * (1.0 - carrier.extra_dc_fraction))
            response = pm.response_to_comb(comb, extra_dc_power_w=p_opt * carrier.extra_dc_fraction)
            p_thz, p_fund = response.total_power_w, response.fundamental_power_w
            if response.over_current:
                warnings.append("BLOCK photocurrent above device limit")
        else:
            p_thz = p_fund = 0.0

        # Hardware model: EDFA (after any rate multiplier) at constant output feeds a
        # 1xN splitter; per-channel VOAs set the level, VOA_BLOCK_DB blocks a port.
        full_share = pm.max_photocurrent_a / pm.responsivity_a_per_w
        channels = []
        for i in range(self.N_CHANNELS):
            on = i in targets and amplitude > 0
            delay = phases[i] / (TAU * carrier.frequency_hz)
            channels.append(ChannelDrive(
                channel_id=i,
                enabled=on,
                optical_power_w=p_opt if on else 0.0,
                attenuation_db=(10.0 * math.log10(full_share / p_opt)) if on else VOA_BLOCK_DB,
                thz_phase_rad=float(phases[i]) if on else 0.0,
                optical_delay_s=delay if on else 0.0,
                fiber_stretch_m=delay * C0 / self.fiber_group_index if on else 0.0,
                thz_power_w=p_thz if on else 0.0,
                thz_fundamental_w=p_fund if on else 0.0,
            ))
        n_on = sum(c.enabled for c in channels)
        split_db = 10.0 * math.log10(self.N_CHANNELS) + self.splitter_excess_loss_db
        amp_out = full_share * 10.0 ** (split_db / 10.0) if n_on else 0.0
        if amp_out > 5.0:
            warnings.append(f"amplifier output {amp_out:.1f} W exceeds a 5 W EDFA class")
        if carrier.fixed_frequency_device:
            warnings.append("microcomb repetition rate is set by ring geometry; retuning needs another ring")
        efficiency = carrier.details.get("power_efficiency", 1.0)
        if carrier.method == "etalon_filtered_comb" and efficiency < 0.1:
            warnings.append(f"etalon passes {100 * efficiency:.1f} % of seed power; amplify after the filter")
        if carrier.method != "heterodyne" and amplitude > 0:
            peak_ma = i_dc * carrier.metrics.peak_to_average * 1e3
            warnings.append(f"pulsed drive peak photocurrent ~{peak_ma:.0f} mA; space-charge saturation not modeled (L2)")
        if carrier.metrics.subharmonic_suppression_db < MIN_SUBHARMONIC_SUPPRESSION_DB:
            warnings.append(f"carrier is not a clean {carrier.frequency_hz / 1e12:.3f} THz train "
                            f"(subharmonic suppression {carrier.metrics.subharmonic_suppression_db:.1f} dB)")
        if carrier.method == "talbot_shaper_comb" and 2.0 * carrier.details["spectral_fwhm_thz"] * 1e12 > self.shaper_bandwidth_hz:
            warnings.append("seed spectrum (2 x FWHM) exceeds the pulse-shaper bandwidth; raise pulse_duty_cycle")

        # Repeated bursts over a session are treated as continuous exposure (>= 10 s), conservative.
        exposure = assess_skin_exposure(p_thz, self.beam_diameter_at_skin_m, max(duration_s, 10.0))
        if not exposure.compliant:
            warnings.append("BLOCK skin exposure limit exceeded")

        envelope = EnvelopePlan(
            pattern=pattern,
            duration_s=duration_s,
            gate_mechanism="photomixer_bias",
            gate_rise_time_s=1e-9,  # L2 bias driver
            modulation_hz=self.envelope_modulation_hz if pattern == "modulated" else None,
        )
        plan = PhotonicDrivePlan(
            command=dict(command),
            carrier=carrier,
            envelope=envelope,
            channels=tuple(channels),
            amplifier_output_w=amp_out,
            total_thz_power_w=n_on * p_thz,
            exposure=exposure,
            tissue=tissue_transmission(carrier.frequency_hz, self.tissue_stack),
            warnings=tuple(warnings),
        )
        return plan

    def apply_plan(self, plan: PhotonicDrivePlan) -> bool:
        """Load a plan into the EmitterConfig state. Refuses plans that fail the interlock."""
        if not plan.safe:
            self.shutdown_all()
            return False
        self.set_frequency(plan.carrier.frequency_hz)
        if plan.carrier.method != "heterodyne":
            # A pulse train's fundamental can carry up to 4x the sinusoidal-beat power at
            # equal mean photocurrent; raise the per-channel ceiling to this carrier's maximum.
            full = self.photomixer.max_photocurrent_a / self.photomixer.responsivity_a_per_w
            ceiling_mw = self.photomixer.response_to_comb(plan.carrier.comb.scaled_to_power(full)).total_power_w * 1e3
            for e in self.emitters:
                e.max_power_mw = max(e.max_power_mw, ceiling_mw)
        for ch in plan.channels:
            self.set_phase(ch.channel_id, ch.thz_phase_rad)
            self.set_power(ch.channel_id, ch.thz_power_w * 1e3)
            self.enable_channel(ch.channel_id, ch.enabled)
        self.last_plan = plan
        return True


# =============================================================================
# Demo, figure and CLI
# =============================================================================


def demo() -> str:
    rows: List[str] = []
    add = rows.append
    add("PHOTONIC THz PULSE ENGINE")
    add(f"  1 THz beat at 1550 nm needs {wavelength_spacing_for_beat(1e12) * 1e9:.3f} nm separation  [L0]")
    ring = MicroringResonator.for_repetition_rate(1e12)
    add(f"  1 THz Si3N4 ring (n_g=2): L = {ring.circumference_m * 1e6:.1f} um, R = {ring.radius_m * 1e6:.2f} um, "
        f"soliton FWHM {ring.soliton_fwhm_s(5e9) * 1e15:.0f} fs @ 5 GHz detuning, "
        f"dFSR/dT {ring.fsr_thermal_coefficient_hz_per_k / 1e6:.1f} MHz/K  [L1]")
    add("  Pulse sharpening at df = 1 THz (flat, locked):")
    for n in (2, 4, 8, 16, 32):
        m = pulse_metrics(make_comb(1e12, n))
        add(f"    N={n:<3d} FWHM {m.fwhm_s * 1e15:7.1f} fs   peak/avg {m.peak_to_average:5.1f}")
    seed, r = design_seed_comb(1e12, 25e9, 0.1)
    m_s = default_samples_per_period(seed.n_lines, r)
    fiber, shaper = TalbotMultiplier(r, implementation="fiber"), TalbotMultiplier(r, implementation="shaper")
    add(f"  Talbot x{r}, 25 GHz -> 1 THz, 100 fs target: GDD {fiber.required_gdd_s2(seed.spacing_hz) * 1e24:.2f} ps^2 "
        f"= {fiber.fiber_length_m(seed.spacing_hz):.0f} m SMF-28  [L0/L2]")
    for label, mult in (("shaper", shaper), ("fiber ", fiber)):
        pm_ = pulse_metrics(mult.apply(seed), m_s)
        add(f"    {label}: FWHM {pm_.fwhm_s * 1e15:6.1f} fs  peak/avg {pm_.peak_to_average:5.2f}  "
            f"subharmonic suppression {pm_.subharmonic_suppression_db:5.1f} dB")
    add(f"    fiber cubic phase at half FWHM: {fiber.third_order_phase_rad(seed.spacing_hz, 2.2e12):.0f} rad  [L1]")
    et = FabryPerotEtalon.for_multiplication(25e9, 40)
    add(f"  Etalon 25 GHz -> 1 THz: L = {et.length_m * 1e6:.0f} um, adjacent suppression "
        f"{et.adjacent_suppression_db(25e9):.1f} dB, throughput ~1/40  [L0/L2]")
    gate = KerrGate(KERR_MEDIA["fused_silica_1mm"])
    add(f"  Kerr gate, 1 mm silica: I_pi = {gate.pi_intensity_w_per_m2 / 1e16:.2f} TW/cm^2  [L1]")
    pm = PHOTOMIXERS["utc_pd_inp_1550"]
    add("  UTC-PD photomixer max output (10 mA):  " + "   ".join(
        f"{f:.1f} THz {pm.max_power_w(f * 1e12) * 1e6:5.1f} uW" for f in (0.8, 1.0, 1.2)) + "  [L1/L2]")
    comb, _ = HeterodynePair.symmetric(1e12, total_power_w=1.0).to_comb()
    _, p_or = OpticalRectifier(CRYSTALS["gap_1550nm"]).response_to_comb(comb)
    add(f"  Optical rectification, 1 W CW heterodyne in 1 mm GaP: {p_or.sum():.2e} W  [L1 upper bound]")
    for layer in tissue_transmission(1e12):
        add(f"  tissue {layer.name:<7s} bottom {layer.depth_bottom_m * 1e3:4.0f} mm  "
            f"-{layer.attenuation_at_bottom_db:8.1f} dB  (1/alpha {layer.penetration_depth_m * 1e6:6.0f} um)")
    array = PhotonicTHzEmitterArray()
    for pattern in PATTERNS:
        plan = array.plan({"frequency_thz": 1.0, "amplitude_normalized": 0.5, "duration_ms": 150,
                           "pattern": pattern, "emitters": [0, 1, 2, 3], "phase_offset_rad": 0.0})
        add(f"  plan '{pattern}':")
        rows.extend("    " + line for line in plan.summary().splitlines())
    return "\n".join(rows)


def render_figure(path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference chart chrome and categorical slots 1-4 in fixed order (validated, light mode).
    ink, muted, grid, axis = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
    series = ("#257de0", "#f47230", "#00b894", "#f2a600")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": axis, "axes.labelcolor": muted,
                         "axes.titlecolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": grid, "grid.linewidth": 0.6, "legend.labelcolor": muted})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    a = ax[0, 0]
    for color, n in zip(series, (2, 4, 8, 32)):
        t, p = intensity_waveform(make_comb(1e12, n), n_periods=3, samples_per_period=2048)
        a.plot(t * 1e12, p / p.max(), color=color, lw=1.4, label=f"N = {n}")
    a.set(xlabel="time (ps)", ylabel="P(t) / P_peak", title="Locking N IR lines at 1 THz spacing")
    a.legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    a.set_ylim(0, 1.18)

    a = ax[0, 1]
    seed, factor = design_seed_comb(1e12, 25e9, 0.1)
    m = default_samples_per_period(seed.n_lines, factor)
    t, p_in = intensity_waveform(seed, 1, m)
    sel = t < 5e-12
    for color, impl, label in ((series[0], "shaper", "line-by-line phase mask"),
                               (series[1], "fiber", "SMF-28 incl. third-order dispersion")):
        mult = TalbotMultiplier(factor, implementation=impl)
        _, p_out = intensity_waveform(mult.apply(seed), 1, m)
        if impl == "shaper":
            p_out = np.roll(p_out, -(int(np.argmax(p_out)) - int(0.5e-12 / (seed.period_s / m))))
            shaper_par = p_out.max() / p_out.mean()
        a.plot(t[sel] * 1e12, p_out[sel] / p_in.mean(), color=color, lw=1.4, label=label)
    fiber_len = TalbotMultiplier(factor, implementation="fiber").fiber_length_m(seed.spacing_hz)
    a.set(xlabel="time (ps)", ylabel="P(t) / <P>", title=f"Talbot x{factor}: 25 GHz seed to 1 THz train")
    a.set_ylim(0, 15)
    a.text(0.99, 0.99, f"seed peak/avg {p_in.max() / p_in.mean():.0f}\nmask output peak/avg {shaper_par:.1f}\n"
           f"fiber option: {fiber_len:.0f} m", transform=a.transAxes, ha="right", va="top", color=muted, fontsize=8)
    a.legend(frameon=False, loc="upper left", fontsize=8)

    a = ax[1, 0]
    f = np.linspace(0.2e12, 2.0e12, 400)
    for color, key in zip(series, PHOTOMIXERS):
        p_uw = PHOTOMIXERS[key].max_power_w(f) * 1e6
        a.semilogy(f / 1e12, p_uw, color=color, lw=1.4, label=key)
        j = int(np.searchsorted(f, 1.45e12))
        a.text(f[j] / 1e12, p_uw[j] * 1.4, key, ha="left", va="bottom", color=muted, fontsize=8)
    a.axvspan(0.8, 1.2, color=grid, alpha=0.8, lw=0)
    a.text(1.0, a.get_ylim()[1] * 0.6, "spec band", ha="center", va="top", color=muted, fontsize=8)
    a.set(xlabel="frequency (THz)", ylabel="max THz power (uW)", title="Photomixer output ceiling (L1 model, L2 params)")
    a.legend(frameon=False, loc="lower left")

    a = ax[1, 1]
    depth = np.linspace(0, sum(layer.thickness_m for layer in DEFAULT_HEAD_STACK), 600)
    for color, fr in zip(series, (0.5e12, 1.0e12, 1.5e12)):
        prof = attenuation_profile_db(fr, depth)
        a.plot(depth * 1e3, prof, color=color, lw=1.4, label=f"{fr / 1e12:.1f} THz")
        a.text(depth[-1] * 1e3, prof[-1], f" {fr / 1e12:.1f} THz", ha="left", va="center", color=muted, fontsize=8)
    for layer in tissue_transmission(1e12):
        a.axvline(layer.depth_bottom_m * 1e3, color=axis, lw=0.8)
        a.text(0.5 * (layer.depth_top_m + layer.depth_bottom_m) * 1e3, 0, layer.name, ha="center", va="bottom",
               color=muted, fontsize=8)
    a.set(xlabel="depth (mm)", ylabel="attenuation (dB, down)", title="THz attenuation through nominal head stack")
    a.set_xlim(0, depth[-1] * 1e3 * 1.12)
    a.invert_yaxis()
    a.legend(frameon=False, loc="lower left")

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Photonic THz pulse engine demo")
    parser.add_argument("--figure", help="write the four-panel summary figure to this path")
    args = parser.parse_args(argv)
    print(demo())
    if args.figure:
        print(f"figure: {render_figure(args.figure)}")


if __name__ == "__main__":
    main()
