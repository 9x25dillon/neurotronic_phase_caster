#!/usr/bin/env python3
"""Two-RTD autonomous HB and loaded three-port Nyquist analysis (SI internally).

Use --help for the two entry points. See CHANGELOG.md for model boundaries,
evidence levels and assumptions. No AEDT dependency. Peak positive-frequency
phasors, exp(+jwt), currents positive into each component, real 50-ohm power waves.
"""

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import skrf
from scipy import linalg
from scipy.constants import c, epsilon_0
from scipy.interpolate import interp1d
from scipy.optimize import brentq, least_squares
from skrf.network import z2s
from skrf.vectorFitting import VectorFitting

GHZ = 1e9
TAU = 2 * np.pi
SCALE = TAU * 300 * GHZ  # L0 numerical scaling of s, rad/s
Z0 = 50.0  # L0 declared reference impedance, ohms
BANDS = (("low", 1, 100), ("mid", 100, 450), ("high", 450, 950))
ODD = np.array([1.0, -1.0]) / np.sqrt(2)
EVEN = np.array([1.0, 1.0]) / np.sqrt(2)
REF = 1 / Z0
FENCE_BAND = (296.1e9, 307.7e9)  # L1 tuning range supplied in section 5
GAP_C = epsilon_0 * (1.1e-6) ** 2 / (7.3e-9)  # L1 empty port gap, see section 1
BIAS_R, BIAS_L = 1.0, 1e-9  # L1 series line at the physical bias pad


@dataclass(frozen=True)
class Device:
    a: float = 0.0182  # L2 midpoint negative slope, S
    b: float = 4 * 0.0182 / (3 * 0.4**2)  # L2 cubic from section 1, A/V^3
    cap: float = 6.5e-15  # L2 per junction, F
    rs: float = 7.0  # L2 swept terminal resistance, ohms
    lb: float = 3.5e-12  # L1 series bridge in circuit-only models
    delay: float = 0.0  # L1 memoryless midpoint benchmark, s

    bias_offset: float = 0.0  # L1 prescribed junction DC offset from the NDR midpoint, V

    @property
    def gain(self):
        return self.a - 3 * self.b * self.bias_offset**2

    def caps(self, mismatch):
        if abs(mismatch) >= 2:
            raise ValueError("Both junction capacitances must be positive.")
        return self.cap * np.array([1 - mismatch / 2, 1 + mismatch / 2])

    def transport(self, s):
        return (1 - s * self.delay / 2) / (1 + s * self.delay / 2)

    def terminal_y(self, s, mismatch=0.0):
        s = np.asarray(s)[..., None]
        intrinsic = -self.gain * self.transport(s) + s * self.caps(mismatch)
        return intrinsic / (1 + (self.rs + s * self.lb) * intrinsic)

    @property
    def stab_limit(self):
        return (1 / self.gain - self.rs) / 2


def hermitian(m):
    return (m + np.swapaxes(m.conj(), -1, -2)) / 2


def passivity(y, frequencies, tolerance=1e-12):
    """Power passivity uses the Hermitian part; elementwise real is insufficient."""
    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite admittance.")
    ev = np.linalg.eigvalsh(hermitian(y))[:, 0]
    i = np.argmin(ev)
    if ev[i] < -tolerance:
        raise ValueError(
            f"Non-passive at {frequencies[i] / GHZ:.9g} GHz: "
            f"minimum loss eigenvalue {ev[i]:.9g} S [L1]."
        )
    return float(ev[i])


class Network:
    def __init__(self, f, y):
        self.f, self.y = np.asarray(f, float), np.asarray(y, complex)
        if (
            self.f.ndim != 1
            or len(self.f) < 2
            or np.any(~np.isfinite(self.f))
            or np.any(np.diff(self.f) <= 0)
            or self.y.ndim != 3
            or self.y.shape[0] != len(self.f)
            or self.y.shape[1] != self.y.shape[2]
        ):
            raise ValueError("Expected strictly increasing Hz and square Y(f) matrices.")
        passivity(self.y, self.f)
        self._interp = interp1d(self.f, self.y, axis=0, bounds_error=True)

    def at(self, f):
        f = np.asarray(f, float)
        if np.any(~np.isfinite(f)) or np.any(f < self.f[0]) or np.any(f > self.f[-1]):
            raise ValueError(
                f"Frequency or harmonic outside data [{self.f[0] / GHZ:g}, "
                f"{self.f[-1] / GHZ:g}] GHz [L0]."
            )
        return self._interp(f)


def stitch(folder, ports, step_hz=GHZ):
    """Compare coincident band endpoints before averaging. Keep native knots."""
    if step_hz <= 0:
        raise ValueError("Grid step must be positive.")
    ff, yy = [], []
    for name, lo, hi in BANDS:
        path = Path(folder) / f"{name}.s{ports}p"
        part = skrf.Network(str(path))
        if part.nports != ports or not np.allclose(part.z0, Z0, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{path}: expected {ports} ports with real {Z0:g}-ohm references.")
        f, y = np.array(part.f), np.array(part.y)
        if not np.allclose(f[[0, -1]], np.array([lo, hi]) * GHZ, rtol=0, atol=2.0):
            raise ValueError(f"{path}: sweep endpoints differ from the export contract.")
        f[0], f[-1] = lo * GHZ, hi * GHZ
        Network(f, y)
        if ff:
            relative = np.linalg.norm(y[0] - yy[-1][-1]) / max(
                np.linalg.norm(yy[-1][-1]), np.linalg.norm(y[0]), 1e-30
            )
            # Also check weak entries which a matrix norm can conceal.
            scale = np.maximum(np.maximum(abs(y[0]), abs(yy[-1][-1])), 1e-12)
            worst = np.max(abs(y[0] - yy[-1][-1]) / scale)
            if max(relative, worst) > 0.01:
                warnings.warn(
                    f"[L1] Y join at {lo:g} GHz: Frobenius {relative:.2%}, "
                    f"largest entry {worst:.2%}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            yy[-1][-1] = (yy[-1][-1] + y[0]) / 2
            f, y = f[1:], y[1:]
        ff.append(f)
        yy.append(y)
    native = Network(np.concatenate(ff), np.concatenate(yy))
    grid = np.unique(np.r_[native.f, np.arange(native.f[0], native.f[-1], step_hz)])
    return Network(grid, native.at(grid))


def write_synthetic_snp(path, f, y):
    """Exact Z-to-power-S conversion for invertible synthetic fixtures."""
    n = skrf.Network(
        frequency=skrf.Frequency.from_f(f, unit="hz"), s=z2s(np.linalg.inv(y), z0=Z0), z0=Z0
    )
    n.write_touchstone(filename=str(path), form="ri", write_z0=False)


def load_bias(y3, s):
    """Schur complement at the pad. Internal film, CPW, bridge and MIM are in Y3."""
    yline = 1 / (BIAS_R + s * BIAS_L)
    return (
        y3[..., :2, :2]
        - y3[..., :2, 2, None] * y3[..., None, 2, :2] / (y3[..., 2, 2] + yline)[..., None, None]
    )


def ring_modes(f, angle, radius_um=46.6, alpha=0.0):
    if not 0 < angle < 360 or radius_um <= 0:
        raise ValueError("Nonzero arcs and positive radius required.")
    f = np.asarray(f)
    lengths = radius_um * 1e-6 * np.deg2rad([angle, 360 - angle])
    arg = (alpha + 1j * TAU * f[..., None] * np.sqrt(6.8) / c) * lengths
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        singular = np.any(abs(np.sinh(arg)) < 1e-10, axis=-1)
        ye = np.sum(np.tanh(arg / 2), axis=-1) / 65.0
        yo = np.sum(1 / np.tanh(arg / 2), axis=-1) / 65.0
    return np.where(singular, np.nan + 0j, ye), np.where(singular, np.nan + 0j, yo)


def first_root(fun, lo, hi, splits=(), points=401):
    """Split at known poles and require a small residual; a sign change can be a pole."""
    boundaries = np.unique([lo, *[x for x in splits if lo < x < hi], hi])
    for a, b in pairwise(boundaries):
        grid = np.linspace(a, b, points)
        if a != lo:
            grid[0] = np.nextafter(a, b)
        if b != hi:
            grid[-1] = np.nextafter(b, a)
        val = np.asarray(fun(grid))
        for k in range(len(grid)):
            if not np.isfinite(val[k]):
                continue
            if abs(val[k]) < 1e-13:
                return float(grid[k])
            if k == 0 or not np.isfinite(val[k - 1]) or val[k] * val[k - 1] >= 0:
                continue
            try:
                root = brentq(fun, grid[k - 1], grid[k], xtol=1e-3)
            except ValueError:
                continue
            if np.isfinite(fun(root)) and abs(fun(root)) < 1e-9:
                return float(root)
    return np.nan


def analytic_odd(angle, lo=220e9, hi=380e9, device=Device(), radius_um=46.6):
    """Section 2 lossless startup reactance, with a small-signal gain check."""
    if not 0 < lo < hi:
        raise ValueError("Invalid search band.")
    lens = np.deg2rad([angle, 360 - angle]) * radius_um * 1e-6
    poles = []
    for length in lens:
        spacing = c / (2 * np.sqrt(6.8) * length)
        poles.extend(spacing * np.arange(1, int(hi / spacing) + 1))
    if device.lb * device.cap > 0:
        poles.append(1 / (TAU * np.sqrt(device.lb * device.cap)))

    def residual(f):
        _, yo = ring_modes(f, angle, radius_um)
        w = TAU * np.asarray(f)
        den = 1 - w * w * device.lb * device.cap
        with np.errstate(divide="ignore", invalid="ignore"):
            return yo.imag + np.where(abs(den) < 1e-12, np.nan, w * device.cap / den)

    root = first_root(residual, lo, hi, poles)
    if np.isfinite(root):
        _, lossy = ring_modes(root, angle, radius_um, alpha=327.0)
        if (lossy + device.terminal_y(1j * TAU * root)[0]).real >= 0:
            return np.nan
    return root


def axial_ratio(e):
    """Major/minor electric-field ellipse axes. Infinity for a line, NaN at a null."""
    axes = np.linalg.svd(np.column_stack((e.real, -e.imag)), compute_uv=False)
    if axes[0] == 0:
        return np.nan
    return np.inf if axes[1] <= 1e-13 * axes[0] else float(axes[0] / axes[1])


class Radiation:
    """Embedded Jones columns for 1-W incident waves; Hermitian radiation power form."""

    def __init__(self, f, eta, accepted_odd, rad_matrix, jones):
        f = np.asarray(f)
        eta, accepted_odd = np.asarray(eta), np.asarray(accepted_odd)
        rad_matrix, jones = np.asarray(rad_matrix), np.asarray(jones)
        if (
            np.any(np.diff(f) <= 0)
            or len(f) < 2
            or rad_matrix.shape != (len(f), 2, 2)
            or jones.shape != rad_matrix.shape
            or not all(np.all(np.isfinite(a)) for a in (f, eta, accepted_odd, rad_matrix, jones))
            or np.any((eta < 0) | (eta > 1))
            or np.any(accepted_odd <= 0)
        ):
            raise ValueError("Invalid radiation export or source normalization.")
        if np.max(abs(rad_matrix - hermitian(rad_matrix))) > 1e-10:
            raise ValueError("Radiation form must be Hermitian.")
        if np.min(np.linalg.eigvalsh(rad_matrix)) < -1e-9:
            raise ValueError("Radiation power form is non-passive.")
        odd = np.array([1.0, -1.0])  # 1 W per incident source, not 1 W total
        ref = (odd @ rad_matrix @ odd).real
        if not np.allclose(ref, eta * accepted_odd, rtol=0.01, atol=1e-12):
            raise ValueError("Odd-drive efficiency and independent radiation drives disagree.")
        self.f = f
        self.interps = [
            interp1d(f, x, axis=0, bounds_error=True)
            for x in (eta, accepted_odd, rad_matrix, jones)
        ]

    @classmethod
    def read(cls, path):
        d = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))

        def z(n):
            return d[n + "_re"] + 1j * d[n + "_im"]

        h, j = (np.empty((len(d), 2, 2), complex) for _ in range(2))
        h[:, 0, 0], h[:, 1, 1], h[:, 0, 1] = d["rad11"], d["rad22"], z("rad12")
        h[:, 1, 0] = h[:, 0, 1].conj()
        for k, pol in enumerate(("theta", "phi")):
            for n in range(2):
                j[:, k, n] = z(f"{pol}{n + 1}")
        if not np.allclose(
            d["radiated_odd_W"], d["accepted_odd_W"] * d["eta_odd"], rtol=0.005, atol=1e-12
        ):
            raise ValueError("Radiated/accepted power inconsistent with exported efficiency.")
        return cls(d["f_Hz"], d["eta_odd"], d["accepted_odd_W"], h, j)

    def evaluate(self, f, v, i):
        eta, po, h, j = (fn(f) for fn in self.interps)
        a = (v + Z0 * i) / (2 * np.sqrt(2 * Z0))  # peak V to sqrt(W)
        odd_norm = (2 * ODD @ h @ ODD).real
        if odd_norm > 1e-15:
            p = float(eta * po * np.vdot(a, h @ a).real / odd_norm)
        else:  # a dark odd drive can coexist with bright even drive
            p = float(np.vdot(a, h @ a).real)
        accepted = np.vdot(v, i).real / 2
        if p < -1e-14 or p > accepted + max(1e-13, 0.01 * abs(accepted)):
            raise ValueError("Radiated power violates accepted-power bound.")
        return p, axial_ratio(j @ a)


@dataclass
class HBResult:
    f: float
    u: np.ndarray
    v: np.ndarray
    i: np.ndarray  # currents into RTDs
    p_accepted: np.ndarray  # per harmonic, both ports
    p_rs: float
    residual_A: float
    odd_purity: float

    @property
    def ratio(self):
        return self.v[0, 1] / self.v[0, 0]


def harmonic_balance(
    net, dev, mismatch=0.0, harmonics=3, band=(280e9, 315e9), seed=None, gap_cap=0.0
):
    """Nontrivial periodic root. U11 is real positive; residual is amplitude-normalized."""
    if harmonics < 1 or not 0 < band[0] < band[1]:
        raise ValueError("Invalid HB band or harmonic order.")
    net.at([band[0], harmonics * band[1]])  # refuse incomplete harmonic coverage
    orders = np.arange(1, harmonics + 1)
    basis = np.exp(1j * TAU * np.outer(np.arange(8 * harmonics + 1) / (8 * harmonics + 1), orders))
    if dev.gain <= 0:
        raise ValueError("Prescribed DC operating point is outside NDR.")
    caps = dev.caps(mismatch)
    count = 2 * harmonics - 1
    lo, hi = np.array(band) / GHZ
    if seed is None:
        u0 = np.zeros((harmonics, 2), complex)
        u0[0] = [1, -1]
        amp = np.sqrt(4 * dev.gain / (3 * dev.b))
        guess = np.clip(300.0, lo + (hi - lo) * 0.01, hi - (hi - lo) * 0.01)
    else:
        amp = abs(seed.u[0, 0])
        u0 = seed.u / seed.u[0, 0]
        guess = seed.f / GHZ
    rest = u0.ravel()[1:]
    init = np.r_[guess, np.log(amp), rest.real, rest.imag]

    def fields(x):
        f, amp = x[0] * GHZ, np.exp(x[1])
        u = amp * np.r_[1, x[2 : 2 + count] + 1j * x[2 + count :]].reshape(harmonics, 2)
        s = 1j * TAU * orders[:, None] * f
        time = (basis @ u).real
        nonlinear = dev.b * time**2 * (time + 3 * dev.bias_offset)
        conduction = -dev.gain * u + (2 / len(time)) * basis.conj().T @ nonlinear
        i = dev.transport(s) * conduction + s * caps * u
        v = u + (dev.rs + s * dev.lb) * i
        iraw = np.einsum("hjk,hk->hj", net.at(orders * f), v)
        kcl = iraw - s * gap_cap * v + i
        return u, v, i, iraw, kcl

    def residual(x):
        kcl = fields(x)[-1] / (np.exp(x[1]) * dev.gain)
        return np.r_[kcl.real.ravel(), kcl.imag.ravel()]

    lower = np.r_[lo, np.log(1e-6), np.full(2 * count, -np.inf)]
    upper = np.r_[hi, np.log(5.0), np.full(2 * count, np.inf)]
    sol = least_squares(
        residual,
        init,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=800,
    )
    u, v, i, iraw, kcl = fields(sol.x)
    purity = abs(ODD @ v[0]) ** 2 / np.vdot(v[0], v[0]).real
    if (
        not sol.success
        or np.max(abs(residual(sol.x))) > 1e-7
        or abs(u[0, 0]) <= 2e-6
        or purity <= 0.5
    ):
        raise RuntimeError(
            f"No validated nonzero odd-branch HB solution in [{lo:g},{hi:g}] GHz: {sol.message}, "
            f"scaled residual {max(abs(residual(sol.x))):.3g}, purity {purity:.3g} [L1]."
        )
    return HBResult(
        sol.x[0] * GHZ,
        u,
        v,
        i,
        (v.conj() * iraw).sum(axis=1).real / 2,
        float(dev.rs * np.sum(abs(i) ** 2) / 2),
        float(np.max(abs(kcl))),
        float(purity),
    )


def mismatch_row(net, radiation, dev, m, sol):
    y = net.at(sol.f)
    split = EVEN @ y @ EVEN - ODD @ y @ ODD
    if abs(split) < 1e-12:
        ep = complex(np.nan, np.nan)
    else:
        ep = 1j * TAU * sol.f * dev.cap * m / (2 * split)
    radiated, ar = radiation.evaluate(sol.f, sol.v[0], y @ sol.v[0])
    phase = np.degrees(np.angle(sol.ratio)) % 360
    row = dict(
        rs_ohm=dev.rs,
        junction_bias_offset_V=dev.bias_offset,
        deltaC_over_C=m,
        f_GHz=sol.f / GHZ,
        phase_deg=phase,
        amplitude_ratio=abs(sol.ratio),
        epsilon_re=ep.real,
        epsilon_im=ep.imag,
        AR_linear=ar,
        AR_dB=20 * np.log10(ar),
        Paccepted1_W=sol.p_accepted[0],
        Prad1_W=radiated,
        PRs_all_W=sol.p_rs,
        KCL_A=sol.residual_A,
        level="L2 powers / L1 other model outputs",
    )
    for h in range(2, len(sol.v) + 1):
        f = h * sol.f
        v = sol.v[h - 1]
        row[f"Paccepted{h}_W"] = sol.p_accepted[h - 1]
        row[f"Prad{h}_W"] = radiation.evaluate(f, v, net.at(f) @ v)[0]
    return row


@dataclass
class RationalY:
    """Y(x)=D+xE+sum(Rk/(x-pk)), x=s/SCALE. Poles include conjugates."""

    poles: np.ndarray
    residues: np.ndarray
    d: np.ndarray
    e: np.ndarray

    def at(self, s):
        x = np.asarray(s) / SCALE
        return (
            self.d
            + x[..., None, None] * self.e
            + np.einsum("...k,kij->...ij", 1 / (x[..., None] - self.poles), self.residues)
        )

    def realization(self):
        n = self.d.shape[0]
        blocks, inputs, outputs = [], [], []
        for pole, r in zip(self.poles, self.residues, strict=True):
            u, sv, vh = linalg.svd(r, full_matrices=False)
            keep = sv > max(1e-26, sv[0] * 1e-12)
            if np.any(keep):
                weight = np.sqrt(sv[keep])
                blocks.append(np.full(sum(keep), pole))
                inputs.append(weight[:, None] * vh[keep])
                outputs.append(u[:, keep] * weight)
        if not blocks:
            return np.empty((0, 0)), np.empty((0, n)), np.empty((n, 0))
        return np.diag(np.concatenate(blocks)), np.vstack(inputs), np.hstack(outputs)


def rational_fit(net, order, tol):
    """Withhold alternate native/grid samples. Check all sampled and midpoint losses."""
    where = np.unique(np.r_[np.arange(0, len(net.f), 2), len(net.f) - 1])
    train = skrf.Network(
        frequency=skrf.Frequency.from_f(net.f[where], unit="hz"), y=net.y[where], z0=Z0
    )
    vf = VectorFitting(train)
    vf.vector_fit(
        n_poles_real=3,
        n_poles_cmplx=order,
        parameter_type="y",
        fit_constant=True,
        fit_proportional=True,
        init_pole_spacing="log",
        enforce_dc=False,
    )
    poles, residues = [], []
    d = vf.constant_coeff.reshape(3, 3).astype(complex)
    e = vf.proportional_coeff.reshape(3, 3).astype(complex) * SCALE
    far_limit = 100 * TAU * net.f[-1] / SCALE
    for k, raw_p in enumerate(vf.poles):
        p = raw_p / SCALE
        r = vf.residues[:, k].reshape(3, 3) / SCALE
        terms = [(p, r)] if not p.imag else [(p, r), (p.conjugate(), r.conj())]
        for pole, res in terms:
            if pole.real >= 0:
                raise ValueError("Unstable passive fit pole.")
            if abs(pole) > far_limit:
                # Outside the identifiable range; retain constant and slope, then revalidate.
                d -= res / pole
                e -= res / pole**2
            elif np.linalg.norm(res) / abs(pole.real) > 1e-12:
                poles.append(pole)
                residues.append(res)
    model = RationalY(np.array(poles), np.array(residues).reshape(-1, 3, 3), d, e)
    if not np.all(model.poles.real < 0):
        raise ValueError("Passive macromodel has an unstable or marginal pole.")
    s = 1j * TAU * net.f
    y = model.at(s)

    def worst(pred, truth):
        a = np.linalg.norm((pred - truth).reshape(len(s), -1), axis=1)
        b = np.linalg.norm(truth.reshape(len(s), -1), axis=1)
        return float(np.max(a / np.maximum(b, 1e-7)))

    loaded, target = load_bias(y, s), load_bias(net.y, s)
    errors = [
        worst(y, net.y),
        worst(loaded, target),
        worst(ODD @ loaded @ ODD, ODD @ target @ ODD),
        worst(EVEN @ loaded @ EVEN, EVEN @ target @ EVEN),
    ]
    if max(errors) > tol:
        raise ValueError(f"L1 fit error max(raw, loaded, odd, even) = {errors} > {tol}.")
    grid = np.sort(np.r_[net.f, (net.f[1:] + net.f[:-1]) / 2])
    passivity(model.at(1j * TAU * grid), grid)
    return model, errors


def pencil(model, dev, gap_cap=0.0, reference=False, mismatch=0.0):
    """Generalized nodal equations M0 + x M1. The reference retains open RTD poles."""
    a, b, c0 = model.realization()
    ns = len(a)
    v = np.arange(ns, ns + 3)  # barrier ports, then bias pad
    u = np.arange(ns + 3, ns + 5)  # intrinsic junction voltage
    i = np.arange(ns + 5, ns + 7)  # RTD terminal current
    g = np.arange(ns + 7, ns + 9)  # delayed conduction current
    ib = ns + 9  # bias-line current
    m0 = np.zeros((ib + 1, ib + 1), complex)
    m1 = np.zeros_like(m0)
    m0[:ns, :ns], m1[:ns, :ns] = -a, np.eye(ns)
    m0[:ns, v], m0[v, :ns] = -b, c0
    m0[np.ix_(v, v)], m1[np.ix_(v, v)] = model.d, model.e
    m1[v[:2], v[:2]] -= SCALE * gap_cap
    if reference:
        m0[v[:2], v[:2]] += REF
    else:
        m0[v[:2], i] = 1
    m0[v[2], ib] = 1
    # v = u + (Rs + s Lb)i, i = g + s C u, (1+s t/2)g = -a(1-s t/2)u.
    m0[u, v[:2]], m0[u, u], m0[u, i], m1[u, i] = 1, -1, -dev.rs, -SCALE * dev.lb
    m0[i, i], m0[i, g], m1[i, u] = 1, -1, -SCALE * dev.caps(mismatch)
    m0[g, g], m0[g, u] = 1, dev.gain
    m1[g, g], m1[g, u] = SCALE * dev.delay / 2, -dev.gain * SCALE * dev.delay / 2
    m0[ib, v[2]], m0[ib, ib], m1[ib, ib] = 1, -BIAS_R, -SCALE * BIAS_L
    return m0, m1, v[:2]


def modal_roots(model, dev, gap_cap, reference=False):
    m0, m1, idx = pencil(model, dev, gap_cap, reference)
    ab, vec = linalg.eig(-m0, m1, homogeneous_eigvals=True)
    keep = abs(ab[1]) > 1e-13 * np.maximum(abs(ab[0]), 1e-15)
    roots = SCALE * ab[0, keep] / ab[1, keep]
    volts = vec[idx][:, keep]
    norm = np.sum(abs(volts) ** 2, axis=0)
    purity = abs(ODD @ volts) ** 2 / np.maximum(norm, 1e-300)
    return roots, purity


def return_difference(model, dev, s, gap_cap):
    y = load_bias(model.at(s), s) - s[..., None, None] * gap_cap * np.eye(2)
    yd = dev.terminal_y(s)[..., None, :] * np.eye(2)
    # Unit phases of det[(Yenv+Y_RTD)/(Yenv+Gref I)] avoid large determinant magnitudes.
    a, loga = np.linalg.slogdet(y + yd)
    b, logb = np.linalg.slogdet(y + REF * np.eye(2))
    if not np.all(np.isfinite(loga - logb)):
        raise ValueError("Singularity on Nyquist contour. Move the band boundary.")
    return a * b.conj()


def nyquist(model, dev, f0, gap_cap=0.0):
    roots, purity = modal_roots(model, dev, gap_cap)
    poles, _ = modal_roots(model, dev, gap_cap, True)
    low, high = TAU * GHZ, TAU * f0
    # Include every finite fitted RHP root on the right of the contour.
    right = max(2 * high, 1.25 * max(np.r_[0.0, roots.real, poles.real]))
    both = np.r_[roots, poles]
    finite_tol = 2e-10 * SCALE

    def in_band(r):
        return (r.imag > low) & (r.imag < high) & (r.real > 0) & (r.real < right)

    if np.any((abs(both.real) < finite_tol) & (both.imag >= low) & (both.imag <= high)):
        raise ValueError("Marginal pole/zero on imaginary-axis contour; stability inconclusive.")
    if np.any(
        (both.real >= 0) & (np.minimum(abs(both.imag - low), abs(both.imag - high)) < finite_tol)
    ):
        raise ValueError("Pole/zero on a band boundary; stability inconclusive.")
    nz, npoles = int(np.sum(in_band(roots))), int(np.sum(in_band(poles)))

    def edge(start, end, count):
        delta = end - start
        projections = (both - start) / delta
        t = list(np.linspace(0, 1, count))
        for q in projections:
            if -0.1 < q.real < 1.1:
                for factor in (-8, -2, -0.5, 0, 0.5, 2, 8):
                    t.append(np.clip(q.real + factor * max(abs(q.imag), 1e-12), 0, 1))
        return start + delta * np.unique(t)

    corners = [1j * low, right + 1j * low, right + 1j * high, 1j * high, 1j * low]
    previous = None
    for count in (129, 257, 513, 1025, 2049):
        contour = np.concatenate([edge(a, b, count) for a, b in pairwise(corners)])
        phase = return_difference(model, dev, contour, gap_cap)
        increments = np.angle(phase[1:] * phase[:-1].conj())
        turns = float(np.sum(increments) / TAU)
        winding = int(round(turns))
        conjugate = return_difference(model, dev, contour[::-1].conj(), gap_cap)
        neg_inc = np.angle(conjugate[1:] * conjugate[:-1].conj())
        negative_turns = float(np.sum(neg_inc) / TAU)
        if (
            previous == winding
            and abs(turns - winding) < 1e-6
            and abs(negative_turns - winding) < 1e-6
            and max(np.max(abs(increments)), np.max(abs(neg_inc))) < np.pi / 2
            and winding + npoles == nz
        ):
            break
        previous = winding
    else:
        raise ValueError("Winding/pencil cross-check failed; refine frequency sampling or fit.")
    # A conjugate partner has the same growth rate. Keep all positive partners and real roots.
    rhp = np.flatnonzero((roots.real > finite_tol) & (roots.imag >= -finite_tol))
    candidates = [
        i
        for i in rhp
        if FENCE_BAND[0] * 0.97 <= roots[i].imag / TAU <= FENCE_BAND[1] * 1.03 and purity[i] > 0.90
    ]
    intended = min(candidates, key=lambda i: abs(roots[i].imag / TAU - f0), default=None)
    modes = []
    for i in sorted(rhp, key=lambda j: (roots[j].imag, roots[j].real)):
        freq = max(0.0, roots[i].imag / TAU)
        modes.append(
            dict(
                sigma_per_s=float(roots[i].real),
                frequency_GHz=float(freq / GHZ),
                has_conjugate=bool(freq > finite_tol / TAU),
                odd_voltage_fraction=float(purity[i]),
                intended=bool(i == intended),
                region="below data / real axis"
                if freq < GHZ
                else ("counted band" if freq < f0 else "above counted band"),
            )
        )
    return dict(
        level="L1 fitted model only",
        positive_frequency_CCW=winding,
        negative_frequency_CCW=int(round(negative_turns)),
        open_RHP_poles_per_half=npoles,
        closed_RHP_zeros_per_half=nz,
        band_GHz=[1.0, f0 / GHZ],
        all_fitted_RHP_modes=modes,
        other_RHP_modes=[m for m in modes if not m["intended"]],
    )


def save_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=rows[0].keys())
        output.writeheader()
        output.writerows(rows)


def analyze(folder, out, step, orders, fit_tol, delay_ps, bias_offset=0.0, hb_band=(280e9, 315e9)):
    meta = json.loads((folder / "model.json").read_text())
    if (
        meta["schema"] != "ringslot.barrier.v1"
        or meta["ports"] != ["P1", "P2", "Bias"]
        or meta["bias_ohm_H"] != [BIAS_R, BIAS_L]
        or not meta["bridge_in_em"]
        or meta["partition"] != ["R1", "R2", "CPW", "MIM"]
    ):
        raise ValueError("Unexpected embedding, port order or bias termination.")
    gap = float(meta["empty_gap_F"])
    if not np.isclose(gap, GAP_C, rtol=1e-6, atol=0):
        raise ValueError("Active-volume subtraction does not match the port geometry.")
    two, three = stitch(folder, 2, step), stitch(folder, 3, step)
    freq = np.unique(np.r_[two.f, three.f])
    schur = load_bias(three.at(freq), 1j * TAU * freq)
    target = two.at(freq)
    err = np.linalg.norm(schur - target, axis=(1, 2)) / np.maximum(
        np.linalg.norm(target, axis=(1, 2)), 1e-12
    )
    if err.max() > 0.01:
        raise ValueError(f"Physical bias termination mismatch: {err.max():.2%} [L1].")
    rad = Radiation.read(folder / "radiation.csv")
    out.mkdir(parents=True, exist_ok=True)
    rows, checks, devices = [], [], []
    for rs in (3.7, 7.0):
        dev = Device(
            rs=rs, lb=0.0, delay=delay_ps * 1e-12, bias_offset=bias_offset
        )  # bridge already present in both extractions
        base = harmonic_balance(two, dev, gap_cap=gap, band=hb_band)
        sweep = {0.0: base}
        for sign in (-1, 1):
            seed = base
            for percent in range(1, 11):
                m = sign * percent / 100
                seed = harmonic_balance(two, dev, m, seed=seed, gap_cap=gap, band=hb_band)
                sweep[m] = seed
        rows.extend(mismatch_row(two, rad, dev, m, sol) for m, sol in sorted(sweep.items()))
        devices.append((dev, base.f))
    save_rows(out / "hb_mismatch.csv", rows)
    good = []
    rejected = []
    for order in orders:
        try:
            model, errors = rational_fit(three, order, fit_tol)
            reports = [nyquist(model, dev, f0, gap) for dev, f0 in devices]
        except (ValueError, np.linalg.LinAlgError) as exc:
            rejected.append(dict(order=order, reason=str(exc)))
            warnings.warn(
                f"L1 fit or contour failed at order {order}: {exc}", RuntimeWarning, stacklevel=2
            )
            continue
        for report in reports:
            report.update(fit_order_requested=order, max_relative_errors=errors)
        good.append(reports)
        if len(good) == 2:
            break
    if len(good) < 2:
        (out / "stability.json").write_text(
            json.dumps(
                dict(status="inconclusive", rejected=rejected, accepted_fit_reports=good),
                indent=2,
                allow_nan=False,
            )
        )
        raise ValueError(
            "Insufficient validated fit orders for stability; HB and diagnostics were saved."
        )
    for index, (dev, _) in enumerate(devices):
        reports = [fits[index] for fits in good]

        def inband(report):
            return [m for m in report["all_fitted_RHP_modes"] if m["region"] == "counted band"]

        first, second = [inband(report) for report in reports]
        agrees = len(first) == len(second) and all(
            abs(x["frequency_GHz"] - y["frequency_GHz"]) <= 0.01 * max(1.0, x["frequency_GHz"])
            and abs(x["sigma_per_s"] - y["sigma_per_s"]) <= 0.10 * max(1.0, x["sigma_per_s"])
            for x, y in zip(first, second, strict=True)
        )
        checks.append(
            dict(
                rs_ohm=dev.rs,
                limit_ohm=dev.stab_limit,
                rejected_orders=rejected,
                margin=1 - 18.75 / dev.stab_limit,
                fit_orders_agree_in_band=agrees,
                fits=reports,
            )
        )
    (out / "stability.json").write_text(json.dumps(checks, indent=2, allow_nan=False))
    if not all(check["fit_orders_agree_in_band"] for check in checks):
        raise RuntimeError(
            "In-band modes depend on fit order; see stability.json. Verdict inconclusive."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analytic", help="L1 section 2 angle sweep (NaN for a missing root)")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--lo-ghz", type=float, default=220.0)
    a.add_argument("--hi-ghz", type=float, default=380.0)
    h = sub.add_parser("analyze", help="HB and Nyquist on one builder export directory")
    h.add_argument("folder", type=Path)
    h.add_argument("--out", required=True, type=Path)
    h.add_argument(
        "--hb-band-ghz", type=float, nargs=2, default=(280.0, 315.0), metavar=("LOW", "HIGH")
    )
    h.add_argument("--grid-ghz", type=float, default=1.0)
    h.add_argument("--fit-orders", type=int, nargs="+", default=[6, 10, 16, 22, 30])
    h.add_argument("--fit-tol", type=float, default=0.01)
    h.add_argument(
        "--delay-ps", type=float, default=0.0, help="Optional causal Pade conduction delay"
    )
    h.add_argument(
        "--bias-offset-v",
        type=float,
        default=0.0,
        help="Fixed junction voltage offset from the cubic midpoint",
    )
    args = parser.parse_args()
    if args.cmd == "analytic":
        save_rows(
            args.out,
            [
                dict(
                    angle_deg=t,
                    odd_GHz=analytic_odd(t, args.lo_ghz * GHZ, args.hi_ghz * GHZ) / GHZ,
                    level="L1",
                )
                for t in np.arange(35.0, 325.1, 0.5)
            ],
        )
    else:
        if args.delay_ps < 0 or args.fit_tol <= 0 or len(set(args.fit_orders)) < 2:
            parser.error("Require nonnegative delay, positive tolerance and distinct fit orders.")
        analyze(
            args.folder,
            args.out,
            args.grid_ghz * GHZ,
            list(dict.fromkeys(args.fit_orders)),
            args.fit_tol,
            args.delay_ps,
            args.bias_offset_v,
            np.asarray(args.hb_band_ghz) * GHZ,
        )


if __name__ == "__main__":
    main()
