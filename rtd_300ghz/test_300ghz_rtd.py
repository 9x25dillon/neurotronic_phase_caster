"""L0 regression checks. All electrical fixtures and outputs here are synthetic L1.

The baseline power targets are L2 in the supplied specification. No AEDT license
is needed. Run from this directory with python -m pytest -q test_300ghz_rtd.py.
"""

import numpy as np
import pytest
from scipy.constants import c

import build_300ghz_rtd_ringslot as cad
import sim_300ghz_rtd_cosim as sim

F = np.arange(1, 951) * sim.GHZ
QO = np.outer(sim.ODD, sim.ODD)
QE = np.outer(sim.EVEN, sim.EVEN)


def fixture(yodd, yeven):
    return sim.Network(
        F, yodd[:, None, None] * QO + np.broadcast_to(yeven, F.shape)[:, None, None] * QE
    )


def synthetic_fields(path, net, eta):
    """Independent drive-power export. H=eta*(I-S†S); unit Jones basis is explicit."""
    n = len(net.f)
    ident = np.broadcast_to(np.eye(2), (n, 2, 2))
    sy = np.linalg.solve(ident + sim.Z0 * net.y, ident - sim.Z0 * net.y)
    acc = ident - sy.conj().swapaxes(1, 2) @ sy
    rad = eta * acc
    odd = np.array([1.0, -1.0])
    ac = (odd @ acc @ odd).real
    rows = []
    for k, f in enumerate(net.f):
        rows.append(
            dict(
                f_Hz=f,
                eta_odd=eta,
                radiated_odd_W=eta * ac[k],
                accepted_odd_W=ac[k],
                rad11=rad[k, 0, 0].real,
                rad22=rad[k, 1, 1].real,
                rad12_re=rad[k, 0, 1].real,
                rad12_im=rad[k, 0, 1].imag,
                theta1_re=1.0,
                theta1_im=0.0,
                theta2_re=0.0,
                theta2_im=0.0,
                phi1_re=0.0,
                phi1_im=0.0,
                phi2_re=1.0,
                phi2_im=0.0,
            )
        )
    sim.save_rows(path, rows)
    return sim.Radiation.read(path)


@pytest.mark.parametrize("h", [1, 3])
def test_cubic_analytic_odd_solution(h):
    dev = sim.Device(rs=0.0, lb=0.0)
    fg, g, ct = 287e9, 0.004, 1e-12
    w = sim.TAU * F
    induct = 1 / ((sim.TAU * fg) ** 2 * (ct + dev.cap))
    net = fixture(g + 1j * (w * ct - 1 / (w * induct)), 0.055)
    hb = sim.harmonic_balance(net, dev, harmonics=h)
    f_analytic = 1 / (sim.TAU * np.sqrt(induct * (ct + dev.cap)))
    amp_squared = 4 * (dev.a - g) / (3 * dev.b)
    # Two ports, each with P=(1/2)G|Vpk|². At 3f the high-Q tank pins the voltage.
    assert hb.f == pytest.approx(f_analytic, rel=0.001)
    assert hb.p_accepted[0] == pytest.approx(g * amp_squared, rel=0.001)
    assert hb.residual_A < 1e-9
    assert hb.p_rs == 0.0


def test_corrected_baseline_three_harmonics(tmp_path):
    # Section 2 supplies Bodd=(79.5-93.4) mS at 300 GHz and G=0.91+0.10 mS.
    # L1 synthetic dispersion: a parallel 10 fF, with L calculated from that susceptance.
    # Section 1 supplies a_RF=a_DC*cos(w*tau); keep the DC-calibrated cubic coefficient.
    w, w0 = sim.TAU * F, sim.TAU * 300e9
    ct = 10e-15
    induct = 1 / (w0 * (w0 * ct + 0.0139))
    net = fixture(0.00101 + 1j * (w * ct - 1 / (w * induct)), 0.052 + 1j * 0.173 * F / 300e9)
    dev = sim.Device(a=sim.Device().a * np.cos(w0 * 0.2e-12), rs=7.0)
    hb = sim.harmonic_balance(net, dev)
    rad = synthetic_fields(tmp_path / "radiation.csv", net, 0.10)
    row = sim.mismatch_row(net, rad, dev, 0.0, hb)
    assert row["f_GHz"] == pytest.approx(300.06, rel=0.001)
    assert row["phase_deg"] == pytest.approx(180.0, abs=0.001)
    assert row["amplitude_ratio"] == pytest.approx(1.0, abs=1e-7)
    assert row["Prad1_W"] == pytest.approx(11e-6, rel=0.05)
    assert row["PRs_all_W"] == pytest.approx(0.172e-3, rel=0.05)
    assert row["epsilon_re"] == 0 and row["epsilon_im"] == 0
    assert np.isinf(row["AR_linear"])
    assert row["PRs_all_W"] > dev.rs * np.sum(abs(hb.i[0]) ** 2) / 2  # harmonic loss included
    plus = sim.harmonic_balance(net, dev, 0.1, seed=hb)
    minus = sim.harmonic_balance(net, dev, -0.1, seed=hb)
    rr = sim.mismatch_row(net, rad, dev, 0.1, plus)
    assert plus.ratio * minus.ratio == pytest.approx(1, abs=1e-7)
    assert np.isfinite(rr["AR_linear"]) and rr["AR_linear"] > 1
    assert abs(complex(rr["epsilon_re"], rr["epsilon_im"])) > 1e-4


def test_entire_modal_scan_and_clearance():
    # p = 0 regression against the original TM_mn0 values at 71.5 um.
    m = cad.cavity_modes(71.5, f_max=1100)
    assert [m["TM", *idx, 0] for idx in ((0, 1), (1, 1), (2, 1), (0, 2))] == pytest.approx(
        [453.90, 723.22, 969.33, 1041.90], abs=0.015
    )
    # Axial modes: h = 70.7 um puts TM111 (degenerate with TE011) in the h=3 guard.
    assert cad.axial_step(cad.TS) == pytest.approx(599.675, abs=0.001)
    assert m["TM", 1, 1, 1] == pytest.approx(939.50, abs=0.01)
    assert m["TE", 0, 1, 1] == pytest.approx(m["TM", 1, 1, 1], rel=1e-12)  # j'_01 = j_11
    assert not cad.slot_coupled(("TE", 0, 1, 1)) and cad.slot_coupled(("TM", 1, 1, 1))
    for r, mode in [(71.5, "TM111"), (78, "TM210"), (85, "TM110")]:
        with pytest.raises(ValueError, match=mode):
            cad.check_modes(r)
    # 70 um clears at nominal thickness only; TM111 enters the guard inside +2 um.
    cad.check_modes(70.0, h_tol=0)
    with pytest.raises(ValueError, match="TM111 enters a guard at substrate 70.829"):
        cad.check_modes(70.0)
    assert cad.clear_radii(60.0, 95.0, 0.05, h_tol=0) == pytest.approx(
        np.round(np.arange(66.65, 70.0501, 0.05), 2)
    )
    assert cad.clear_radii(60.0, 95.0, 0.05) == pytest.approx(
        np.round(np.arange(68.55, 68.8501, 0.05), 2)
    )
    info = cad.preflight(cad.Design())
    assert cad.Design().rc == 68.75
    assert info["modes_GHz"] == pytest.approx(
        {"TM010": 472.06, "TE111": 700.17, "TM110": 752.15, "TM011": 763.18, "TE211": 847.97},
        abs=0.01,
    )
    assert info["clear_substrate_window_um"] == pytest.approx([68.5074, 72.8954], abs=1e-4)
    assert info["nominal_clearance_um"] == pytest.approx(10.65)
    assert info["corner_clearance_um"] == pytest.approx(10.3812, abs=1e-4)
    assert info["via_count"] == 22


def test_mode_enumeration_is_complete():
    # Independent brute force: generous Bessel root tables, every p, no early exits.
    from scipy.special import jn_zeros, jnp_zeros

    rc, h, f_max = 68.75, 69.3, 1500.0
    scale = c / (2 * np.pi * rc * 1e-6 * np.sqrt(12.5)) / 1e9
    fz = cad.axial_step(h)
    brute = {}
    for family, zeros, p0 in (("TM", jn_zeros, 0), ("TE", jnp_zeros, 1)):
        for mm in range(30):
            for n, x in enumerate(zeros(mm, 30), 1):
                for p in range(p0, 30):
                    f = np.hypot(x * scale, p * fz)
                    if f <= f_max:
                        brute[family, mm, n, p] = f
    assert cad.cavity_modes(rc, h, f_max) == pytest.approx(brute, rel=1e-12)


@pytest.mark.parametrize("rc", [67.0, 68.5, 68.75, 69.0, 70.0])
def test_thickness_window_matches_sampling(rc):
    lo, hi, below, above = cad.thickness_window(rc)
    assert lo < cad.TS < hi and (below, above) == ("TE211", "TM111")

    def inside(h):
        return any(
            cad.slot_coupled(k) and any(a <= f <= b for a, b in cad.guarded_bands())
            for k, f in cad.cavity_modes(rc, h).items()
        )

    assert not any(inside(h) for h in np.linspace(lo + 1e-6, hi - 1e-6, 41))
    # The window edges are exact: 1 nm outside, a coupled mode sits in a guard band.
    assert all(inside(h) for h in (lo - 1e-3, lo - 1e-6, hi + 1e-6, hi + 1e-3))
    assert cad.thickness_window(71.5)[:2] == (cad.TS, cad.TS)  # nominal itself forbidden


def test_no_root_and_singularity():
    assert np.isnan(sim.analytic_odd(90.0, 299.9e9, 300.1e9))
    assert np.isnan(sim.first_root(lambda f: np.ones_like(f), 220e9, 380e9))
    r_um = c / (2 * np.sqrt(6.8) * 300e9 * np.pi) * 1e6
    assert all(np.isnan(v) for v in sim.ring_modes(300e9, 180, r_um))
    # A sign flip across a pole does not count as a root.
    assert np.isnan(sim.first_root(lambda f: 1 / (f - 1.45), 1.0, 2.0, [1.45]))


def test_nonpassive_network():
    # Positive diagonals do not ensure passivity; the odd mode has negative damping.
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]]) * 1e-3
    with pytest.raises(ValueError, match="Non-passive"):
        sim.Network([1e9, 2e9], np.stack([matrix, matrix]))
    # Purely imaginary off-diagonal entries in a non-reciprocal counterexample.
    m = np.array([[1.0, 2j], [-2j, 1.0]]) * 1e-3
    with pytest.raises(ValueError, match="Non-passive"):
        sim.passivity(np.stack([m, m]), np.array([1e9, 2e9]))


@pytest.mark.parametrize("variant,angle", [("si_lens", 95.0), ("cavity_via", 135.0)])
@pytest.mark.parametrize("layout", ["cpw_gap", "flank_10deg"])
def test_layouts(variant, angle, layout):
    d = cad.Design(variant, layout)
    p = cad.preflight(d)
    assert p["theta_R2_deg"] == angle / 2 + 180
    assert p["plate_area_um2"] == pytest.approx(3600)
    bodies = {obj.name: obj for obj in cad.device_solids(d, cad.layout(d))}
    assert bodies["MIM_Top_Plate"].fp.equals(bodies["MIM_SiO2_Dielectric"].fp)
    assert bodies["Choke_to_MIM_Via"].z == cad.TM
    assert p["raw_film_ohm"] == pytest.approx([37.383, 74.766, 74.766], rel=1e-4)
    assert ("via_count" in p) == (variant == "cavity_via")


def test_stitch_join_passivity_and_harmonic_coverage(tmp_path):
    for label, lo, hi in sim.BANDS:
        f = np.linspace(lo, hi, 5) * sim.GHZ
        y = np.tile(np.eye(2) * (0.011 if label == "mid" else 0.010), (len(f), 1, 1))
        sim.write_synthetic_snp(tmp_path / f"{label}.s2p", f, y)
    with pytest.warns(RuntimeWarning, match="join") as records:
        net = sim.stitch(tmp_path, 2, 4e9)
    assert len(records) == 2
    assert np.sum(net.f == 100e9) == 1
    assert net.at(100e9)[0, 0] == pytest.approx(0.0105)
    with pytest.raises(ValueError, match="outside data"):
        net.at(3 * 320e9)
    with pytest.raises(ValueError, match="outside data"):
        sim.harmonic_balance(net, sim.Device(), band=(280e9, 320e9))


def rational_fixture():
    p = np.zeros((3, 3))
    p[:2, :2] = QO
    e = np.zeros_like(p)
    e[:2, :2] = QE
    ct = 30e-15
    induct = 1 / ((sim.TAU * 250e9) ** 2 * (ct + 6.5e-15))
    r = 0.5
    pad = np.array([0.5, 0.5, -1.0])
    d = 0.003 * p + 0.055 * e + 0.001 * np.outer(pad, pad)
    d[2, 2] += 0.002
    return (
        sim.RationalY(
            np.array([-r / (sim.SCALE * induct)]),
            np.array([p / (sim.SCALE * induct)]),
            d,
            ct * sim.SCALE * p,
        ),
        ct,
        induct,
        r,
    )


def test_nyquist_and_independent_polynomial():
    model, ct, induct, r = rational_fixture()
    dev = sim.Device(rs=0, lb=0)
    result = sim.nyquist(model, dev, 300e9)
    # (s Ctot + G-a)(s L + r) + 1 = 0.
    cap = ct + dev.cap
    coef = [cap * induct, cap * r + (0.003 - dev.a) * induct, 1 + (0.003 - dev.a) * r]
    root = max(np.roots(coef), key=lambda z: z.imag)
    assert result["positive_frequency_CCW"] == result["negative_frequency_CCW"] == 1
    assert result["open_RHP_poles_per_half"] == 0
    [mode] = result["other_RHP_modes"]
    assert mode["sigma_per_s"] == pytest.approx(root.real, rel=1e-8)
    assert mode["frequency_GHz"] == pytest.approx(root.imag / (sim.TAU * sim.GHZ), rel=1e-8)
    rs_result = sim.nyquist(model, sim.Device(rs=7.0, lb=0), 300e9)
    assert rs_result["other_RHP_modes"][0]["frequency_GHz"] < mode["frequency_GHz"]
    s = 1j * sim.TAU * 5e9
    y = model.at(s)
    result = sim.load_bias(y, s)
    v = np.array([1.0, 1.0])
    vp = -y[2, :2] @ v / (y[2, 2] + 1 / (sim.BIAS_R + s * sim.BIAS_L))
    assert result @ v == pytest.approx((y @ np.r_[v, vp])[:2])
    assert sim.Device().stab_limit == pytest.approx(23.9725, abs=0.001)
    assert sim.Device(rs=3.7).stab_limit == pytest.approx(25.6225, abs=0.001)


def test_field_polarization_and_imported_efficiency(tmp_path):
    net = fixture(np.full_like(F, 0.003), 0.04)
    v = np.array([0.2, -0.2j])
    i = net.at(300e9) @ v
    r1 = synthetic_fields(tmp_path / "a.csv", net, 0.12)
    r2 = synthetic_fields(tmp_path / "b.csv", net, 0.24)
    assert r2.evaluate(300e9, v, i)[0] == pytest.approx(2 * r1.evaluate(300e9, v, i)[0])
    assert sim.axial_ratio(np.array([1.0, 1j])) == pytest.approx(1.0)
    assert np.isinf(sim.axial_ratio(np.array([1.0, 0.3])))
    assert np.isnan(sim.axial_ratio(np.zeros(2, complex)))


def test_review_shunt_normalizations():
    g = cad.r2_estimates(cad.Design())
    for key, pair in [("cpw_gap_review_5um", 0.180), ("flank_10deg", 0.474)]:
        est = g[key]
        assert est["pair_mS"] == pytest.approx(pair, abs=0.001)
        assert est["pair_mS"] == pytest.approx(2 * est["per_port_mS"])
    assert g["physical_cpw_transverse_center_um"] == 6.5


def test_end_to_end_loaded_exports(tmp_path):
    import json
    import warnings

    s = 1j * sim.TAU * F
    w0 = sim.TAU * 300e9
    ct = 15e-15
    induct = 1 / (w0 * (w0 * ct + 0.0125))
    yo = 0.00101 + 1 / (0.05 + s * induct) + s * (ct + sim.GAP_C)
    ye = 0.052 + s * (90e-15 + sim.GAP_C)
    y = np.zeros((len(F), 3, 3), complex)
    y[:, :2, :2] = yo[:, None, None] * QO + ye[:, None, None] * QE
    pad = np.array([0.5, 0.5, -1.0])
    y += np.outer(pad, pad) / (1 + s[:, None, None] * 79e-12)
    y[:, 2, 2] += 0.001 + s * 1.24e-12
    two = sim.Network(F, sim.load_bias(y, s))
    for lab, lo, hi in sim.BANDS:
        idx = (F >= lo * sim.GHZ) & (F <= hi * sim.GHZ)
        for ports, mtx in [(2, two.y), (3, y)]:
            sim.write_synthetic_snp(tmp_path / f"{lab}.s{ports}p", F[idx], mtx[idx])
    synthetic_fields(tmp_path / "radiation.csv", two, 0.10)
    meta = dict(
        schema="ringslot.barrier.v1",
        ports=["P1", "P2", "Bias"],
        bias_ohm_H=[1, 1e-9],
        bridge_in_em=True,
        partition=["R1", "R2", "CPW", "MIM"],
        empty_gap_F=sim.GAP_C,
    )
    (tmp_path / "model.json").write_text(json.dumps(meta))
    out = tmp_path / "results"
    with warnings.catch_warnings(record=True):
        # Over-ordering an exact low-order fixture can make pole relocation ill-conditioned.
        # The numerical checks below remain mandatory for both independent fits.
        sim.analyze(tmp_path, out, 1e9, [2, 3, 4, 6], 0.01, 0.0)
    hb = np.genfromtxt(
        out / "hb_mismatch.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    assert set(hb["rs_ohm"]) == {3.7, 7.0}
    assert len(hb) == 42
    for rs in (3.7, 7.0):
        sample = hb[hb["rs_ohm"] == rs]
        assert sample["deltaC_over_C"] == pytest.approx(np.arange(-10, 11) / 100)
        assert np.all(sample["KCL_A"] < 1e-9)
    reports = json.loads((out / "stability.json").read_text())
    assert all(r["fit_orders_agree_in_band"] for r in reports)
    for r in reports:
        for fit in r["fits"]:
            assert (
                fit["positive_frequency_CCW"] + fit["open_RHP_poles_per_half"]
                == fit["closed_RHP_zeros_per_half"]
            )
            assert max(fit["max_relative_errors"]) < 0.01


def test_second_harmonic_off_midpoint():
    f = F
    w = sim.TAU * f
    w0 = sim.TAU * 300e9
    ct = 15e-15
    induct = 1 / (w0 * (w0 * ct + 0.0139))
    net = fixture(0.00101 + 1j * (w * ct - 1 / (w * induct)), 0.052 + 1j * 0.173 * f / 300e9)
    centered = sim.harmonic_balance(net, sim.Device())
    biased = sim.harmonic_balance(net, sim.Device(bias_offset=0.035), seed=centered)
    assert abs(centered.v[1]).max() < 1e-9
    assert abs(biased.v[1]).max() > 0.001
    assert biased.p_accepted[1] > 1e-8
    assert sim.Device(bias_offset=0.035).stab_limit > sim.Device().stab_limit
