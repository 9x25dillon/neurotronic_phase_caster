"""Calibration, integrity and regression checks for src/water_structure_assay.py.

The assay is only worth running if it (a) stays at chance when nothing is there,
(b) finds an effect when one is injected, and (c) raises integrity flags when the
pipeline leaks. Each property is tested here on the simulator.
"""

import dataclasses
import json
import math

import numpy as np
import pytest

from src import water_structure_assay as wa
from src.photonic_thz_pulse_engine import PhotonicTHzEmitterArray

SMALL_CB = wa.SymbolCodebook((0.9e12, 1.1e12), (0, 2))
FAST = wa.Preregistration(n_permutations=1000, n_bootstrap=500, slots_per_arm_per_block=8)


@pytest.fixture(scope="module")
def null_run():
    return wa.simulate_assay(FAST, SMALL_CB, injected_magnitude=0.0, payload_text="hi", seed=11)


@pytest.fixture(scope="module")
def positive_run():
    return wa.simulate_assay(FAST, SMALL_CB, injected_magnitude=0.05, payload_text="hi", seed=11)


# --- cell, geometry, codebook ------------------------------------------------------------


def test_cell_time_constants():
    cell = wa.SampleCell()
    assert cell.heat_capacity_j_per_k == pytest.approx(4.18e6 * math.pi * 2.5e-3**2 * 100e-6)
    assert cell.thermal_time_constant_s == pytest.approx(1.64, rel=1e-2)
    assert cell.lateral_diffusion_time_s == pytest.approx(2.5e-3**2 / 1.43e-7)
    assert cell.classical_memory_s == pytest.approx(5 * cell.lateral_diffusion_time_s)
    assert float(cell.alpha_per_m(1e12, wa.T_REF_K + 10)) == pytest.approx(220.3e2 * 1.05)


def test_layline_charge_controls_on_axis_field():
    cell, ill = wa.SampleCell(), wa.RingIllumination()
    p = np.full(12, 10e-6)
    centre = ill.grid_points // 2
    i0, _ = ill.intensity_map(cell, 1e12, p, wa.SymbolCodebook.layline_phases(0, 12))
    i2, _ = ill.intensity_map(cell, 1e12, p, wa.SymbolCodebook.layline_phases(2, 12))
    assert i0[centre, centre] == pytest.approx(i0.max(), rel=1e-6)  # charge 0 peaks on axis
    assert i2[centre, centre] < 1e-6 * i2.max()  # charge 2 has a vortex null
    # A vortex moves power off axis without discarding it: both charges absorb ~ the incoherent sum
    captured = 1 - math.exp(-2 * cell.aperture_radius_m**2 / ill.spot_radius_m**2)
    expected = p.sum() * captured * float(cell.absorbed_fraction(1e12))
    for charge in (0, 1, 2, 3):
        got = ill.absorbed_power_w(cell, 1e12, p, wa.SymbolCodebook.layline_phases(charge, 12))
        assert got == pytest.approx(expected, rel=0.05)


def test_codebook_roundtrip_and_command():
    cb = wa.SymbolCodebook()
    assert cb.size == 16 and cb.bits_per_symbol == 4
    for s in range(cb.size):
        assert cb.compose(*cb.decompose(s)) == s
    cmd = cb.command(7, 0.5, 1000.0, 12)
    node, charge = cb.decompose(7)
    assert cmd["frequency_thz"] == pytest.approx(cb.frequency_nodes_hz[node] / 1e12)
    assert np.allclose(cmd["phase_pattern_rad"], wa.SymbolCodebook.layline_phases(charge, 12))
    assert PhotonicTHzEmitterArray().plan(cmd).safe
    with pytest.raises(ValueError):
        cb.decompose(16)


def test_equalisation_removes_absorbed_power_differences():
    cb, cell, ill, arr = wa.SymbolCodebook(), wa.SampleCell(), wa.RingIllumination(), PhotonicTHzEmitterArray()
    cal = wa.calibrate_codebook(cb, arr, cell, ill, 1.0, equalize=True)
    raw = np.array(cal.unequalised_absorbed_power_w)
    assert raw.max() / raw.min() > 1.5  # nodes and laylines differ a lot before equalisation
    assert np.allclose(cal.absorbed_power_w, cal.absorbed_power_w[0], rtol=1e-12)
    assert max(cal.amplitudes) <= 1.0 + 1e-12
    for s in (0, 5, 15):  # model check: the amplitude really produces the equalised power
        plan = arr.plan(cb.command(s, cal.amplitudes[s], 1000.0, 12))
        assert ill.absorbed_power_from_plan(cell, plan) == pytest.approx(cal.absorbed_power_w[s], rel=1e-9)


# --- streams -------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 4, 16])
def test_text_roundtrip(m):
    text = "hello, water é"
    assert wa.symbols_to_text(wa.text_to_symbols(text, m), m) == text
    stream = wa.TextStream("hi", m)
    got = [stream.next_symbol(m) for _ in range(len(stream))]
    assert got == wa.text_to_symbols("hi", m)
    with pytest.raises(StopIteration):
        stream.next_symbol(m)


def test_text_stream_validation():
    with pytest.raises(ValueError):
        wa.TextStream("x", 6)
    with pytest.raises(ValueError):
        wa.TextStream("x", 16).next_symbol(4)


@pytest.mark.parametrize("freq,expected", [(2.0, 0), (6.0, 1), (10.0, 2), (20.0, 3), (40.0, 4)])
def test_eeg_stream_maps_dominant_band_to_substrate(freq, expected):
    fs = 250.0
    t = np.arange(int(fs * 4)) / fs
    rng = np.random.default_rng(0)
    eeg = np.vstack([np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size) for _ in range(8)])
    stream = wa.EEGSubstrateStream(eeg, fs, window_s=2.0)
    assert stream.next_symbol(16) == expected
    assert stream.substrate_sequence()[0] == wa.SUBSTRATES[expected]
    assert not stream.retro_valid


def test_random_stream():
    a, b = wa.RandomStream(3), wa.RandomStream(3)
    assert [a.next_symbol(16) for _ in range(20)] == [b.next_symbol(16) for _ in range(20)]
    assert wa.RandomStream().retro_valid
    assert 0 <= wa.RandomStream().next_symbol(16) < 16


# --- simulator and thermal calibration -------------------------------------------------------


def test_heater_reaches_predicted_steady_state():
    cell = wa.SampleCell()
    inst = wa.SimulatedInstrument(cell, wa.RingIllumination(), wa.ReceiverModel(thermistor_noise_k=0.0))
    inst.apply(wa.DriveAction("heater", heater_power_w=5e-3))
    inst.wait(cell.thermal_time_constant_s)
    assert inst.temperature_k - cell.holder_temperature_k == pytest.approx(5e-3 * 200 * (1 - math.e**-1), rel=1e-9)
    inst.wait(50 * cell.thermal_time_constant_s)
    assert inst.read(np.array([1e12])).temperature_k == pytest.approx(cell.holder_temperature_k + 1.0, abs=1e-9)


def test_simulator_refuses_drive_without_plan():
    inst = wa.SimulatedInstrument(wa.SampleCell(), wa.RingIllumination())
    with pytest.raises(ValueError):
        inst.apply(wa.DriveAction("thz", None))
    with pytest.raises(ValueError):
        inst.apply(wa.DriveAction("teleport"))


def test_simulator_identifies_symbol_from_plan():
    cb = wa.SymbolCodebook()
    inst = wa.SimulatedInstrument(wa.SampleCell(), wa.RingIllumination(), injected=wa.InjectedEffect(cb))
    arr = PhotonicTHzEmitterArray()
    for s in range(cb.size):
        assert inst._symbol_from_plan(arr.plan(cb.command(s, 0.5, 1000.0, 12))) == s


def test_thermal_calibration_recovers_kappa():
    cell = wa.SampleCell(kappa_per_k=0.006)
    inst = wa.SimulatedInstrument(cell, wa.RingIllumination(), seed=4)
    cal = wa.thermal_calibration(inst, cell, wa.Preregistration().readout_frequencies_hz)
    assert np.median(cal.slope_rel_per_k) == pytest.approx(0.006, rel=0.05)
    assert np.median(cal.r_squared) > 0.95
    assert np.allclose(cal.alpha_ref_per_m, cell.alpha_per_m(cal.frequencies_hz, cell.holder_temperature_k), rtol=2e-3)


# --- preregistration, blinding, schedule -------------------------------------------------------


def test_preregistration_digest_is_stable_and_sensitive():
    a, b = wa.Preregistration(), wa.Preregistration()
    assert a.digest() == b.digest()
    assert dataclasses.replace(a, sesoi_rel_alpha=3e-3).digest() != a.digest()
    assert a.trial_duration_s() == pytest.approx(600 + 5 + 30 + 300)


def test_preregistration_validation_against_cell():
    cell = wa.SampleCell()
    assert wa.Preregistration().validate(cell) == []
    bad = wa.Preregistration(intertrial_s=60.0, post_delays_s=(1.0, 10.0))
    issues = bad.validate(cell)
    assert any("intertrial" in i for i in issues) and any("no post delay" in i for i in issues)


def test_allocation_key_commitment():
    k1, k2 = wa.make_allocation_key(1), wa.make_allocation_key(2)
    assert k1.commitment() == wa.make_allocation_key(1).commitment()
    assert k1.commitment() != k2.commitment()
    assert {k1.arm_of(c) for c, _ in k1.codes} == set(wa.ARMS)
    assert len(wa.make_allocation_key().salt) == 32


def test_schedule_structure():
    key = wa.make_allocation_key(0)
    sched = wa.make_schedule(wa.Preregistration(n_blocks=3), 4, key, payload_length=5, seed=0)
    cal = [t for t in sched if t.phase == "calibration"]
    assert len(cal) == 3 * 3 * 4
    for block in range(3):
        for arm in wa.ARMS:
            assert sum(t.block == block and t.arm_code == key.code_of(arm) for t in cal) == 4
    pay = [t for t in sched if t.phase == "payload"]
    assert len(pay) == 10
    for p in range(5):
        assert sorted(key.arm_of(t.arm_code) for t in pay if t.payload_index == p) == ["active", "sham"]
    assert [t.index for t in sched] == list(range(len(sched)))


def test_run_refuses_invalid_preregistration():
    cell, ill = wa.SampleCell(), wa.RingIllumination()
    bad = wa.Preregistration(intertrial_s=10.0)
    key = wa.make_allocation_key(0)
    with pytest.raises(ValueError):
        wa.run_assay(wa.SimulatedInstrument(cell, ill), PhotonicTHzEmitterArray(), cell, ill, SMALL_CB, bad,
                     wa.make_schedule(bad, SMALL_CB.size, key), key)


# --- statistics ---------------------------------------------------------------------------------


def test_holm_matches_hand_calculation():
    assert np.allclose(wa.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_permutation_tost_and_ci():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
    shifted = b + 3.0
    assert wa.permutation_p(a, shifted, 2000, rng) < 0.001
    assert wa.permutation_p(a, b, 2000, rng) > 0.05
    assert wa.tost_p(a, b, 1.0) < 0.05
    assert wa.tost_p(a, shifted, 1.0) > 0.5
    lo, hi = wa.bootstrap_ci(a, shifted, 2000, rng)
    assert lo < a.mean() - shifted.mean() < hi < 0


def test_mutual_information_and_decoder():
    y = np.repeat(np.arange(4), 25)
    assert wa.mutual_information_bits(y, y, 4)[0] == pytest.approx(2.0)
    assert wa.mutual_information_bits(y, np.zeros_like(y), 4)[0] == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(1)
    centres = rng.normal(0, 5, (4, 6))
    x = centres[y] + rng.normal(0, 0.3, (y.size, 6))
    groups = np.tile(np.arange(5), 20)
    preds = wa.nearest_centroid_cv(x, y[None, :], groups, 4)
    assert (preds[0] == y).mean() == 1.0


# --- end to end: null, positive control, integrity -------------------------------------------------


def test_null_run_is_an_informative_null(null_run):
    rep = null_run.report
    assert rep.persistence_verdict.startswith("NULL")
    assert rep.channel_verdict.startswith("no information")
    assert rep.integrity_flags == ()
    assert all(d.p_holm >= FAST.alpha for d in rep.decoding)
    assert abs(np.mean([d.mi_bits for d in rep.decoding])) < 0.1


def test_positive_control_is_detected(positive_run):
    rep = positive_run.report
    assert rep.persistence_verdict.startswith("DETECTED")
    assert rep.channel_verdict.startswith("ANOMALOUS")
    assert rep.integrity_flags == ()
    active = {(w, lag): (acc, p) for w, lag, acc, _, p in rep.channel_map("active")}
    assert active[("post@300s", 0)][1] < FAST.alpha
    assert active[("pre", 0)][1] >= FAST.alpha  # nothing readable before the symbol is drawn
    assert all(p >= FAST.alpha for w, lag, _, _, p in rep.channel_map("sham"))
    msg = {m.arm: m for m in rep.messages}
    assert msg["active"].symbol_accuracy > 2 * msg["active"].chance and msg["active"].p_binomial < 0.01
    assert msg["sham"].p_binomial > 0.05


def test_short_memory_effect_is_classified_as_within_classical_horizon():
    res = wa.simulate_assay(FAST, SMALL_CB, injected_magnitude=0.05, injected_memory_s=15.0, seed=11)
    assert res.report.channel_verdict.startswith("information present only within the classical memory horizon")
    assert not res.report.persistence_verdict.startswith("DETECTED")


def test_label_leak_into_pre_window_is_flagged(positive_run):
    ds = positive_run.dataset
    leaked_alpha = ds.alpha_per_m.copy()
    leaked_alpha[:, 0, :] = ds.alpha_per_m[:, -1, :]
    leaked_t = ds.temperature_k.copy()
    leaked_t[:, 0] = ds.temperature_k[:, -1]
    leaked = dataclasses.replace(ds, alpha_per_m=leaked_alpha, temperature_k=leaked_t)
    analysis = wa.analyze(leaked, FAST, positive_run.thermal)
    report = wa.unblind(analysis, positive_run.key, FAST)
    assert any(f.startswith("RETRO") for f in report.integrity_flags)
    assert report.channel_verdict.startswith("INTEGRITY")


class _LeakySham(wa.SimulatedInstrument):
    """Sham path that still writes the symbol, e.g. an IR or electrical side channel."""

    def apply(self, action):
        if action.kind == "sham":
            super().apply(wa.DriveAction("thz", action.plan))
            self._heat_in_w = 0.0
        else:
            super().apply(action)


def test_information_in_sham_arm_is_flagged():
    cell, ill, prereg = wa.SampleCell(), wa.RingIllumination(), FAST
    inst = _LeakySham(cell, ill, injected=wa.InjectedEffect(SMALL_CB, 0.05), seed=5)
    arr = PhotonicTHzEmitterArray()
    thermal = wa.thermal_calibration(inst, cell, prereg.readout_frequencies_hz)
    key = wa.make_allocation_key(5)
    ds = wa.run_assay(inst, arr, cell, ill, SMALL_CB, prereg, wa.make_schedule(prereg, SMALL_CB.size, key, seed=5),
                      key, wa.RandomStream(6))
    report = wa.unblind(wa.analyze(ds, prereg, thermal), key, prereg)
    assert any("sham" in f for f in report.integrity_flags)
    assert not report.channel_verdict.startswith("ANOMALOUS")


def test_tampering_is_refused(null_run):
    with pytest.raises(ValueError):
        wa.analyze(null_run.dataset, dataclasses.replace(FAST, sesoi_rel_alpha=1e-2), null_run.thermal)
    with pytest.raises(ValueError):
        wa.unblind(null_run.analysis, wa.make_allocation_key(999), FAST)


def test_false_positive_rate_is_controlled():
    rates = wa.power_analysis([0.0], runs=6, prereg=FAST, codebook=SMALL_CB, seed=300)[0.0]
    assert rates["h1_rate"] <= 1 / 6 and rates["h2_rate"] <= 1 / 6


def test_report_serialises(positive_run):
    body = json.loads(positive_run.report.to_json())
    assert body["prereg_digest"] == FAST.digest()
    assert set(body["arm_codes"]) == set(wa.ARMS)
    assert "temporal" not in body  # schema guard: only the documented keys
    assert {"persistence_verdict", "channel_verdict", "decoding", "contrasts", "messages"} <= set(body)


def test_demo_runs_quickly():
    text = wa.demo("ok", 0.05)
    assert "NULL (no effect in nature)" in text and "POSITIVE CONTROL" in text
