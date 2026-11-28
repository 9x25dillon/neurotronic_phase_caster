"""Regression and physics checks for src/photonic_thz_pulse_engine.py.

Run from the repository root:  python -m pytest -q tests
"""

import asyncio
import math

import numpy as np
import pytest

from src import photonic_thz_pulse_engine as pe
from src.thz_coherence_wearable_spec import EmitterArray, THzCoherenceWearable


# --- L0 numbers quoted in the design brief -------------------------------------------


def test_1thz_beat_needs_8nm_at_1550():
    dl = pe.wavelength_spacing_for_beat(1e12, 1550e-9)
    assert dl == pytest.approx(8.014e-9, rel=1e-3)
    small_signal = (1550e-9) ** 2 * 1e12 / pe.C0
    assert dl == pytest.approx(small_signal, rel=1e-4)


def test_1thz_ring_geometry():
    ring = pe.MicroringResonator.for_repetition_rate(1e12, group_index=2.0)
    assert ring.circumference_m == pytest.approx(149.9e-6, rel=1e-3)
    assert ring.radius_m == pytest.approx(23.86e-6, rel=1e-3)
    assert ring.fsr_hz == pytest.approx(1e12, rel=1e-12)


def test_carrier_frequency_at_1550():
    assert pe.wavelength_to_frequency(1550e-9) == pytest.approx(193.414e12, rel=1e-5)


# --- comb synthesis and pulse sharpening ---------------------------------------------


def test_two_line_beat_is_sinusoidal():
    m = pe.pulse_metrics(pe.make_comb(1e12, 2))
    assert m.repetition_rate_hz == pytest.approx(1e12)
    assert m.fwhm_s == pytest.approx(0.5e-12, rel=1e-3)
    assert m.peak_to_average == pytest.approx(2.0, rel=1e-9)
    assert m.modulation_depth == pytest.approx(1.0, abs=1e-9)
    t, p = pe.intensity_waveform(pe.make_comb(1e12, 2, total_power_w=1.0), 1, 512)
    expected = 1.0 + np.cos(2 * np.pi * 1e12 * t)
    assert np.max(np.abs(p - expected)) < 1e-9


@pytest.mark.parametrize("n", [2, 3, 4, 8, 16, 32, 64])
def test_locked_comb_matches_dirichlet_kernel(n):
    m = pe.pulse_metrics(pe.make_comb(1e12, n))
    assert m.peak_to_average == pytest.approx(n, rel=1e-9)
    assert m.fwhm_s == pytest.approx(pe.flat_comb_fwhm_s(n, 1e12), rel=1e-2)


def test_large_n_fwhm_asymptote():
    assert pe.flat_comb_fwhm_s(256, 1e12) == pytest.approx(0.8859e-12 / 256, rel=1e-3)


def test_fwhm_shrinks_monotonically_with_n():
    widths = [pe.pulse_metrics(pe.make_comb(1e12, n)).fwhm_s for n in (2, 4, 8, 16, 32)]
    assert all(a > b for a, b in zip(widths, widths[1:]))


def test_unlocked_lines_do_not_form_pulses():
    locked = pe.pulse_metrics(pe.make_comb(1e12, 32, phases="locked"))
    free = pe.pulse_metrics(pe.make_comb(1e12, 32, phases="random", seed=7))
    assert free.average_power_w == pytest.approx(locked.average_power_w)
    assert free.peak_to_average < 0.5 * locked.peak_to_average


def test_comb_state_is_immutable():
    comb = pe.make_comb(1e12, 4)
    with pytest.raises(ValueError):
        comb.amplitudes[0] = 0.0
    with pytest.raises(Exception):
        comb.spacing_hz = 2e12


def test_comb_validation():
    with pytest.raises(ValueError):
        pe.CombState(193e12, 0.0, [1.0])
    with pytest.raises(ValueError):
        pe.make_comb(1e12, 4, phases=[0.0, 1.0])


def test_power_harmonics_parseval():
    comb = pe.make_comb(1e12, 16, envelope="gaussian", phases="random", seed=3)
    t, p = pe.intensity_waveform(comb, 1, 1024)
    acf = pe.line_autocorrelation(comb.amplitudes)
    assert acf[0].real == pytest.approx(p.mean(), rel=1e-12)
    fft = np.fft.fft(p) / p.size
    assert np.allclose(np.abs(fft[1:16]), np.abs(acf[1:16]), atol=1e-12 * p.mean())


# --- heterodyne pair ---------------------------------------------------------------------


def test_heterodyne_modulation_depth_and_polarization():
    base = pe.HeterodynePair.symmetric(1e12, total_power_w=5e-3)
    assert base.modulation_depth == pytest.approx(1.0)
    imbalanced = pe.HeterodynePair(pe.LaserLine(1550e-9, 4e-3), pe.LaserLine(1558e-9, 1e-3))
    assert imbalanced.modulation_depth == pytest.approx(0.8)
    crossed = pe.HeterodynePair(pe.LaserLine(1550e-9, 1e-3), pe.LaserLine(1558e-9, 1e-3, polarization_rad=math.pi / 2))
    assert crossed.modulation_depth == pytest.approx(0.0, abs=1e-12)
    comb, dc = crossed.to_comb()
    assert dc == pytest.approx(1e-3)
    assert comb.total_power_w + dc == pytest.approx(crossed.mean_power_w)


def test_heterodyne_retune_and_linewidth():
    pair = pe.HeterodynePair.symmetric(1e12, linewidth_hz=200e3)
    moved = pair.retuned(0.85e12)
    assert moved.beat_frequency_hz == pytest.approx(0.85e12, rel=1e-9)
    assert moved.laser_a == pair.laser_a
    assert pair.beat_linewidth_hz == pytest.approx(400e3)


# --- microresonator ----------------------------------------------------------------------


def test_soliton_requires_anomalous_dispersion():
    ring = pe.MicroringResonator.for_repetition_rate(1e12, beta2_s2_per_m=+50e-27)
    with pytest.raises(ValueError):
        ring.soliton_tau_s(5e9)


def test_soliton_duration_and_comb():
    ring = pe.MicroringResonator.for_repetition_rate(1e12)
    fwhm = ring.soliton_fwhm_s(5e9)
    assert 10e-15 < fwhm < 40e-15
    m = pe.pulse_metrics(ring.soliton_comb(5e9))
    assert m.repetition_rate_hz == pytest.approx(1e12, rel=1e-9)
    assert m.fwhm_s == pytest.approx(fwhm, rel=0.05)


def test_microcomb_thermal_tuning_is_small():
    ring = pe.MicroringResonator.for_repetition_rate(1e12)
    assert abs(ring.fsr_thermal_coefficient_hz_per_k) < 20e6  # far below the 0.4 THz band


# --- repetition-rate multiplication ----------------------------------------------------------


@pytest.mark.parametrize("factor", [2, 3, 4, 10, 40])
def test_talbot_shaper_multiplies_rate_losslessly(factor):
    seed = pe.make_comb(25e9, 24 * factor, envelope="gaussian", envelope_fwhm_hz=24 * factor * 25e9 * 0.2775)
    out = pe.TalbotMultiplier(factor, implementation="shaper").apply(seed)
    samples = pe.default_samples_per_period(seed.n_lines, factor)
    m_in, m_out = pe.pulse_metrics(seed, samples), pe.pulse_metrics(out, samples)
    assert m_out.repetition_rate_hz == pytest.approx(factor * 25e9, rel=1e-9)
    assert m_out.peak_power_w == pytest.approx(m_in.peak_power_w / factor, rel=1e-2)
    assert out.total_power_w == pytest.approx(seed.total_power_w, rel=1e-12)
    assert m_out.subharmonic_suppression_db > 40.0


def test_talbot_fiber_quadratic_matches_shaper():
    seed = pe.make_comb(25e9, 120, envelope="gaussian", envelope_fwhm_hz=120 * 25e9 * 0.2775)
    fiber = pe.TalbotMultiplier(5, implementation="fiber").apply(seed, include_third_order=False)
    m = pe.pulse_metrics(fiber, pe.default_samples_per_period(120, 5))
    assert m.repetition_rate_hz == pytest.approx(125e9, rel=1e-9)
    assert m.subharmonic_suppression_db > 40.0


def test_talbot_fiber_length_and_validation():
    tal = pe.TalbotMultiplier(100, implementation="fiber")
    assert tal.required_gdd_s2(10e9) == pytest.approx((1e-10) ** 2 / (2 * np.pi * 100))
    assert tal.fiber_length_m(10e9) == pytest.approx(733.4, rel=1e-3)
    with pytest.raises(ValueError):
        pe.TalbotMultiplier(4, numerator=2)


def test_third_order_dispersion_breaks_fiber_talbot_at_thz_rates():
    seed, r = pe.design_seed_comb(1e12, 25e9, 0.1)
    samples = pe.default_samples_per_period(seed.n_lines, r)
    clean = pe.pulse_metrics(pe.TalbotMultiplier(r, implementation="shaper").apply(seed), samples)
    tod = pe.pulse_metrics(pe.TalbotMultiplier(r, implementation="fiber").apply(seed), samples)
    assert clean.subharmonic_suppression_db > 40.0
    assert tod.subharmonic_suppression_db < 10.0
    assert pe.TalbotMultiplier(r, implementation="fiber").third_order_phase_rad(seed.spacing_hz, 2.2e12) > 1.0


def test_design_seed_comb_hits_target_rate_and_duty():
    seed, r = pe.design_seed_comb(1.07e12, 25e9, 0.2)
    assert r == 43
    out = pe.TalbotMultiplier(r).apply(seed)
    m = pe.pulse_metrics(out, pe.default_samples_per_period(seed.n_lines, r))
    assert m.repetition_rate_hz == pytest.approx(1.07e12, rel=1e-9)
    assert m.duty_cycle == pytest.approx(0.2, rel=0.03)


def test_etalon_keeps_every_mth_line():
    seed = pe.make_comb(25e9, 161, envelope="gaussian")
    etalon = pe.FabryPerotEtalon.for_multiplication(25e9, 8, finesse=400.0)
    out = etalon.apply(seed)
    assert etalon.length_m == pytest.approx(pe.C0 / (2 * 200e9))
    m = pe.pulse_metrics(out)
    assert m.repetition_rate_hz == pytest.approx(200e9)
    assert out.total_power_w / seed.total_power_w == pytest.approx(1 / 8, rel=0.1)
    s = math.sin(math.pi / 8) ** 2
    assert etalon.adjacent_suppression_db(25e9) == pytest.approx(10 * math.log10(1 + (800 / math.pi) ** 2 * s))


# --- all-optical gates ----------------------------------------------------------------------


def _gaussian_pump(peak_w, fwhm_s, n=4096, window_s=4e-12):
    t = np.arange(n) * (window_s / n)
    return t, peak_w * np.exp(-4 * math.log(2) * ((t - window_s / 4) / fwhm_s) ** 2)


def test_kerr_gate_full_switching_at_pi_power():
    medium = pe.KerrMedium("instant", 2.7e-20, 1e-3)
    gate = pe.KerrGate(medium)
    t, pump = _gaussian_pump(gate.pi_peak_power_w, 100e-15)
    res = gate.switch(t, pump, periodic=False)
    assert res.peak_transmission == pytest.approx(1.0, abs=1e-4)
    assert res.lag_s == pytest.approx(0.0, abs=2e-15)
    # sin^2(pi g / 2) crosses 1/2 where g = 1/2, so the window equals the pump FWHM at full drive
    assert res.window_fwhm_s == pytest.approx(100e-15, rel=1e-2)
    assert res.contrast_db > 45.0


def test_kerr_gate_small_signal_window_is_pump_over_sqrt2():
    gate = pe.KerrGate(pe.KerrMedium("instant", 2.7e-20, 1e-3))
    t, pump = _gaussian_pump(0.05 * gate.pi_peak_power_w, 100e-15)
    res = gate.switch(t, pump, periodic=False)
    assert res.window_fwhm_s == pytest.approx(100e-15 / math.sqrt(2), rel=2e-2)


def test_kerr_gate_without_pump_only_leaks():
    gate = pe.KerrGate(pe.KERR_MEDIA["fused_silica_1mm"])
    t, pump = _gaussian_pump(0.0, 100e-15)
    res = gate.switch(t, pump)
    assert np.allclose(res.transmission, gate.polarizer_leakage)


def test_kerr_delayed_response_leaves_slow_tail():
    slow = pe.KerrGate(pe.KERR_MEDIA["cs2_1mm"])
    fast = pe.KerrGate(pe.KerrMedium("instant", slow.medium.n2_m2_per_w, slow.medium.length_m))
    t, pump = _gaussian_pump(0.3 * slow.pi_peak_power_w, 100e-15)
    r_slow, r_fast = slow.switch(t, pump, periodic=False), fast.switch(t, pump, periodic=False)
    after = int(np.argmax(pump)) + int(1e-12 / (t[1] - t[0]))
    assert r_slow.lag_s > 0.0
    assert r_slow.transmission[after] > 10 * slow.polarizer_leakage
    assert r_fast.transmission[after] == pytest.approx(fast.polarizer_leakage, rel=1e-6)


def test_kerr_damage_flag():
    gate = pe.KerrGate(pe.KERR_MEDIA["fused_silica_1mm"])
    t, pump = _gaussian_pump(1e3 * gate.pi_peak_power_w, 100e-15)
    assert gate.switch(t, pump).damage_risk


def test_saturable_absorber_opens_and_recovers():
    gate = pe.SaturableAbsorberGate(pe.SATURABLE_ABSORBERS["graphene_monolayer"])
    t, pump = _gaussian_pump(2e3, 100e-15, n=8192, window_s=12e-12)
    res = gate.switch(t, pump, periodic=False)
    ab = gate.absorber
    small_signal = (1 - ab.modulation_depth) * (1 - ab.nonsaturable_loss)
    assert res.transmission[0] == pytest.approx(small_signal, rel=1e-9)
    assert res.peak_transmission > small_signal + 0.25 * ab.modulation_depth
    q = gate.loss_trace(t, pump, periodic=False)
    tail = slice(int(0.6 * t.size), int(0.9 * t.size))
    y = np.log(ab.modulation_depth - q[tail])
    slope = np.polyfit(t[tail], y, 1)[0]
    assert -1 / slope == pytest.approx(ab.recovery_time_s, rel=1e-3)


def test_lowpass_periodic_steady_state():
    x = np.zeros(1000)
    x[:100] = 1.0
    y = pe._first_order_lowpass(x, 1e-15, 50e-15, periodic=True)
    y_next = pe._first_order_lowpass(np.tile(x, 2), 1e-15, 50e-15, periodic=True)[1000:]
    assert np.allclose(y, y_next, atol=1e-12)


# --- converters -----------------------------------------------------------------------------


def test_photomixer_matches_closed_form():
    pm = pe.PHOTOMIXERS["utc_pd_inp_1550"]
    assert float(pm.rolloff(0.0)) == pytest.approx(1.0)
    comb, dc = pe.HeterodynePair.symmetric(1e12, total_power_w=50e-3).to_comb()
    out = pm.response_to_comb(comb, dc)
    i_dc = pm.responsivity_a_per_w * 50e-3
    expected = 0.5 * pm.antenna_resistance_ohm * i_dc**2 * float(pm.rolloff(1e12))
    assert out.dc_photocurrent_a == pytest.approx(i_dc)
    assert out.fundamental_power_w == pytest.approx(expected, rel=1e-12)
    assert not out.over_current


def test_photomixer_inverse_and_ceiling():
    pm = pe.PHOTOMIXERS["utc_pd_inp_1550"]
    p_max = float(pm.max_power_w(1e12))
    assert 5e-6 < p_max < 50e-6  # literature: >10 uW at 1 THz for UTC-PDs
    p_opt = pm.optical_power_for(p_max, 1e12)
    assert p_opt * pm.responsivity_a_per_w == pytest.approx(pm.max_photocurrent_a, rel=1e-12)
    band = pm.max_power_w(np.array([0.8e12, 1.0e12, 1.2e12]))
    assert band[0] > band[1] > band[2]


def test_pulse_train_fundamental_exceeds_sinusoid_at_equal_photocurrent():
    pm = pe.PHOTOMIXERS["utc_pd_inp_1550"]
    seed, r = pe.design_seed_comb(1e12, 25e9, 0.1)
    train = pe.TalbotMultiplier(r).apply(seed).scaled_to_power(50e-3)
    beat, dc = pe.HeterodynePair.symmetric(1e12, total_power_w=50e-3).to_comb()
    ratio = pm.response_to_comb(train).fundamental_power_w / pm.response_to_comb(beat, dc).fundamental_power_w
    assert 3.0 < ratio <= 4.0


def test_optical_rectification_limits():
    rect = pe.OpticalRectifier(pe.CRYSTALS["gap_1550nm"])
    lossless = pe.OpticalRectifier(pe.NonlinearCrystal("x", 1e-12, 3.0, 3.1, 3.3, 0.0, 1550e-9), 1e-3)
    f = np.array([0.5e12, 1e12, 2e12])
    dk = lossless.delta_k(f)
    assert np.allclose(lossless.effective_length_sq(f), (2 * np.sin(dk * 1e-3 / 2) / dk) ** 2)
    assert float(rect.coherence_length_m(1e12)) == pytest.approx(pe.C0 / (2e12 * (3.34 - 3.13)), rel=1e-9)
    comb, _ = pe.HeterodynePair.symmetric(1e12, total_power_w=1.0).to_comb()
    _, p = rect.response_to_comb(comb)
    assert p.sum() < 1e-9  # CW optical rectification is negligible next to photomixing
    ln = pe.OpticalRectifier(pe.CRYSTALS["linbo3_800nm"])
    assert math.degrees(ln.tilted_pulse_front_angle_rad) == pytest.approx(63.0, abs=1.0)


# --- tissue and exposure ----------------------------------------------------------------------


def test_water_absorption_table_and_extrapolation():
    assert float(pe.water_absorption_per_m(1e12)) == pytest.approx(220.3e2)
    assert float(pe.water_absorption_per_m(0.5e12)) == pytest.approx(157.8e2)
    a = pe.water_absorption_per_m(np.array([0.3e12, 0.8e12, 1.2e12, 3e12]))
    assert np.all(np.diff(a) > 0)


def test_tissue_attenuation_is_monotone_and_extreme():
    layers = pe.tissue_transmission(1e12)
    db = [l.attenuation_at_bottom_db for l in layers]
    assert all(a < b for a, b in zip(db, db[1:]))
    assert layers[0].penetration_depth_m == pytest.approx(1 / (0.7 * 220.3e2))
    assert db[-1] > 300.0
    z = np.linspace(0, 17e-3, 341)
    prof = pe.attenuation_profile_db(1e12, z)
    assert np.all(np.diff(prof) >= 0)
    assert prof[-1] == pytest.approx(db[-1], rel=1e-9)


def test_skin_exposure_limits():
    assert pe.skin_exposure_limit_w_per_m2(10.0) == pytest.approx(1000.0)
    assert pe.skin_exposure_limit_w_per_m2(9.999) * 9.999 == pytest.approx(5600 * 9.999**0.25)
    assert pe.skin_exposure_limit_w_per_m2(100.0, irradiated_area_m2=0.05) == pytest.approx(200.0)
    assert pe.skin_exposure_limit_w_per_m2(100.0, irradiated_area_m2=0.5) == pytest.approx(100.0)
    assert pe.assess_skin_exposure(5e-3, 1e-3, 60.0).compliant  # averaged over the 3.5 mm aperture
    assert not pe.assess_skin_exposure(20e-3, 1e-3, 60.0).compliant
    with pytest.raises(ValueError):
        pe.skin_exposure_limit_w_per_m2(0.0)


# --- array backend and controller integration ---------------------------------------------------------


def _cmd(**kw):
    base = {"frequency_thz": 1.0, "amplitude_normalized": 0.25, "duration_ms": 120.0,
            "pattern": "continuous", "emitters": [0, 3, 6, 9], "phase_offset_rad": math.pi}
    base.update(kw)
    return base


def test_array_is_emitter_array_and_clips_to_physics():
    arr = pe.PhotonicTHzEmitterArray()
    assert isinstance(arr, EmitterArray)
    assert arr.emitters[0].max_power_mw == pytest.approx(arr.channel_max_power_w() * 1e3)
    arr.set_power(0, 5.0)
    assert arr.emitters[0].current_power_mw == pytest.approx(arr.emitters[0].max_power_mw)
    assert arr.power_shortfall_mw()[0] == pytest.approx(5.0 - arr.emitters[0].max_power_mw)


def test_array_frequency_range_enforced():
    arr = pe.PhotonicTHzEmitterArray()
    with pytest.raises(ValueError):
        arr.set_frequency(1.5e12)
    with pytest.raises(ValueError):
        arr.plan(_cmd(frequency_thz=0.5))
    fa = arr.pair.laser_a.frequency_hz
    arr.set_frequency(0.9e12)
    assert arr.emitters[5].frequency_thz == pytest.approx(0.9)
    assert arr.pair.laser_a.frequency_hz == fa  # reference laser untouched
    assert arr.pair.laser_b.frequency_hz == pytest.approx(fa + 0.9e12, rel=1e-12)


def test_phase_to_delay_mapping():
    arr = pe.PhotonicTHzEmitterArray()
    arr.set_phase(2, math.pi)
    assert arr.optical_delay_s(2) == pytest.approx(0.5e-12)
    assert arr.fiber_stretch_m(2) == pytest.approx(0.5e-12 * pe.C0 / 1.4682)
    assert arr.thermal_phase_drift_rad_per_k(1.0) == pytest.approx(2 * math.pi * 1e12 * 1e-5 / pe.C0)


def test_plan_continuous_amplitude_mapping():
    arr = pe.PhotonicTHzEmitterArray()
    plan = arr.plan(_cmd())
    on = [c for c in plan.channels if c.enabled]
    assert [c.channel_id for c in on] == [0, 3, 6, 9]
    assert on[0].thz_fundamental_w == pytest.approx(0.25 * arr.channel_max_power_w(), rel=1e-9)
    assert on[0].attenuation_db == pytest.approx(10 * math.log10(2.0))  # sqrt(0.25) of full photocurrent
    assert on[0].optical_delay_s == pytest.approx(0.5e-12)
    assert plan.carrier.details["wavelength_separation_nm"] == pytest.approx(8.01, abs=0.02)
    assert plan.safe and plan.exposure.fraction_of_limit < 0.01
    mcu = plan.to_mcu_command()
    assert mcu["bias_enable"].count(True) == 4 and mcu["interlock_ok"]


@pytest.mark.parametrize("method,kwargs", [
    ("talbot_shaper", {}),
    ("etalon", {"etalon_finesse": 2000.0}),
    ("microcomb", {}),
])
def test_plan_pulsed_methods(method, kwargs):
    arr = pe.PhotonicTHzEmitterArray(pulsed_method=method, **kwargs)
    plan = arr.plan(_cmd(pattern="pulsed", amplitude_normalized=1.0))
    assert plan.carrier.metrics.repetition_rate_hz == pytest.approx(1e12, rel=1e-6)
    assert plan.carrier.metrics.subharmonic_suppression_db > pe.MIN_SUBHARMONIC_SUPPRESSION_DB
    assert plan.safe
    assert arr.apply_plan(plan)
    assert arr.get_total_power() == pytest.approx(plan.total_thz_power_w * 1e3, rel=1e-9)


@pytest.mark.parametrize("method,kwargs", [("talbot_fiber", {}), ("etalon", {"etalon_finesse": 400.0})])
def test_plan_flags_unclean_trains(method, kwargs):
    plan = pe.PhotonicTHzEmitterArray(pulsed_method=method, **kwargs).plan(_cmd(pattern="pulsed"))
    assert any("not a clean" in w for w in plan.warnings)


def test_plan_accepts_helical_phase_pattern():
    arr = pe.PhotonicTHzEmitterArray()
    pattern = [2 * math.pi * 2 * n / 12 for n in range(12)]
    plan = arr.plan(_cmd(emitters=list(range(12)), phase_offset_rad=0.0, phase_pattern_rad=pattern))
    phases = [c.thz_phase_rad for c in plan.channels]
    assert np.allclose(phases, np.mod(pattern, 2 * math.pi))
    assert plan.channels[3].optical_delay_s == pytest.approx(phases[3] / (2 * math.pi * 1e12))
    with pytest.raises(ValueError):
        arr.plan(_cmd(phase_pattern_rad=[0.0] * 5))


def test_plan_zero_amplitude_disables_everything():
    plan = pe.PhotonicTHzEmitterArray().plan(_cmd(amplitude_normalized=0.0))
    assert not any(c.enabled for c in plan.channels)
    assert plan.amplifier_output_w == 0.0 and plan.total_thz_power_w == 0.0


def test_apply_plan_refuses_unsafe_plan():
    arr = pe.PhotonicTHzEmitterArray(beam_diameter_at_skin_m=3.5e-3)
    plan = arr.plan(_cmd(amplitude_normalized=1.0))
    unsafe = pe.PhotonicDrivePlan(plan.command, plan.carrier, plan.envelope, plan.channels, plan.amplifier_output_w,
                                  plan.total_thz_power_w, pe.assess_skin_exposure(1.0, 1e-3, 60.0), plan.tissue,
                                  plan.warnings)
    assert not arr.apply_plan(unsafe)
    assert arr.get_total_power() == 0.0


def test_wearable_spec_accepts_photonic_backend():
    wearable = THzCoherenceWearable(emitters=pe.PhotonicTHzEmitterArray())
    assert wearable.start_session(10.0)
    assert wearable.apply_substrate_pattern(np.array([3.0, 1.0, 2.5, 1.5, 4.0]))
    assert wearable.safety.is_safe_to_operate()
    assert wearable.get_status()["total_power_mw"] <= 12 * wearable.emitters.emitters[0].max_power_mw + 1e-12


def test_realtime_controller_uses_backend():
    from src.wearable_controller import NeurotronicPhaseCasterWearable, THzPulseParameters

    backend = pe.PhotonicTHzEmitterArray()
    controller = NeurotronicPhaseCasterWearable(emitter_backend=backend)
    pulse = THzPulseParameters(frequency=1.1, amplitude=0.6, duration=150.0, pattern="pulsed", target_emitters=[0, 1, 2])
    asyncio.run(controller._emit_thz_pulse(pulse))
    plan = controller.last_drive_plan
    assert plan is not None and backend.last_plan is plan
    assert plan.carrier.frequency_hz == pytest.approx(1.1e12, rel=1e-9)
    assert sum(e.enabled for e in backend.emitters) == 3


def test_demo_runs():
    text = pe.demo()
    assert "8.014 nm" in text and "plan 'pulsed'" in text
