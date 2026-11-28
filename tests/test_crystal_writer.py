"""Physics and decoding checks for src/crystal_writer.py."""

import math

import numpy as np
import pytest

from src import crystal_writer as cw
from src.water_structure_assay import RingIllumination, SampleCell, SymbolCodebook
from src.photonic_thz_pulse_engine import PhotonicTHzEmitterArray

GE = cw.GROWTH_SYSTEMS["ge_ga_czochralski"]
KDP = cw.GROWTH_SYSTEMS["kdp_aqueous_solution"]
FULL = cw.WriterCodebook()
SPOT_RING = cw.WriterCodebook(levels=(0.5, 1.0), layline_charges=(0, 3))


# --- growth physics -----------------------------------------------------------------------


def test_bps_limits_and_monotonicity():
    v = np.logspace(-9, -4, 50)  # physical growth rates
    k = GE.k_eff(v)
    assert GE.k_eff(0.0) == pytest.approx(GE.k0)
    assert GE.k_eff(1e-2) == pytest.approx(1.0, abs=1e-6)  # fast growth traps everything
    assert np.all(np.diff(k) > 0)


@pytest.mark.parametrize("system", [GE, KDP])
def test_uptake_sensitivity_matches_numerical_derivative(system):
    v0 = system.nominal_velocity_m_s
    h = 1e-6
    numeric = (math.log(system.k_eff(v0 * (1 + h))) - math.log(system.k_eff(v0 * (1 - h)))) / (2 * h)
    assert system.dlnk_dlnv == pytest.approx(numeric, rel=1e-5)


def test_melt_growth_responds_to_temperature_rate():
    dt = 0.01
    steady = np.full((100, 3), 0.005)
    assert np.allclose(GE.velocity(steady, dt), GE.nominal_velocity_m_s)
    ramp = (np.arange(100) * dt * 0.01)[:, None] * np.ones((1, 3))  # 10 mK/s
    v = GE.velocity(ramp, dt)
    assert np.allclose(v[1:-1], GE.nominal_velocity_m_s - 0.01 / GE.thermal_gradient_k_m)


def test_solution_growth_responds_to_temperature_level():
    v = KDP.velocity(np.array([[0.0], [KDP.stall_limit_k()]]), 1.0)
    assert v[0, 0] == pytest.approx(KDP.nominal_velocity_m_s)
    assert v[1, 0] == pytest.approx(0.0, abs=1e-15)
    small = 1e-4
    dlnv = math.log(KDP.velocity(np.array([[small]]), 1.0)[0, 0] / KDP.nominal_velocity_m_s) / small
    assert abs(dlnv) * KDP.dlnk_dlnv == pytest.approx(KDP.contrast_per_kelvin(), rel=1e-3)


def test_stall_limits():
    assert GE.stall_limit_k() == pytest.approx(3000 * 2e-5 * 0.2)
    assert KDP.stall_limit_k() == pytest.approx(math.log(1.05) / 0.02)


def test_record_lifetime_and_arrhenius():
    d300, d600 = GE.solid_diffusivity(300.0), GE.solid_diffusivity(600.0)
    assert d600 / d300 == pytest.approx(math.exp(GE.solid_ea_ev / cw.K_B_EV * (1 / 300 - 1 / 600)))
    assert GE.record_lifetime_s(40e-6, 300.0) == pytest.approx((40e-6) ** 2 / (4 * math.pi**2 * d300))
    assert GE.record_lifetime_s(40e-6, 300.0) > 3.15e7 * 1e6  # far beyond a million years at room temperature


def test_growth_system_validation():
    with pytest.raises(ValueError):
        cw.GrowthSystem("x", "plasma", 300, 0.1, 1e-9, 1e-4, 1e-7, 1e-6, 1.0, 1.0, 1e-4)
    with pytest.raises(ValueError):
        cw.GrowthSystem("x", "melt", 300, 0.1, 1e-9, 1e-4, 1e-7, 1e-6, 1.0, 1.0, 1e-4)  # no gradient


# --- codebook, actuator ----------------------------------------------------------------------


def test_layline_profiles_are_spot_and_rings():
    act = cw.make_actuator("x", 1.0, KDP, FULL)
    centre = act.x_m.size // 2
    assert np.allclose(act.profiles.max(axis=1), 1.0)
    assert np.argmax(act.profiles[0]) == centre  # charge 0: central spot
    assert act.profiles[3, centre] < 0.6 * act.profiles[3].max()  # charge 3: ring with a central dip


def test_frequency_nodes_map_to_levels_through_absorbed_power():
    cb = cw.WriterCodebook.from_frequency_nodes(SymbolCodebook(), PhotonicTHzEmitterArray(), RingIllumination(),
                                                SampleCell())
    assert cb.levels[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(cb.levels, cb.levels[1:]))  # photomixer roll-off with frequency


def test_actuator_strengths():
    assert 1e-3 < cw.photonic_full_scale_k() < 0.1  # tens of mK at most from microwatt THz
    assert cw.ir_pump_full_scale_k(12e-3) == pytest.approx(12e-3 * cw.SPOT_THERMAL_RESISTANCE_K_PER_W)


# --- writing and reading -----------------------------------------------------------------------


def test_schedule_is_iid_preamble_then_payload():
    sched = cw.make_schedule(FULL, [1, 2, 3], n_train=5, seed=1)
    assert sched.n_slots == 5 * 16 + 3
    assert sched.train_group[:16] == (0,) * 16 and sched.train_group[-3:] == (-1, -1, -1)
    assert sched.symbols[-3:] == (1, 2, 3)
    with pytest.raises(ValueError):
        cw.make_schedule(FULL, [16])
    with pytest.raises(ValueError):
        cw.make_schedule(FULL, [], n_train=1)


def test_undriven_crystal_is_uniform_and_markers_are_present():
    act = cw.make_actuator("x", 0.0, GE, FULL)
    sched = cw.make_schedule(FULL, [], n_train=2, slot_on_s=1.0, slot_off_s=1.0)
    rec = cw.write_crystal(GE, act, FULL, sched)
    assert np.allclose(rec.concentration_rel, 1.0)
    assert rec.length_m == pytest.approx(GE.nominal_velocity_m_s * sched.duration_s, rel=1e-6)
    assert rec.clock_concentration_rel.min() < 0.95  # marker striations exist on the clock track
    assert not rec.stalled


def test_overdrive_stalls_the_interface():
    act = cw.make_actuator("x", 3.0 * GE.stall_limit_k(), GE, FULL)
    rec = cw.write_crystal(GE, act, FULL, cw.make_schedule(FULL, [], n_train=2, slot_on_s=1.0, slot_off_s=1.0))
    assert rec.stalled


def test_markers_locate_slot_boundaries():
    act = cw.make_actuator("x", 8e-3, GE, FULL)
    sched = cw.make_schedule(FULL, [], n_train=2, slot_on_s=1.0, slot_off_s=1.0, seed=2)
    rec = cw.write_crystal(GE, act, FULL, sched)
    z, _, clock = rec.section()
    bounds, ok = cw.locate_markers(z, clock, sched.n_slots, 0.5 * GE.nominal_velocity_m_s * GE.interface_thermal_time_s)
    steps = int(round(sched.slot_s / (rec.t_s[1] - rec.t_s[0])))
    true = rec.depth_m[::steps][: sched.n_slots]
    assert ok
    assert np.max(np.abs(bounds[1:-1] - true[1:])) < 0.1 * GE.nominal_velocity_m_s * sched.slot_s


def test_anneal_erases_striations():
    z = np.linspace(0, 400e-6, 4001)
    c = 1.0 + 0.1 * np.sin(2 * np.pi * z / 40e-6)
    tau = GE.record_lifetime_s(40e-6, 1200.0)  # ~1.4 days near the melting point
    hot = cw.anneal(z, c[:, None], GE, 1200.0, 3.0 * tau)[:, 0]
    cold = cw.anneal(z, c[:, None], GE, 300.0, 3.15e9)[:, 0]  # a century at room temperature
    assert np.ptp(hot[500:-500]) == pytest.approx(np.ptp(c) * math.exp(-3.0), rel=0.05)
    assert np.ptp(cold) == pytest.approx(np.ptp(c), rel=1e-6)


@pytest.fixture(scope="module")
def ge_run():
    act = cw.make_actuator("Peltier", 8e-3, GE, FULL)
    return cw.run_writer(GE, act, FULL, message="crystal", seed=3, n_permutations=300)[0]


def test_melt_grown_crystal_carries_a_message(ge_run):
    assert not ge_run.stalled and ge_run.sync_ok
    assert ge_run.payload_accuracy > 0.6 and ge_run.payload_p < 1e-6
    assert ge_run.training_p < 0.01 and ge_run.training_excess_bits > 1.0
    assert ge_run.retro_p > 0.01  # nothing from the future


def test_photomixer_array_alone_cannot_write_solution_growth():
    act = cw.make_actuator("THz", cw.photonic_full_scale_k(), KDP, FULL)
    rep, _ = cw.run_writer(KDP, act, FULL, message="hi", seed=3, n_permutations=300)
    assert rep.training_p > 0.05
    assert rep.required_full_scale_k > 10 * rep.full_scale_k


def test_spot_ring_layline_survives_in_solution_growth():
    act = cw.make_actuator("IR", cw.ir_pump_full_scale_k(12e-3), KDP, SPOT_RING)
    rep, _ = cw.run_writer(KDP, act, SPOT_RING, message="hi", seed=3, n_permutations=300)
    assert not rep.stalled
    assert rep.layline_accuracy > 0.9
    assert rep.payload_p < 0.01


def test_required_full_scale_formula():
    r = cw.Readout()
    need = cw.required_full_scale_k(KDP, FULL, r, snr=5.0)
    noise = r.relative_noise / math.sqrt(r.lateral_samples * r.depth_samples_per_slot)
    assert need * KDP.contrast_per_kelvin() * 0.25 == pytest.approx(5.0 * noise)


def test_demo_runs():
    text = cw.demo("ok")
    assert "Peltier" in text and "THz photomixer array" in text
