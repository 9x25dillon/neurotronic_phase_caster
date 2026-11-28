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
SPOT_RING = cw.WriterCodebook.spot_ring()


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


def test_report_carries_all_three_neighbour_controls(ge_run):
    for acc, p in ((ge_run.retro_accuracy, ge_run.retro_p), (ge_run.forward_accuracy, ge_run.forward_p),
                   (ge_run.shuffled_accuracy, ge_run.shuffled_p)):
        assert p > 0.01
        assert abs(acc - ge_run.chance) < 0.06
    text = ge_run.summary()
    assert "retro k-1" in text and "forward k+1" in text and "shuffled" in text


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


def test_spot_ring_mode_writes_exact_messages_in_melt_growth():
    act = cw.make_actuator("Peltier", 8e-3, GE, SPOT_RING)
    for seed in range(3):
        rep, _ = cw.run_writer(GE, act, SPOT_RING, message="crystal", code=cw.HAMMING_GRAY, seed=seed,
                               n_permutations=50)
        assert rep.message_exact and rep.received_text == "crystal"
        assert rep.payload_symbols == cw.HAMMING_GRAY.n_symbols(56, SPOT_RING) == 49
        assert rep.layline_accuracy > 0.97


def test_required_full_scale_formula():
    r = cw.Readout()
    need = cw.required_full_scale_k(KDP, FULL, r, snr=5.0)
    noise = r.relative_noise / math.sqrt(r.lateral_samples * r.depth_samples_per_slot)
    assert need * KDP.contrast_per_kelvin() * 0.25 == pytest.approx(5.0 * noise)


def test_demo_runs():
    text = cw.demo("ok")
    assert "Peltier" in text and "THz photomixer array" in text


# --- channel coding ----------------------------------------------------------------------------


def test_codebook_presets():
    assert cw.CODEBOOKS["spot_ring_4"] == cw.WriterCodebook((0.5, 1.0), (0, 3))
    assert cw.CODEBOOKS["full_16"].size == 16 and cw.CODEBOOKS["spot_ring_4"].size == 4


@pytest.mark.parametrize("codebook", [FULL, SPOT_RING])
def test_gray_mapping_flips_one_bit_per_one_step_misread(codebook):
    words = cw.symbol_words(codebook, gray=True)
    assert sorted(words.tolist()) == list(range(codebook.size))
    nc = len(codebook.layline_charges)
    for s in range(codebook.size):
        for t in range(codebook.size):
            if abs(s // nc - t // nc) + abs(s % nc - t % nc) == 1:  # neighbouring level or neighbouring layline
                assert bin(int(words[s] ^ words[t])).count("1") == 1
    sym = np.arange(codebook.size)
    assert np.array_equal(cw.bits_to_symbols(cw.symbols_to_bits(sym, codebook), codebook), sym)


def test_natural_mapping_flips_two_bits_on_some_neighbour_misreads():
    words = cw.symbol_words(FULL, gray=False)
    assert bin(int(words[1] ^ words[2])).count("1") == 2  # layline 1 -> 2 at the same level


def test_bit_mapping_needs_power_of_two_axes():
    with pytest.raises(ValueError):
        cw.symbol_words(cw.WriterCodebook((0.3, 0.6, 1.0), (0, 3)))


def test_hamming_corrects_every_single_bit_error():
    import itertools
    data = np.array(list(itertools.product([0, 1], repeat=4))).ravel()
    code = cw.hamming74_encode(data)
    assert code.size == 7 * 16
    assert np.array_equal(cw.hamming74_decode(code)[0], data)
    for pos in range(7):
        bad = code.reshape(-1, 7).copy()
        bad[:, pos] ^= 1
        decoded, corrected = cw.hamming74_decode(bad.ravel())
        assert np.array_equal(decoded, data) and corrected == 16


@pytest.mark.parametrize("depth", [None, 1, 3, 4])
def test_interleaver_roundtrip(depth):
    bits = np.arange(7 * 11)
    assert np.array_equal(cw.deinterleave(cw.interleave(bits, 7, depth), 7, depth), bits)
    assert np.array_equal(cw.interleave(bits, 7, 1), bits)


@pytest.mark.parametrize("code", [cw.UNCODED, cw.HAMMING_GRAY, cw.ChannelCode(interleave_depth=1),
                                  cw.ChannelCode(fec="none", gray=True)])
@pytest.mark.parametrize("codebook", [FULL, SPOT_RING])
def test_channel_code_roundtrip(code, codebook):
    for text in ("crystal", "Ge:Ga", "\u00e9t\u00e9"):
        symbols = code.encode_text(text, codebook)
        assert symbols.max() < codebook.size
        assert code.decode_text(symbols, codebook, len(text.encode()))[0] == text


def test_uncoded_mapping_matches_original_text_mapping():
    from src.water_structure_assay import text_to_symbols
    for codebook in (FULL, SPOT_RING):
        assert cw.UNCODED.encode_text("crystal", codebook).tolist() == text_to_symbols("crystal", codebook.size)


@pytest.mark.parametrize("codebook", [FULL, SPOT_RING])
def test_interleaved_hamming_survives_any_single_band_misread(codebook):
    """Full interleaving puts each of a band's bits into a different codeword, so one wrong band, read as
    any other symbol, is always repaired."""
    code = cw.HAMMING_GRAY
    sent = code.encode_text("crystal", codebook)
    for k in range(sent.size):
        for wrong in range(codebook.size):
            if wrong == sent[k]:
                continue
            rx = sent.copy()
            rx[k] = wrong
            assert code.decode_text(rx, codebook, 7)[0] == "crystal"


def test_without_interleaving_a_diagonal_misread_can_break_a_codeword():
    code = cw.ChannelCode(interleave_depth=1)
    sent = code.encode_text("crystal", SPOT_RING)
    failures = 0
    for k in range(sent.size):
        rx = sent.copy()
        rx[k] = 3 - rx[k]  # both bits flip: other level and other layline
        failures += code.decode_text(rx, SPOT_RING, 7)[0] != "crystal"
    assert failures > 0


# --- neighbour controls ----------------------------------------------------------------------------


def _synthetic(shift, n=240, m=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, m, n)
    f = 0.3 * rng.standard_normal((n, m))
    src = np.roll(y, shift)  # band k carries symbol k - shift
    f[np.arange(n), src] += 1.0
    return y, f, np.repeat(np.arange(6), n // 6)


def test_neighbour_controls_detect_what_they_are_built_to_detect():
    clean = cw.temporal_controls(*_synthetic(0), n_permutations=200)
    assert clean.retro.p_value > 0.01 and clean.forward.p_value > 0.01 and clean.shuffled.p_value > 0.01
    carry = cw.temporal_controls(*_synthetic(1), n_permutations=200)  # band k+1 holds symbol k
    assert carry.forward.p_value < 0.01 and carry.retro.p_value > 0.01
    leak = cw.temporal_controls(*_synthetic(-1), n_permutations=200)  # band k-1 holds symbol k
    assert leak.retro.p_value < 0.01 and leak.forward.p_value > 0.01
    assert leak.shuffled.p_value > 0.01  # shuffling destroys any adjacency effect


# --- drift ----------------------------------------------------------------------------------


def test_growth_drift_shapes():
    ramp = cw.GrowthDrift(0.1)
    assert ramp.factor(np.array([0.0, 50.0, 100.0]), 100.0) == pytest.approx([0.9, 1.0, 1.1])
    sine = cw.GrowthDrift(0.2, period_s=40.0)
    assert sine.factor(np.array([10.0]), 100.0)[0] == pytest.approx(1.2)
    with pytest.raises(ValueError):
        cw.GrowthDrift(1.0)
    with pytest.raises(ValueError):
        cw.GrowthDrift(0.1, period_s=-1.0)


def test_drift_moves_the_record_and_the_clock_track_together():
    act = cw.make_actuator("x", 0.0, GE, FULL)
    sched = cw.make_schedule(FULL, [], n_train=2, slot_on_s=1.0, slot_off_s=1.0)
    rec = cw.write_crystal(GE, act, FULL, sched, drift=cw.GrowthDrift(0.1))
    assert rec.length_m == pytest.approx(GE.nominal_velocity_m_s * sched.duration_s, rel=1e-3)  # ramp averages out
    n = rec.t_s.size
    assert rec.interface_velocity_m_s[n // 10] < rec.interface_velocity_m_s[9 * n // 10]
    # undriven data track and the clock track's baseline (end of each slot, marker relaxed) drift together
    steps = int(round(sched.slot_s / (rec.t_s[1] - rec.t_s[0])))
    ends = np.arange(1, sched.n_slots + 1) * steps - 1
    assert np.allclose(rec.concentration_rel[ends, 0], rec.clock_concentration_rel[ends], rtol=1e-4)
    assert np.ptp(rec.clock_concentration_rel[ends]) > 0.01  # and the drift is visible in both


def _sweep(system, scale, codebook, code, drift, ref, seeds):
    act = cw.make_actuator("a", scale, system, codebook)
    out = []
    for s in seeds:
        rep, _ = cw.run_writer(system, act, codebook, message="crystal", code=code, drift=drift,
                               readout=cw.Readout(clock_reference=ref), seed=s, n_permutations=20)
        out.append(rep)
    return out


def _pooled(reps, codebook):
    n_tr = 6 * codebook.size
    return float(np.mean([(r.training_accuracy * n_tr + r.payload_accuracy * r.payload_symbols)
                          / (n_tr + r.payload_symbols) for r in reps]))


@pytest.mark.parametrize("amplitude", [0.1, 0.2])
@pytest.mark.parametrize("codebook,code", [(FULL, cw.UNCODED), (SPOT_RING, cw.HAMMING_GRAY)],
                         ids=["16-symbol-uncoded", "4-symbol-hamming"])
def test_drift_sweep_clock_reference(codebook, code, amplitude):
    """Ramped growth-rate drift on Ge:Ga: markers alone lose the level baseline, the clock-track reference
    restores it. Run on both codebooks."""
    seeds = range(8)
    drift = cw.GrowthDrift(amplitude)
    with_ref = _sweep(GE, 8e-3, codebook, code, drift, True, seeds)
    without = _sweep(GE, 8e-3, codebook, code, drift, False, seeds)
    a_ref, a_raw = _pooled(with_ref, codebook), _pooled(without, codebook)
    assert a_ref > a_raw + 0.2
    assert all(r.sync_ok and not r.stalled for r in with_ref)
    if codebook is SPOT_RING:
        exact = np.mean([r.message_exact for r in with_ref])
        if amplitude == 0.1:
            assert a_ref > 0.97 and exact >= 7 / 8
        else:
            assert a_ref > 0.9 and exact >= 2 / 8
        assert not any(r.message_exact for r in without)
    else:
        assert a_ref > (0.5 if amplitude == 0.1 else 0.35)


def test_oscillating_drift_needs_a_preamble_that_spans_its_period():
    """The reference removes the drifting baseline; the drift's change of response gain is learned only
    if the training preamble samples the whole drift cycle."""
    slot = 10 * GE.interface_thermal_time_s
    drift = cw.GrowthDrift(0.1, 60 * slot)
    act = cw.make_actuator("a", 8e-3, GE, SPOT_RING)

    def payload(n_train):
        return np.mean([cw.run_writer(GE, act, SPOT_RING, message="crystal", code=cw.HAMMING_GRAY, drift=drift,
                                      n_train=n_train, seed=s, n_permutations=20)[0].payload_accuracy
                        for s in range(6)])

    short, long = payload(6), payload(36)  # 24 bands (0.4 period) against 144 bands (2.4 periods)
    assert long > 0.98
    assert long > short + 0.05


def test_clock_reference_helps_solution_growth_under_strong_drift():
    ir = cw.ir_pump_full_scale_k(12e-3)
    seeds = range(8)
    with_ref = _sweep(KDP, ir, SPOT_RING, cw.HAMMING_GRAY, cw.GrowthDrift(0.2), True, seeds)
    without = _sweep(KDP, ir, SPOT_RING, cw.HAMMING_GRAY, cw.GrowthDrift(0.2), False, seeds)
    assert _pooled(with_ref, SPOT_RING) > _pooled(without, SPOT_RING)
