"""Known-value checks for src/order_records.py: each operational definition against L0 physics."""

import math

import numpy as np
import pytest

from src import order_records as o


# --- geometric absence in reciprocal space -----------------------------------------------


def test_ice_ih_systematic_absences_match_space_group_rule():
    absent = set(o.systematic_absences(3))
    predicted = {(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)
                 if (h, k, l) != (0, 0, 0) and o.ice_ih_absence_rule(h, k, l)}
    assert absent == predicted
    assert (0, 0, 1) in absent and (1, 1, 1) in absent  # 6_3 screw axis and c-glide
    assert (0, 0, 2) not in absent and (1, 0, 0) not in absent


@pytest.mark.parametrize("z", [0.0625, 0.03, 0.11, 0.2])
def test_absences_hold_for_any_free_coordinate(z):
    frac = o.ice_ih_oxygen_fractional(z)
    assert abs(o.structure_factor((0, 0, 1), frac)) < 1e-12
    assert abs(o.structure_factor((2, -1, 3), frac)) < 1e-12  # h - k = 3, l odd: 4f special condition
    assert abs(o.structure_factor((0, 0, 2), frac)) == pytest.approx(4 * abs(math.cos(4 * math.pi * z)), abs=1e-12)


def test_pauling_entropy():
    assert o.pauling_residual_entropy() == pytest.approx(3.371, abs=1e-3)


# --- local order -------------------------------------------------------------------------


def test_ice_supercell_geometry():
    cell = o.HexagonalCell(4.497e-10, 4.497e-10 * o.ICE_IH_IDEAL_C_OVER_A)
    pos, box = o.ice_ih_supercell(3, 2, 2, cell)
    assert pos.shape == (3 * 2 * 2 * 8, 3)
    assert np.all((pos >= 0) & (pos < box))
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pos, boxsize=box).query(pos, k=5)
    assert np.allclose(d[:, 1:], 3 * cell.c_m / 8, rtol=1e-9)  # four equal O-O bonds in ideal ice


def test_tetrahedral_order_limits():
    cell = o.HexagonalCell(4.497e-10, 4.497e-10 * o.ICE_IH_IDEAL_C_OVER_A)
    pos, box = o.ice_ih_supercell(4, 3, 3, cell)
    q_ideal = o.tetrahedral_order(pos, box)
    assert np.allclose(q_ideal, 1.0, atol=1e-9)
    real_pos, real_box = o.ice_ih_supercell(4, 3, 3, o.HexagonalCell())
    assert o.tetrahedral_order(real_pos, real_box).mean() > 0.99
    gas = np.random.default_rng(1).uniform(0, 1, (5000, 3)) * box
    q_gas = o.tetrahedral_order(gas, box)
    assert abs(q_gas.mean()) < 0.05
    assert q_gas.max() <= 1.0 + 1e-12


# --- coarse-grained field and order deficit -------------------------------------------------


def test_coarse_grain_and_deficit():
    rng = np.random.default_rng(0)
    box = np.array([10.0, 10.0])
    pts = rng.uniform(0, 10, (4000, 2))
    const = o.coarse_grain(pts, np.full(4000, 0.8), box, 32, 0.8)
    assert np.allclose(const, 0.8, atol=1e-9)
    values = np.where(pts[:, 0] < 5, 1.0, 0.0)
    m = o.coarse_grain(pts, values, box, 32, 0.5)
    deficit = o.order_deficit(m, 1.0)  # grid axes follow coordinate order: first index is x
    assert deficit[8, :].mean() < 0.05 and deficit[24, :].mean() > 0.95
    assert m.shape == (32, 32)


# --- topological defects ------------------------------------------------------------------


def test_perfect_triangular_lattice_has_no_defects():
    lat = o.triangular_lattice(20)
    d = o.hexatic_defects(lat, 2.5)
    assert d.defect_indices.size == 0
    assert np.allclose(np.abs(d.psi6[d.interior]), 1.0, atol=1e-9)


def test_vacancy_is_a_neutral_defect_cluster():
    lat = o.triangular_lattice(20)
    removed = 210
    vac = np.delete(lat, removed, axis=0)
    d = o.hexatic_defects(vac, 2.5)
    assert d.defect_indices.size > 0
    assert d.net_charge == 0  # absence of one site carries no net disclination charge
    near = np.linalg.norm(vac - lat[removed], axis=1) < 1.01
    assert np.abs(d.psi6[near]).mean() < 0.9  # local order deficit around the absent site


# --- phase ordering and the geometric clock --------------------------------------------------


def test_model_b_conserves_order_parameter_and_model_a_does_not_need_to():
    snaps_b = o.simulate_phase_ordering("B", [200.0], n=64, seed=1)
    assert abs(snaps_b[200.0].mean()) < 1e-10
    snaps_a = o.simulate_phase_ordering("A", [200.0], n=64, seed=1)
    assert np.abs(snaps_a[200.0]).mean() > 0.8  # domains have formed
    with pytest.raises(ValueError):
        o.simulate_phase_ordering("C", [1.0])


@pytest.fixture(scope="module")
def clocks():
    return {  # model A coarsens faster, so it needs a larger grid to hold enough domains
        "A": o.CoarseningClock.calibrate("A", (25, 50, 100, 200), n=256),
        "B": o.CoarseningClock.calibrate("B", (100, 200, 400, 800)),
    }


def test_growth_exponents_distinguish_conservation_laws(clocks):
    ea, _ = clocks["A"].exponent
    eb, _ = clocks["B"].exponent
    assert 0.35 < ea < 0.55  # asymptote 1/2, finite-time effective value lower
    assert 0.27 < eb < 0.38  # asymptote 1/3
    assert ea - eb > 0.04


@pytest.mark.parametrize("model,t,n", [("A", 100.0, 256), ("B", 400.0, 128)])
def test_clock_reads_elapsed_time_from_frozen_snapshot(clocks, model, t, n):
    readings = [clocks[model].read(o.simulate_phase_ordering(model, [t], n=n, seed=s)[t]) for s in range(20, 26)]
    vouched = [r for r in readings if r.reliable]  # the clock declines snapshots with too few domains
    assert len(vouched) >= 4
    assert sum(r.low <= t <= r.high for r in vouched) >= len(vouched) - 1
    assert np.median([abs(r.estimate / t - 1.0) for r in vouched]) < 0.25


def test_clock_flags_undersampled_snapshots(clocks):
    phi = o.simulate_phase_ordering("A", [200.0], n=64, seed=1)[200.0]
    assert not clocks["A"].read(phi).reliable


def test_frozen_configuration_is_a_record_of_elapsed_time():
    past, feats = [], []
    for seed in range(8):
        snaps = o.simulate_phase_ordering("A", [50.0, 100.0, 200.0], seed=100 + seed)
        for cls, t in enumerate((50.0, 100.0, 200.0)):
            past.append(cls)
            feats.append([math.log(o.coarsening_length(snaps[t]))])
    rec = o.record_fidelity(np.array(past), np.array(feats), n_permutations=500)
    assert rec.accuracy > 0.8 and rec.p_value < 0.01 and rec.excess_bits > 0.5


# --- fractal scaling with diagnostics ------------------------------------------------------------


def test_box_counting_known_sets():
    square = np.ones((256, 256), dtype=bool)
    r = o.box_counting_dimension(square)
    assert r.dimension == pytest.approx(2.0, abs=1e-9) and not r.approximately_fractal
    line = np.zeros((256, 256), dtype=bool)
    line[128, :] = True
    r = o.box_counting_dimension(line)
    assert r.dimension == pytest.approx(1.0, abs=1e-9) and not r.approximately_fractal
    carpet = o.box_counting_dimension(o.sierpinski_carpet(5), box_sizes=(1, 3, 9, 27))
    assert carpet.dimension == pytest.approx(math.log(8) / math.log(3), abs=1e-9)
    assert carpet.approximately_fractal
    with pytest.raises(ValueError):
        o.box_counting_dimension(square, box_sizes=(2, 4))


def test_coarsening_interfaces_are_not_fractal():
    phi = o.simulate_phase_ordering("A", [200.0], n=256, seed=3)[200.0]
    s = np.sign(phi)
    interface = (s != np.roll(s, 1, 0)) | (s != np.roll(s, 1, 1))
    r = o.box_counting_dimension(interface)
    assert r.local_slopes[0] == pytest.approx(1.0, abs=0.1)  # smooth curves below the domain size
    assert not r.approximately_fractal


# --- records ------------------------------------------------------------------------------------


def test_record_fidelity_detects_and_rejects():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    centres = rng.normal(0, 3, (4, 5))
    strong = o.record_fidelity(y, centres[y] + rng.normal(0, 1, (200, 5)), n_permutations=300)
    assert strong.accuracy > 0.8 and strong.p_value < 0.01 and strong.excess_bits > 1.0
    null = o.record_fidelity(y, rng.normal(0, 1, (200, 5)), n_permutations=300)
    assert null.p_value > 0.01 and abs(null.excess_bits) < 0.15


def test_record_lifetime_fit():
    t = np.linspace(0, 100, 11)
    tau, i0 = o.record_lifetime(t, 2.0 * np.exp(-t / 37.0))
    assert tau == pytest.approx(37.0, rel=1e-9) and i0 == pytest.approx(2.0, rel=1e-9)
    assert o.record_lifetime([0, 1, 2], [1.0, 1.0, 1.0])[0] == math.inf
    with pytest.raises(ValueError):
        o.record_lifetime([0, 1], [0.0, 1.0])


def test_kramers_and_landauer():
    kt = o.K_B * 300.0
    assert o.kramers_lifetime(10 * kt, 300.0, 1e-12) == pytest.approx(1e-12 * math.exp(10))
    barrier = o.barrier_for_lifetime(3600.0, 273.15)
    assert o.kramers_lifetime(barrier, 273.15) == pytest.approx(3600.0, rel=1e-12)
    assert barrier / (o.K_B * 273.15) == pytest.approx(math.log(3600 / 1e-12))
    assert o.landauer_cost_j(1, 300.0) == pytest.approx(2.871e-21, rel=1e-3)
