# Crystal Writer

Writes a symbol stream into a growing crystal as growth-rate modulation, and reads it back from the dopant striation profile. Prediction P3 of [ORDER_RECORDS_THEORY.md](ORDER_RECORDS_THEORY.md), built for crystals.

Module: `src/crystal_writer.py` · Tests: `tests/test_crystal_writer.py` (54) · Run: `python src/crystal_writer.py`

Evidence levels: **L0** standard physics; **L1** model; **L2** parameters to measure; **L3** speculation.

---

## 1. Why crystals already record time

A crystal's growth zoning is a record of its growth history (L0). Melt-grown semiconductors show striations from interface temperature fluctuations of millikelvin size. Peltier interface demarcation passes current pulses through the melt–crystal interface to write time stamps into the crystal on purpose, so the growth rate can be measured afterwards. This module makes that record carry symbols.

---

## 2. Physics chain

```
 text ──► bits ──► Hamming(7,4) ──► interleave ──► Gray-mapped symbols     ChannelCode (optional)
 symbol ──► (drive level, layline charge)                        WriterCodebook
        ──► interface temperature rise dT(t, x)                   actuator: Peltier pulses, 1550 nm pump light, THz photomixers
              first-order lag tau, lateral heat spreading, ring-array geometry for the layline
        ──► growth velocity
              melt growth (isotherm-bound):      v = v_pull − (1/G) d(dT)/dt     records the RATE of change
              solution growth (kinetics-bound):  v = v0 (sigma(T)/sigma0)^g      records the LEVEL
              slow growth-rate drift:            v → v (1 + a f(t))              GrowthDrift (optional)
        ──► dopant uptake (Burton–Prim–Slichter): k_eff = k0 / (k0 + (1 − k0) exp(−v delta / D_l))
        ──► striation record c(z, x): flat front, local uptake; clock track with one marker per band
        ──► cool-down and storage: solid-state diffusion D_s = D0 exp(−Ea/kT)
        ──► read: PSF, noise, markers locate bands, features referenced to the clock track's baseline,
            nearest-centroid decoding, de-interleave, Hamming correction
```

**Melt and solution growth write different formats.**

- **Melt growth.** The interface sits on an isotherm. A heat step moves the isotherm, so the interface briefly slows, then recovers. The striation marks the *edges* of each pulse, which is why the Peltier technique works with millikelvin pulses.
- **Solution growth.** The growth rate follows supersaturation, which follows temperature, so the striation marks the *level* of each pulse. The uptake sensitivity is the Péclet number Pe = v delta / D_l times a BPS factor. For KDP at Pe 0.05 it is weak.

---

## 3. Results (simulator, 48 runs per configuration)

Each run writes "crystal" after an i.i.d. training preamble (6 × codebook size bands), stores the crystal for one year at 300 K, and reads it back. Accuracy is pooled over the cross-validated preamble and the payload. Seeds 0–47; the clock-track reference is on unless stated.

An earlier version of this page quoted single runs (86 % for Ge:Ga, 79 % for KDP). The multi-run means below replace them.

**Ge:Ga Czochralski, Peltier pulses, 8 mK peak (meltback limit 12 mK)**
- 16 symbols (4 levels × ring charges ℓ = 0, 1, 2, 3), uncoded: 65 % symbol accuracy (chance 6.25 %); level 84 %, layline 73 %. No run reads the 56-bit message exactly. This mode stays uncoded as the degradation benchmark. The demo's seed 3 reads "crfstal" (86 % on 14 payload bands), a favourable draw.
- 4 symbols (2 levels × spot/ring), uncoded: 98.5 % (SEM 0.4); level 99 %, layline 99 %; 88 % of messages exact.
- 4 symbols with Hamming(7,4), Gray mapping, full interleave: 98.4 % raw; 92 % of messages exact.

**KDP from aqueous solution, THz photomixer array alone, about 21 mK**
- At chance. SNR 5 needs about 0.65 K. The photomixers deliver microwatts; heating a growth interface by a kelvin takes milliwatts.

**KDP, 1550 nm pump light at 12 mW absorbed, 1.6 K peak (dissolution limit 2.4 K)**
- 16 symbols: 30 % (4.8× chance), 28 % with the clock reference.
- 4 symbols, spot/ring: 77 %, 75 % with the clock reference; layline 97 %, level 77 %. Hamming(7,4) brings 8 % of messages through exactly. The level channel is the weak one in solution growth, and at 23 % symbol error no short code rescues the message.

**Why the fine ring charges fail.** Lateral heat spreading at the interface (0.3 mm Gaussian) blurs the rings of charges ℓ = 1, 2 and 3 into one another, while a central spot (ℓ = 0) against a ring stays distinct. In Ge:Ga (12-run error analysis), 46 % of 16-symbol errors are a neighbouring ring charge at the same level, 23 % a neighbouring level, 24 % both, and none a distant level. The crystal resolves geometry at the coarse level only. The 4-symbol spot/ring codebook (`WriterCodebook.spot_ring()`, `CODEBOOKS["spot_ring_4"]`) is the operating mode.

**Record lifetime** (solid-state diffusion, band period 40 µm in Ge)
- At room temperature: effectively permanent (~10³⁶ years).
- At 1100 K: about 22 days.
- Near the melting point: about 1.4 days.

The record is erasable only by annealing, as theory §2.3 requires.

The THz array's role, then, is to set the frequency node and the layline geometry. The writing power must come from a stronger actuator: the system's own 1550 nm pump light, or Peltier current at a melt interface.

---

## 4. Channel coding

`ChannelCode(fec, gray, interleave_depth)` sits between text and symbols. `UNCODED` is the original mapping (byte bits straight onto symbols). `HAMMING_GRAY` is Hamming(7,4) with Gray mapping and whole-message interleaving.

- **Gray mapping.** Each symbol carries a bit word: the Gray code of its level index in the high bits, the Gray code of its layline index in the low bits. The crystal's misreads are one step along one axis, and under Gray each such error flips one bit. On the 16-symbol code, bits flipped per symbol error drop from 1.52 to 1.32 (same 12-run analysis).
- **Hamming(7,4).** Corrects one bit per 7-bit codeword at rate 4/7. "crystal" takes 49 bands in the 4-symbol code, against 28 uncoded.
- **Interleaving.** A block interleaver sends codewords column by column, so the 2 bits of one band, and a run of drift-damaged bands, land in different codewords. With full interleaving, any single misread band, read as any other symbol, is repaired (tested exhaustively). Without it, a diagonal misread (other level and other layline) can break a codeword. Interleaving matters under drift bursts. With a sine drift and a short preamble, 31 % of messages come through exactly with it and 4 % without.
- **What the code can and cannot do.** In Ge:Ga 4-symbol, Hamming raises exact messages from 88 % to 92 % without drift, and from 85 % to 96 % under a ±10 % ramp. A failure needs two errors in one codeword, so the remaining failures are runs with several errors. At the 16-symbol mode's 35 % symbol error and KDP's 23 %, the message is lost with or without the code.
- **Reed–Solomon was dropped.** At 35 % raw error on 16 symbols, RS(15,3) would add 5× overhead to rescue a mode that lateral heat spreading has already ruled out.

---

## 5. Growth-rate drift

`GrowthDrift(amplitude, period_s)` multiplies the growth rate of the data and clock tracks by 1 + a f(t). `period_s = 0` gives a linear ramp from 1 − a to 1 + a across the run (melt level falling, solute depletion). `period_s > 0` gives a sinusoid.

Real growth always drifts, so the reader has two defences:
- **Markers fix where a band is.** Boundaries computed from the nominal rate collapse to near chance at ±10 % (Ge:Ga 16-symbol, 6-run analysis: 70 % → 8 %).
- **The clock-track reference fixes what zero means in it.** The clock track drifts with the data track but carries no symbol. The reader takes its mean log uptake over the last half of each band's window, interpolates that baseline to every depth bin, and subtracts it (`Readout.clock_reference`, on by default). Without drift it costs about 0 points in Ge:Ga and 2 points in KDP.

Ge:Ga, 48 runs each, with the reference (and without it):
- 16 symbols, uncoded: no drift 65 % (65 %); ±10 % ramp 58 % (24 %); ±20 % ramp 44 % (16 %).
- 4 symbols, Hamming: no drift 98.4 % (98.7 %), 92 % exact; ±10 % ramp 98.0 % (69 %), 96 % exact (0 %); ±20 % ramp 95.3 % (45 %), 75 % exact (0 %).
- 4 symbols, uncoded: ±10 % ramp 98.1 %, 85 % exact; ±20 % ramp 95.6 %, 56 % exact.

KDP 4 symbols, uncoded, with the reference (and without it): ±10 % ramp 70 % (68 %); ±20 % ramp 70 % (60 %).

**Oscillating drift needs a preamble that spans it.** A ±10 % sine with a period of 60 bands leaves the preamble perfectly decoded (99 %) but drops the Ge:Ga 4-symbol payload to 86 %, with 31 % of messages exact. The reference removes the drifting baseline, but the drift also changes the gain of the edge response, and a 24-band preamble samples only 0.4 of the cycle. A 144-band preamble (2.4 cycles) restores 98.9 % with 100 % exact. In a real growth run the remedy is pilots: training bands distributed through the message instead of only ahead of it.

---

## 6. Reader design and the integrity controls

Every report decodes each training band's symbol from three places other than its own band (`temporal_controls`):

- **Retro (band k−1 → symbol k).** Must sit at chance. Anything above is a reader leak: the reader let a later symbol shape an earlier band.
- **Forward (band k+1 → symbol k).** Measures carry-over. A heat tail, a solute boundary layer or diffusion can push symbol k into the next band. That is physical and causal, but it is interference.
- **Shuffled (retro after randomising band order).** Calibrates the estimator and its permutation test.

Unit tests plant each effect in synthetic data and confirm that each control detects only its own.

The 48-run sweep has 54 configurations and three controls each, 162 cells:
- **Mean control accuracy** sits within 0.015 of chance on 16 symbols and within 0.041 on 4 symbols, where a 24-band preamble gives only 23 pairs per run.
- **p < 0.05** occurs in 3.7 % of runs on average (90th percentile 6 %).
- **Six cells** reach ≥ 6 hits in 48 runs, where chance alone predicts about 5. Three of the six are the KDP 16-symbol forward check, in configurations that share seeds. An independent 32-run check with a longer preamble put it at 3 %. It stays flagged and unresolved.

The controls catch the two ways neighbours mix:
- **Off-period of 2τ in KDP.** The heat tail reaches the next band: forward p < 0.05 in 16 % of runs, retro in 22 %. The default off-period is 5τ.
- **Depth resolution of 2.5 µm against KDP's 10 µm bands.** Blur carries band k+1's onset into band k's window: retro in 41 % of runs. At the default 1 µm, retro sits at the null. Keep the PSF at or below a tenth of the band period.

During development the retro check caught four leak paths. All four are fixed and covered by tests:

1. **Boundaries late by the marker lag.** Solution-growth marker dips peak at the end of the pulse. Without correction each window swallowed the next symbol's onset. Fix: shift boundaries back by the protocol's known lag (zero for melt growth).
2. **Windows ending at the next marker.** That marker's position depends on the next symbol. Fix: fixed-length windows from each band's own start.
3. **Windows longer than thin bands.** Heated bands grow thinner in solution growth. Fix: window length set by the thinnest bands (5th percentile).
4. **Permutation preamble.** Drawing each repeat as a permutation makes each symbol partly predictable from its predecessors. Fix: i.i.d. draws (the same fix the water assay needed).

Two further checks were made in this revision:
- **Interpolated reference.** Late bins of band k−1 take part of their baseline from band k's window tail. This was checked directly for a backward leak: retro rates match a per-band constant reference in every configuration tested.
- **Causality of the melt model.** The melt velocity used a central difference, so v(t) saw dT(t + dt), one 0.02 s step ahead. It is now a backward difference. Results are unchanged (98.52 % either way), but a simulator whose controls test for temporal leaks must itself be causal.

Features are bin-averaged log concentrations over 15 lateral × 8 depth positions per band, decoded without z-scoring. One instrument gives every value the same noise, so z-scoring would only up-weight empty positions.

---

## 7. Model limits

- **Flat front with local uptake (L1).** Real faceted growth proceeds by step flow, and local temperature changes step velocities and step densities. Impurity uptake in solution growth often depends more strongly on growth rate than BPS predicts (kinetic trapping). So the KDP results are a conservative case.
- **Lumped interface thermal lag (L1).** In melt growth, lateral heat spreading in the melt is set by convection; here it is one Gaussian width.
- **Drift (L1).** One multiplicative factor on the growth rate. Real drift also changes the thermal gradient and the boundary layer, so the response gain moves differently from the baseline.
- **Presets (L2).** Every preset needs measurement for the actual growth system: k0, D_l, delta, G, pull rate, solid diffusion, and readout noise and resolution.
- **Interface stall.** Meltback or dissolution overwrites the record. The writer flags any stall, and `stall_limit_k()` gives the peak dT that keeps the interface moving.

---

## 8. API

```python
from src import crystal_writer as cw

system = cw.GROWTH_SYSTEMS["ge_ga_czochralski"]
codebook = cw.CODEBOOKS["spot_ring_4"]                          # 2 levels x spot/ring
actuator = cw.make_actuator("Peltier", 8e-3, system, codebook)  # 8 mK peak interface dT
report, record = cw.run_writer(system, actuator, codebook, message="crystal",
                               code=cw.HAMMING_GRAY,             # Hamming(7,4), Gray, full interleave
                               drift=cw.GrowthDrift(0.1))        # +/-10 % growth-rate ramp
print(report.summary())  # preamble, retro/forward/shuffled controls, raw and corrected payload

# the 16-symbol degradation benchmark, uncoded
full = cw.CODEBOOKS["full_16"]
cw.run_writer(system, cw.make_actuator("Peltier", 8e-3, system, full), full, message="crystal")

# frequency nodes from the THz engine become drive levels through absorbed power (theory P2)
nodes = cw.WriterCodebook.from_frequency_nodes(SymbolCodebook(), PhotonicTHzEmitterArray(),
                                               RingIllumination(), SampleCell())
```
