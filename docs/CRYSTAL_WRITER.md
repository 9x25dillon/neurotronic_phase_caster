# Crystal Writer

Writes a symbol stream into a growing crystal as growth-rate modulation, and reads it back from the dopant striation profile. Prediction P3 of [ORDER_RECORDS_THEORY.md](ORDER_RECORDS_THEORY.md), built for crystals.

Module: `src/crystal_writer.py` · Tests: `tests/test_crystal_writer.py` (21) · Run: `python src/crystal_writer.py`

Evidence levels: **L0** standard physics; **L1** model; **L2** parameters to measure; **L3** speculation.

---

## 1. Why crystals already record time

A crystal's growth zoning is a record of its growth history (L0). Melt-grown semiconductors show striations from interface temperature fluctuations of millikelvin size. Peltier interface demarcation passes current pulses through the melt–crystal interface to write time stamps into the crystal on purpose, so the growth rate can be measured afterwards. This module makes that record carry symbols.

---

## 2. Physics chain

```
 symbol ──► (drive level, layline charge)                        WriterCodebook
        ──► interface temperature rise dT(t, x)                   actuator: Peltier pulses, 1550 nm pump light, THz photomixers
              first-order lag tau, lateral heat spreading, ring-array geometry for the layline
        ──► growth velocity
              melt growth (isotherm-bound):      v = v_pull − (1/G) d(dT)/dt     records the RATE of change
              solution growth (kinetics-bound):  v = v0 (sigma(T)/sigma0)^g      records the LEVEL
        ──► dopant uptake (Burton–Prim–Slichter): k_eff = k0 / (k0 + (1 − k0) exp(−v delta / D_l))
        ──► striation record c(z, x): flat front, local uptake; clock track with one marker per band
        ──► cool-down and storage: solid-state diffusion D_s = D0 exp(−Ea/kT)
        ──► read: PSF, noise, markers locate bands, bin-averaged features, nearest-centroid decoding
```

**Melt and solution growth write different formats.**

- **Melt growth.** The interface sits on an isotherm. A heat step moves the isotherm, so the interface briefly slows, then recovers. The striation marks the *edges* of each pulse, which is why the Peltier technique works with millikelvin pulses.
- **Solution growth.** The growth rate follows supersaturation, which follows temperature, so the striation marks the *level* of each pulse. The uptake sensitivity is the Péclet number Pe = v delta / D_l times a BPS factor. For KDP at Pe 0.05 it is weak.

---

## 3. Results (simulator, default parameters)

The message "crystal" was written after a 96-symbol i.i.d. training preamble, stored for one year at 300 K, then read back.

**Ge:Ga Czochralski, Peltier pulses, 8 mK peak (meltback limit 12 mK), 16 symbols**
- Read back "crfstal".
- Payload: 86 % symbol accuracy (chance 6.25 %), binomial p 3×10⁻¹³.
- Preamble: 1.64 bits/symbol.
- Retro check: at chance.

**KDP from aqueous solution, THz photomixer array alone, about 21 mK**
- At chance.
- SNR 5 needs about 0.65 K. The photomixers deliver microwatts; heating a growth interface by a kelvin takes milliwatts.

**KDP, 1550 nm pump light at 12 mW absorbed, 1.6 K peak (dissolution limit 2.4 K)**
- 16 symbols: 36 % (5.7× chance), p 0.001.
- 4 symbols (two levels × spot/ring laylines): 79 %, p 4×10⁻⁹, read back "Cp1st@l". The layline alone is read correctly in 100 % of the training preamble. In solution growth the geometric layline is the strongest carrier, because the crystal keeps the lateral uptake pattern where level contrast is weak.

**Record lifetime** (solid-state diffusion, band period 40 µm in Ge)
- At room temperature: effectively permanent (~10³⁶ years).
- At 1100 K: about 22 days.
- Near the melting point: about 1.4 days.

The record is erasable only by annealing, as theory §2.3 requires.

The THz array's role, then, is to set the frequency node and the layline geometry. The writing power must come from a stronger actuator: the system's own 1550 nm pump light, or Peltier current at a melt interface.

---

## 4. Reader design, and four leaks the integrity check caught

Every band's symbol is also decoded from the band *before* it (the retro check), which must sit at chance. During development it caught four distinct leak paths. All four are fixed and covered by tests:

1. **Boundaries late by the marker lag.** Solution-growth marker dips peak at the end of the pulse. Without correction each window swallowed the next symbol's onset. Fix: shift boundaries back by the protocol's known lag (zero for melt growth).
2. **Windows ending at the next marker.** That marker's position depends on the next symbol. Fix: fixed-length windows from each band's own start.
3. **Windows longer than thin bands.** Heated bands grow thinner in solution growth. Fix: window length set by the thinnest bands (5th percentile).
4. **Permutation preamble.** Drawing each repeat as a permutation makes each symbol partly predictable from its predecessors. Fix: i.i.d. draws (the same fix the water assay needed).

After the fixes, retro p < 0.05 occurs in about 5 % of runs, the rate expected by chance.

Features are bin-averaged log concentrations over 15 lateral × 8 depth positions per band, decoded without z-scoring. One instrument gives every value the same noise, so z-scoring would only up-weight empty positions.

---

## 5. Model limits

- **Flat front with local uptake (L1).** Real faceted growth proceeds by step flow, and local temperature changes step velocities and step densities. Impurity uptake in solution growth often depends more strongly on growth rate than BPS predicts (kinetic trapping). So the KDP results are a conservative case.
- **Lumped interface thermal lag (L1).** In melt growth, lateral heat spreading in the melt is set by convection; here it is one Gaussian width.
- **Presets (L2).** Every preset needs measurement for the actual growth system: k0, D_l, delta, G, pull rate, solid diffusion, and readout noise and resolution.
- **Interface stall.** Meltback or dissolution overwrites the record. The writer flags any stall, and `stall_limit_k()` gives the peak dT that keeps the interface moving.

---

## 6. API

```python
from src import crystal_writer as cw

system = cw.GROWTH_SYSTEMS["ge_ga_czochralski"]
codebook = cw.WriterCodebook()                                  # 4 levels x 4 laylines
actuator = cw.make_actuator("Peltier", 8e-3, system, codebook)  # 8 mK peak interface dT
report, record = cw.run_writer(system, actuator, codebook, message="crystal")
print(report.summary())

# frequency nodes from the THz engine become drive levels through absorbed power (theory P2)
nodes = cw.WriterCodebook.from_frequency_nodes(SymbolCodebook(), PhotonicTHzEmitterArray(),
                                               RingIllumination(), SampleCell())
```
