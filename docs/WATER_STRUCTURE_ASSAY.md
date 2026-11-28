# Water Structure Assay

A preregistered, blinded test of whether a THz drive leaves liquid water in a persistent, information-bearing state, and whether a symbol stream written into that state can be read back across time.

Module: `src/water_structure_assay.py` · Tests: `tests/test_water_structure_assay.py` · Engine: `src/photonic_thz_pulse_engine.py`

Evidence levels: **L0** standard physics and published values; **L1** model calculations; **L2** parameters that need measurement; **L3** speculation.

---

## 1. Executive summary

The assay turns three ideas from the project into measurable quantities:

- **Stream of consciousness → symbols.** `TextStream` (any text, as base-M digits), `EEGSubstrateStream` (the dominant EEG band per window, mapped to the five ABCR substrates), or `RandomStream` (OS entropy).
- **Informational frequency node × geometric layline → drive.** Each symbol selects a THz beat frequency (the node) and a helical phase pattern across the 12-channel ring, phi_n = 2 pi l n / 12 (the layline, with topological charge l). Charge 0 focuses on axis; charge l ≠ 0 forms a vortex ring with an on-axis null. The default codebook has 4 nodes × 4 laylines = 16 symbols (4 bits).
- **Communication across temporal dimensions → temporal channel map.** For every arm, the decoder asks whether the symbol can be read: before it was chosen (pre-window and negative lags), during the drive, at 1, 10, 60 and 300 s after shutdown, and in the pre-windows of later trials (positive lags).

Two hypotheses are fixed before data exist and hashed (SHA-256):

- **H1, persistence.** Active and sham differ in band-mean absorption by at least 0.2 % after the classical memory horizon.
- **H2, channel.** Symbols are decodable from the active arm after the horizon, and from neither control.

The physics baseline (L0) is that water's hydrogen-bond network rearranges in about a picosecond, and the only known memory of a driven cell is heat. For the default cell the slowest thermal process is lateral diffusion (44 s), so the classical memory horizon is 5 × 44 = **219 s**. Only readouts after that count toward H1 and H2.

On the simulator, the pipeline:

- returns a formal **equivalence null** (TOST within ±0.2 %) with every decoding test at chance when nothing is there;
- detects an injected hypothetical effect (5 % peak absorption change, 1 h memory) at 300 s, and writes and reads back a text message at 75 % symbol accuracy against 6.25 % chance;
- raises integrity flags when labels leak into the pre-window or information appears in the sham arm.

Power and false-positive rates are in §7.

---

## 2. Architecture

```
 stream ──► symbol ──► SymbolCodebook ──► equalised command ──► PhotonicTHzEmitterArray.plan()
 (text, EEG,           node f_i, layline l     (same absorbed power         (interlocked drive plan)
  random)                                       for every symbol)
                                                        │
 Preregistration ── digest ──┐                          ▼
 AllocationKey ── commitment ┤        Instrument: SimulatedInstrument now, bench rig later
                             │          apply(DriveAction)  wait(s)  read(frequencies)
 ThermalCalibration ─────────┤                          │
 (heater steps, unblinded)   │                          ▼
                             └──────────────► AssayDataset (codes only)
                                                        │
                                   analyze() ──► BlindedAnalysis (per-code stats)
                                                        │
                                   unblind(key) ──► AssayReport
                                                   H1 verdict · H2 verdict · integrity flags
                                                   temporal channel map · recovered message
```

### Bench rig the `Instrument` interface expects

```
  12 photomixer fibers on a 20° cone ──► sample cell: 100 µm water between THz windows,
                                          heater film on one window, thermistor in the water,
                                          TEC-stabilised holder
                                                    │
  probe beam (low power, same laser pair) ──► second photomixer as coherent receiver
                                              ──► transmission at 16 frequencies, 0.3–2.0 THz
```

A hardware implementation subclasses `Instrument` and provides three methods:

- `apply(action)`: action kinds are `thz` (load the plan to the emitter MCU and enable bias), `sham` (lasers, splitter and stretchers set identically, photomixer bias off), `heater` (drive the film at the matched power) and `off`.
- `wait(seconds)`.
- `read(frequencies)`: returns time, cell temperature and a transmission spectrum.

Everything downstream stays unchanged.

---

## 3. Protocol

### Arms

- **Active.** THz drive, using the symbol's node and layline at the equalised amplitude.
- **Sham.** Everything identical except photomixer bias: same lasers, IR power, fiber state, timing and symbol bookkeeping. No THz is produced. This isolates the THz field from every other part of the apparatus.
- **Thermal.** No THz. A heater film deposits the same power the active drive would absorb. This isolates heat.

### Codebook equalisation

Absorbed power differs between symbols before equalisation: the photomixer output falls with frequency, water absorbs more at higher frequency, and laylines distribute power differently over the aperture. In the model the spread is over 1.5×. `calibrate_codebook` scales each symbol's amplitude so every symbol deposits identical absorbed power. The thermal side channel is closed by design, and the thermal arm matches that single power.

### Trial timeline

1. **Settle.** Intertrial interval of 600 s, longer than the 219 s horizon, so heat from one trial never reaches the next.
2. **Pre-window.** A readout at 5 s, recorded before the symbol exists.
3. **Draw.** The symbol is drawn i.i.d. uniform (OS CSPRNG on the bench). i.i.d. draws are required: drawing without replacement makes later symbols in a block predictable from earlier ones, and any slow memory in the sample would then leak into the retro test.
4. **Drive.** Active, sham or thermal for 30 s, with a readout at the end ("during").
5. **Post-windows.** Shutdown, then readouts at 1, 10, 60 and 300 s.

**Blocks.** Four blocks; arm order is shuffled within each block. The default runs 16 slots per arm per block, for 192 calibration trials.

**Payload (message).** Each stream symbol is sent once under active and once under sham, in random order. A decoder trained on the calibration trials of the same arm reads the payload back at the preregistered `message_window` (post@300s).

**Bench time.** About 15.6 min per trial; the default run is ~52 h. A 4-symbol codebook with 8 slots per arm per block takes ~25 h.

### Blinding and preregistration

- `Preregistration` holds every analysis choice: windows, frequencies, SESOI, alpha, permutation counts, lags, message window and seed. `digest()` is computed before the run. `analyze()` refuses to proceed if the digest differs from the one stored in the dataset.
- Arms carry opaque codes. `AllocationKey.commitment()` (salted SHA-256) is stored in the dataset; the key stays with the run controller. `analyze()` sees codes only. `unblind()` verifies the key against the commitment and refuses any other key.
- `ThermalCalibration` (a heater step response) is measured before the main run, so the temperature correction never depends on arm identity.

---

## 4. Analysis

**Features.** Absorption from transmission, alpha = -ln T / d. The thermally corrected residual is r = alpha / alpha_ref - 1 - s(f) (T - T_holder), with alpha_ref(f) and s(f) taken from the heater calibration at each frequency. For the during and post windows, the trial's own pre-window residual is subtracted. The pre-window uses the absolute residual.

**H1 contrasts.** For every window and pair of codes: band-mean difference, bootstrap 95 % CI, Hedges g, permutation p, and TOST equivalence at ±SESOI.

**H2 decoding.** Leave-one-block-out nearest-centroid on the 16-point residual spectrum. The permutation null is vectorised across permutations, and labels are exchangeable because draws are i.i.d. Information is reported as excess bits: the plug-in MI minus its mean under the permutation null. Raw plug-in MI is biased upward by about 1.5 bits at this sample size.

**Multiple comparisons.** Holm within families fixed by test type, defined blind:

- `beyond_horizon`: the primary claims.
- `within_horizon`: during the drive and post windows inside the horizon.
- `integrity`: the pre-window and negative lags.
- `carryover`: positive lags.

A leak detector that shared a family with the claims it protects would be diluted by them. That failure appeared in testing (53 % against 25 % chance read as Holm p 0.12) and is why the families are separate.

### Verdict rules, applied after unblinding

**H1:**
- **DETECTED** when active minus sham beyond the horizon has Holm p < alpha, its magnitude is at least SESOI, its CI excludes 0, and the thermal arm shows no such difference from sham.
- **NULL** when TOST shows equivalence at every beyond-horizon delay.
- **INCONCLUSIVE** otherwise.

**H2, evaluated in this order:**
1. **INTEGRITY:** any integrity flag voids every channel claim.
2. **ANOMALOUS CHANNEL:** active decodable beyond the horizon and neither control decodable there.
3. **Classical thermal channel:** active and thermal both decodable during the drive.
4. **Within horizon only:** decodable during the drive or inside the horizon.
5. **No information beyond chance.**

### Integrity flags

- **RETRO.** The symbol is readable before it was drawn. With a predictable stream this becomes LEAK: the pipeline is leaking labels.
- **CONTROL.** Information appears in the sham or thermal arm after shutdown: an IR, electrical, acoustic or bookkeeping path exists.
- **CONTROL.** The sham arm is decodable during the drive: the THz field is not the carrier.

---

## 5. Calibration evidence (simulator, default preregistration)

- Heater calibration recovers the water absorption temperature coefficient at 0.489 %/K against 0.500 %/K true (median R² 0.986). The residual slope error contributes under 1e-4 to any feature at the drive's temperature rise.
- **Null run:** H1 NULL (equivalent within ±2e-3); H2 no information; every decoding test at chance; mean excess information about 0 bits.
- **Positive control (L3 effect injected: 5 % peak, 1 h memory):**
  - H1 DETECTED: +6.5e-3 at 300 s, CI +5.8e-3 to +7.2e-3, Holm p 0.0015.
  - H2 ANOMALOUS: about 0.6 accuracy against 0.0625 chance at 300 s.
  - Message 'hi' read back at 0.75 symbol accuracy (binomial p 0.0009).
  - The sham arm stays at chance.
- **Effect with 15 s memory:** classified as within-horizon information with no persistence claim, which is the correct outcome for something that heat-like dynamics could explain.
- **Leak injected** (post-window copied into pre-window): RETRO flag raised and channel verdict voided.
- **Leaky sham** (sham path still writes the symbol): CONTROL flag raised, no anomalous claim.

**Power and false-positive rate** (`power_analysis`, default preregistration, 12 simulated runs per point; peak relative absorption change of the injected effect, 1 h memory). Each rate is k of 12, so the binomial uncertainty is about ±10–14 points.

- **0 (no effect):** H1 0/12, H2 0/12. False positives were controlled in this sample; the 95 % upper bound from 12 runs is about 22 %, so run more nulls before a bench campaign.
- **0.25 %:** H1 0/12, H2 6/12.
- **0.5 %:** H1 0/12, H2 11/12.
- **1 %:** H1 0/12, H2 10/12.
- **2 %:** H1 10/12, H2 11/12.

H2 reads spectral shape and is the sensitive test. H1 is gated on purpose by the SESOI (0.2 % band-mean, reached near a 2 % peak change), so it never reports a difference too small to matter.

---

## 6. Failure modes

- **Spatial heating (L1, not in the lumped simulator).** Laylines deposit power in different spatial patterns. Equal total power can still leave different local temperatures at the probe spot for up to the lateral diffusion time (44 s). The horizon rule (5 × 44 s) is what protects the claim. On the bench, map the probe spot with a thermal camera, or probe the full aperture.
- **Receiver gain drift.** This is frequency-flat in -ln T and appears in relative absorption as a 1/alpha_ref(f) shape. Over a 50 h run it accumulates to about 3e-3, which is why the pre-window (absolute) features are noisier than the within-trial differences. Use a reference cell or a reference beam on the bench.
- **Sample state.** Evaporation, dissolved gas, bubbles, contamination and window etalon fringes all change THz absorption slowly. They affect all arms equally only if arm order is randomised, which it is. Seal the cell, degas, and log ambient humidity.
- **Temperature control.** At 0.5 %/K, a 2 mK thermistor error is 1e-5, negligible. A holder drift of 0.4 K equals the full SESOI, so the TEC loop must hold better than ±50 mK over the run.
- **Electrical pickup.** Photomixer bias switching (active versus sham) can couple into the receiver. The sham arm catches only effects that differ from bias-off, so add a bias-on, lasers-blocked arm if the receiver shares ground with the emitter.
- **Experimenter degrees of freedom.** Closed by the digest check. Any change to windows, thresholds, frequencies or decoders requires a new preregistration and new data.

---

## 7. Open questions

- Which property of water should a morphic or structured state change: absorption, refractive index, the shape of the OH-stretch Raman band, NMR relaxation? The assay reads THz absorption because the hardware already exists. A second readout would make a positive result much harder to attribute to the receiver.
- Which samples? Pure water is the hardest test. Protein solutions are where measured long-range hydration effects already exist, and are the more natural place to look for something new.
- Who replicates? A positive result means an independent rig, an independent operator, and the same preregistration digest.

---

## 8. Evidence vs speculation

- **L0.** Hydrogen-bond lifetimes of about 1 ps; water's THz absorption and its rise with temperature; Beer-Lambert; vortex fields from helical phase arrays; permutation, TOST and Holm statistics; SHA-256 commitments.
- **L1.** Lumped thermal model, the classical memory horizon, the ring-illumination field model, codebook equalisation.
- **L2.** Cell geometry, thermal resistance, temperature coefficient, receiver noise and drift, illumination geometry.
- **L3.** Morphic fields, consciousness-to-water coupling, and communication across temporal dimensions. The assay is built so these claims can fail cleanly or pass under conditions where no known physics explains the result. `InjectedEffect` exists only in the simulator, as a positive control. Real hardware has no such knob.
