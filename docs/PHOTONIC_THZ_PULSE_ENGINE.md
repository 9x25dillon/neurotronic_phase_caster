# Photonic THz Pulse Engine

All-optical synthesis of terahertz-rate infrared pulse trains, conversion to free-space THz, and a phase-coherent 12-channel emitter backend for the Neurotronic Phase Caster.

Module: `src/photonic_thz_pulse_engine.py` · Tests: `tests/test_photonic_thz_pulse_engine.py` (60 checks) · Figure: `docs/figures/photonic_thz_pulse_engine.png`

Evidence levels follow the repository convention in `rtd_300ghz/CHANGELOG.md`: **L0** identities, standard formulas and published values; **L1** model calculations; **L2** device and material parameters that need measurement; **L3** speculation.

---

## 1. Executive summary

Electronic switches and electro-optic modulators roll off near 100 GHz, so a 1 THz (1 ps period) IR envelope has to be built optically by interfering phase-locked IR lines. This module implements every route in the design brief as composable pure functions on an immutable `CombState`, then exposes the result as `PhotonicTHzEmitterArray`, a drop-in `EmitterArray` subclass that the existing `THzCoherenceWearable` and `NeurotronicPhaseCasterWearable` accept without behavioral changes.

Numbers the module reproduces or derives (run `python src/photonic_thz_pulse_engine.py`):

- **L0** A 1 THz beat at 1550 nm needs 8.014 nm line separation (carrier 193.414 THz).
- **L0** A 1 THz Si3N4 ring (n_g = 2) is 149.9 µm around, radius 23.86 µm.
- **L0** Locking N flat lines at 1 THz gives peak/average = N and FWHM 500, 228, 111, 55.5, 27.7 fs for N = 2, 4, 8, 16, 32.
- **L1** Soliton FWHM in that ring is about 21 fs at 5 GHz detuning (beta2 = -60 ps²/km, L2). The FSR moves only -12 MHz/K, so a microcomb is effectively a fixed-frequency source.
- **L0/L1** Talbot x40 from a 25 GHz seed to 1 THz needs 6.37 ps² of group-delay dispersion (293 m SMF-28). In fiber, third-order dispersion adds hundreds of radians of cubic phase across a 100 fs pulse spectrum and destroys the train (0.5 dB subharmonic suppression). A line-by-line phase mask gives the same quadratic phase with no cubic term: 100.7 fs pulses, 68.8 dB suppression. The planner therefore defaults to the mask.
- **L1/L2** A UTC-PD photomixer at its 10 mA thermal limit gives 34.8, 15.4 and 7.7 µW at 0.8, 1.0 and 1.2 THz. That is consistent with the published ">10 µW at 1 THz", and roughly 300x below the 5 mW per emitter in `THZ_WEARABLE_HARDWARE_SPEC.md`.
- **L1** At equal mean photocurrent, a 10 % duty pulse train puts 3.7x more power into the 1 THz fundamental than a sinusoidal beat. Peak photocurrent rises by the peak/average ratio, so space-charge saturation (L2, not modeled) caps the gain.
- **L1** CW optical rectification of a 1 W heterodyne pair in 1 mm GaP yields about 1e-13 W. At THz repetition rates photomixing wins by eight orders of magnitude.
- **L0/L1** Liquid water absorbs 220 cm⁻¹ at 1 THz. In the nominal head stack the 1/e depth in scalp is 65 µm, and attenuation reaches 335 dB at the bottom of 5 mm of scalp and 888 dB at the bottom of the cortex layer.

---

## 2. Architecture

### 2.1 Hardware topology (CW / modulated patterns)

```
 laser A  DFB/ECL, reference, fixed ─┐
                                     ├─ PM coupler ─ EDFA (~1.0 W, constant) ─ 1x12 splitter ─┐
 laser B  DFB/ECL, tuned to f_THz ───┘                                                       │
                                                                                             ▼
  per channel n:  VOA (level) ─ fiber stretcher (THz phase = 2 pi f tau_n) ─ UTC-PD + log-spiral
                  antenna ─ IR blocker ─ collimating lens ─ skin

  photomixer bias gate (ns rise)  =  millisecond burst envelope from the controller
```

Both lasers feed every channel, so all 12 THz outputs share one phase reference. A channel's THz phase is set by delaying its whole optical envelope: 0.5 ps (0.10 mm of fiber) is pi at 1 THz.

### 2.2 Hardware topology (pulsed pattern)

```
 CW laser ─ EO comb (25 GHz seed, retuned to f/r) ─ spectral broadening ─ Talbot phase mask (period 2r lines)
           ─ EDFA ─ 1x12 splitter ─ per-channel VOA / stretcher / photomixer  (as above)

 alternatives:  Fabry-Perot etalon (FSR = r x seed, keeps 1/r of power)
                Kerr soliton microring (FSR = f, one ring per frequency)
```

### 2.3 Software module map

```
 CombState (frozen)  ──make_comb / design_seed_comb / HeterodynePair.to_comb / MicroringResonator.soliton_comb
      │
      ├── TalbotMultiplier.apply ── FabryPerotEtalon.apply           (CombState -> CombState)
      │
      ├── intensity_waveform, pulse_metrics, line_autocorrelation    (analysis)
      │
      ├── KerrGate.switch, SaturableAbsorberGate.switch              (time domain, GateResult)
      │
      ├── Photomixer.response_to_comb, OpticalRectifier.response_to_comb
      │
      ├── tissue_transmission, attenuation_profile_db                (depth model)
      ├── skin_exposure_limit_w_per_m2, assess_skin_exposure         (interlock)
      │
      └── PhotonicTHzEmitterArray(EmitterArray)
              plan(hardware_cmd) -> PhotonicDrivePlan (frozen)  ->  apply_plan(plan)
                                    .to_mcu_command()  .summary()
```

The array is the only stateful object; it holds state because `EmitterArray` does. Everything upstream of it is a pure function of frozen inputs.

---

## 3. Core theory

**Comb envelope (L0).** With line k at f0 + (k - (N-1)/2) df and complex amplitude a_k (sqrt W), the envelope is A(t) = sum a_k exp(i 2 pi k df t) and P(t) = abs(A)². Equivalently P(t) = sum_q P_q exp(i 2 pi q df t) with the line autocorrelation P_q = sum_k a_{k+q} conj(a_k). P_0 is the mean power and abs(P_q) is the optical power modulation at harmonic q.

**Pulse sharpening (L0).** For flat, locked lines P(t) is the Dirichlet kernel sin²(N pi df t) / sin²(pi df t). Peak/average equals N exactly; FWHM goes from T/2 at N = 2 to 0.8859 / (N df) at large N. Random line phases keep the mean power and lose the pulses.

**Heterodyne (L0).** Modulation depth m = 2 sqrt(Pa Pb) abs(cos theta) / (Pa + Pb); free-running Lorentzian linewidths add in the beat.

**Kerr soliton (L1).** f_rep = c / (n_g L). Bright dissipative solitons need beta2 < 0; the sech parameter is tau = sqrt(-beta2 c / (2 n_g delta_omega)), FWHM = 1.763 tau.

**Etalon (L0).** Airy transmission T = 1 / (1 + (2F/pi)² sin²(pi delta / FSR)). Setting FSR = r x seed passes every r-th line; the adjacent-line rejection is -10 log10 T(seed).

**Temporal Talbot (L0).** Group-delay dispersion Phi2 = (s/r) T² / (2 pi) with gcd(s, r) = 1 gives line phases pi (s/r) k² and an r-fold multiplied train with every sub-pulse at 1/r of the input peak. The operation is all-pass, so it loses no energy. A line-by-line mask applies pi (s/r) k² mod 2 pi directly and repeats every 2r lines.

**Kerr gate (L0/L1).** T = leak + (1 - leak) sin²(2 psi) sin²(dphi/2), dphi = (2 pi / lambda) L b n2 I_eff, where I_eff is the response-filtered pump intensity: instantaneous plus a delayed exponential fraction. Full switching needs I_pi = lambda / (2 b n2 L), about 2.2 TW/cm² for 1 mm of silica.

**Saturable absorber (L1).** dq/dt = (q0 - q)/tau - q I/F_sat, integrated with the exact per-sample exponential solution (unconditionally stable), iterated to periodic steady state.

**Photomixer (L1).** P(f) = 1/2 R_A I_ac² / ((1 + (w tau)²)(1 + (w R_A C)²)). For a comb, each harmonic carries I_q = 2 R abs(P_q). For two lines this reduces to I_ac = m I_dc.

**Optical rectification (L1, upper bound).** P_THz(q) = 2 d² W_q² abs(P_q)² L_eff² T_out / (n_opt² n_THz eps0 c³ A), with L_eff² = abs(exp((i dk - alpha/2) L) - 1)² / (dk² + alpha²/4) and dk = (W/c)(n_THz - n_g). THz diffraction is ignored.

**Tissue (L0 data, L1 model).** Beer-Lambert through layered water fractions, using the cited water table (157.8, 220.3, 270.6, 316.9 cm⁻¹ at 0.5, 1.0, 1.5, 2.0 THz), log-log interpolation inside the table and power-law extrapolation outside, and one Fresnel loss at a skin index of 2. Dry-matter absorption defaults to zero, so the attenuation is a lower bound.

**Skin exposure limit (L0).** IEC 60825-1 and ICNIRP 2013 cover 180 nm to 1 mm, which includes 0.8 to 1.2 THz (250 to 375 µm). For 1.4 µm to 1 mm: H = 5600 t^0.25 J/m² for 100 ns to 10 s; E = 1000 W/m² beyond 10 s, reduced for irradiated areas above 100 cm². Irradiance is averaged over a 3.5 mm aperture.

---

## 4. Implementation strategy

- **Data structure.** `CombState(center_hz, spacing_hz, amplitudes)` is frozen, with a read-only complex array. Every optical element maps CombState to CombState, so pipelines compose: `photomixer.response_to_comb(TalbotMultiplier(40).apply(seed))`.
- **Waveforms.** Inverse FFT on a grid of M >= 16N samples per period (a power of two, times r when the train is r-fold multiplied, so periodicity is exact on the grid).
- **Repetition rate detection.** Taken from the harmonic domain, independent of sampling: the dominant harmonic argmax abs(P_q), with purity reported as the ratio to the strongest harmonic that is not its multiple (subharmonic suppression, dB).
- **Autocorrelation.** FFT, O(N log N).
- **Gates.** First-order responses use `scipy.signal.lfilter`, with a pure-numpy fallback. Periodic steady state is solved in closed form for the linear filter and iterated to tolerance for the nonlinear absorber.
- **Planner contract.** `plan()` consumes exactly `THzPulseParameters.to_hardware_command()`. `amplitude_normalized` a maps to photocurrent sqrt(a) I_max, so THz power scales linearly with a and can never exceed the device limit. The EDFA runs at constant output; per-channel VOAs set the level (60 dB blocks a port).
- **Interlock.** `apply_plan()` refuses any plan whose exposure check fails or that carries a `BLOCK` warning, and shuts all channels down. Repeated bursts are assessed against the continuous (>= 10 s) limit, which is the conservative choice.
- **Portability.** Python 3.8+, numpy only (scipy optional). No language-specific tricks: every kernel is an FFT, a cumulative recurrence or a closed form, and maps directly to Rust, Julia, C++ or the STM32H7 firmware.

### Integration

```python
from src.photonic_thz_pulse_engine import PhotonicTHzEmitterArray
from src.thz_coherence_wearable_spec import THzCoherenceWearable
from src.wearable_controller import NeurotronicPhaseCasterWearable

# Hardware-spec controller with the photonic backend
wearable = THzCoherenceWearable(emitters=PhotonicTHzEmitterArray())

# Real-time controller: every emitted pulse becomes a PhotonicDrivePlan
backend = PhotonicTHzEmitterArray(pulsed_method="talbot_shaper")
controller = NeurotronicPhaseCasterWearable(emitter_backend=backend)

plan = backend.plan({"frequency_thz": 1.05, "amplitude_normalized": 0.5, "duration_ms": 150,
                     "pattern": "pulsed", "emitters": [0, 1, 2, 3], "phase_offset_rad": 0.0})
print(plan.summary())
mcu = plan.to_mcu_command()   # laser nm, VOA dB, stretcher um, bias gate, interlock flag
```

Existing code paths are unchanged when no backend is passed. `EmitterArray.set_power` requests in mW are clipped to the photomixer ceiling, and `power_shortfall_mw()` reports the gap instead of hiding it.

### Complexity

- `intensity_waveform`: O(M log M); `pulse_metrics`: O(N log N + M log M).
- `line_autocorrelation`: O(N log N); photomixer and rectifier responses: O(N log N).
- Etalon and Talbot: O(N). Kerr gate: O(n). Saturable absorber: O(n) per pass, typically 2 to 5 passes.
- Measured `plan()` latency (CPython, desktop CPU): 0.2 ms for the heterodyne path, 0.34 ms for the pulsed Talbot path at 1 THz (319 seed lines, 5120 samples). The full demo, including all three patterns, runs in under 1 s.

---

## 5. Safety review

- **THz skin exposure.** At the default 3.5 mm spot, a UTC-PD channel at full drive produces about 1.6 W/m², 0.16 % of the 1000 W/m² continuous limit. The interlock is wired into `plan()` and `apply_plan()` so the margin is checked rather than assumed. A different photomixer class or a tighter focus is re-checked automatically.
- **IR at 1550 nm.** About 1 W of amplified IR sits inside the fiber harness, which is Class 4 territory if a connector is opened or a fiber breaks. Required: APC connectors with shutters, a fiber-break or loss-of-light interlock on the EDFA, and an IR blocker (black PE or similar) in front of every antenna. Residual IR leaving the emitter must be measured separately; THz power meters need the same IR filter. Not modeled here.
- **Photodiode damage.** A pulsed drive raises peak photocurrent by the peak/average ratio (about 33 to 48 mA at 10 mA mean in the demo plans). The planner warns; the device limit must come from the photomixer datasheet (L2).
- **Kerr gate media.** `GateResult.damage_risk` flags pump intensities at or above the preset damage threshold (L2 values).
- **Scope of the limits.** The limit values are engineering interlocks taken from the standards, not a clearance for human exposure. Verify them against the current editions, under a laser safety officer, before any person is exposed.

---

## 6. Testing strategy

`python -m pytest -q tests` (60 tests, < 1 s):

- **L0 reproduction.** 8.014 nm, 149.9 µm / 23.86 µm ring, 193.414 THz carrier.
- **Analytic identities.** The two-line beat equals 1 + cos exactly; FWHM matches the Dirichlet kernel for N = 2 to 64; peak/average = N; the 0.8859/N asymptote holds; Parseval agreement between FFT harmonics and line autocorrelation.
- **Talbot.** r = 2, 3, 4, 10, 40: rate exactly r x seed, sub-pulse peak 1/r, energy conserved to 1e-12, suppression > 40 dB. Fiber without TOD matches the mask; fiber with TOD fails, as it physically should; seed design hits 1.07 THz at 20 % duty.
- **Etalon.** Every m-th line kept, throughput 1/m, suppression formula.
- **Gates.** Full switching at I_pi; the window equals the pump FWHM at pi and pump/sqrt(2) in the small-signal limit; the delayed medium leaves a tail; damage flag; absorber recovery time recovered to 0.1 % from the trace; periodic steady state.
- **Converters.** The photomixer matches the closed form; inverse and ceiling are consistent; the in-band ceiling is monotone; the pulse train gives 3 to 4x the sinusoidal fundamental; the rectifier's L_eff reduces to L² sinc² when lossless; the LiNbO3 tilted-pulse-front angle is 63°.
- **Tissue and limits.** Table values exact, monotone attenuation, profile consistent with the layer totals, limit continuity at 10 s, area reduction, aperture averaging.
- **Integration.** `EmitterArray` subclass, clipping and shortfall, frequency range, phase-to-delay mapping, amplitude mapping, every pulsed method, unclean trains flagged, unsafe plans refused, `THzCoherenceWearable` injection, `NeurotronicPhaseCasterWearable._emit_thz_pulse` producing a plan.

Hardware verification to add when parts exist: a calibrated THz power meter per channel versus `channel_max_power_w`; an optical spectrum analyzer on the seed and mask output; intensity autocorrelation of the pulse train; a two-channel THz interferometer to confirm phase-to-delay mapping and measure thermal drift.

---

## 7. Failure modes

- **Third-order dispersion in fiber Talbot.** Detected via `subharmonic_suppression_db` and flagged. Use the mask, a chirped fiber Bragg grating with TOD compensation, or a lower duty cycle.
- **Etalon leakage.** Finesse 400 at r = 40 leaves 18 dB suppression (flagged); finesse 2000 clears 20 dB. Throughput 1/r forces amplification after the filter, which adds ASE.
- **Phase drift.** 0.21 rad/K per metre of fiber at 1 THz (L1). Twelve uncontrolled fiber paths decohere within a few kelvin. Mitigations: matched short fibers in one thermal block, or closed-loop phase locking against a reference THz detector.
- **Polarization drift.** Modulation depth falls as abs(cos theta). Use PM fiber end to end, or a polarizer before each photomixer.
- **Laser wavelength drift.** The THz frequency drifts 1:1 with laser B's absolute frequency (DFB: about -12 GHz/K). Lock B to A with an offset-phase lock or a wavemeter loop.
- **Shaper bandwidth.** At 10 % duty and 1 THz the seed needs about 4.4 THz FWHM, more than a C-band shaper covers. The default duty is 20 % (2.2 THz FWHM), and the planner warns above the configured shaper bandwidth.
- **Photomixer saturation.** Thermal (mean current) and space-charge (peak current) limits are both L2. Only the thermal limit is enforced.
- **Microcomb retuning.** About -12 MHz/K cannot span 0.8 to 1.2 THz; each frequency needs its own ring.

---

## 8. Optimization opportunities

- Replace the per-port VOA with per-channel EDFAs or SOAs if channel counts or amplitudes diverge strongly. That trades amplifier count for splitter loss (11.8 dB at 1x12).
- Travelling-wave UTC-PDs: modeled at about 10 dB more output at 1 THz than lumped UTC-PDs in the literature. Swap the `Photomixer` preset when a measured response exists.
- Store `CombState` amplitudes as float32 complex64 on the MCU; every kernel is an FFT or recurrence that fits in CMSIS-DSP.
- Cache `pulse_metrics` for the pulsed carrier per (frequency, duty), since `plan()` recomputes it per pulse.

---

## 9. Future evolution

- Multiple frequency groups: one laser pair per group, or wavelength-selective-switch pairs from one 25 GHz comb, which also keeps groups mutually coherent.
- A closed-loop phase controller using one THz detector and dithered stretchers (pilot-tone phase locking).
- A measured-response import: `Photomixer.from_csv(frequency, power_at_current)` replacing the lumped model.
- A Lugiato-Lefever split-step solver behind `MicroringResonator`, to model soliton existence and access in place of the closed-form duration.
- Space-charge saturation in the photomixer (peak-current limit).

---

## 10. Open questions

- Which photomixer part, and what is its measured P(f, I)? Every absolute THz power in this document is an L1 estimate on L2 parameters until that measurement exists.
- What irradiance at what depth does the intervention model require? Without that target, no source choice can be sized.
- Is the intended agent the THz field, or the THz-rate IR envelope delivered directly to tissue? This module models the first. For the second, any absorber turns the envelope into heat with thermal time constants of microseconds to milliseconds, which averages a 1 ps modulation away (L0 thermal physics). A different mechanism would have to be specified and measured.
- How many independent frequencies must run at once? That determines the number of laser pairs.

---

## 11. Evidence vs speculation

- **L0.** Comb, Dirichlet, Talbot, Airy, heterodyne and Kerr-gate identities; the water absorption table; standard exposure limits; the 8.014 nm and 23.86 µm geometry.
- **L1.** The lumped photomixer, plane-wave rectification, soliton duration formula, first-order gate responses, Beer-Lambert stack, fiber thermal phase drift, and every absolute THz power quoted here.
- **L2.** Every preset device and material parameter (responsivity, transit time, capacitance, current limit, n2, damage thresholds, beta2, beta3, finesse, tissue thicknesses and water fractions), each marked in code.
- **L3.** Any neural, cognitive or therapeutic effect of the emitted field. The module contains no such model, and its tissue calculation shows that 0.8 to 1.2 THz radiation from any source deposits its energy in the outer tenths of a millimetre of skin.

### What would reduce uncertainty

Measured photomixer response, measured beam profile at the skin plane, measured tissue optical properties at the target sites, and a stated, measurable endpoint for whatever effect the system is meant to produce.

---

## References

- Water absorption at 0.5 to 2 THz: Diagnostics 12:2395 (2022), Table 2, PMC9600075.
- UTC-PD photomixers, ">10 µW at 1 THz": H. Ito, NTT Photonics Laboratories; PIN and UTC-PD comparison to 1 THz: J. Infrared Millim. Terahertz Waves (2019), doi:10.1007/s10762-019-00638-5.
- THz QCL maximum operating temperature about 261 K: Khalatpour et al., Appl. Phys. Lett. 122, 161101 (2023).
- 1 THz Si3N4 soliton microcomb: Drake et al., Phys. Rev. X 9, 031023 (2019).
- Temporal Talbot condition and line-by-line implementation: Optics Letters 36(6), 858 (2011); Zajnulina and Böhm, arXiv:2405.09160.
- Soliton duration: Herr et al., Nature Photonics 8, 145 (2014).
- Laser exposure limits, 180 nm to 1 mm: ICNIRP, Health Physics 105(3) (2013); IEC 60825-1:2014.
