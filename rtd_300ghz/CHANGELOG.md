# Ring-slot RTD transceiver change record

Against Randy Lynn’s design specification, §§1, 3, 4 and 5. **L0** covers identities, specified inputs, numerical settings and observed test results. **L1** covers model calculations. **L2** covers physical properties and performance requiring measurement. **L3** denotes speculation. Each row or paragraph labels all its quantities. Frequencies below use GHz.

## §1 · Device and reference plane

| Level | Change and derivation |
|---|---|
| L1 | At 300 GHz, $C_{eff}=C/(1-\omega^2L_bC)=7.072$ fF with $C=6.5$ fF and $L_b=3.5$ pH. The synthetic circuit adds the inductor. The barrier-port HFSS extraction contains the physical bridge. |
| L1 | The barrier-reference sheet spans $z=0.2950$–$0.3023$ µm. Remove its empty parallel-plate capacitance, $C_g=\epsilon_0(1.1\,\mu m)^2/(7.3\,nm)=1.468$ fF per port, via $Y_{ext}=Y_{EM}-sC_gI$. EM retains fringe capacitance. |
| L2 | For $a=18.2$ mS and $\Delta V=0.4$ V, $b=4a/(3\Delta V^2)=0.15167$ A/V³. At fixed junction offset $v_b$, the incremental cubic is $(-a+3bv_b^2)v+3bv_bv^2+bv^3$. At midpoint the quadratic term vanishes. |

## §3 · Stabilization and R2 loss

| Level | Change and derivation |
|---|---|
| L1 | $R_{max}=(1/a-R_s)/2=23.973$ Ω at $R_s=7.0$ Ω and $25.623$ Ω at $3.7$ Ω. Nominal $R_{stab}=(1/37.5+2/75)^{-1}=18.75$ Ω gives margins $1-R_{stab}/R_{max}=21.79\%$ and $26.82\%$. |
| L1, film properties L2 | $R=\rho L/(Wt)=37.383$ Ω for R1 and $74.766$ Ω per R2 film, with $\rho=2\times10^{-5}$ Ω·m, $L=10$ µm, $t=0.5$ µm and $W=10.7$ or $5.35$ µm. At $0.9$ V, $J=V/(\rho L)=4.5\times10^5$ A/cm². Both layouts have an Au ground strap. |
| L1 | At 300 GHz, $k=\sin(\beta s)/\sin(\beta L_{long}/2)$, $\beta=2\pi f\sqrt{6.8}/c$. The review’s $s=5$ µm CPW sample gives $k=0.08209$. The flank’s $s=46.6\pi(10/180)=8.133$ µm gives $k=0.13329$. $G_{pair}=2k^2/75=0.180$ and $0.474$ mS. The physical CPW centers lie at $\pm(3/2+10/2)=\pm6.5$ µm transverse, or $\pm46.6\arctan(6.5/48.9)=\pm6.16$ µm of ring arc. |
| L0 normalization, L1 estimates | For equal and opposite port peaks and $k=V_{film}/V_{port}$, $q=(1,-1)/\sqrt2$ gives $\Delta G_o=\Re(q^\dagger\Delta Yq)=G_{pair}/2=0.090$ and $0.237$ mS per port. Holding the spec’s per-port conductances fixed gives $\eta=0.10/(0.10+0.91+\Delta G_o)=0.091$ and $0.080$. Each HFSS layout has its own open-film control. The comparison reports both normalizations. |

**L1 stability.** The companion extraction exposes the bias pad and retains the films, choke, MIM and bridges. For $Y_b=1/(1\,\Omega+s\,1\,nH)$, $Y_L=Y_{aa}-Y_{ab}(Y_{bb}+Y_b)^{-1}Y_{ba}$. The directly terminated two-port checks this reduction. Closed positive- and negative-frequency RHP contours count $D=\det(Y_{ext}+Y_{RTD})/\det(Y_{ext}+G_rI)$, $G_r=(50\,\Omega)^{-1}$. The independent pencil must satisfy $N_{CCW}+P=Z$. The report lists every fitted RHP mode, its conjugate where applicable, and its relationship to the 1 GHz–fundamental band. Below-band and real-axis modes depend on the fitted continuation.

## §4 · Harmonic guards and layout clearance

**L1 cavity model.** The back reflector, top metallization and via fence form a pillbox of radius $R_c$ and height $h$. It supports $f_{mnp}=\frac{c}{2\pi\sqrt{12.5}}\sqrt{(x_{mn}/R_c)^2+(p\pi/h)^2}$ with $x_{mn}=j_{mn}$ for TM$_{mnp}$, $p\ge0$, and $x_{mn}=j'_{mn}$ for TE$_{mnp}$, $p\ge1$. The axial step is $c/(2h\sqrt{12.5})=599.675$ GHz at $h=70.7$ µm, so $p=1$ modes reach the second and third guards. The check enumerates every mode up to the top guard edge, with no fixed $m$, $n$ or $p$ limit. A brute-force test confirms completeness.

**L1 coupling rule.** The circumferential slot field is radial, so its equivalent magnetic current is azimuthal. It couples through $H_\phi$ at the top wall. TE$_{0np}$ has $H_\phi\equiv0$ and is excluded. Every other family is checked. The feed channel, films and port gaps break this symmetry, and HFSS remains the arbiter.

**L1 guards.** The guards $[0.97h(296.1),1.03h(307.7)]$ equal $[287.217,316.931]$, $[574.434,633.862]$ and $[861.651,950.793]$ GHz. A radius passes only if no slot-coupled mode enters a guard at nominal $h$ or anywhere in $h=70.7\pm2$ µm, the step-7 thinning tolerance. Each $(\text{mode},\text{band})$ pair forbids the closed interval $h\in[p k/\sqrt{f_{hi}^2-f_r^2},\,p k/\sqrt{\max(f_{lo}^2-f_r^2,0)}]$, where $k=c/(2\sqrt{12.5})$ and $f_r$ is the transverse cutoff. The clear thickness window is therefore exact, not sampled.

| $R_c$ (µm) | L1 result |
|---|---|
| 71.5 | Reject (superseded value). TM₁₁₁ = 939.50 GHz lies in the third guard at nominal $h$. It is degenerate with the uncoupled TE₀₁₁ because $j'_{01}=j_{11}$. It stays inside over the whole $\pm2$ µm range, at 950.74–929.06 GHz. The earlier TM$_{mn0}$-only scan passed this radius. |
| 70.0 | Nominal pass, tolerance reject. TM₁₁₁ enters the third guard at $h=70.829$ µm. |
| **68.75** | **Pass (new default).** Modes below the top guard: TM₀₁₀ 472.06, TE₁₁₁ 700.17, TM₁₁₀ 752.15, TM₀₁₁ 763.18 and TE₂₁₁ 847.97 GHz. TE₂₁₁ is 13.68 GHz below the third guard. TM₁₁₁ = 961.95 GHz is 11.15 GHz above it. The clear window is $h\in[68.507,72.895]$ µm, a margin of $-2.193/+2.195$ µm limited by TE₂₁₁ and TM₁₁₁. |
| 78.0 | Reject. TM₂₁₀ = 888.55 GHz enters the third guard. |
| 85.0 | Reject. TM₁₁₀ = 608.36 GHz and TM₀₂₀ = 876.42 GHz enter the second and third guards. |

**L1 feasible radii.** On a $60$–$95$ µm grid at $0.05$ µm, $66.65$–$70.05$ µm clear at nominal $h$ only. Only $68.55$–$68.85$ µm clear over $\pm2$ µm. $68.75$ µm balances the two thickness margins and becomes the default. The radius window is $\pm0.15$ µm wide. That is comparable to the L1 uncertainty in the effective wall radius of a $10$ µm via fence at $19$ µm pitch, so confirm it with an HFSS $R_c$ sweep across $68.55$–$68.85$ µm. Either tightening the thinning tolerance or relaxing the third guard widens the window. `--thickness-tol-um 0` restores a nominal-only check.

**L1 clearance.** The inner via edge is $68.75-5=63.75$ µm. Clearance is $63.75-(46.6+5+1.5)=10.65$ µm nominal and $63.75-53.369=10.38$ µm to the outermost drawn film corner. With the spec’s $30$ µm feed opening, $1+\lceil(2\pi R_c-30)/20\rceil=22$ vias give $19.141$ µm ordinary pitch. A radius sweep sizes its via count for the largest radius.

## §5 · Process, extraction and verification

**L2 process.** Form the front-side via ring after front alignment and before the films, oxide and bridge plating. Protect the epitaxy, etch blind vias to $75$–$80$ µm, plate, fill and planarize. Tie the lands to ground in step 6. In step 7, thinning to $70.7\pm2$ µm exposes $75-72.7=2.3$ to $80-68.7=11.3$ µm of fill before reflector deposition. Qualify fill, planarity and contact continuity. The lens branch thins and bonds Si.

**L1 extraction.** The riser starts at $t_{metal}$ inside the CPW. The subcollector begins at $R_{in}$. Full plate overlap gives $C_{MIM}=\epsilon_0(3.9)(60\,\mu m)^2/(100\,nm)=1.243$ pF. Solid cross-sections and AEDT strict validation check intersections. Embedded Jones fields give broadside AR. Radiation uses imported $\eta_o=P_{rad,o}/P_{acc,o}$ and the complex radiation matrix for the actual drive. HB reports $P_{acc,h}=\Re(V_h^\dagger I_h)/2$ and $P_{R_s}=R_s\sum_{h,p}|I_{h,p}|^2/2$. At $\delta=0.05$, $\epsilon=j\omega C\delta/[2(Y_e-Y_o)]=0.001528+j0.000414$ for the spec’s $Y_e-Y_o=50.6+j186.8$ mS.

**L1 synthetic benchmark.** Section 2 supplies $G_o=0.91+0.10=1.01$ mS and $B_o=79.5-93.4=-13.9$ mS. The fixture assumes $C_t=10$ fF, giving $L_t=[\omega_0(\omega_0C_t-B_o)]^{-1}=16.199$ pH. It uses the §1 narrowband slope $a_{RF}=a\cos(2\pi f\tau)=16.922$ mS at $300$ GHz and $0.2$ ps, keeps the DC-calibrated cubic coefficient, and imports a synthetic efficiency of $0.10$. The three-harmonic solve gives **300.014 GHz, 180.000°, amplitude ratio 1.000, 11.243 µW radiated and 0.16841 mW series loss**. **L0 verification.** All 22 pytest cases passed. AEDT Desktop was unavailable. Physical R2 loss, spectra, polarization and the device stability verdict await field exports.

### Remaining assumptions

- **L1 physical model.** Ideal circular PEC walls at the via-centre radius screen the finite fence. The cavity has PEC top and bottom walls. Slot coupling follows the azimuthal-symmetry rule of §4. Isotropic, lossless dielectrics, stock AEDT gold, equipotential mesa stubs, a gap-free uncoated $6$ mm Si hemisphere with zero extension and $\epsilon_r=11.7$. Broadside is $+z$ for the cavity and $-z$ for the lens. Contact resistance and film resistivity require L2 measurements.
- **L1 drawing.** The CPW films center at $46.6+2.3=48.9$ µm. Contacts use $1.5$ µm overlap, $0.25$ µm shoulders and $2$ µm tapers. Strap feet occupy $x=57$–$59$ and $|y|=5.5$–$7$ µm in the local feed frame, with $1.2$ µm air clearance. The choke runs from ring mean to riser center. Short MIM tongues and the pad-edge RL reference represent the bias hookup.
- **L1 device.** Fixed junction DC voltage and temperature. Default production HB uses the memoryless midpoint cubic. An optional causal delay applies $(1-s\tau/2)/(1+s\tau/2)$ to conduction in HB and Nyquist. Off-midpoint HB holds the junction DC voltage fixed during rectification. Physical powers remain L2.
- **L0 solver settings.** Discrete sweeps cover $1$–$100$, $100$–$450$ and $450$–$950$ GHz at nominal $1$ GHz spacing. Mesh adaptation uses $300/600/900$ GHz and $\Delta S=0.01$. FE-BI encloses the device. Far-field steps are $2°$ cavity and $1°$ lens. HB uses peak phasors, $H=3$, $8H+1=25$ time samples, a $280$–$315$ GHz root window and $1\%$ mismatch increments. The analytic window is $220$–$380$ GHz.
- **L0 stability settings.** Join warning and raw/loaded/modal fit tolerances are $1\%$. Loss-eigenvalue roundoff tolerance is $10^{-12}$ S. Poles beyond $100$ times the sampled angular-frequency ceiling contribute their constant and slope before revalidation. Two fit orders must agree on in-band frequency within $1\%$ and growth within $10\%$. At most one pair inside the tuning guard, nearest the HB fundamental with over $90\%$ odd voltage fraction, receives the intended-mode designation.

### Run

Keep the Python files together. Inspected versions **L0** were Python 3.12, PyAEDT 1.10.0 and scikit-rf 2.2.0. [HFSS lumped ports](https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.lumped_port.html) · [HFSS antenna quantities](https://ansyshelp.ansys.com/public/views/secured/electronics/v261/en/subsystems/hfss/Content/ReportsandPostProc/SelectingAntennaParameterstoPlot.htm) · [scikit-rf power waves](https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.network.z2s.html)

```bash
python -m pip install 'pyaedt==1.10.0' 'scikit-rf==2.2.0' 'shapely>=2.1,<3' numpy scipy pytest
python -m pytest -q test_300ghz_rtd.py
python build_300ghz_rtd_ringslot.py api-check
python build_300ghz_rtd_ringslot.py preflight --compare-r2
python build_300ghz_rtd_ringslot.py build --compare-r2 --solve --out runs
python build_300ghz_rtd_ringslot.py build --variant si_lens --solve --out runs
python sim_300ghz_rtd_cosim.py analytic --out angles.csv
python sim_300ghz_rtd_cosim.py analyze runs/cavity_via/cpw_gap --out analysis
```

Read `R2_comparison_at_300GHz.csv` for the side-by-side HFSS increments, `hb_mismatch.csv` for the oscillator sweep and `stability.json` for windings, roots and fit diagnostics.
