# Physics and models

This document states the equations the analytic backend implements, the
parameter values it ships with, and the limits of what it can tell you.

Every quantity in the library's public API is SI. The command line accepts the
units an engineer actually types (micrometres, nanometres, milliwatts) and
converts at the boundary.

## 1. Conventions

Each component contributes a complex **field** transfer coefficient. Power is
the squared magnitude of the accumulated field:

```
P_out = |E_out|^2
```

Losses are quoted in dB of power, so a loss `L_dB` multiplies the field by
`10^(-L_dB/20)` and the power by `10^(-L_dB/10)`.

## 2. Waveguide

### Effective index

The effective index is a second-order Taylor expansion about a reference
wavelength, the form used by the SiEPIC compact models:

```
n_eff(lam) = c0 + c1 (lam - lam0) + c2 (lam - lam0)^2
```

with the offset `(lam - lam0)` expressed in **micrometres**. Defaults describe
the fundamental TE mode of a 220 nm x 500 nm SOI strip waveguide:

| Parameter | Value      |
| --------- | ---------- |
| `c0`      | 2.4379     |
| `c1`      | -1.1217    |
| `c2`      | -0.04565   |
| `lam0`    | 1.55 um    |

### Group index

```
n_g = n_eff - lam * dn_eff/dlam
```

At 1550 nm this gives `n_g = 4.1765`, consistent with the 4.1-4.3 typically
measured for this geometry. The test suite checks the analytic derivative
against a finite-difference derivative, and checks the value falls in a
physically sensible range.

### Propagation

Over a length `L` the waveguide contributes phase and attenuation:

```
E_out / E_in = 10^(-alpha * L_cm / 20) * exp(-j * 2 pi n_eff(lam) L / lam)
```

The default propagation loss is `alpha = 2.4 dB/cm`.

## 3. Grating coupler

A Gaussian spectral response about the design wavelength:

```
T(lam) = 10^(-IL/10) * exp(-4 ln2 ((lam - lam_c) / BW)^2)
```

`BW` is the full width at half maximum, so `T` is exactly half its peak at
`lam_c +/- BW/2` -- which the test suite verifies. Defaults: `IL = 3 dB`,
`lam_c = 1550 nm`, `BW = 40 nm`.

A circuit has two couplers, one in and one out, so their loss enters twice.

## 4. Y-branch

Modelled as an ideal 50/50 power split with a wavelength-independent excess
loss of 0.15 dB. Each branch carries a field amplitude of

```
10^(-EL/20) / sqrt(2)
```

This is the model's most significant simplification: a real Y-branch has a
small, wavelength-dependent imbalance between its arms, which is what limits
the extinction ratio of a fabricated MZI. See "Limitations" below.

## 5. Circuit solutions

### Straight link

```
E_out = t_gc * t_wg(L) * t_gc
```

Peak loss is therefore `2 * IL + alpha * L`, verified directly by the tests.

### Mach-Zehnder interferometer

The splitter and combiner each contribute `1/sqrt(2)` per branch:

```
E_out = t_gc * t_y * ( t_1 e^{-j phi_1} + t_2 e^{-j phi_2} ) * t_y * t_gc
```

With lossless, ideal components this reduces to the familiar cosine-squared
law:

```
T(lam) = cos^2( pi n_eff(lam) dL / lam ),    dL = L_1 - L_2
```

The test suite asserts this to within 1e-12 using a lossless parameter set, so
a regression in the interference maths shows up immediately.

### Free spectral range

Adjacent nulls are separated by

```
FSR = lam^2 / (n_g dL)
```

For the default 150/50 um MZI (`dL = 100 um`) this predicts **5.75 nm** at
1550 nm. Measuring the spacing of minima in the simulated trace gives
**5.75 nm**, agreeing to better than 0.1 %. Because `FSR` scales with `lam^2`,
it drifts by a few percent across a 1500-1600 nm sweep; the reported figure is
the mean over the sweep.

The library reports the closed-form prediction as
`result.metadata["predicted_fsr_m"]` alongside the value measured from the
trace, so the two can be compared at a glance.

## 6. Monte Carlo

Waveguide width and thickness variation both appear, to first order, as an
offset in effective index. The simulator draws an independent Gaussian offset
`dn ~ N(0, sigma)` per waveguide and applies it as the equivalent optical-path
change:

```
dL_equivalent = L * dn / n_eff
```

which produces exactly the same phase as perturbing the index directly.

The default `sigma_n_eff = 2e-3` corresponds to a few nanometres of width and
thickness variation, taking sensitivities of order `1e-3` per nm of width and
`2e-3` per nm of thickness. **This is an order-of-magnitude figure**, chosen so
the default plot shows a meaningful spread. Calibrate it against wafer data
before drawing any yield conclusion.

Runs are seeded by default, so a Monte Carlo study is reproducible; the tests
assert that two runs with the same seed are bit-identical.

### What the default study shows

At `sigma_n_eff = 2e-3` on a 150 um arm, the index offset alone shifts that
arm's phase by roughly

```
dphi = 2 pi dn L / lam = 2 pi * 2e-3 * 150e-6 / 1.55e-6 ~ 1.2 rad
```

and the two arms vary independently, so the differential phase routinely
exceeds a full fringe period. The consequence is visible in the figure: the
*envelope* is stable, since it is set by the grating couplers, while the
*absolute fringe position* is essentially uniformly distributed. Free spectral
range, which depends on the group index rather than on accumulated phase,
stays tight — a few picometres of spread on 5.75 nm.

This is the real behaviour of an unbalanced SOI interferometer, and it is why
fabricated devices of this kind need thermal tuning to land on a target
wavelength. Reduce `sigma_n_eff` to around 5e-4 to see the tight-band regime
instead.

## 7. Sampling

An interferometer's fringes narrow as the arm imbalance grows. If the sweep
resolution approaches the fringe spacing, the trace aliases and the plot is
misleading rather than merely coarse. The library warns
(`UnderSampledSweepWarning`) when fewer than eight points fall within one
predicted FSR. As a rule of thumb the sweep needs

```
points > 8 * (span * n_g * dL) / lam^2
```

For a 100 nm span and a 100 um imbalance that is about 140 points; for a
1000 um imbalance, about 1400.

## 8. Limitations

These matter if you plan to compare against measurement.

1. **Nominal, not PDK-calibrated.** The coefficients above describe a typical
   SOI strip waveguide, not a specific foundry process. The *shape* of the
   response (fringe spacing, envelope width, loss slopes) is trustworthy; the
   *absolute* fringe positions are not, because they depend on `n_eff * L`,
   where a 0.1 % index error shifts fringes by a substantial fraction of one
   period. Running the same circuit through the `simphony` backend shows this
   directly: the two agree on FSR to better than 1 % while their fringes sit
   at visibly different absolute positions.

2. **Ideal splitters.** Extinction ratio is limited only by arithmetic, so the
   analytic backend reports very high values (tens of dB) where a fabricated
   device would be limited to roughly 20-30 dB by splitter imbalance. The
   `simphony` backend, which uses measured S-parameters, is the better guide
   here.

3. **Single polarisation, single mode.** TE only. No polarisation conversion,
   no higher-order modes, no back-reflection, and no thermal or free-carrier
   effects.

4. **Two topologies.** Straight links and single-stage MZIs. Ring resonators,
   cascaded or nested interferometers, and directional couplers are not
   implemented; the netlist validator rejects them explicitly rather than
   silently producing a wrong answer.

5. **Monte Carlo is uncorrelated.** Each waveguide varies independently. Real
   wafer variation is spatially correlated, so nearby arms drift together;
   that correlation both reduces and reshapes the spread relative to this
   model.

## References

The compact-model forms and the parameter ranges used here follow standard
silicon-photonics practice, in particular:

- Chrostowski & Hochberg, *Silicon Photonics Design: From Devices to Systems*,
  Cambridge University Press, 2015 -- waveguide compact models, MZI transfer
  functions, and the SiEPIC parameter conventions.
- The SiEPIC EBeam PDK component library, whose measured S-parameters back the
  `simphony` backend.
