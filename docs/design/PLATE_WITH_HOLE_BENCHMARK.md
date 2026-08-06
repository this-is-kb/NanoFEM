# NanoFEM Verification Benchmark: Plate with a Circular Hole (v0.19.0)

**Status:** implemented and tested. This is the final Stage-3 gap: Step 8's "infinite plate with
a hole" and "mesh convergence" requirements, both satisfied by one deliberately-chosen benchmark
rather than two separate ones (the directive's own "do not implement unnecessary benchmark
examples" caution). Companion to every increment this session shipped: `ELEMENT_FACTORY.md`
(`Model`-driven Q4 solve), `POSTPROCESSING.md` (stress recovery), `TRACTION_LOADS.md` (the
`TractionLoad` the remote tension is applied through).

---

## 1. Why this one benchmark, not two

Stage 3's constant-strain patch test (v0.12.0/v0.13.0) and the beam cantilever benchmarks
(v0.10.0/v0.11.0) were already real, shipped, and verified. What remained were "infinite plate
with a hole" and "mesh convergence" for the 2-D continuum elements specifically. A constant-strain
field is, by construction, exactly reproducible on *any* mesh (that is what "constant strain
patch test" means), so it cannot demonstrate convergence - there is nothing to converge *to*
beyond the exact answer a single element already gives. The Kirsch plate-with-hole problem has a
genuinely non-constant stress field with a real gradient, so a coarse mesh cannot reproduce it
exactly and refinement must visibly close the gap - one problem, two checklist items, no
duplicated benchmark machinery.

## 2. Two numerically-caught details, before any test code was trusted

**A finite-width plate's stress concentration factor is genuinely, not buggily, higher than 3.**
The classical infinite-plate Kirsch result (`sigma_theta_theta = 3S` at the hole boundary,
`theta = 90` deg) only holds in the limit of an infinitely large plate. A first attempt at `W/a =
4` (plate half-width four times the hole radius) converged - cleanly, monotonically, to a stable
number - at **~3.58**, not 3.0. This was investigated as a suspected bug before being confirmed as
the real, well-documented finite-width correction (Peterson's charts): a relatively large hole
relative to the plate genuinely raises the true concentration factor above 3. Increasing to `W/a
= 10` (verified to reduce the correction to within a percent or two) was the fix - not a code
change, a benchmark-design change, made only after confirming the smaller-`W/a` result was real
physics and not an artifact.

**Uniform radial mesh spacing converges far slower than graded spacing for a stress
concentration.** A first mesh with uniform radial subdivision (equal-width rings from the hole
outward) was still visibly climbing toward 3 at 64x128 elements, not yet close. Switching to a
quadratically-graded radial spacing (`t = (i/n_r)^2`, clustering elements near the hole where the
gradient is steepest) converged to within a few percent by 32x64 elements - the standard,
textbook-documented fix for exactly this kind of problem (a boundary-layer-like feature needs
mesh density near the feature, not uniformly across the domain), confirmed by direct numerical
comparison of both grading schemes before choosing the graded one for the shipped test.

## 3. The model

Double symmetry reduces the problem to a quarter-plate: `u_y = 0` on the bottom edge (`theta =
0`), `u_x = 0` on the left edge (`theta = 90` deg), the hole boundary traction-free (natural BC),
and the remote tension applied as a real `TractionLoad` (`(S, 0)`) on the lower portion of the
right outer edge - the top outer edge is traction-free by construction, since the remote stress
state has `sigma_yy = 0` there. Two independent analytical checkpoints are read from the two
corner elements touching the hole (each belongs to exactly one Q4 element in this mesh topology,
so element-average and nodal-average recovery coincide there): `sigma_xx = 3S` at the top of the
hole, `sigma_yy = -S` at the side - the *different* Cartesian component at each point, since the
tangential direction the Kirsch formula is stated in rotates with `theta`.

---

## 4. Verification

`tests/unit/test_plate_with_hole_benchmark.py` (3 tests): mesh-convergence (three refinement
levels, `(6,12) -> (16,32) -> (32,64)`, asserting the error against the 3.0 target strictly
decreases and the finest level is within 10%); the independent side-of-hole checkpoint against
-1.0 (within 15%, a looser tolerance since it is checked at a single, coarser mesh level, not the
finest); global equilibrium (reactions sum to exactly minus the total applied force, to
`rtol=1e-9` - this one is exact at any mesh density, since it follows from the assembled system's
own algebra, not from mesh resolution).
