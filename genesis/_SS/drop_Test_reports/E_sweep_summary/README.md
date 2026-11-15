# E sweep: sphere drop damping vs. mud stiffness (2026-10-27)

Goal: find a Young's modulus E (yield stress fixed at 80 Pa) at which the sphere's depth after impact is
critically damped or overdamped, i.e. it settles without springing back or bobbing.

All runs: configs/sphere_drop.yaml defaults (1 kg, r 0.05 m sphere dropped from 0.5 m; elasto_plastic mud,
nu 0.35, rho 1500, yield 80 Pa; grid_density 48; dt 0.005 s; 1 s settle + 1.5 s recorded), changing only
mud.E and sim.substeps. Each run folder has its own config.yaml.

- depth_vs_time_by_E.png: depth after contact, one row per substep count, with a zoom on the impact.
- summary.txt / summary.json: per-run metrics. "max mm" is the depth where the sphere first stops,
  "springback mm" how far it rises after that, "revs" the number of velocity reversals over 2 mm/s,
  "creep mm" how far it is still sinking over the last 0.5 s.
  Verdict: underdamped if springback > 0.5 mm or more than one reversal.
- analyze_esweep.py: regenerates these: python analyze_esweep.py ../2026-10-27_23-*/ --out .

| run folder | substeps | E (Pa) |
|---|---|---|
| 2026-10-27_23-02-07 | 15 | 2e4 |
| 2026-10-27_23-02-32 | 15 | 5e4 |
| 2026-10-27_23-03-10 | 15 | 1e5 |
| 2026-10-27_23-03-46 | 15 | 2e5 |
| 2026-10-27_23-04-23 | 15 | 5e5 |
| 2026-10-27_23-04-59 | 15 | 1e6 |
| 2026-10-27_23-08-26 | 45 | 2e4 |
| 2026-10-27_23-09-20 | 45 | 5e4 |
| 2026-10-27_23-06-20 | 45 | 1e5 |
| 2026-10-27_23-10-12 | 45 | 2e5 |
| 2026-10-27_23-11-04 | 45 | 5e5 |
| 2026-10-27_23-07-18 | 45 | 1e6 |
| 2026-10-27_23-11-55 | 90 | 2e4 |
| 2026-10-27_23-15-21 | 90 | 1e5 |
| 2026-10-27_23-13-13 | 90 | 1e6 |
| 2026-10-27_23-16-37 | 180 | 2e4 |

(2026-10-27_22-28-17 is an earlier short CPU test of the video fix, not part of the sweep.)

## Findings

1. Raising E does not monotonically reduce bounce. Springback is smallest around E = 1e5-2e5 and grows
   again above that (5e5, 1e6 spring back 4-13 mm).
2. Critically/overdamped (no springback, no reversals): E = 1e5 and 2e5 at 45 substeps; E = 2e4 and 1e5
   at 90; E = 2e4 at 180. The calibrated E = 2e4 bobs at 15 and 45 substeps but not at 90+, so most of the
   bounce at the default 15 substeps is a time-step artifact.
3. Results are not converged in substeps. Final depth at E = 2e4: 82 mm (15), 57 mm (45), 41 mm (90),
   27 mm (180), with peak force rising 66 -> 100 N. Penetration depth is being set by the time step as much
   as by the material.
4. The sphere never comes to rest: every run is still sinking 1-4 mm per 0.5 s at the end. An 80 Pa yield
   stress cannot hold a 1 kg, 5 cm sphere statically, so it creeps down.
