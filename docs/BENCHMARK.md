# Benchmark matrix (P6)

Controller x scenario x disturbance, **12 episodes/cell**, vision/estimate pipeline (no ground truth in the loop). Produced by `scripts/benchmark.py`. Geometric is the default controller; alternatives are for reference.

## Single-drone landing (geometric)

| configuration | success | horiz-err (succ, m) | mean impact (m/s) | non-success outcomes |
|---|---|---|---|---|
| ground (cruising rover) | 83% | 0.17 | 0.09 | {'out_of_bounds': 1, 'crash': 1} |
| ground-hard (fast/random) | 8% | 0.36 | 0.10 | {'off_platform': 3, 'crash': 4, 'out_of_bounds': 3, 'timeout': 1} |
| ship / calm | 100% | 0.08 | 0.09 | - |
| ship / moderate | 92% | 0.13 | 0.13 | {'out_of_bounds': 1} |
| ship / rough | 83% | 0.21 | 0.14 | {'out_of_bounds': 2} |
| ship / moderate + green-deck | 92% | 0.13 | 0.13 | {'out_of_bounds': 1} |
| offshore OSV | 100% | 0.12 | 0.12 | - |
| inclined / gentle 6deg | 100% | 0.10 | 0.11 | - |
| inclined / moderate 12deg | 0% | - | 0.10 | {'timeout': 12} |
| USV (maneuver + rock) | 83% | 0.17 | 0.16 | {'out_of_bounds': 1, 'crash': 1} |
| moving truck | 100% | 0.11 | 0.10 | - |

## Controllers (ground & ship)

| configuration | success | horiz-err (succ, m) | mean impact (m/s) | non-success outcomes |
|---|---|---|---|---|
| ground / geometric | 83% | 0.17 | 0.09 | {'out_of_bounds': 1, 'crash': 1} |
| ground / IBVS | 58% | 0.28 | 0.05 | {'timeout': 2, 'out_of_bounds': 1, 'off_platform': 2} |
| ground / MPC | 33% | 0.15 | 0.10 | {'out_of_bounds': 8} |
| ground / RL residual | 100% | 0.14 | 0.12 | - |
| ship / geometric | 92% | 0.13 | 0.13 | {'out_of_bounds': 1} |
| ship / IBVS | 83% | 0.16 | 0.10 | {'out_of_bounds': 2} |

## Disturbance rejection & safety (ship/offshore)

| configuration | success | horiz-err (succ, m) | mean impact (m/s) | non-success outcomes |
|---|---|---|---|---|
| ship (baseline) | 92% | 0.13 | 0.13 | {'out_of_bounds': 1} |
| ship + air-wake | 58% | 0.23 | 0.23 | {'out_of_bounds': 5} |
| ship + air-wake + DOB | 83% | 0.17 | 0.20 | {'out_of_bounds': 2} |
| ship + spectral waves (B1) | 83% | 0.08 | 0.09 | {'out_of_bounds': 2} |
| ship + shield (B2) | 92% | 0.13 | 0.13 | {'out_of_bounds': 1} |
| offshore + sense-and-avoid | 100% | 0.14 | 0.10 | - |

## Swarm (kinematic, no-cheats sensing) — all-landed % over episodes

| configuration | all-landed | mean recovered | min-sep (m) | obstacle clear (m) |
|---|---|---|---|---|
| 4 drones / ship | 100% | 100% | 1.04 | - |
| 6 drones / ship | 100% | 100% | 0.87 | - |
| 6 drones / consensus (A2) | 100% | 100% | 0.93 | - |
| 5 drones / offshore + avoid (P3) | 100% | 100% | 0.82 | +0.25 |
| 9 drones -> 3 decks (deck 2 fouls) | 100% | 100% | 0.76 | - |

_Produced in 1131s._
