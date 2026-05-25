# Command-line reference

The project ships **two console commands**, installed by `pip install -e .`:

| command | purpose | module |
|---|---|---|
| **`drone`** | single-drone landing simulator | `drone_landing` |
| **`swarm`** | multi-drone flight-deck recovery (separate module) | `drone_landing_swarm` |

`drone` has the aliases `dl` and `drone-landing`. Without an install, run `python -m drone_landing ...`
and `python -m drone_landing_swarm ...` with `PYTHONPATH=src`.

Every run is the **full closed-loop vision autopilot** (ArUco → EKF → supervisor → control) on simulated
onboard sensors; ground truth is read **only to score outcomes**, never inside a control decision.

- **Headless vs. live:** `run` / `parallel` / `verify` / `eval` print metrics tables; `watch` opens an
  interactive MuJoCo (or matplotlib) window.
- **Quick discovery:** `drone list`, `drone info`, `swarm info` print the live capability summary.

---

## `drone` — single-drone simulator

```
drone {list,info,run,watch,eval,reachability,safety,train,parallel} [options]
```

### Discovery

| command | what it shows |
|---|---|
| `drone list` | scenarios, controllers, sea states, all named presets, and the flag summary |
| `drone info` | build status: worlds found, optional deps (OpenCV/CasADi/Gym/SB3), compute device (GPU/CPU), trained RL policies |

### `drone run` — headless batch of episodes

```bash
drone run [PRESET] [--scenario ...] [--controller ...] [flags] [--episodes N] [--seed S] [--json] [--quiet]
```
Runs `--episodes` (default 10) from `--seed` and prints a success / touchdown-error / impact-velocity table
with a non-success outcome breakdown. `--json` emits one summary line (for scripting); `--quiet` hides the
per-episode lines.

### `drone watch` — live MuJoCo viewer

```bash
drone watch [PRESET] [flags] [--episodes N] [--seed S] [--speed X]
```
Opens one interactive window (drag to orbit, scroll to zoom, Tab cycles cameras). `--speed` is a real-time
multiplier.

### Configuration flags (shared by `run` and `watch`)

| flag | values / default | meaning |
|---|---|---|
| `--scenario` | `ground` `ship` `offshore` `inclined` `usv` `truck` | landing target |
| `--controller` | `geometric` (default) `mpc` `ibvs` `rl` | control law |
| `--sea` | `calm` `moderate` `rough` | sea state (seakeeping scenarios) |
| `--incline` | `gentle` `moderate` `steep` | tilt level (`inclined` scenario) |
| `--sea-model` | `sinusoid` (default) `spectral` | `spectral` = JONSWAP/PM spectrum + RAOs (B1) |
| `--airwake` | flag | ship air-wake (burble) turbulence over the deck |
| `--green-deck` | flag | time the commit to a low-motion (green) deck window |
| `--motion-data` | path to CSV | drive the deck from a recorded 6-DOF seakeeping trace |
| `--vmax` `--amax` `--jerk` | floats | ground-rover motion limits (m/s, m/s², m/s³) — stress-test knobs |
| `--no-wind` | flag | disable the (default-on) correlated wind gusts, for ablation |
| `--dob` | flag | wind-aware disturbance-observer feedforward |
| `--markerless` | flag | classical deck-pad fallback when the ArUco can't be decoded |
| `--cnn-markerless` | flag | learned CNN deck detector fallback (B3; `runs/cnn/deck_cnn.pt`) |
| `--shield` | flag | reachability safe-descent shield (B2) — clamp descent so a soft touchdown stays reachable |
| `--avoid` | flag | in-loop sense-and-avoid (higher-order CBF) + obstacle-abort around the OSV superstructure (P3) |
| `--fail-rotor` | `0..3` | kill this rotor mid-flight (rotor-out stress test) |
| `--fail-rotor-mode` | `descent` (default) `floquet` | rotor-out handling: bounded spinning descent, or the averaged-precession controller |
| `--controller rl` + `--rl-policy` | path | use a specific trained policy `.zip` (default: `runs/rl/<scenario>/ppo_final.zip`) |

### `drone eval` — green-deck ablation

```bash
drone eval --seas moderate rough [--episodes N] [--seed S]
```
Baseline vs. green-deck commit timing, per sea state, side by side.

### `drone reachability` — formal safety (B2)

```bash
drone reachability [--d-max 2.0] [--w-land 0.6] [--trials 200]
```
Computes the Hamilton–Jacobi safe-landing set (printed as an ASCII map with the analytic braking curve) and
runs a shielded-vs-unshielded A/B against a reckless dive under random wind. `--d-max` = disturbance bound
(m/s²), `--w-land` = max soft-touchdown speed (m/s).

### `drone safety` — sense-and-avoid + contingency (P3)

```bash
drone safety [--demo avoid|contingency|all]
```
`avoid` = higher-order-CBF obstacle avoidance demo; `contingency` = the PX4-style failsafe FSM
(geofence / low-battery RTL / lost-comms / abort / rotor-out); `all` = both.

### `drone train` — reinforcement learning

```bash
drone train [--scenario ground|ship] [--algo ppo|recurrent_ppo] [--timesteps N] [--n-envs N]
            [--device auto|cpu|cuda] [--no-curriculum] [--normalize] [--anneal-lr]
            [--resume CKPT] [--save-dir DIR] [--seed S]
drone train --eval runs/rl/ground/ppo_final.zip [--scenario ...] [--algo ...]   # score vs baseline
```
Residual PPO on the geometric+supervisor baseline. `--algo recurrent_ppo` is the LSTM variant for the
partial-observability POMDP. `--eval` scores a checkpoint (RL-residual vs. zero-action baseline) on the hard
regime instead of training. `--normalize`/`--anneal-lr` exist but are **off by default** (they hurt this
shaped reward — see [RESEARCH_NOTES.md](RESEARCH_NOTES.md)).

### `drone parallel` — many presets at once

```bash
drone parallel PRESET [PRESET ...]      # or:
drone parallel --all [--episodes N] [--workers W] [--seed S]
```
Runs each preset in its own process (`ProcessPoolExecutor`) and prints a combined table.

### Named presets (`drone run <preset>`)

| preset | configuration |
|---|---|
| `ground` / `ground-mpc` / `ground-ibvs` | ground rover with geometric / MPC / IBVS |
| `ground-hard` | fast random rover (`vmax 1.5, amax 0.8, jerk 2.0`) — the stress regime |
| `ship-calm` / `ship` / `ship-rough` | ship deck at calm / moderate / rough sea |
| `ship-green` / `ship-rough-green` | + green-deck commit timing |
| `ship-spectral` | JONSWAP/PM spectral waves (B1) |
| `ship-airwake` | + ship air-wake turbulence |
| `offshore` / `offshore-rough` / `offshore-green` | offshore OSV vessel |
| `inclined` / `inclined-moderate` / `inclined-steep` | tilted deck (gentle/moderate/steep) |
| `inclined-shield` | steep incline + reachability shield |
| `usv` | agile surface craft (maneuvers + rocks) |
| `truck` | road vehicle on a smooth loop |

Flags override a preset, e.g. `drone run ship --sea rough --green-deck --dob`.

---

## `swarm` — multi-drone flight-deck recovery

A separate module that reuses the single-drone components without modifying them. Coordination runs on
each drone's **onboard estimates + latency/dropout-limited neighbour broadcasts** — no ground truth in any
decision. See [SWARM.md](SWARM.md) for the design.

```
swarm {info,run,watch,verify,multi,marl,gnn,active} [options]
```

### Common flags (`run`, `watch`, `verify`)

| flag | default | meaning |
|---|---|---|
| `--drones` | 4 | number of drones |
| `--scenario` | `ship` | `ship` or `ground` |
| `--sea` | `moderate` | `calm` / `moderate` / `rough` |
| `--slots` | 1 | drones allowed to land at once |
| `--engine` | `mujoco` | `mujoco` (real X2 physics + contact) or `kinematic` (fast point-mass) |
| `--comms` | ∞ | comms/sensing range in m (finite = hard partial-observability regime) |
| `--margin` | 0.0 | extra CBF separation margin (m) — robustness to sensing noise |
| `--consensus` | flag | A2: distributed Kalman-Consensus deck estimation |
| `--vision` | flag | cooperative perception: real per-drone onboard vision (mujoco only) |
| `--cameras N` | — | heterogeneous fleet: only the first N drones carry a camera (needs `--vision`) |
| `--offshore` | flag | orange OSV-vessel look (visual only) |
| `--avoid` | flag | P3: fold the OSV superstructure into the non-bypassable CBF as keep-outs |
| `--seed` | 0 | base seed |

### Subcommands

| command | purpose | key extra flags |
|---|---|---|
| `swarm info` | capability summary | — |
| `swarm run` | recovery episodes (headless metrics) | `--episodes N` |
| `swarm watch` | watch one recovery (MuJoCo 3-D, or matplotlib for kinematic) | `--policy CKPT` (deploy a GNN policy on every drone, kinematic) |
| `swarm verify` | **A3** formal safety: sweep seeds, assert no separation violation | `--seeds N` |
| `swarm multi` | **A5** M drones → K moving decks with dynamic re-tasking | `--decks K`, `--foul-deck K`, `--foul-time S`, `--engine`, `--watch`, `--episodes N` |
| `swarm marl` | train/eval the decentralized MARL avoidance policy | `--eval CKPT`, `--timesteps N`, `--drones N`, `--comms R`, `--device` |
| `swarm gnn` | **A4** train/eval the GNN policy (generalizes across swarm size) | `--eval CKPT`, `--sizes 10 14 18`, `--timesteps N`, `--device` |
| `swarm active` | **A6/P5** learned active cooperative perception (attention fusion + value-of-information gating) | `--eval CKPT`, `--steps N`, `--episodes N`, `--outlier-p F`, `--device` |

### Examples

```bash
swarm run --drones 6 --scenario ship --sea rough        # N drones, real MuJoCo physics
swarm run --drones 6 --consensus                        # A2 distributed deck estimation
swarm run --drones 5 --vision --cameras 2               # heterogeneous fleet: 2 cameras guide 3 camera-less
swarm run --drones 5 --offshore --avoid                 # P3 sense-and-avoid around the OSV superstructure
swarm verify --drones 6 --seeds 25                      # A3 formal-safety separation sweep
swarm multi --drones 9 --decks 3 --foul-deck 2          # A5 K decks + fouled-deck re-tasking
swarm multi --drones 6 --decks 2 --engine mujoco        # A5 on real physics
swarm gnn --eval runs/marl_gnn/gnn_final.zip            # A4 GNN policy across swarm sizes
swarm active                                            # A6 learned active cooperative perception
swarm watch --drones 4 --scenario ship                  # live 3-D viewer
```

---

## Where outputs go

- **Benchmarks:** `python scripts/benchmark.py` → [BENCHMARK.md](BENCHMARK.md).
- **Demo media:** `python scripts/make_demo_gifs.py` → `media/`.
- **Trained policies:** `runs/rl/<scenario>/`, `runs/marl_gnn/`, `runs/cnn/`, `runs/active/` (the keeper
  policies are included so `--controller rl`, `swarm gnn --eval`, etc. work on a fresh clone).
- **Ablation scripts:** `scripts/eval_*.py` reproduce the per-capability A/Bs in [RESULTS.md](RESULTS.md).
