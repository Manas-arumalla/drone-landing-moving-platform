# Position against the published baseline

This project reproduces and extends the problem setup of a widely-cited recent baseline in
RL-based landing on a moving platform. This document lays out the comparison honestly —
including where the baseline is stronger.

## Baseline

**Goldschmid & Ahmad, "Reinforcement learning based autonomous multi-rotor landing on
moving platforms," _Autonomous Robots_ 48(4), 2024** — DOI
[`10.1007/s10514-024-10162-8`](https://doi.org/10.1007/s10514-024-10162-8),
[arXiv 2302.13192](https://arxiv.org/abs/2302.13192),
code: [`robot-perception-group/rl_multi_rotor_landing`](https://github.com/robot-perception-group/rl_multi_rotor_landing).

Their contribution is a **tabular double-Q-learning** controller with a sequential curriculum
and multi-resolution state discretization, decomposing the 2-D landing task into two independent
1-D policies. They evaluate on both a Gazebo/RotorS simulation and a physical Vicon-tracked
rail platform.

## Scope comparison

| dimension | Goldschmid & Ahmad (2024) | this repository |
|---|---|---|
| **Physics engine** | Gazebo 11 + RotorS (Hummingbird quadrotor) | MuJoCo 3.8 (validated Skydio X2) |
| **Platform types** | one — rectilinear/figure-8 ground rail | six — ground rover, ship (6-DOF seakeeping), offshore OSV, inclined, USV, truck |
| **Platform speeds tested** | 0.2–1.6 m/s | ground: 0.4–1.5 m/s (`--vmax`); seakeeping ship: JONSWAP/PM spectrum |
| **Observations** | **state-based** (position, velocity) | **vision-based** (ArUco → EKF), IMU/rangefinder/optical-flow |
| **State noise / observation noise** | additive Gaussian: σ = {0.1 m position, 0.25 m/s velocity} | sensor-level (per-sensor bias/dropout/quantization) + estimator innovations |
| **Wind / aero disturbances** | not modeled | correlated wind gusts (OU) by default; ship air-wake / burble (opt-in) |
| **Sea state / wave dynamics** | not modeled | JONSWAP / Pierson–Moskowitz spectra + RAOs; data-driven CSV replay |
| **Formal safety** | not addressed | Hamilton–Jacobi reachability shield + higher-order-CBF avoidance |
| **Multi-drone / cooperative** | not addressed | separate swarm module (consensus, GNN policy, cooperative perception) |
| **Real-hardware validation** | **yes** — Vicon-tracked rail platform | **no** — simulation only |

## Reported results — like-for-like where possible

On their translational ground-platform scenario, the paper reports:

| condition | their method (Double-Q + curriculum) | their cascaded-PI baseline |
|---|---|---|
| simulation, no noise, RPM 0.2 (v_p ≈ 0.2 m/s) | **99%** | **100%** |
| simulation, no noise, higher RPMs | 99% typical | 100% across the range |
| simulation, with {0.1 m, 0.25 m/s} noise, RPM 0.4 | (+4% vs RL baseline) | **100%** (unaffected by noise) |
| simulation, with noise, RPM 1.2 | (+20% vs RL baseline) | **100%** |
| real hardware (Vicon), all speeds | 99–100% | — |

Notably, **the paper itself finds that a well-tuned cascaded PI controller matches or beats
the learned policy across all their simulation scenarios**, and is "unaffected by the specified
noise level." Their contribution is not "RL beats classical" — it is "learned control can be
robust to noise, and generalizes to real hardware."

The closest comparable measurements on our side (from [`BENCHMARK.md`](BENCHMARK.md), 12
episodes per cell, vision pipeline, no ground truth in the loop):

| condition | our geometric baseline | our residual PPO / LSTM-PPO |
|---|---|---|
| ground rover, cruising motion | **83%** (with wind on by default) | **100%** (residual PPO) |
| ground rover, hard regime (fast + wind) | **86%** (well-tuned classical) | **94%** (PPO) / **96%** (LSTM-PPO) |
| ship, moderate sea, wind on | **92%** | — |
| ship + air-wake + DOB | 58% → **83%** with disturbance observer | — |

Our numbers are lower per-cell than theirs because we run a **strictly harder observation
regime** (perception + estimation in the loop, wind on by default, and seakeeping motion).
Their setup provides the policy with 0.1 m / 0.25 m/s Gaussian state noise on ground truth;
ours forces the policy to close the loop on an ArUco/EKF estimate that can lose tracking or
diverge under aggressive maneuvers.

## Where the baseline is stronger

- **Real-hardware validation.** They deploy on a Vicon-tracked rail platform; we are
  simulation-only. This is the single largest gap on our side.
- **Cleaner, publishable RL story.** Their tabular double-Q + curriculum is a well-scoped
  algorithmic contribution with a clean training-time story. Our RL is a residual on a strong
  classical baseline, which is honest but is a smaller algorithmic claim.
- **Peer-reviewed publication.** Their work is in _Autonomous Robots_.

## Where this repository extends the problem

- **Vision-in-the-loop.** No ground-truth pose reaches any controller. ArUco → EKF is the sole
  positional signal. This is the biggest practical gap between typical RL-landing papers and
  real deployment on a non-Vicon platform.
- **Wind and air-wake modeling.** Correlated wind gusts by default; ship air-wake / burble as
  an opt-in. Both meaningfully move headline success rates (86% → 58% with air-wake, recovered
  to 83% with the disturbance observer).
- **Seakeeping.** 6-DOF ship deck driven by JONSWAP / PM wave spectra and RAOs, plus
  data-driven CSV replay — a much richer platform-motion model than a translational cart.
- **Formal safety.** Hamilton–Jacobi reachability shield (reckless dive: 100% hard impacts →
  8% under the shield) and higher-order CBFs for obstacle avoidance — orthogonal to their RL
  contribution.
- **Multi-drone coordination.** A separate swarm module with distributed Kalman-Consensus,
  a non-bypassable CBF-QP safety filter, a permutation-invariant GNN policy, and cooperative
  perception; entirely outside the scope of the baseline.
- **Controller breadth.** Geometric SE(3), nonlinear MPC, tube / DOB-MPC (~80× tighter under
  wind), IBVS, and residual PPO / LSTM-PPO — all switchable from the same CLI on the same
  worlds.

## Honest caveats

- **Not an apples-to-apples reproduction.** The comparison above uses the closest platform
  motion we ship (translating ground rover with `--vmax`), not a bit-for-bit port of their
  Gazebo config. A future ablation could add a `platform="rl_multi_rotor_baseline"` scenario
  with matching rectilinear/figure-8 profile, then rerun our controllers under matched noise.
- **Their cascaded-PI baseline is a very strong baseline.** Our geometric SE(3) is in the
  same family and reaches 83–92% on our scenarios, which suggests classical control is
  robustly competitive on both problem statements when tuned. This is a shared finding, not
  a rebuttal of either paper.
- **We cannot claim a real-hardware win.** Any "sim-only ≥ their sim" number needs the
  matching hardware caveat.

## Suggested future work

1. **Port their exact Gazebo/RotorS scenario config to MuJoCo** (`scripts/eval_rl_baseline.py`)
   and run our best classical + best learned controller under matched conditions and noise
   level. Publish the resulting head-to-head table.
2. **Reproduce their curriculum-driven double-Q learner** as an alternative controller in this
   repo, then compare training-time / sample-efficiency against our residual PPO on the same
   scenario.
3. **Hardware bring-up** (PX4 SITL → ROS 2 → a small physical testbed) — the highest-value
   upgrade for closing the gap identified above.
