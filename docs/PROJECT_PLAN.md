# Autonomous Drone Landing on Moving Platforms — Project Plan & Architecture

> Status: **approved 2026-05-23**. This is the master plan. See [REALISM_CHARTER.md](REALISM_CHARTER.md) for the governing simulation-fidelity rules.

## 1. Vision & positioning

A research-grade autonomy stack that lands a quadrotor on a **randomly-moving ground platform** and a
**wave-driven ship deck**, using **vision-based relative-state estimation** (ArUco + markerless, fused
with IMU) and a **benchmarked suite of controllers** (PID floor → geometric SE(3) → nonlinear MPC → RL →
MPC+residual-RL). The headline contribution is a rigorous, reproducible comparison of model-based vs
learning-based landing under realistic disturbances, plus a novel maritime **"green-deck" timing**
algorithm that commits to touchdown only during predicted low-motion windows.

## 2. Approved decisions

| Decision | Choice |
| --- | --- |
| Control philosophy | **Hybrid** — model-based (NMPC + geometric SE(3)) AND RL, benchmarked head-to-head against a PID floor. |
| Deployment | **Pure simulation**, but with the [Realism Charter](REALISM_CHARTER.md) — no cheats. |
| Perception | **ArUco** fiducials + markerless visual tracking, fused with IMU via EKF/UKF. |
| Compute | Local NVIDIA GPU (RTX 5060). MPC/classical are CPU/CasADi; RL uses the GPU. |
| Drone | **Skydio X2** (real MuJoCo Menagerie model) as the primary platform. |
| MPC backend | **CasADi** (already installed; avoids acados/Windows pain). acados is a future optimization. |

## 3. System architecture

```
  Down camera ─►  PERCEPTION (ArUco solvePnP 6-DOF + markerless fallback)
  IMU/Baro/Range ─►  STATE ESTIMATION (EKF/UKF: drone state + RELATIVE platform state + motion forecast)
                       │
                       ▼
                  LANDING SUPERVISOR (FSM: SEARCH→ACQUIRE→APPROACH→DESCEND→COMMIT→SECURED, +ABORT)
                       │
                       ▼
                  GUIDANCE/PLANNING (rendezvous trajectory; maritime green-deck commit timing)
                       │
                       ▼
                  POSITION CONTROLLER  ── A) NMPC  B) Geometric SE(3)  C) RL  D) MPC+residual-RL
                       │  → thrust + attitude/rate setpoints
                       ▼
                  ATTITUDE/RATE CONTROL → CONTROL ALLOCATION (mixer) → 4 motor thrusts
                       ▼
                  MuJoCo PHYSICS (X2 + ground/ship platform + wind/air-wake + frictional contact)
```

The stack is scenario-agnostic; only the world (platform dynamics + disturbances) changes between ground and ship.

## 4. Components

- **Perception:** multi-scale ArUco board on the deck → `cv2.aruco` + `solvePnP` → 6-DOF deck pose; markerless
  contour/optical-flow (+ optional small CNN) fallback; realistic camera model (intrinsics, noise, blur, latency, FOV).
  Optional Phase-2.5 downward stereo pair for marker-independent metric depth.
- **Estimation:** EKF/UKF fusing IMU strapdown propagation + ArUco pose + markerless bearing + range/baro, with a
  pluggable platform-motion model (CV/CA for ground; harmonic/ship-RAO for maritime) and latency compensation.
  Outputs the relative platform state + short-horizon forecast the controllers consume.
- **Control:** shared inner attitude/rate loop + control allocation on the X2 actuator model. Four interchangeable
  outer controllers: (A) CasADi nonlinear MPC tracking the predicted platform with thrust/tilt constraints and a
  robust/disturbance-observer variant; (B) geometric SE(3) tracking control + Kalman platform tracker (training-free
  floor); (C) RL (PPO/SAC, asymmetric actor-critic, curriculum, domain randomization); (D) MPC + residual RL.
- **Planning:** rendezvous trajectory optimization to a moving target; maritime green-deck quiescent-window detection
  and commit timing; landing supervisor FSM with go-around/abort logic.

## 5. Simulation worlds

- **Drone:** Skydio X2 + landing gear + downward camera (+ optional stereo) + IMU/rangefinder, validated inertia/actuators preserved.
- **Ground platform:** rigid rover driven by randomized bounded-accel/jerk trajectory; true contact deck.
- **Ship deck:** 6-DOF seakeeping motion from a wave spectrum (JONSWAP/PM) + response amplitude operators (heave/roll/pitch
  dominant); true contact deck. Tunable by sea state.
- **Disturbances:** Dryden turbulence + gusts; ship air-wake approximation aft of the superstructure.
- **Sensors:** cameras (MuJoCo renderer + noise/blur/rate), IMU (+bias/noise), rangefinder, synthesized baro/GPS with dropout.
- **Fast model:** the existing state-space sim is retained for design/debug and as the RL critic's privileged model.

## 6. Repository layout (target)

```
src/drone_landing/
  sim/        worlds (MJCF), platforms (ground, ship_seakeeping), sensors, disturbances, mjcf loader
  perception/ aruco_detector, markerless, camera_model
  estimation/ ekf, ukf, platform_models, forecaster
  control/    allocation, rate_attitude, geometric_se3, pid_cascade (legacy), mpc/, rl/
  planning/   rendezvous, quiescent (green-deck), supervisor (FSM)
  envs/       gym_state, gym_sensor, wrappers
  evaluation/ metrics, benchmark, scenarios
  config/     typed YAML configs
  cli/        run, train, evaluate, render, watch
assets/mujoco/  meshes/, worlds/, textures/ (ArUco)
docs/  tests/  .github/workflows/
```

Tooling: typed configs (replacing single JSON), pytest + ruff + mypy + pre-commit, GitHub Actions CI,
Weights & Biases + TensorBoard, MuJoCo renderer → mp4 demos, Optuna tuning.

## 7. Evaluation

- **Metrics:** success rate %, touchdown position error (m), contact velocity (vertical + relative horizontal),
  tilt at touchdown, time-to-land, control energy, abort/go-around rate, slide-off & tip-over rate, settle success.
- **Protocol:** 500+ seeded episodes per cell; means with confidence intervals.
- **Matrix:** {PID, Geometric+KF, NMPC, RL, MPC+ResidualRL} × {Ground, Ship} × {disturbance levels}.
- **Robustness/generalization:** speed, sea state, wind, sensor noise & latency, marker occlusion, init offset, mass/inertia;
  evaluate on trajectories/sea-states unseen in training.
- **Ablations:** reward terms, curriculum stages, residual-RL on/off, fusion vs marker-only, MPC horizon, green-deck timing.

## 8. Phased roadmap

| Phase | Focus | Deliverable |
| --- | --- | --- |
| 0 | Foundation: repo layout, configs, CI, adopt X2 model, charter & metrics | Loadable X2 landing scene, clean repo |
| 1 | High-fidelity ground world: X2+gear+cameras+sensors, randomized rover, strict contact (no lock) | Physically honest ground env |
| 2 | Perception + estimation: ArUco solvePnP, markerless fallback, EKF fusion | Estimator validated vs ground truth; controller flies on estimated state |
| 3 | Classical advanced floor: geometric SE(3) + Kalman tracker, rendezvous, supervisor FSM | Strong training-free baseline + ground benchmark |
| 4 | Nonlinear MPC (CasADi) + robust variant | NMPC beating classical floor |
| 5 | Maritime: seakeeping ship deck, air-wake, green-deck commit timing | Ship landing demo + timing ablation |
| 6 | RL: domain randomization, asymmetric actor-critic curriculum, residual RL, e2e PPO/SAC | Trained policies + curves |
| 7 | Evaluation: full benchmark matrix, robustness, ablations | Results section |
| 8 | Polish: Optuna, demo videos, dashboards, report, README, reproducibility | Portfolio-ready repo + report |

## 9. Key risks & mitigations

- **CUDA torch on RTX 5060 (Blackwell):** current torch is CPU-only. Needs a recent CUDA 12.8+ wheel before Phase 6 (RL).
- **Vision-RL rendering throughput:** train on the fast state model + estimator output first; e2e pixels-to-action is an ablation.
- **Ship seakeeping fidelity vs tractability:** spectral/RAO motion model is the documented tradeoff; CFD/buoyancy out of scope.
- **"No cheats" vs RL needing ground truth:** asymmetric actor-critic; evaluation is always sensor-only.
- **Scope:** each phase ships an independently valuable result.
