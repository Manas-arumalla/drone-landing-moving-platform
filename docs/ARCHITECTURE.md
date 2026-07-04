# Architecture

A walkthrough of the full stack, from physics up through perception, estimation, control,
learning, safety, and the swarm layer. The guiding constraint throughout is the
[no-cheats realism charter](REALISM_CHARTER.md): the **deployable stack runs on onboard,
sensor-derived state only** — simulator ground truth is used for physics, contact, and
metrics, but never inside a control or coordination decision.

```
                                  ┌──────────────────────────────────────────────────────────────┐
                                  │                       ONBOARD AUTOPILOT                        │
camera (ArUco grid + nested  ─┐   │                                                                │
  centre marker)             │   │   ┌───────────┐   ┌──────────────────┐   ┌──────────────────┐  │   ┌──────────┐
IMU / AHRS                   ├───┼──►│ perception├──►│ RelativeStateEKF ├──►│ LandingSupervisor├──┼──►│ control  ├──► 4 motor
rangefinder                  │   │   │ (PnP/CNN) │   │ (relative state) │   │ FSM (commit/cut) │  │   │ + mixer  │    thrusts
gear-contact                 │   │   └───────────┘   └──────────────────┘   └──────────────────┘  │   └──────────┘
stabilized nadir gimbal +   ─┘   │         ▲                                         │            │        ▲
  optical-flow velocity          │         │              green-deck heave predictor ┘            │  reachability shield
                                  └─────────┼─────────────────────────────────────────────────────┘  + CBF safety filter
                                            │
                                   MuJoCo physics (truth, used only for sensing models, contact, metrics)
```

---

## 1. Physics & simulation (`src/drone_landing/sim/`)

- **Vehicle:** the validated Skydio X2 quadrotor (mesh + dynamics from MuJoCo Menagerie) with added
  landing gear (four legs + frictional feet) for true contact touchdown — no weld/teleport snapping.
- **Worlds** (`assets/mujoco/worlds/`): a dynamic servo-driven **ground rover**, a **6-DOF seakeeping
  ship deck**, and an **offshore OSV** vessel. `sim/world.py` auto-selects the model from the scenario
  name and drives 3 (ground) or 6 (ship) DOFs.
- **Platform motion** (`sim/platforms/`): pluggable `PlatformMotion` subclasses — ground rover, ship
  (sum-of-sinusoids and spectral), inclined deck, USV, truck, and a data-driven CSV replay — so new
  landing targets are additive and never touch the validated worlds.
- **Sensors** (`sim/sensors/models.py`): each model takes the true physical quantity and returns a
  realistically corrupted measurement — white noise, random-walk bias, finite range/rate, dropout,
  quantization. Covers IMU/AHRS, rangefinder, barometer, GNSS, and a downward optical-flow + laser unit.
- **Disturbances:** correlated wind gusts (Ornstein–Uhlenbeck) on by default, plus the ship air-wake.

## 2. Maritime fidelity (`sim/platforms/wave_spectrum.py`, `sim/airwake.py`)

- **Wave spectra → motion:** JONSWAP / Pierson–Moskowitz spectra feed response-amplitude operators
  (RAOs) to synthesize per-DOF deck motion, normalized so the validated landing rates are preserved
  (`--sea-model spectral`). The spectrum recovers the design significant wave height and period.
- **Air-wake / burble:** a position-dependent turbulence field over the deck — the dominant real
  shipboard-landing disturbance. Toggled with `--airwake`; the disturbance observer (`--dob`) recovers
  the success rate it costs.

## 3. Perception (`src/drone_landing/perception/`)

- **Fiducial:** an ArUco **grid board** with a **nested centre marker** for close range, decoded with
  OpenCV and `solvePnP` to a relative pose (~1 cm at altitude).
- **Stabilized nadir gimbal** (`perception/camera.py`): a kinematic body held at the drone's belly but
  always world-level, decoupling camera pointing from body attitude — this fixed the tilt→vision-loss
  coupling that otherwise breaks aggressive approaches.
- **Optical flow** (`perception/optical_flow.py`): relative-velocity estimate from the fiducial's image
  flow, used by IBVS and fused into the EKF.
- **Learned CNN detector** (`perception/cnn_detector.py`): a small conv net with a **spatial-softmax**
  (soft-argmax) head regresses the deck-pixel + visibility, back-projected to a relative fix — an
  ArUco-independent markerless fallback (`--cnn-markerless`). Self-labeled from rendered frames.
- **Classical markerless** (`perception/markerless.py`): segments the bright pad and back-projects the
  centroid; border-gated so a clipped pad never biases the fix.

## 4. Estimation (`src/drone_landing/estimation/ekf.py`)

A **relative-state Extended Kalman Filter** tracks deck-relative position and velocity. It is
**innovation-gated** (rejects outlier measurements), **IMU-aided** in the prediction step, and fuses
ArUco PnP, markerless/CNN fixes, the rangefinder (keeps altitude observable through touchdown), and the
optical-flow velocity. Everything downstream consumes the EKF estimate, never the truth.

## 5. Control (`src/drone_landing/control/`, `planning/`)

- **Geometric SE(3)** (`control/geometric.py`, default): a robust attitude/position controller with a
  leaky integrator and conditional-integration anti-windup.
- **Nonlinear MPC** (`control/mpc/nmpc.py`): a CasADi predictive horizontal tracker; the **tube / DOB-MPC**
  variant (`control/mpc/tube_mpc.py`) folds a disturbance estimate into the prediction model and tightens
  the acceleration bound, giving ~80× tighter station-keeping under steady wind.
- **IBVS** (`control/ibvs.py`): image-based visual servoing on image-space position + optical-flow
  velocity, which removes the velocity-spike fly-offs.
- **Flatness / minimum-snap planning** (`planning/minsnap.py`, `--controller minsnap`): the quadrotor
  is differentially flat in (x, y, z, yaw), so a snap-minimizing polynomial from the current relative
  state to a platform rendezvous gives a smooth reference whose acceleration doubles as attitude
  feedforward (Mellinger & Kumar). Planned receding-horizon in the platform-relative frame, fed
  through the same `a_xy_override` path as the MPC (drop-in comparable; supervisor commit logic
  untouched). Boundary conditions are sanitized (low-passed + clipped velocity, clipped initial
  accel): a planner bakes its boundary state into seconds of feedforward, so raw estimate spikes
  destabilize it where the every-step-replanning MPC merely stumbles. Its **attitude-matched
  touchdown** uses the flatness fact that terminal acceleration sets touchdown attitude: the deck
  normal is measured by the same ArUco `solvePnP` fix that supplies position
  (`perception.board_normal_world`, low-passed over grid detections), and on a tilted deck the
  commit descent pre-tilts toward the normal over the final ~15 cm while the press pushes along the
  normal — unlocking the 12° inclined deck (0% → 67%).
- **Landing-supervisor FSM** (`planning/supervisor.py`): owns `APPROACH → DESCEND → COMMIT → SECURED /
  GO_AROUND`, the commit/press/cut logic, and — for ships — a **green-deck** strategy that perches and
  waits for a low-motion window, then performs a **heave-synchronized descent** fed by an onboard deck
  motion predictor (`planning/deck_predictor.py`).
- **Control allocation** (`control/allocation.py`): motor mixing, including a fault-tolerant 3-rotor mode.

## 6. Learning (`src/drone_landing/rl/`)

- **Environment** (`rl/landing_env.py`): a Gymnasium env where the policy outputs guidance executed by the
  geometric inner loop. It trains on a **calibrated estimator-noise surrogate** (rendering is ~100× too
  slow for RL), then is evaluated honestly on the full vision pipeline.
- **Residual policy** (`rl/policy.py`): the RL action perturbs the classical baseline (action = 0 → the
  proven controller), so learning only refines the hard regime instead of relearning to fly. Horizontal-only
  authority — the supervisor keeps full ownership of descent/commit/cut.
- **Training** (`rl/train.py`): PPO and **recurrent (LSTM) PPO** (Stable-Baselines3 / sb3-contrib) with a
  difficulty curriculum and domain randomization, GPU-accelerated. Potential-based reward shaping.
- **Result:** classical baseline 86% on the hard regime → **94%** residual PPO → **96%** recurrent LSTM-PPO.

## 7. Formal safety (`src/drone_landing/control/reachability.py`, `safety/`)

- **Hamilton–Jacobi reachability:** a robust discrete-time HJ DP computes the safe-landing set for the
  vertical channel under a disturbance bound; `safe_action` is a runtime-assurance **shield** that wraps
  any controller (`drone reachability`, `--shield`). Reckless dive + random wind: 100% hard impacts →
  8% under the shield.
- **Higher-order CBFs** (`safety/avoid.py`): a relative-degree-2 acceleration QP (latency look-ahead) for
  sense-and-avoid past obstacles, with an onboard 2-D scanning rangefinder returning noisy surface points
  (not obstacle IDs).
- **Contingency FSM** (`safety/contingency.py`): a PX4-style priority machine
  (`ROTOR_OUT > LOW_BATTERY > GEOFENCE > LOST_COMMS > OBSTACLE_ABORT > NOMINAL`).

## 8. Swarm (`src/drone_landing_swarm/`)

A **separate package** with its own `swarm` CLI that reuses the single-drone components without modifying
them. Coordination runs entirely on per-drone onboard estimates plus latency/dropout-limited neighbour
broadcasts — no ground truth in any decision.

- **Scheduling** (`scheduler.py`, `holding.py`): Hungarian assignment of landing slots + a deconflicted
  holding stack.
- **Consensus** (`consensus.py`): distributed Kalman-Consensus deck estimation; blind drones recover the
  deck from neighbours' shared fixes.
- **Safety** (`safety.py`): a non-bypassable CBF-QP filter (Hildreth dual QP) every command passes
  through, with a provable discrete-time separation certificate.
- **Learning** (`marl_gnn.py`): a permutation-invariant message-passing GNN policy that generalizes across
  swarm size.
- **Multi-deck** (`multi_deck.py`, `multideck_*`): M drones → K moving decks with auction re-tasking;
  recovers a fouled deck (100% vs 0% for static assignment), on both a kinematic layer and real MuJoCo physics.
- **Cooperative perception** (`vision.py`, `active_perception.py`): per-drone rendered cameras → pad
  detection → shared fixes; learned attention fusion that rejects confident outliers + value-of-information
  communication gating (V2VNet / Where2comm-style).

---

For the *why* behind these method choices — and the debugging lessons (e.g. the three RL bugs only
caught by evaluating checkpoints) — see [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md). For measured numbers and
honest limitations, see [`RESULTS.md`](RESULTS.md).
