# Workshop-paper outline (draft)

**Target venue:** ICRA/IROS workshop on aerial robotics or field robotics; RSS demo track as fallback.
4 pages + references, workshop format.

**Working title:**
*Spin-Averaged Precession Control for Rotor-Failure Descent: Why Fixed-Point Attitude Regulation
Fails and What a Vision Pipeline Can(not) Recover*

**One-sentence pitch:** landing a quadrotor with a dead rotor is blocked in practice not by control
but by close-range position sensing — we show a precession controller that steers the spin-averaged
thrust axis flies the descent through a full vision pipeline, and precisely characterize the terminal
sensing gap that prevents touchdown on the pad.

---

## 1. Introduction (0.5 p)

- Motivation: single-rotor failure is the most common actuator fault; the underactuated vehicle must
  spin (yaw is uncontrollable), so classical attitude control does not transfer.
- Prior art: Mueller & D'Andrea's spinning-equilibrium control (state-based, mocap); learned recovery
  policies; none address the *vision-in-the-loop landing* problem during the spin.
- Contributions:
  1. A negative result with a diagnosis: a reduced-attitude LQR about a fixed point keeps tilt bounded
     but drifts tens of meters — because the rotor-out equilibrium is a **limit cycle**, not a fixed point.
  2. A simple averaged/precession controller: low-pass the body-z axis over spin periods, command an
     inertial torque via the phase map τ_body = Rᵀ·τ_world, with the **signed** spin rate in
     L = J_z·Ω (sign errors invert the precession direction and tumble the vehicle).
  3. Full-pipeline evaluation (ArUco → EKF → supervisor): the controller flies the spinning descent
     at ~0.2–0.5 m median horizontal error **even under wind**, but on-deck landing is capped by the
     <0.3 m fiducial blind zone — a **sensing** limit, not a control limit. Optical-flow velocity
     fusion does not unblock it (velocity ≠ position); we argue the fix is a close-range position
     reference.

## 2. Problem setup (0.5 p)

- Skydio X2 model in MuJoCo, one rotor commanded to zero mid-flight; spin reaches ~15–20 rad/s.
- Definitions: spin-averaged thrust axis, precession dynamics dL/dt = τ, reduced-attitude state.
- The landing task: moving deck, vision-only relative state (no mocap / no ground truth in the loop).

## 3. Why fixed-point regulation fails (0.75 p)

- The reduced-attitude LQR (linearized about the spin, gyroscopic coupling by measured spin rate,
  Riccati gains): bounded instantaneous tilt (~26°) but unbounded drift + climb.
- Analysis: the controller regulates the instantaneous axis to a fixed point while the true equilibrium
  is a periodic orbit; the period-averaged thrust vector is left uncontrolled.
- Figure: tilt trace + drift trace, LQR vs proposed controller (data exists in RESULTS.md §8 runs).

## 4. Spin-averaged precession control (0.75 p)

- Axis estimate: first-order low-pass of body-z over a few spin periods.
- Steering law: τ_world = J_z·Ω·(k_a·(a_des − a)⊥ − k_da·ȧ); realized instantaneously by the
  phase-mapped body torque τ_body = Rᵀ·τ_world (yaw component free).
- The signed-Ω subtlety (negative spin ⇒ L anti-parallel to the averaged axis) — the decisive bug class.
- Center-then-descend commit logic + PD fallback below the spin threshold.
- Seeded results (truth state, wind-off, clean hover; 10 seeds, `scripts/eval_rotor_out_paper.py`):
  precession controller descends to the deck in 6/10 episodes with **1.21 m median drift** and no
  tumble; the fixed-point LQR keeps tilt bounded (median max 26.9°) but **flies off ~130 m median**;
  the reduced-attitude PD tumbles (median max tilt 61°, 0/10 down). The precession law is the only
  one of the three that keeps the vehicle near the deck through the descent.

## 5. Full vision pipeline evaluation (0.75 p)

- Mid-flight failure onset during a real approach; ArUco → EKF relative state; supervisor commit logic.
- Key positive (seeded, 12 episodes/cell): the controller transfers — the median horizontal error
  holds ~0.2–0.5 m from 2 m altitude down to ~0.35 m, **nearly identical with wind on and off**
  (fig_herr_vs_alt.png).
- Key negative, precisely characterized: inside the <0.3 m fiducial blind zone the median error
  jumps to ~1.5–1.8 m and no seeded episode lands on the pad (0/12 wind-off, 0/12 wind-on;
  failures split between off-platform drift and near-deck hard contact). By contrast the default
  spinning-descent contingency drifts out of bounds in 24/24 episodes (median ~6 m) — the precession
  controller is strictly closer, and the binding constraint is terminal position sensing.
  Optical-flow velocity fusion into the EKF does not restore position (velocity, not position).
- The general lesson: for underactuated terminal descent, close-range *position* sensing is the
  binding constraint; velocity or altitude sensors do not substitute.

## 6. Discussion & future work (0.25 p)

- Proposed fix: nested fiducial decodable inside 0.3 m or a beacon/range-bearing reference
  (perception redesign, not control).
- Hardware path: PX4 SITL first, then a small testbed.
- Honest scope: simulation-only; unbounded spin (no rotor aero drag) slowly erodes precession authority.

## Figures / tables inventory (all data already exists in the repo)

1. Fig: limit cycle illustration — instantaneous body-z cone vs spin-averaged axis.
2. Fig: LQR vs precession controller, tilt + drift traces.
3. Fig: full-pipeline descent, horizontal error vs altitude, wind on/off.
4. Table: clean-hover results (LQR / PD / precession).
5. Table: full-pipeline ablations (blind-commit, flow fusion, press-to-contact).
6. Fig: detector trace showing the <0.3 m blind zone (found=False region).

## Related work to cite

- Mueller & D'Andrea (ICRA 2014) — relaxed-hover / rotor-failure control.
- Sun et al. — incremental NDI rotor-failure flight (wind-tunnel validated).
- Goldschmid & Ahmad (Auton. Robots 2024) — RL landing baseline (see docs/LITERATURE.md).
- Faessler et al. — thrust-direction control of underactuated multirotors.
- Fiducial-marker close-range limits literature (ArUco / AprilTag scale-range analyses).

## Writing plan

- [x] Re-run the three headline experiments with fixed seeds; export CSV + plots —
      `scripts/eval_rotor_out_paper.py` (outputs land in `runs/paper/`: per-episode CSVs,
      20 Hz traces, `fig_clean_hover.png`, `fig_herr_vs_alt.png`, `fig_outcomes.png`,
      and a `manifest.json` pinning commit/seeds/versions).
- [ ] Draft §3–§5 first (the technical core), then intro/discussion.
- [ ] Internal red-team pass: does every claim have a number in BENCHMARK/RESULTS?
