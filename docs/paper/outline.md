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
  3. Full-pipeline evaluation (ArUco → EKF → supervisor): the controller flies the spinning descent at
     ~0.1 m horizontal error **even under wind**, but on-deck landing is capped by the <0.3 m fiducial
     blind zone — a **sensing** limit, not a control limit. Optical-flow velocity fusion does not
     unblock it (velocity ≠ position); we argue the fix is a close-range position reference.

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
- Results (truth state, wind-off, clean hover): lands 5/10 on deck, 3/10 on the 0.4 m pad,
  ~1 m median drift, no tumble (vs 66 m drift for the LQR).

## 5. Full vision pipeline evaluation (0.75 p)

- Mid-flight failure onset during a real approach; ArUco → EKF relative state; supervisor commit logic.
- Key positive: the controller transfers — holds ~0.1 m horizontal error down to ~0.7 m altitude,
  **including wind**.
- Key negative, precisely characterized: below ~0.3 m the fiducial is too zoomed to decode;
  the underactuated vehicle drifts off the deck during the blind terminal descent (~0–8% full-pipeline
  landing). Optical-flow velocity fusion into the EKF does not restore position; press-to-contact
  makes it worse. Table: ablations (blind-commit / flow-fusion / press).
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

- [ ] Re-run the three headline experiments with fixed seeds; export CSV + plots via a new
      `scripts/eval_rotor_out_paper.py`.
- [ ] Draft §3–§5 first (the technical core), then intro/discussion.
- [ ] Internal red-team pass: does every claim have a number in BENCHMARK/RESULTS?
