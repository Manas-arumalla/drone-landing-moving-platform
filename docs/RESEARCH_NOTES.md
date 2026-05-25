# Research Notes — How real moving-platform landing systems are built

Synthesized from the literature (2026-05) to understand why our early closed-loop attempts struggled
and to align the design with proven real-world approaches.

## The central lesson: IBVS, not PBVS

Our first closed-loop design was **Position-Based Visual Servoing (PBVS)**: estimate the platform's
3-D position *and velocity*, then track it with a geometric controller. This needs a good velocity
estimate; differentiating 33 Hz vision is noisy and spikes (1–3 m/s), and the controller's damping
term turns those spikes into fly-offs. This caused the majority of our failures.

Real systems overwhelmingly use **Image-Based Visual Servoing (IBVS)**: control *directly in the
image plane* (regulate where the target's features appear in the image). The literature is explicit
that IBVS is **"less sensitive to depth estimation"** and **"can overcome the lack of … the velocity
feedback of the target by exploiting error transformation in the image space."** IBVS therefore
sidesteps the exact weakness (3-D velocity estimation) that broke our PBVS design.

Demonstrated capability of IBVS landing systems:
- Landing on **ship decks moving at ~15 km/h, oscillating in waves, 19/22 success, vision-only**
  (no RTK / motion capture).
- Robust to wind, illumination, and weather in outdoor flights.

## Techniques real systems add

1. **Target-velocity feed-forward** — estimate the platform/ship velocity (often fusing GPS + image
   + a platform motion model in a Kalman filter) and add it as a feed-forward term in the controller.
2. **Adaptive IBVS gain + sliding-mode control** — robustness to ground effect, wind, oscillation.
3. **Stabilized / virtual camera** — a downward camera kept level (gimbal or "virtual plane") so the
   image features are not corrupted by the drone's own tilt. (We added a software gimbal for exactly
   this; it also removes the rotational component of optical flow.)
4. **Circle / image-moment features** — rotation-invariant features that decouple translation from
   rotation, simplifying the control law.

## The learning-based path (the other proven approach)

- `robot-perception-group/rl_multi_rotor_landing` (Springer 2024 / arXiv 2302.13192): **curriculum**
  RL with a sequential 1-D decomposition + state-space discretization (Double Q-learning), sim-to-real.
- Detection-informed deep RL and meta-RL reach **~94% landing success** on moving platforms in sim,
  with zero-shot or few-shot sim-to-real transfer.
- Takeaway: RL handles the noisy-perception + fast-tracking robustness that hand-tuned classical
  control cannot, at the cost of a heavier training pipeline (and, here, a CUDA PyTorch install).

## How this maps onto our project

- **Adopt IBVS-flavored control:** with our nadir gimbal, the fiducial's image error maps directly to
  a horizontal command, and the fiducial's **optical flow** (validated here at ~0.16–0.28 m/s vs
  truth) gives a robust relative-velocity term — replacing the spiky differentiated 3-D velocity.
- **Use a realistic platform-motion profile** as the primary scenario (a vehicle cruising for a
  landing, not darting randomly at 1.5 m/s). On realistic motion the existing stack already lands
  reliably; the adversarial random rover remains as a stress test.
- **Keep RL (Phase 6)** as the route to the highest robustness on the hardest motion.

## Maritime scenario: green-deck timing and deck-motion prediction

Landing on a ship at sea adds a disturbance the ground rover does not have: the deck **heaves, rolls,
and pitches** with the waves. Real shipboard recovery (manned and unmanned) does not fight this
continuously — it **waits for a quiescent ("green-deck") window**, the brief lull between wave groups
when deck motion is small, and commits the touchdown into it. Naval rotorcraft recovery formalizes
this as the **Energy/Quiescent-period detection** problem and uses short-horizon **deck-motion
prediction** to anticipate the lull.

Key facts that shaped our design:
- **Deck-motion prediction is short-horizon.** Ship motion is a narrow-band random process (a wave
  spectrum shaped by the hull's RAOs), so autoregressive / sinusoid-fit predictors are useful only
  for a few seconds; velocity (the derivative) degrades fastest. We confirmed this empirically — a
  multi-sinusoid least-squares fit nowcasts the deck vertical velocity well (~0.09 m/s RMS) but its
  +1 s forecast error grows to ~0.17 m/s.
- **Implication for our controller.** Rather than trust a long forecast to time a slow descent, we
  (a) **perch** at the commit altitude and wait for a predicted low-motion window, and (b) use a
  **heave-synchronized descent**: feed the *nowcast* deck vertical velocity forward into the descent
  command so the drone rides the heave and closes at a small, constant *relative* rate. Impact
  velocity then depends on the accurate nowcast, not the unreliable multi-second forecast.
- **Honest estimation (no cheats).** The deck-motion estimate is reconstructed **onboard** from the
  relative-altitude signal the EKF already tracks (downward camera + rangefinder); the simulator's
  wave model is never read by the controller (see [REALISM_CHARTER.md](REALISM_CHARTER.md)). The
  prescribed wave motion only drives the *plant* (the deck is far heavier than the drone).

Implemented as `sim/platforms/ship.py` (6-DOF seakeeping deck), `planning/deck_predictor.py` (onboard
heave estimator + green-window logic), and the supervisor/autopilot green-deck path
(`--ship --green-deck`).

## Method choices for the RL phase (Phase 6) — from the literature

Searched the recent (2024–25) literature to pick the best-suited methods (targeting SOTA approaches, not
quick hacks). Findings and the resulting decisions:

- **Algorithm: PPO is the right backbone.** Across quadrotor-landing studies PPO, TD3, and SAC are all
  used; **PPO shows superior, more stable performance** for landing (incl. inclined/static), and is the
  standard for this task. We use PPO as the robust baseline.
- **Partial observability → recurrent policy.** Vision-based *ship* landing is a POMDP (noisy, dropping
  observations, deck motion). The 2024–25 SOTA embeds an **LSTM in PPO** ("robust policy optimization",
  recurrent PPO) to cope. Our estimator surrogate has exactly this (noise + dropout), so a **recurrent
  PPO** (`sb3-contrib RecurrentPPO`) is the planned upgrade over feed-forward PPO.
- **Curriculum training accelerates learning** (proven for landing): start easy (slow/static, calm) and
  widen to hard (fast/random rover, rough seas). Implemented via a `difficulty` schedule that scales the
  domain-randomization ranges over training.
- **Domain randomization** (mass, wind, platform motion, estimator noise) for robustness + sim-to-real —
  already in the env.
- **Asymmetric actor–critic** (privileged critic sees truth, actor sees only onboard estimate) is the
  standard sim trick to speed learning while keeping the deployed policy onboard-only; we expose truth in
  `info` for this (full asymmetric critic = an enhancement; SB3 needs a custom policy).
- **Residual RL on the model-based controller** (learn a correction on top of geometric/MPC) is the
  safe, sample-efficient option and pairs with our MPC benchmark; a candidate once the direct policy works.
- Reference: the curriculum + 1-D sequential-decomposition Double-Q approach reaches **~94%** on moving
  platforms (arXiv 2302.13192 / Springer Auton. Robots 2024).

## Method choices for the swarm (separate module) — from the literature

- **Coordination: hierarchical (task-allocation + planning) baseline, MARL extension.** 2024 work splits
  the problem into **task allocation + path planning** (hierarchical) and uses **MARL with CTDE**
  (centralized training, decentralized execution; e.g. MAPPO) for collaboration — *MARLander* (2024)
  reports ~2.3 cm (static) / ~3.9 cm (moving) landing accuracy. We build the **classical hierarchical
  baseline first** — optimal slot scheduling (K-lowest-cost / **Hungarian** for multi-platform) + an
  **exact CBF-QP collision-avoidance safety filter** (Hildreth dual QP) + a deconflicted holding stack,
  reusing the single-drone autopilot — then add **MARL (MAPPO/CTDE)** for the coordination policy as the
  research-grade extension. Recent filings also use **local consensus propagation** to cut inter-UAV
  comms — aligns with our planned cooperative (consensus) deck estimate.

## Sources

- [Visual Servoing Approach to Autonomous UAV Landing on a Moving Vehicle (MDPI Sensors 2022)](https://www.mdpi.com/1424-8220/22/17/6549)
- [Robust visual servoing control for quadrotors landing on a moving target (ScienceDirect 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0016003221000223)
- [Autonomous ship-deck landing via feed-forward IBVS (ScienceDirect 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1270963822005430)
- [Robust Image-Based Landing on an Unpredictable Moving Vehicle Using Circle Features (IEEE 2022)](https://ieeexplore.ieee.org/document/9791477/)
- [Autonomous Landing of a VTOL UAV on a Moving Platform Using IBVS (IEEE)](https://ieeexplore.ieee.org/document/6224828/)
- [rl_multi_rotor_landing — RL autonomous multi-rotor landing on moving platforms (GitHub / arXiv 2302.13192)](https://arxiv.org/pdf/2302.13192)
- [Closing the Sim-to-Real Gap: Detection-Informed Deep RL for UAV Precision Landing (Springer 2024)](https://link.springer.com/chapter/10.1007/978-3-031-66694-0_11)
- [Robust RL Control for Vision-Based Ship Landing of VTOL-UAVs (recurrent/LSTM PPO, 2024-25)](https://www.researchgate.net/publication/388863927_Robust_Reinforcement_Learning_Control_for_Vision-Based_Ship_Landing_of_VTOL-UAVs)
- [Autonomous Landing on a Mobile Platform via Meta Reinforcement Learning (2024)](https://www.researchgate.net/publication/379278546_Autonomous_Landing_of_the_Quadrotor_on_the_Mobile_Platform_via_Meta_Reinforcement_Learning)
- [Deep RL sim-to-real for VTOL-UAV offshore docking (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1568494624006173)
- [RL-based autonomous multi-rotor landing on moving platforms (Springer Auton. Robots 2024 / arXiv 2302.13192)](https://link.springer.com/article/10.1007/s10514-024-10162-8)
- [MARLander: local path planning for drone swarms via multi-agent deep RL, CTDE (arXiv 2406.04159, 2024)](https://arxiv.org/html/2406.04159v1)
- Maritime / deck-motion: shipboard recovery & quiescent-period ("green-deck") detection; ship-motion
  prediction as a narrow-band process shaped by response-amplitude operators (RAOs) over a wave
  spectrum — basis for the sum-of-sinusoids deck model and the short-horizon onboard heave predictor.
