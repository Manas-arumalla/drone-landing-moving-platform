# Results & Findings

A consolidated, **honest** summary of what works, by how much, and *why* — every number is from the
no-cheats pipeline (controllers/coordinators act on onboard-estimated state, never ground truth; see
[REALISM_CHARTER.md](REALISM_CHARTER.md)). Where a method does *not* help, that is stated plainly: the
recurring theme of this project is that the **classical stack is excellent, and learning/extra machinery
wins only where the bottleneck is genuinely algorithmic, not physical or perceptual.**

The full controller × scenario × disturbance **benchmark matrix** (P6) is produced into
[BENCHMARK.md](BENCHMARK.md) by `python scripts/benchmark.py`; the tables below are the per-feature A/Bs
gathered as each capability was built. Demo GIFs (`python scripts/make_demo_gifs.py`) live in
[`media/`](../media/) and are embedded in the README.

## 1. Single-drone landing (vision-only, light wind)

| scenario | success | notes |
| --- | --- | --- |
| ground rover (realistic motion) | ~90% | ArUco→EKF→supervisor→geometric, stabilized gimbal |
| ship deck, moderate sea | ~90% | 6-DOF seakeeping deck + green-deck heave-sync descent |
| ship deck, rough sea | ~83% | geometric baseline |
| hard regime (fast/random rover + wind) | **86% → 94%** | residual RL beats the classical baseline |
| inclined deck — gentle ~6° / moderate ~12° (P4) | **100% / ~0%** | level-attitude press seats on a gentle slope, fails on a steeper one (motivates attitude-matched touchdown; pairs with `--shield`) |
| moving truck — smooth loop, 0.45 m/s (P4) | **~100%** | continuously *translating* target; drone rides the moving bed through descent |
| USV — maneuvering + lively rocking (P4) | **~83%** | translates *and* rocks (8° roll, ~3 s) — the hardest new platform |

**Why MPC isn't the default:** the nonlinear MPC tracks 3.4× tighter with *clean* state but is *worse*
closed-loop — the bottleneck is estimation, not control. Geometric SE(3) is more robust in-loop and stays
the default. IBVS removes velocity-spike fly-offs by using image-space position + optical-flow velocity.

**Alternative-controller success (12-ep, no truth in loop), for reference — geometric is the default:**
geometric **ground 88% / ship 92%**; IBVS **ground 58% / ship 83%**; MPC **ground ~67%** (estimation-
limited, confidence-gated). IBVS smooths the velocity (no fly-off spikes).

**IBVS commit-descent fix (traced + resolved).** Diagnosis: IBVS *lands* the drone softly (reaches SECURED
on the deck) but on the fast **random** rover it used to touch down ~0.44 m off-centre — just **outside the
0.40 m success radius** — scoring `timeout`. Root cause: at COMMIT it is centred (~0.18 m), but the
commit phase switched to **open-loop velocity-hold**, so during the ~0.7 s descent the rover slid out and
it drifted ~0.26 m before contact. **Fix:** while the fiducial is still tracked, IBVS keeps **closing the
loop on the image position through the commit descent** (instead of velocity-hold), falling back to
velocity-hold only if the marker is lost. **Result: ground IBVS 25% → 58%, and ship IBVS 75% → 83%** (the
same centring help benefits the moving ship deck) — a clean win, **no regression** (geometric/MPC/RL commit
paths untouched; the change is gated to the IBVS controller). Geometric still leads (88% ground), so it
stays the default; IBVS is now a much stronger alternative.

**Tube / DOB-MPC (B4).** Folding the disturbance-observer estimate into the MPC prediction model + a
constraint-tightening tube gives **guaranteed tracking under bounded wind**: station-keeping under a steady
0.5–2.0 m/s² wind, DOB-MPC holds ~0.001–0.002 m vs plain MPC's 0.05–0.19 m standing offset (**~80× tighter**),
until the wind approaches the actuator limit and leaves the robustly-controllable set. Validated with clean
state (`scripts/eval_tube_mpc.py`); closed-loop vision integration deferred — estimation-limited like plain MPC.

## 2. Reinforcement learning (residual on the classical baseline)

| regime | classical baseline | residual RL | verdict |
| --- | --- | --- | --- |
| hard ground (fast rover + wind), surrogate | 86% | **94%** (½ the crashes) | RL **wins** |
| hard ground — **recurrent-PPO (LSTM)** | 86% | **96%** (2 crashes) | LSTM **edges the MLP** (memory helps the POMDP) |
| default ground, full vision pipeline | — | **100% (8/8)** | transfers from surrogate |
| hardest fixed motion, full pipeline | 17% | **33%** | RL ~2× where vision-estimation bites |
| ship (already-strong baseline) | 96% | 44% | RL **loses** — no headroom; geometric stays default |

**Findings.** (1) RL adds value only where the baseline is weak (fast/random motion); on the near-ceiling
ship baseline its residual noise just causes fly-offs. (2) VecNormalize — the usual "best practice" — *hurt*
(36% vs 94%): reward normalization diluted the success-bonus-dominated shaped reward. (3) Three separate RL
bugs (reward-accumulation suicide, obs sign convention, vz-authority/hover) were each caught only by
*evaluating checkpoints*, never by the reward curve — always eval.

## 3. Disturbance rejection & maritime fidelity (B1)

| condition | baseline | with `--dob` | note |
| --- | --- | --- | --- |
| strong steady wind | 33% | **42%** | disturbance-observer feedforward, +9 pts |
| ship **air-wake** (burble) over the deck | 86% → **50%** | **→ 86%** | DOB fully recovers the burble |

The air-wake (the dominant real shipboard-landing disturbance) is modeled as a position-dependent
turbulence field strongest just over the deck. The wave model is a **JONSWAP / Pierson–Moskowitz**
spectrum passed through response-amplitude operators; it reproduces the target significant height `Hs`
exactly and is calibrated to the validated deck-motion RMS (so the spectral model is a physically-correct
drop-in, `--sea-model spectral`). The DOB's air-wake recovery (50%→86%) is its strongest, most physical win.

## 4. Formal safety (B2 — Hamilton–Jacobi reachability)

A robust discrete-time HJ reachability (`∃a ∀d` differential-game DP) computes the safe-landing set for the
vertical channel under a disturbance bound; the grid set matches the analytic worst-case braking curve. A
runtime-assurance **shield** overrides any controller that would leave the set:

| reckless dive controller + random wind | hard landings | mean touchdown speed |
| --- | --- | --- |
| unshielded | 100% | 5.25 m/s |
| **shielded** | **8–12%** | **0.4 m/s** (13× softer) |

The residual few % are marginal grid/discretization boundary cases. `drone reachability` prints the safe
set + this A/B. This complements the swarm CBF (A3) with an offline-verified landing envelope.

## 5. Swarm: M drones onto moving decks (separate module, no-cheats)

| feature | result |
| --- | --- |
| **A1** onboard sensing (cheat fix) | coordination loop reads *zero* truth; 100% all-landed, separation kept on both kinematic + MuJoCo physics |
| **A3** non-bypassable CBF + certificate | perfect inputs → certificate ≥ 0 (matches the forward-invariance proof); realistic sensing → **0/25 separation violations** |
| **A2** consensus deck estimation | network deck-estimate error **0.295 → 0.127 m (57% lower)**; blind drones (no direct view) recover ~0.14 m purely from neighbours; coordinator min-sep 0.86 → 0.94 m |
| **A5** K decks + dynamic re-tasking | 9 drones → 3 decks balanced (3/3/3); when a deck **fouls** mid-recovery, re-tasking recovers all (**100%** vs **0%** static) |
| **A5 on real physics (P4.1)** | `swarm multi --engine mujoco`: K servo decks + N real X2 drones, true contact — **6→3 decks 6/6 (2/2/2), 4→2 decks 4/4 (2/2)**, balanced, separation kept (removes the kinematic shortcut) |
| **A4** GNN policy | permutation-invariant, **size-agnostic** message-passing policy: one policy trained on N∈[6,10] runs *unchanged* at N=10/14/18 and **ties classical** sep-kept at every size (100% / 46.7% / 0%) — generalizes across N; doesn't beat classical (comms-physics-limited, residual under the CBF) |
| **A6** learned active cooperative perception (P5) | attention fusion **rejects confident outliers** (inverse-variance 0.47 m → **learned 0.16 m**, 3×) and **ties** the optimal baseline on homogeneous Gaussian (0.12 vs 0.11 m); value-of-information gating reaches near-full accuracy at **3 of 8** messages and beats random at every budget (`swarm active`) |

**Honest swarm finding.** The classical coordination (scheduling + CBF + holding) is already excellent.
MARL **ties** it on separation in the dense, short-comms hard regime because the residual violations there
are **comms-physics-limited** (you cannot avoid what you cannot see in time), not algorithmic — the CBF sits
underneath any learned policy. The GNN's contribution is the SOTA *representation* and **generalization
across swarm size**, not beating the classical separation metric. **A6 active cooperative perception
reinforces the same lesson from the other side:** the learned attention fusion only *ties* the
inverse-variance baseline on homogeneous Gaussian fixes (where that baseline is already optimal) and *wins*
(3×) precisely on the **heterogeneous / confident-outlier** regime the Gaussian model cannot handle —
learning earns its keep where the classical assumption breaks, not where it holds.

## 6. Cooperative perception — real per-drone onboard vision (P2)

Each drone runs **real vision on its own rendered, gimbal-stabilized downward camera** (threshold-centroid
pad detection → nadir back-projection → world deck fix), fused by the A2 consensus filter, so a drone that
cannot see the pad lands on its neighbours' shared fix. `swarm run --vision` (MuJoCo engine only).

| capability | result |
| --- | --- |
| **P2.1** gimbal-stabilized camera | tilt-invariant deck fix — **0.008 m** at 0° and at 20° roll (a body-fixed cam shifts ~0.7 m or loses the pad) |
| **P2.2** wide-FOV + consensus motion-prediction | deck-position estimate error **1.99 m → 0.22 m (9×)**; coverage 0.2 → 0.6 of 3 drones |
| **P2.3** vision landing on a translating deck | **0% → ~25–33% all-landed** on the fast 0.4 m/s deck |
| **CP** blind-drone recovery (pad out of FOV) | isolated **4.24 m** (stuck at prior) → cooperative **0.34 m** |
| **P2.4** heterogeneous fleet (`--cameras N`) | camera-less drones with **no camera at all** recover **4.24 m → 0.34 m** purely from camera drones' fixes (camera drones drift 0.01 → 0.26 m — a real consensus-coupling cost) |

**Honest status.** The **perception/estimation** sharing is proven and accurate (0.22 m on a moving deck;
heterogeneous recovery to 0.34 m). But **full-swarm vision *landing* on a moving deck is only partially
reliable** (~1/3, high variance): the modeled-sensing swarm with a continuous estimate lands 100%;
vision-grade, intermittent, partial-coverage estimates + a translating deck cap the closed loop. Reliable
sequential-touchdown coordination tuned for vision-grade estimates is the hard open part. See
[SWARM.md](SWARM.md).

## 7. Sense-and-avoid + contingency failsafes (P3)

The real-robotics layer a setpoint tracker lacks: an **onboard 2-D scanning rangefinder** (noisy surface
points, no obstacle identities — no-cheats), a **higher-order CBF** that bends the *acceleration* command
(the static-obstacle barrier is relative-degree-2 for an accel-controlled quad), latency-aware, and a
PX4-style **contingency failsafe FSM**. `drone safety`; details in [SAFETY.md](SAFETY.md).

| mode (transit past the OSV superstructure, noisy onboard scan) | reached pad | collided | min clearance |
| --- | --- | --- | --- |
| no-avoid (setpoint tracker) | yes | **yes** | **−0.24 m** (inside a structure) |
| **AVOID (higher-order CBF)** | yes | **no** | **+0.32 m** |

The failsafe FSM (`ROTOR_OUT > LOW_BATTERY > GEOFENCE > LOST_COMMS > OBSTACLE_ABORT > NOMINAL`) takes over
guidance on each injected fault with correct priority (`drone safety --demo contingency`). **Honest scope:**
a reactive CBF guarantees *safety* but can deadlock on a head-on obstacle (a clear approach sector must
exist; global liveness needs a planner).

**In-loop integration (done).** The HOCBF + obstacle-abort are spliced into the live autopilot via
`drone run offshore --avoid` (latent guard: passthrough when clear → **offshore `--avoid` = baseline**,
engages near the structure). The offshore superstructure is now **collidable** (`hit_structure` termination)
so the guard acts against *true contact*, not a software-only field — nominal offshore landing is unchanged
(~100%) because the validated approach stays clear of the fore structure. 17 safety tests.

**Swarm sense-and-avoid (done).** `SafetyFilter.filter(obstacles=...)` folds sensed static keep-outs (the
OSV superstructure) into the *same* CBF-QP; each drone places them at its own deck estimate + offset (no
ground truth). `swarm run --offshore --avoid` (5 drones): **100% all-landed, separation kept, worst
obstacle clearance +0.25 m** — the swarm skirts the superstructure and still recovers. P3 sense-and-avoid
is now wired into **all three** loops: the single-drone autopilot (`drone run --avoid`, collidable
superstructure), the swarm (`swarm --avoid`), and multi-deck (`swarm multi --avoid`, per-vessel keep-outs
at each drone's assigned deck) — `swarm multi --drones 6 --decks 2 --avoid` → 100% all-landed, kept clear.

## 8. Parked honestly (not claimed working)

- **Rotor-out recovery:** fault-tolerant 3-rotor allocation + failure injection are built, but a quad with
  a dead rotor is **underactuated** and a per-step thrust-axis tracker cannot hold the period-averaged
  thrust direction under the fast yaw spin; reliable "land on 3 rotors" needs a full Mueller-style LQR
  around the spinning equilibrium (a dedicated build, deferred). **Near-term handling (now wired into the
  live `--fail-rotor` loop):** the rotor-out **controlled spinning-descent contingency** (Option D) — stop
  chasing the deck (that smears the thrust vector under the spin and flew the drone off → `out_of_bounds`
  with no contact) and instead hold the thrust axis vertical while descending at a bounded sink, optionally
  shielded. **Effect:** the violent fly-off becomes a **controlled soft descent (contact ~0.3 m/s)** that
  bounds the impact — graceful degradation, *not* a precision landing (the underactuated drift means it
  comes down near, not on, the pad). **Mueller-LQR landing attempted + diagnosed (honest):** the existing
  reduced-attitude PD *does* stabilize a dead-rotor **hover** from clean state (**7° tilt, 0.6 m drift /
  6 s**); a rate-based "Mueller" variant I tried was *worse* and was discarded. The decisive barrier:
  commanding a **descent** destabilizes it (`vz≈−0.4` → 61° tilt, ~7.6 m drift, ~1/6 land) because reducing
  collective thrust cuts the 3-rotor roll/pitch authority that holds the spin — **hover-stabilizable but
  not simple-descent-stabilizable**. **Equilibrium-LQR built + tested** (`control/rotor_out_lqr.py`, a
  documented research artifact, not wired in): a reduced-attitude LQR (thrust-axis tilt + body rates,
  gyroscopic coupling via the measured spin rate, scipy Riccati) keeps the tilt *magnitude* bounded (~26°)
  but **drifts/climbs** — it regulates the *instantaneous* attitude to a fixed point but **not the
  period-averaged thrust vector**, because the single-rotor-out equilibrium is a **limit cycle**, not a
  fixed point. **Averaged/Floquet control then cracked the landing** (`rotor_out_floquet.py`,
  `--fail-rotor-mode floquet`): steer the **spin-averaged** thrust axis (low-pass of body-z) via
  **phase-mapped precession torque** `τ_body = Rᵀ(J_z·Ω·k(a_des−a))` (the **signed** Ω is decisive — the
  spin is negative) + center-then-descend. **First controller in the project to land the dead-rotor drone:
  wind-off, clean hover — 5/10 on the deck, 3/10 on the 0.4 m pad, ~1 m median drift, no tumble** (vs the
  LQR's 66 m). **Full-pipeline transfer (diagnosed):** in the closed-loop **vision** autopilot with a
  mid-flight failure onset it **does** control the spinning descent — *even with wind* it holds
  **herr ≈ 0.1 m** from the failure (~1.3 m up) down to ~0.7 m. The **last ~0.5 m** is the blocker: the pad
  fills the close-range FOV, the grid clips, **vision tracking is lost**, and the underactuated vehicle
  **drifts ~1 m blind** → off the deck. A blind-commit (hold vertical + damp velocity on tracking loss)
  lifted full-pipeline only to **8% wind-off / 0% wind-on** (a press-to-contact made it *worse*). The limit
  is now precise and **fundamental/sensing, not control**: below **~0.3 m the downward ArUco is too
  close/zoomed to detect at all** (an inherent fiducial blind-zone), and the underactuated vehicle **drifts
  off the deck during that blind final descent**. **Close-range sensor tried (downward optical-flow + laser,
  PMW3901-class):** added as a real no-cheats sensor (`FlowConfig`; measures deck-relative *horizontal
  velocity* down to <0.3 m where the fiducial is dead) and fused into the EKF (`update_velocity_xy`). **It
  does NOT unblock the landing** (full-pipeline still ~0%, wind-off and wind-on): flow supplies *velocity*,
  but the spinning underactuated vehicle needs a close-range *horizontal position* reference. Damping the
  measured velocity only holds it at whatever lateral offset it drifted to; with no absolute position fix it
  cannot re-center, and the spin's lateral force carries it off the deck. A downward LiDAR/laser would give
  *altitude* (range), also not lateral position — so neither sensing upgrade resolves it. **The true fix is a
  close-range POSITION reference** (a nested smaller fiducial that stays decodable inside 0.3 m, or a
  beacon/range-bearing system) — a **perception redesign**, deferred. The optical-flow sensor is **kept** as
  a permanent realistic capability (improves terminal velocity feedback; verified **no regression** — ground
  92% / ship 92% / 112 tests). **Net: control law validated + spinning descent flown in the full pipeline
  (herr ≈ 0.1 m, with wind); on-deck landing remains blocked by the close-range position-sensing gap.**
  Opt-in (`--fail-rotor-mode floquet`); the wind-robust Option-D spinning-descent stays the default.
- **MPC closed-loop:** estimation-limited (see §1); kept available (`--controller mpc`) but not the default.
- **Markerless fusion (classical + learned B3):** the classical centroid fallback is neutral/safe
  (border-gated). The **learned CNN deck detector (B3)** is built, trained, and integrated
  (`--cnn-markerless`): standalone it localizes the deck to median **5.9 px (~8.5 cm)** at 96% confidence
  (spatial-softmax net, self-labelled from rendered pads), but closed-loop on ship moderate it is
  **neutral** (92% with/without — ArUco rarely fails there). A heavy marker-loss A/B to show a closed-loop
  win is future work; the accurate ArUco-independent detector is the deliverable.

## Reproduce

```powershell
python -m pytest -q                          # 112 tests
drone train --eval runs/rl/ground/ppo_final.zip   # §2 RL vs baseline (hard regime)
drone run ship --airwake ; drone run ship --airwake --dob   # §3 air-wake + DOB
python scripts/eval_seakeeping.py            # §3 spectra/RMS table (+ --episodes for the air-wake A/B)
drone reachability                           # §4 safe set + shield A/B
python scripts/eval_tube_mpc.py              # §1 DOB-MPC vs plain MPC tracking under wind (B4)
swarm verify --drones 6 --seeds 25           # §5 A3 separation certificate sweep
python scripts/eval_consensus.py             # §5 A2 consensus vs raw deck error
swarm multi --drones 9 --decks 3 --foul-deck 2   # §5 A5 fouled-deck re-tasking (kinematic)
swarm multi --engine mujoco --drones 6 --decks 3 # §5 A5 on REAL physics (P4.1)
swarm active                                     # §5 A6 learned active cooperative perception (P5)
swarm run --drones 5 --offshore --avoid          # §7 swarm sense-and-avoid of the OSV superstructure (P3)
swarm run --drones 5 --vision --cameras 2    # §6 heterogeneous cooperative perception
python scripts/eval_heterogeneous_perception.py   # §6 camera-less drones recover via sharing
drone safety                                 # §7 sense-and-avoid (HOCBF) + contingency FSM
python scripts/eval_platforms.py             # §1 new platforms: inclined deck / truck / USV (P4)
```

See [PROGRESS_LOG.md](PROGRESS_LOG.md) for the checkpointed history and
[ADVANCED_ROADMAP.md](ADVANCED_ROADMAP.md) for the full feature plan.
