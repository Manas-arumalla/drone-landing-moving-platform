# Progress Log

A running, checkpointed record of the project. Each checkpoint corresponds to a git commit.
Newest entries on top. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the roadmap and
[REALISM_CHARTER.md](REALISM_CHARTER.md) for the simulation-fidelity rules.

---

## Close-range sensor for the rotor-out blind-zone — downward optical-flow added; honest verdict

**Date:** 2026-05-25

**Goal:** the rotor-out thread closed with a precise blocker — below ~0.3 m the downward ArUco is too
zoomed to decode, so the underactuated spinning vehicle drifts off the deck during the blind final descent.
The hypothesized fix was a **close-range sensor that works inside 0.3 m** (the project's LightWare-style
reference). This checkpoint actually builds and tests one.

**Done (a real, no-cheats sensor — kept permanently):**
- `sim/sensors/models.py`: `FlowConfig` + `SensorReading.flow_vel/flow_valid` — a downward **optical-flow +
  laser** unit (PMW3901-class) that measures the drone's **horizontal velocity relative to the surface**
  from image-texture flow. It does **not** need the ArUco to be decodable, so it works in the <0.3 m
  blind-zone; noise grows with altitude and it's valid below 2.5 m. No ground truth — the measurement is
  `true_rel_velocity + noise`, exactly like the IMU/rangefinder.
- `sim/world.py`: `_read_sensors` computes `raw_rel_vel_xy = drone_vel − deck_vel` and passes it to the suite.
- `autopilot.py`: fuses the flow velocity into the EKF via `update_velocity_xy` (a gated Kalman update;
  rel-velocity = `−flow_vel`). **Guard:** flow is a *velocity* fix, not a position fix, so it deliberately
  does **not** refresh `last_good_k` (the marker-tracking liveness signal) — keeping `tracked` alive on
  velocity alone would mask genuine marker loss and change the validated nominal commit behavior.

**Honest verdict — it does NOT unblock the rotor-out landing** (full-pipeline still ~0%: 0/12 wind-off,
0/12 wind-on). The blocker is a missing close-range **horizontal POSITION** reference, and flow supplies
**velocity**. Damping the measured velocity only holds the spinning vehicle at whatever lateral offset it
drifted to; with no absolute position fix it cannot re-center, and the spin's lateral force carries it off
the deck. A downward LiDAR/laser would give **altitude** (range), also not lateral position — so neither
sensing upgrade resolves it. **The true fix is a close-range POSITION reference** (a nested smaller fiducial
that stays decodable inside 0.3 m, or a beacon/range-bearing system) — a **perception redesign**, deferred.

**Kept anyway + verified no regression.** Real landing drones carry optical flow; it improves terminal
velocity feedback, so the sensor stays in the suite. The shared-path changes (SensorReading / world / EKF /
autopilot) were re-validated: **ground 92% / ship 92%**, and the **full suite is 112/112 green**. Opt-in
`--fail-rotor-mode floquet` is unchanged; the wind-robust Option-D spinning-descent stays the **default**.
**The rotor-out research thread is now fully wrapped:** control law validated (and flies the spinning
descent in the full pipeline at herr ≈ 0.1 m, even with wind); the on-deck landing is a documented
close-range position-sensing gap, not a control failure.

---

## Rotor-out AVERAGED/precession control -- the first controller that LANDS the dead-rotor drone

**Date:** 2026-05-25

**Cracked the rotor-out landing at the controller level** (`control/rotor_out_floquet.py RotorOutFloquet`,
`drone run --fail-rotor K --fail-rotor-mode floquet`). The fixed-point LQR failed because the equilibrium is
a **limit cycle** (a fast spin, measured ~15-20 rad/s here); this controller steers the **spin-averaged**
thrust axis instead:

1. low-pass body-z over a few spin periods → the averaged thrust axis `a`;
2. position loop → a tilt-limited desired axis `a_des`;
3. **precession steering**: `dL/dt = τ` with `L = J_z·Ω·a`, so an inertial torque perpendicular to `a`
   precesses it — command `τ_world = J_z·Ω·(k_a·(a_des−a)_perp − k_da·ȧ)` (the **signed** Ω is decisive —
   the spin is *negative*, so `L` points opposite `a`; using `|Ω|` drove it the wrong way and tumbled) and
   realize it with a **phase-mapped body torque** `τ_body = Rᵀ·τ_world` (yaw free);
4. **center-then-descend** (sink only when over the pad) + a PD fallback during spin-up.

**Result (clean-hover, truth-state, wind-off — the validated controller regime):** **lands 5/10 on the
deck (<0.9 m), 3/10 on the tight 0.4 m pad, ~1.0 m median drift, no tumble** — the **first** rotor-out
controller in the project to actually land (vs the LQR's 66 m drift and the PD's tumble-on-descent).

**Full-pipeline transfer — diagnosed deeper (better than first thought, but the landing is still blocked).**
Tracing the closed-loop **vision** autopilot with a **mid-flight** failure onset showed the Floquet
controller actually **transfers and controls the spinning descent** — *even with wind*: from the rotor
failure (~1.3 m above deck) it holds **herr ≈ 0.1 m** and rides the spin down to ~0.7 m. The blocker is the
**last ~0.5 m**: the 1 m pad fills the close-range 90°-FOV, the ArUco grid clips, **vision tracking is
lost**, and the underactuated vehicle then **drifts ~1 m during the blind final descent** → lands off the
0.9 m deck (`off_platform`). Added a **blind-commit** on tracking loss (hold the axis vertical + damp the
residual horizontal velocity, no position chase, commit the descent) — it helped only marginally:
**full-pipeline wind-off 0% → 8%, wind-on 0%**. So the *control law* is validated and it *does* fly the
spinning descent in the full pipeline, but the **close-range vision loss + blind-descent drift** (the same
close-range limit the nominal system mitigates with commit logic, here compounded by the spin) blocks the
on-deck landing. Wind itself is no longer the primary blocker — the controller held herr ≈ 0.1 with wind
during the tracked descent; the close-range blind phase is.

**Close-range touchdown — investigated, and the limit is now precisely understood (fundamental, sensing).**
Traced the detector through the final descent: below **~0.3 m the rangefinder reads ~0.1-0.2 m and the
ArUco marker is too close/zoomed to detect at all** (`found=False`) — *not* a "keep the centre marker"
fix; it is the inherent close-range blind-zone of a downward fiducial camera (the nominal landing has the
same zone, handled by a brief committed velocity-hold). For the **underactuated dead-rotor** vehicle that
final ~0.3 m blind descent is the killer: with no horizontal feedback the spinning vehicle **drifts off the
0.9 m deck faster than it settles**, no matter the terminal law. Tried **velocity-damped blind-commit**
(hold axis vertical + damp the residual velocity) → full-pipeline **8% wind-off**; adding a **press-to-
contact** (settle fast) made it *worse* (off_platform 10/12 — it comes down but still off-deck). Reverted
the press; kept the velocity-damped blind-commit.

**Final honest status of the rotor-out thread.** The averaged/precession control **law is validated**
(clean-hover truth: 5/10 on-deck) and **transfers to the full vision pipeline, flying the spinning descent
at herr ≈ 0.1 m even with wind** — a genuine result. The **on-deck landing is blocked by a fundamental
sensing limit**: the <0.3 m ArUco blind-zone + the underactuated blind-descent drift cap the full pipeline
at ~8% (wind-off). The real-world fix is a **close-range sensor that works inside 0.3 m** (e.g. the
downward LiDAR/laser altimeter in the project's reference image) for the terminal blind phase — a sensing
upgrade, not a control one. Wired **opt-in** (`--fail-rotor-mode floquet`); the wind-robust Option-D
spinning-descent stays the **default**. This wraps the rotor-out research thread.

Wired as **opt-in** (`--fail-rotor-mode floquet`); the **default stays the Option-D spinning-descent
contingency** (wind-robust, bounded). Controller-level test (`RotorOutFloquetTests`: lands wind-off, no
tumble). The fixed-point LQR (`rotor_out_lqr.py`) stays as the documented negative-result artifact.

---

## Rotor-out equilibrium-LQR -- built + tested; honest negative result (limit-cycle, not a fixed point)

**Date:** 2026-05-25

**Took on the textbook 3-rotor *landing* (Mueller & D'Andrea equilibrium-LQR) as a focused build; it does
not yet land, and the *why* is the valuable result.** New `control/rotor_out_lqr.py RotorOutLQR`: linearizes
the **reduced-attitude** dynamics (inertial thrust-axis tilt `n_x, n_y` + body roll/pitch rates `p, q`)
about the spin, keeping the **gyroscopic coupling** (parametrized by the measured spin rate `r`), and solves
a continuous-time-Riccati **LQR** `τ = −K(r)·[n−n_des; p; q]` (cached per spin rate) + a gentle
authority-limited position outer loop + a collective channel. scipy `solve_continuous_are`.

**Findings (clean wind-off A/B vs the existing PD):**
- The **existing PD** (`compute_rotor_out`) holds a dead-rotor **hover at 18° tilt, 0.5 m drift**, and a
  *centred* descent stays bounded (22°) — so it regulates the **period-averaged thrust direction** well.
  (Its limits: the vertical channel climbs rather than cleanly descending, it needs near-centred
  conditions, and **wind destabilizes it badly** — 61° tilt with default wind on.)
- The **LQR**, after rebalancing Q toward heavy rate-damping + fixing the thrust projection (→0 when
  inverted), keeps the **tilt magnitude bounded (~26°)** but **drifts tens of metres and climbs** — it
  regulates the *instantaneous* tilt to a fixed point but **not the period-averaged thrust vector**, because
  the single-rotor-out equilibrium is a **limit cycle (a spin), not a fixed point**.

**Conclusion:** a correct equilibrium-LQR must control the **spin-averaged** thrust direction
(**averaged / Floquet control around the periodic orbit**), with a thrust-preserving descent and wind
robustness — a substantially deeper build than one session. The module is kept as a **documented, tested
research artifact** (honest docstring; **not wired into the live system** — the production rotor-out
handling stays the Option-D bounded spinning-descent contingency). 2 tests (constructs + valid wrench +
gain cache). No regression elsewhere.

---

## P6 dissemination -- benchmark matrix + demo GIFs (the hold was released)

**Date:** 2026-05-25

**Built the controller x scenario x disturbance benchmark + demo GIFs** (P6 was held until explicit
release; released now). Two reusable, additive tools:

- **`scripts/benchmark.py` -> `docs/BENCHMARK.md`** — a curated matrix (12 episodes/cell, vision pipeline,
  no truth in the loop), writing markdown incrementally. Headlines, all consistent with the per-feature
  A/Bs: single-drone landing (ground 83% / ground-hard 8% / ship calm 100%, moderate 92%, rough 83% /
  offshore / inclined gentle ~100%, moderate ~0% / USV / truck); controllers ground (geometric 83%, **RL
  100%**, IBVS 58%, MPC 33%) + ship (geometric 92%, IBVS 83%); disturbance (air-wake 58% -> **+DOB 83%**,
  spectral 83%, shield 92%, **offshore+avoid 100%**); swarm (4/6 drones, consensus, offshore+avoid,
  9->3-deck-foul) all **100% all-landed**. ~19 min to run.
- **`scripts/make_demo_gifs.py` -> `docs/media/*.gif`** — chase-cam GIFs of clean landings (ground / rough
  ship / offshore) + a swarm clip, embedded in the README. (Bug found + fixed: the perception camera must
  render at the autopilot's calibrated `CAM_W x CAM_H`, else the detector intrinsics mismatch and the demo
  flies off; the generator also retries seeds until a successful landing so the demo shows a clean one.)

Wired into README (Demos section + benchmark link), RESULTS (matrix now script-produced, not "held"), and
`drone list`. No code regression (109/109 tests). **All P1-P6 now addressed**; remaining are the deferred
rotor-out equilibrium-LQR (substantial) and PX4/ROS2 (needs WSL2).

---

## Phase 5 / A6 -- Learned active cooperative perception (whom to trust + what to communicate)

**Date:** 2026-05-24

**The research capstone: a learned graph-attention fusion + value-of-information communication gating**
(`drone_landing_swarm/active_perception.py`, CLI `swarm active`). The A2 consensus filter fuses neighbour
deck fixes by **inverse-variance** weighting -- optimal for homogeneous Gaussian noise, but it (a) is fooled
by a **confident outlier** (a reflection/wrong decode reported with high confidence) and (b) ignores
bandwidth. The two collaborative-perception ideas (V2VNet / Where2comm) that fix this:

- **Learned attention fusion** (`AttentionFusion`, torch): permutation-invariant attention over the fix
  set whose output is a **convex combination of the actual fixes** (cannot hallucinate). Its decisive input
  is each fix's **deviation from the group median** -- a translation-invariant *agreement* feature that
  inverse-variance weighting structurally can't use -- so it learns to **reject confident outliers**.
  Trained supervised on synthetic heterogeneous scenes (~4 s CPU); checkpoint `runs/active/fusion.pt`.
- **Value-of-information gating** (`select_broadcasters`): under a bandwidth budget of B messages, pick the
  B broadcasters that most cut the fused variance (greedy by expected precision over the comms graph).

**Honest A/B (this is the headline finding):**

| regime | equal-mean | inverse-variance (consensus) | **learned** |
|---|---|---|---|
| heterogeneous (confident outliers) | 0.40 m | **0.47 m** (worse -- trusts them) | **0.16 m** (3x better) |
| homogeneous (Gaussian) | 0.15 m | 0.12 m | **0.11 m** (a tie) |

So learning **wins exactly where the modelling assumption breaks** (outliers) and **ties the optimal
baseline where it holds** -- the same honest lesson as our MARL (learning helps where the bottleneck is
algorithmic, not where a classical method is already optimal). **Bandwidth Pareto:** value-of-information
selection reaches 0.127 m at **3 of 8 messages** (~ fuse-all's 0.119 m) and beats random selection at every
budget -- the Where2comm bandwidth story. CLI `swarm active [--eval runs/active/fusion.pt]`;
`scripts/eval_active_perception.py`; 6 tests in `tests/test_active_perception.py`. **No regression:
102/102 tests.** Separate swarm module; no ground truth at inference.

---

## Phase 4 / P4.1 -- Multi-deck MuJoCo physics (removes the kinematic A5 shortcut)

**Date:** 2026-05-24

**The A5 multi-deck scenario now runs on real physics.** The kinematic `MultiDeckCoordinator` flew
point-mass drones onto *modeled* decks; new `multideck_world.py` + `multideck_runner.py` put **K
independent 6-DOF servo decks + N real X2 quadrotors** in one MuJoCo model and fly the drones with the
validated geometric controller, landing via **true contact** on the assigned deck.

- `MultiDeckMujocoWorld` (`multideck_world.py`): stamps K decks (each the validated platform on its own
  ring base) + N drones + N nadir gimbals; per-deck joint/geom addressing; `deck_state(k)`,
  `support_feet(i, k)` (feet planted on the *assigned* deck).
- `MujocoMultiDeckCoordinator` (`multideck_runner.py`): **reuses the kinematic A5 decision layer unchanged**
  -- balanced Hungarian start + decentralized auction re-tasking, per-deck `LandingScheduler` +
  `HoldingStack`, no-cheats `SwarmSensing`, non-bypassable `SafetyFilter` -- but flies real X2 physics.
  Distinct **on-deck spots** so a second drone lands *beside* an already-landed one, not on top.
  Fouled-deck event mirrored from the kinematic A5.
- **Results: 6 drones -> 3 decks 6/6 landed (2/2/2), 4 drones -> 2 decks 4/4 (2/2)**, balanced, separation
  kept (min-sep 0.78-1.11 m), no collisions, ~13-20 s. CLI `swarm multi --engine mujoco [--watch]`
  (real MuJoCo viewer). 1 test in `tests/test_swarm.py`.

**Honest note:** the per-deck *shared* deck estimate is truth+sensing-noise (the A5 observer convention,
same as the kinematic version); the drones' own state + neighbour broadcasts carry no truth. The CBF's
discrete-time certificate can dip slightly negative under true second-order physics (the documented
"bounded margin" caveat) but the *true* min-separation stays above d_min. **No regression: 96/96 tests.**

---

## Rotor-out (b) -- Mueller-LQR attempt: honest finding (hover-stabilizable, descent destabilizes)

**Date:** 2026-05-25

**Attempted the textbook 3-rotor *landing* (Mueller & D'Andrea spinning-equilibrium idea); the controller
attempt underperformed, but the investigation produced the decisive physical finding.** Implemented a
rate-based reduced-attitude controller (command roll/pitch *rates* to precess the average-thrust axis + a
gyroscopic feed-forward `ω×Jω`, yaw free) and A/B'd it on a clean dead-rotor hover.

- **The existing reduced-attitude PD already stabilizes a dead-rotor HOVER** from clean state: **7° tilt,
  0.6 m drift over 6 s** (the vehicle holds a controlled spin — it does *not* immediately tumble). The
  earlier "0% / flies off" came from engaging it mid-tumble (rotor fails during a dynamic approach or while
  still in pre-init SEARCH), not from a controller deficiency.
- **My rate-based "Mueller" controller was *worse*** (69° tilt, 5.3 m drift in the same hover) — so I
  **discarded it** (no regression left in the codebase; reverted cleanly).
- **The real barrier (decisive):** commanding a **descent** destabilizes the dead-rotor vehicle — at
  `vz_des = −0.3…−0.5 m/s` it tumbles (61° tilt, ~7.6 m drift, ~1/6 land). Reducing collective thrust to
  descend **reduces the 3-rotor roll/pitch authority that maintains the spin-stabilization**, so it loses
  control on the way down. The vehicle is **hover-stabilizable but not simple-descent-stabilizable**.

**Conclusion (honest):** a reliable 3-rotor *landing* genuinely needs the full Mueller equilibrium-LQR that
manages the descent *within* the spinning equilibrium (maintaining stabilization authority while sinking) —
a substantial dedicated build, confirmed deferred. The practical handling remains the **Option-D bounded
spinning-descent contingency** (soft ~0.3 m/s come-down, not a precision landing). No code change kept
(controller reverted); finding recorded in `rotor-out-decision` memory + RESULTS §8.

---

## P3 swarm hook -- sensed static obstacles folded into the swarm CBF (`swarm ... --avoid`)

**Date:** 2026-05-25

**The last P3 integration piece: the swarm now avoids sensed static obstacles too.** `SafetyFilter.filter`
gained an `obstacles=[(pos, radius), ...]` argument — zero-velocity keep-out volumes folded into the **same
exact CBF-QP** as the inter-drone and deck constraints (additive: an empty list reproduces the original
behaviour exactly). `SwarmConfig.obstacles` holds them as **deck-relative** `(dx, dy, radius)` offsets, and
both the kinematic and MuJoCo runners place them at **each drone's own deck estimate + offset** (so the
obstacle positions carry that drone's deck-estimate noise — **no ground truth**, consistent with the A1
sensing model + a known vessel map). `swarm ... --avoid` auto-populates the OSV superstructure
(wheelhouse + bow); the run summary reports the **true** worst obstacle clearance.

**Result:** `swarm run --offshore --avoid` (5 drones) — **100% all-landed, separation kept, worst obstacle
clearance +0.25 m, kept clear 3/3** (never entered a keep-out) while still recovering the swarm. 3 new
tests (`StaticObstacleTests`: filter deflects a head-on obstacle / empty list is passthrough / a swarm
keeps clear and still lands). **Multi-deck parity done too:** `MultiDeckConfig.obstacles` + `swarm multi
--avoid` place per-vessel keep-outs at each drone's *assigned* deck's estimate (kinematic + MuJoCo
multi-deck runners); `swarm multi --drones 6 --decks 2 --avoid` → 100% all-landed, kept clear. **P3
sense-and-avoid is now fully wired across single-drone (`drone run --avoid` + collidable superstructure),
swarm (`swarm --avoid`), and multi-deck (`swarm multi --avoid`).** 4 obstacle tests; full suite green
(109).

---

## P3.1 completion -- offshore superstructure made COLLIDABLE (sense-and-avoid vs true contact)

**Date:** 2026-05-25

**Closed the P3.1 gap: the OSV superstructure was visual-only (`contype=0`) -> now COLLIDABLE.** The
wheelhouse, its base, the mast and the bow in `x2_landing_offshore.xml` are now solid keep-out volumes
(`contype=1 conaffinity=1 condim=3`); the hull stays visual (it sits at deck level and would perturb the
landing physics). New world plumbing: `LandingWorld` caches `obstacle_gids` (the named superstructure
geoms; empty on the clear ground/ship worlds), `_contact_sets` flags `hit_structure` when any drone geom
touches one, and `_termination` returns a new terminal **`hit_structure`** outcome. So a stray approach
into the structure is now a **real crash**, giving the `--avoid` guard genuine teeth (it was guarding a
ghost before).

**No regression:** nominal offshore landing **~100% (8 ep)** — the validated approach stays high until
aligned over the pad, so it never nears the fore structure (the guard stays *latent*; I did **not** contrive
an adversarial low-transit demo because the controller correctly avoids the structure on its own — the
standalone `drone safety --demo avoid` shows the deflection). 2 new tests
(`CollidableSuperstructureTests`: offshore has >=3 obstacle geoms + a drone dropped into the wheelhouse
terminates `hit_structure`; ground has none). Full suite green.

---

## IBVS commit-descent fix -- closes the loop on image position through commit (ground 25%->58%, ship 75%->83%)

**Date:** 2026-05-25

**Re-verified + fixed IBVS** (it had appeared to perform better in earlier runs). Re-verify: ground 25% / ship 75% (12
ep). Trace showed it is **not broken** — IBVS lands the drone softly (reaches SECURED on the deck) but on
the fast **random** rover touched down ~0.44 m off-centre, just outside the **0.40 m success radius**, so
it scored `timeout`. Root cause: at COMMIT it is centred (~0.18 m), but the commit phase switched to
**open-loop velocity-hold**, so during the ~0.7 s descent the rover slid out and it drifted ~0.26 m before
contact (the marker leaves the close-range FOV and optical-flow velocity degrades). **Fix:** while the
fiducial is still tracked, IBVS keeps **closing the loop on the image position through the commit descent**
(`ibvs_commit` -> `vhold=False` in `autopilot.step`), falling back to velocity-hold only if the marker is
lost. **Result: ground IBVS 25% -> 58%, ship IBVS 75% -> 83%** — a clean win, **no regression** (the change
is gated to the IBVS controller; geometric/MPC/RL commit paths untouched; full suite green). Geometric
still leads (88% ground) and stays the default. NOT a regression from P3/P4/P5 — those never touched IBVS.

---

## Rotor-out -- Option-D controlled spinning-descent contingency wired into the live `--fail-rotor` loop

**Date:** 2026-05-25

**The rotor-out contingency now actually engages in the live autopilot (was standalone-only).** Before:
`--fail-rotor K` ran `compute_rotor_out`, which *chased* the deck; under the fast yaw spin of an
underactuated dead-rotor quad that smears the thrust vector around the rotation, so the drone **flew off
(`out_of_bounds`, no contact)**. Now the autopilot's rotor-out branch runs the **Option-D contingency**
(the choice recorded in the rotor-out decision): **stop the horizontal chase** (zero horizontal target),
hold the thrust axis vertical, and descend at a **bounded sink** (`rotor_out_sink` 0.6 m/s, shielded by the
reachability guard if `--shield`). State reports `ROTOR_OUT`.

**Effect:** the violent fly-off becomes a **controlled soft descent — contact ~0.3 m/s** (the impact is now
recorded, where before there was none). Honest: it is **graceful degradation, not a landing** — a dead
rotor is underactuated, so the vehicle still drifts and comes down *near*, not *on*, the pad (some episodes
contact near centre, some drift out). A reliable 3-rotor **landing** still needs the deferred
Mueller–D'Andrea LQR-around-the-spin. Additive (only the `failed_rotor` branch changed); no test regression.

---

## Phase 3 follow-up -- sense-and-avoid spliced into the LIVE landing loop (`--avoid`)

**Date:** 2026-05-24

**The P3 safety layer now affects the active landing behaviour (was standalone-only).** New autopilot
hook `--avoid` (`AutopilotConfig.use_avoid`): each step, in the **deck-relative frame** (obstacles are
fixed offsets from the deck; the drone's deck-relative pose comes from the EKF — no ground truth), the
onboard `RangeSensor` scans the offshore superstructure (mirrored as a deck-relative `ObstacleField`), the
**higher-order CBF** bends the horizontal command to keep clear, and the **contingency FSM** breaks off the
approach (climb-and-hold, state `AVOID_ABORT`) inside the abort radius. It is a **latent guard**: when no
obstacle is within `avoid_engage` (1.6 m) it is a pure passthrough, so the validated controller is
untouched on a clear deck.

- Wiring: `cli.build()` constructs the offshore `ObstacleField` and passes it to the autopilot when
  `--avoid`; other scenarios get an empty field (passthrough). `swarm`'s `SafetyFilter` already supports
  static obstacles, so the analogous swarm hook is a small follow-on.
- **No regression: offshore `--avoid` lands 88% = offshore baseline 88%** (8 ep) — the guard doesn't
  disturb the clean approach. New test `test_offshore_field_sensed_and_deflects` proves the superstructure
  is sensed and a command aimed at the wheelhouse is bent back. 103/103 tests.

**Honest scope:** in a nominal offshore landing the drone descends onto the pad clear of the fore
superstructure, so the guard rarely engages — it activates when the approach strays near the structure
(the `drone safety --demo avoid` standalone shows the full deflection). Battery/lost-comms/geofence
branches of the FSM stay exercised standalone (`drone safety`) — the sim has no battery/datalink signal to
drive them in-loop; obstacle-abort and rotor-out are the branches with live signals.

---

## Phase 4 fix -- platform control: seed-0 acquisition failure (the "no controller" bug)

**Date:** 2026-05-24

**Fixed the broken-feeling control on `drone watch inclined|usv|truck`.** Symptom: the drone detected
nothing, hovered/drifted, and never landed — most visibly in `watch`, which defaults to **seed 0**. Root
cause (traced, not guessed): each platform's `reset()` draws a different number of RNG values, so the SAME
seed yields a **different spawn offset** per scenario — and seed 0 happened to give inclined/truck/usv a
**~2 m offset** spawn. At the 2.5 m spawn altitude with the **90° camera**, a 2 m-offset marker sits right
at the FOV edge (~42° off-nadir vs 45° half-angle), so the drone never got the initial 4-marker grid lock,
sat in SEARCH, and drifted out of bounds. (Ground got a ~1.3 m offset at seed 0, so it acquired — which is
why ground was fine and these were not.) **Fix (additive, ground/ship untouched):** deploy the drone closer
on these harder moving/tilted decks (`init_xy_spread` 1.5 → **0.8** for inclined/usv/truck only — a
realistic "launched roughly overhead" assumption), and **gentle the truck/USV motion** (truck 0.7 → 0.45
m/s, USV 0.6 → 0.4 m/s, larger loop radius → lower curvature) so an acquired target stays trackable through
descent. **Result: inclined 90% → 100%, truck 50% → 100%, USV 42% → 83%; every seed-0 episode now lands
(watch shows a clean landing).** No regression: ground/ship unchanged, 102/102 tests. (Also verified a
candidate SEARCH-altitude-hold change and **reverted it** — passing zero rel_vel to the controller removed
velocity damping and caused a runaway descent that broke ground; the lesson is logged.)

---

## Phase 4 (part) -- New platform types: inclined deck, USV, moving truck (breadth)

**Date:** 2026-05-24

**Three new landing-target types, all additive and reusing the validated worlds** (no new XML, no change to
ground/ship/offshore): each is a new `PlatformMotion` subclass driving an existing world via the
`platform=` hook. Inclined/USV reuse the 6-DOF ship world (roll/pitch/heave servos); truck reuses the
ground world. Scenarios `inclined` / `usv` / `truck` + presets; `drone run inclined --incline
gentle|moderate|steep`. Eval: `scripts/eval_platforms.py`. 10 tests in `tests/test_platforms.py`.

- **Inclined deck** (`sim/platforms/inclined.py`): a persistently tilted surface (barge ramp / listing
  vessel / sloped pad) + gentle residual motion. **Honest gradient: gentle ~6 deg lands 90%, moderate
  ~12 deg ~0% (timeout).** The level-attitude press can't seat 3 feet on a steeper slope -> motivates
  attitude-matched touchdown (future) and pairs with the `--shield` (preset `inclined-shield`). Default
  preset = gentle (the working baseline).
- **USV** (`sim/platforms/usv.py`): a small agile surface craft -- planar maneuvering (smooth bounded
  loop) **plus** a lively short-period seaway response (8 deg roll, ~3 s). **42% landing** (the target
  both translates *and* rocks -- the hardest of the three; failures are mostly off-platform).
- **Truck** (`sim/platforms/truck.py`): a road vehicle cruising a smooth oval loop (steady 0.7 m/s, level,
  predictable). **50% landing** (a continuously *translating* target is harder than the near-stationary
  rover -- the drone must ride the moving bed through the descent).

**KEY BUG FOUND + FIXED (general lesson):** the moving platforms first scored ~0-25% with everything
ending `out_of_bounds`. Two causes, both fixed: (1) a **hard wall-reflection** in the early path model
snapped the heading 180 deg, and the drone (carrying velocity-feedforward momentum) overshot to the world
edge -> replaced with a **smooth closed loop** (no reflections); (2) the platform **started on the loop
(radius ~2.3 m from origin)** while the drone spawns at the origin, so the marker was never under the
camera and the drone sat in SEARCH and drifted out -> **shifted the loop to start at the origin** (under
the drone) for immediate acquisition. Lesson: a moving-target scenario must spawn the target under the
drone and move *smoothly*, or the vision pipeline never acquires. **No regression: 95/95 tests** (+10).

---

## Phase 3 -- Sense-and-avoid + contingency failsafes (the biggest real-robotics gap)

**Date:** 2026-05-24

**Closed the largest gap between "lands well on a clear pad" and "field-safe autopilot."** New
`src/drone_landing/safety/` package, all decision-only, estimate-driven (no ground truth), and **additive**
(the validated ground/ship/offshore landing loops are untouched). Demo: `drone safety` (+ `--demo
avoid|contingency`); standalone `scripts/eval_obstacle_avoidance.py` / `eval_contingency.py`. Full writeup:
docs/SAFETY.md.

- **P3.1 obstacle field** (`obstacles.py`): the OSV superstructure (wheelhouse + mast + bow, matching
  `x2_landing_offshore.xml`) as vertical primitives with exact ray-cast + signed-distance; pure geometry.
- **P3.2 onboard sensing** (`RangeSensor`): a no-cheats **2-D scanning rangefinder** -- casts beams, returns
  noisy **surface points** within range (blind zone + Gaussian noise + dropout), NOT obstacle identities.
  Horizontal scan plane (2.5-D): a beam at altitude z only hits obstacles spanning z.
- **P3.3/P3.4 higher-order CBF** (`HOCBFAvoider`): the inter-drone CBF is single-integrator (bends a
  *velocity*); a quadrotor is accel-controlled, so the static-obstacle barrier has **relative degree 2** ->
  a proper **HOCBF** (Xiao & Belta) solved as a minimal-intervention accel QP via Hildreth. **Actuation
  latency** handled by a look-ahead (`p + v*latency`) + radius inflation. **A/B (noisy onboard scan, transit
  past the superstructure): no-avoid penetrates -0.24 m (collides); HOCBF holds +0.32 m clearance, 0
  collisions, still reaches the pad.** Honest: a reactive CBF guarantees safety but can deadlock on a
  head-on obstacle (a side must exist -> realistic clear-sector approach); global liveness needs a planner.
- **P3.5 contingency FSM** (`ContingencySupervisor`): PX4/ArduPilot-style priority failsafes -- `ROTOR_OUT >
  LOW_BATTERY > GEOFENCE > LOST_COMMS > OBSTACLE_ABORT > NOMINAL`. **Rotor-out (Option D):** a
  controlled **spinning descent** at bounded sink (graceful degradation, NOT a precision landing -- a quad
  is underactuated with a dead rotor; full Mueller-LQR "land on 3 rotors" deferred). Obstacle-abort has
  hysteresis. Demo prints the state timeline as faults are injected.

**No regression: 85/85 tests** (14 new in `tests/test_safety.py`: ray-cast, altitude gating, surface
returns, max-range blind, clustering; HOCBF passthrough + closed-loop never-penetrates; FSM priority +
hysteresis + rotor-out descent). **Not yet spliced into the live MuJoCo/swarm loops** (the clean next
integration step; kept out for now under the no-regression mandate).

---

## Phase 2 / P2.4 -- Heterogeneous cooperative perception (camera drones guide camera-less drones)

**Date:** 2026-05-24

**A genuinely heterogeneous fleet: only some drones carry a camera.** Until now every drone had its own
camera and a "blind" drone simply had the pad outside its FOV. P2.4 adds a true camera-less class of drone
that carries **no camera at all** and must recover the deck *entirely* from neighbours' shared visual fixes
-- the realistic asymmetric-fleet case (cheap relay/forwarder drones + a few well-equipped sensor drones).

- **`SwarmVision(camera_ids=...)`** -- only the listed drones render + detect; camera-less drones return
  `None` every frame (they have no sensor), so the consensus filter (A2) treats them as observation-free
  nodes that converge purely on their neighbours' fixes.
- **`SwarmConfig.camera_drones`** wired through `mujoco_runner`, plus a **`swarm run --vision --cameras N`**
  CLI flag (only the first `N` drones carry a camera; guarded to require `--vision` and `1..n_drones`).
- **A/B demo** (`scripts/eval_heterogeneous_perception.py`, 5 drones, cameras {0,1}): the camera-less
  drones {2,3,4} go from **isolated 4.24 m deck-position error -> cooperative 0.34 m** -- recovered the
  deck purely from the camera drones' broadcast fixes. (Honest cost: the camera drones themselves drift
  from 0.01 m to 0.26 m as the consensus pulls them toward the shared estimate -- a real coupling
  trade-off, dwarfed by the camera-less majority's 12x gain.)

**Honest status:** this proves the *perception/estimation* sharing across an asymmetric fleet, the same as
the earlier blind-recovery result; closed-loop full-swarm *landing* on a moving deck remains the partially
reliable open part noted in P2.3. **No regression: 71/71 tests** (new
`test_cameraless_drone_recovers_from_camera_drone` -- a no-camera drone parked *over* the pad still returns
no fix and recovers only via the camera drone). Additive only; `camera_drones=None` keeps the all-cameras
default unchanged.

---

## Phase 2 / P2.1-P2.3 -- Cooperative vision on moving decks: gimbal + wide FOV + motion prediction

**Date:** 2026-05-24

**Tackled the fast-deck cooperative-vision limit found earlier (was 0% landing).** Three additive
upgrades to the swarm vision:
- **P2.1 gimbal-stabilized per-drone cameras:** each drone's camera now rides a nadir-held mocap gimbal
  (`world.py _gimbal_bodies` + `drive_gimbals`), like the single-drone. **Tilt-invariant:** the deck fix
  error is 0.008 m whether the drone is level or rolled 20deg (a body-fixed camera would shift the pad
  ~0.7 m or lose it).
- **P2.2 wide FOV + motion prediction:** the holding stack parks drones at radius ~= altitude, so a 90deg
  downward FOV could not see the pad (only 0.2/3 drones saw it). Widening to **120deg** (camera +
  back-projection intrinsics matched) plus the consensus filter's constant-velocity predict step (which
  extrapolates the deck between vision frames and feeds the estimated deck velocity to the guidance) cut
  the **deck-position estimate error from 1.99 m to 0.22 m (9x)** and raised coverage to ~0.6/3.
- **P2.3 vision landing on a translating deck:** **0% -> ~25-33% all-landed** (mean ~1-1.7/3) on the fast
  ship deck (0.4 m/s).

**Honest status:** the **perception** now genuinely tracks a moving deck via shared vision + consensus
prediction (0.22 m), and the cooperative-recovery mechanism is proven (controlled exp: blind drones
4.24->0.34 m). But **full-swarm vision *landing* on a moving deck is only partially reliable** (~1/3 land,
high variance) -- the modeled-sensing swarm with a continuous estimate lands 100%; vision-grade,
intermittent, partial-coverage estimates + a translating deck cap the closed-loop landing. Reliable
full-swarm moving-deck vision landing (sequential-touchdown coordination tuned for vision-grade estimates)
is the hard open part. **No regression: 70/70 tests; gimbal/FOV changes are additive (the controlled
cooperative-perception tests still pass -- blind drones stay blind at 120deg).**

---

## Phase 1 / P1.2+P1.3 -- Data-driven deck motion + seakeeping validation

**Date:** 2026-05-24

**The deck can now be driven by recorded 6-DOF seakeeping data, and the wave model is validated against
it.** New `sim/platforms/data_driven.py` (`DataDrivenDeckMotion`): replays a recorded
`t,x,y,z,roll,pitch,yaw` trajectory (linear-interp pose + finite-diff velocity, loops, `relative_xy`
starts at world origin) -- a drop-in `PlatformMotion`, so a **real sea-trial / MRU log drops straight in**.
`scripts/gen_seakeeping_data.py` generates high-fidelity reference CSVs per sea state
(`assets/seakeeping/{calm,moderate,rough}.csv`, from the JONSWAP/RAO spectral model) and **characterizes**
each trace exactly as you would real data -- significant heave `Hs=4*std(heave)`, peak period `Tp` from
the heave PSD, roll/pitch RMS.

**P1.3 validation:** the recovered peak period matches the design Tp within tolerance (calm 5.4 vs 5.5 s,
moderate 9.1 vs 8.8 s, rough 11.1 vs 12.4 s -- all OK) and roll/pitch RMS match the calibrated targets
(0.71/0.50, 4.53/3.02, 9.07/5.22 deg). **Landing on the data-driven deck: 13/14 = 93%** (ship moderate,
`--motion-data assets/seakeeping/moderate.csv`) -- the autopilot lands on replayed motion. **CLI:**
`drone run ship --motion-data <csv>` (and `offshore`). **Honesty:** the bundled CSVs are
spectrum-synthesized realistic references (clearly labelled, not measured); a real trial/NDBC-derived CSV
uses the identical pipeline. Exposed `DataDrivenDeckMotion` from `drone_landing.sim.platforms`.
**Tests: +2 (DataDrivenMotionTests); 70/70 total. Phase 1 (maritime realism) complete.**

---

## Phase 1 / P1.1+P1.4 -- Offshore-vessel (OSV) scenario: orange ship + sea (drone + swarm)

**Date:** 2026-05-24

**A cinematic orange offshore-support-vessel (OSV) landing scenario**, for both single-drone and swarm,
built **additively so the validated `ship` (90%) is untouched.** New world
`assets/mujoco/worlds/x2_landing_offshore.xml`: an orange OSV (hull + raised bow + white wheelhouse +
mast, on the sea) whose **helideck + ArUco pad + 6-DOF seakeeping joints + servos + mass/inertia are
IDENTICAL to the validated ship** — a compact invisible 60 kg "ballast" carries the body inertia while
the long orange hull/superstructure are **light visual geoms**, so the deck dynamics (and landing
behaviour) match. **Verified: offshore lands 17/20 = 85% on the same seeds as the ship baseline (17/20)
-- physics-equivalent, no regression.** Why a new world (not enlarging `ship`): a bigger collision deck
tripled the pitch inertia and shifted success within seed-noise; rather than risk the validated baseline,
the OSV look is delivered as a separate opt-in scenario.

**CLI:** single-drone scenario `offshore` + presets `offshore`, `offshore-rough`, `offshore-green`
(`drone run offshore`, `drone watch offshore`); world.py recognizes `offshore` as a 6-DOF seakeeping deck.
Swarm: `swarm run/watch --offshore` (orange OSV look in the MuJoCo swarm world; visual-only, physics
unchanged -- 3/3 land). Preview render saved to `runs/preview/offshore.png` (orange OSV confirmed,
mean RGB ~[212,109,17]). No beams/turbines (per request). **68/68 tests still pass.** (P1.4 superstructure
is realized as the OSV wheelhouse; the validated `ship` deliberately keeps its 1.8 m deck.)

---

## Phase A / CP -- Cooperative perception: real per-drone onboard vision + shared deck fixes

**Date:** 2026-05-24

**The swarm now perceives the deck with real per-drone vision (not just a modeled estimate).** Addresses
the honest gap surfaced earlier (the swarm used a calibrated noise surrogate, not images). New:
`world.py` stamps a downward `cam_i` on every drone + a bright landing **pad** on the deck;
`vision.py` (`SwarmVision`) renders each drone's camera, detects the saturated pad (border-gated
centroid), and back-projects it through the pinhole + rangefinder altitude to a **world-frame deck fix**
(validated ~1-2 cm). A drone that can't see the pad (out of FOV / too tilted) returns **None = blind**.
Wired into `MujocoSwarmCoordinator` (`SwarmConfig.vision` / `swarm run --vision`, MuJoCo only): visual
fixes feed the **A2 consensus filter**, so blind drones land on neighbours' shared fixes.

**Cooperative-perception proof** (`scripts/eval_cooperative_perception.py`, 5 drones, 3 blind, wrong 3 m
prior): blind drones recover the deck from neighbours' *real visual* fixes — **4.24 m -> 0.34 m**; seeing
drones stay ~sharp (0.01 -> 0.26 m, the consensus blending cost). **Honest closed-loop finding:**
vision-in-the-loop *landing* works on a low-motion/station-keeping deck but degrades on a **fast-translating
deck** (ship forward_speed 0.4 m/s drifts ~12 m/landing) — intermittent downward vision can't track a fast
target, drones lag and lose the pad; the modeled-sensing swarm (continuous estimate) handles fast decks.
A gimbal / higher vision rate / motion prediction is the open next step. **CLI:** `swarm run --vision`
(guarded to MuJoCo). **Tests: 2 CooperativePerceptionTests** (detection accuracy + blind-drone recovery).

---

## Phase B / B3 -- Learned perception: CNN deck-pose detector (the proper markerless)

**Date:** 2026-05-24

**A trained CNN localizes the deck without ArUco.** New `src/drone_landing/perception/cnn_detector.py` —
`DeckCNN` (RGB 96px -> conv stack -> **spatial-softmax** soft-argmax -> deck-centre pixel + visibility) +
`CNNDeckDetector` that back-projects the predicted pixel through the same pinhole+range geometry as the
classical markerless tracker, so it is a **drop-in** (`.detect(image,range)->.found/.rel_xy`). Trained on
6000 sim-rendered frames at randomized viewpoints; **fused into the autopilot** as `--cnn-markerless`.

**Result:** val pixel error **0.02**; on fresh frames median **5.9 px (~8.5 cm at altitude)**, 96%
detection confidence. **Closed-loop (ship moderate, 12 eps): neutral** — 92% with and without it (matches
the classical markerless; ArUco rarely fails there so the fallback seldom engages — the key property is it
**doesn't hurt**). The accurate standalone learned detector is the deliverable; a heavy marker-loss A/B to
show a closed-loop win is future work. **3 tests** (test_cnn_detector.py); model `runs/cnn/deck_cnn.pt`.

**Two debugging lessons (logged to memory)** — both general traps: (1) **position regression needs spatial
information** — the first net used `AdaptiveAvgPool2d(1)` (global average pool), which averages away *where*
the deck is -> unlearnable; a **spatial-softmax** (differentiable centroid) fixed it. (2) **labels must be
consistent with the image** — forward-projecting the deck body origin to a label pixel was ~30 px off the
actual rendered pad (camera-mount + deck-origin offsets), so the net (and even a threshold-centroid) failed
on held-out frames; **self-labelling from the rendered pad** (threshold-235 centroid, border-gated) fixed
it. The model overfit a 128-frame batch throughout, which is how the data/label bug was isolated from a
code bug. **CLI:** `--cnn-markerless` on `drone run`/`watch`. Phase B recommended items B1-B4 now done.

---

## Phase A / A4 -- GNN-based MARL policy: permutation-invariant, generalizes across swarm size

**Date:** 2026-05-24

**A single size-agnostic GNN policy runs at any swarm size — the SOTA representation.** New
`src/drone_landing_swarm/marl_gnn.py`: a message-passing **graph neural network** over the ego's
neighbour graph (per-neighbour shared encoder + masked **mean+max aggregation** -> permutation-invariant,
size-agnostic), trained by parameter-sharing PPO on a **randomized-N** ego env (DummyVecEnv — Windows
'spawn' SubprocVecEnv dies on this fast kinematic env). `swarm gnn` CLI. Trained 500k steps on CPU.

**Cross-N generalization eval (the success metric)** — deploy the *one* policy (trained on N∈[6,10]) at
several sizes vs the classical coordinator:

| swarm size | classical sep-kept | GNN sep-kept | min-sep (classical / GNN) |
| --- | --- | --- | --- |
| N=10 | 100% | 100% | 0.738 / 0.745 |
| N=14 | 46.7% | 46.7% | 0.542 / 0.536 |
| N=18 | 0% | 0% | 0.342 / 0.338 |

The policy runs **unchanged at N=14 and 18** (no retraining) — permutation-invariant + size-agnostic, the
A4 deliverable. It **ties** classical at every size: honestly, it does *not beat* classical because the
dense/short-comms regime is **comms-physics-limited** (you can't avoid what you can't see in time) and the
learned residual sits under the non-bypassable CBF (A3). The GNN's contribution is the SOTA *representation*
+ cross-N generalization, not the separation metric — consistent with the project's recurring theme (RL/MARL
wins only where the bottleneck is algorithmic, not physical). Model: `runs/marl_gnn/gnn_final.zip`; eval
`swarm gnn --eval <zip> --sizes 10 14 18`. **Phase A swarm series (A1-A5) now complete.**

---

## Phase 6 (d) -- Recurrent-PPO (LSTM) policy: OOM fixed, trains to 1M, beats the MLP on the POMDP

**Date:** 2026-05-24

**The LSTM policy edges out the feedforward MLP on the hard regime — the partial-observability hypothesis
holds.** The recurrent-PPO (`MlpLstmPolicy`) ground run had OOM'd at ~254k steps on the 8 GB laptop GPU
(RecurrentPPO backprop-through-time + variable-length-sequence CUDA-allocator fragmentation). **Fixed** by
a memory-frugal recurrent config in `rl/train.py` (`n_steps` 1024->256, `batch_size` 1024->256,
`net_arch` [256,256]->[128,128], lstm_hidden 128) -> ~8x less BPTT memory -> fits 8 GB, and by running it
**solo** on the GPU (a concurrent GPU+CPU training had crashed the laptop). (`PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` is **not supported on Windows CUDA** — the frugal config is the actual fix.)

Trained to ~1M steps (reward 84 -> **124**), no OOM. **Eval (hard regime, 50 episodes, surrogate env):**

| policy | success | failures |
| --- | --- | --- |
| classical baseline (supervisor + geometric) | 86% | 5 crash, 2 timeout |
| feedforward MLP residual (earlier) | 94% | — |
| **recurrent-PPO (LSTM) residual** | **96%** | 2 crash |

The LSTM **beats the MLP (96% vs 94%)** with fewer failures — its memory helps in the noisy/dropping-vision
POMDP, as hypothesized (the 2024-25 SOTA choice for vision-based ship landing). Model:
`runs/rl/ground_lstm/recurrent_ppo_final.zip`; eval via `drone train --eval <zip> --algo recurrent_ppo`.
**Lessons logged** (memory): the frugal config (not the unsupported env var) is the Windows OOM fix; run one
GPU training at a time; never pipe a long background train through `| tail` (buffers until EOF). The A4 GNN
swarm training was then launched on CPU (GPU now free), one training at a time.

---

## Phase B / B4 -- Robust control: tube / disturbance-observer MPC

**Date:** 2026-05-24

**DOB-MPC cancels the wind-induced tracking offset that plain MPC leaves.** New
`src/drone_landing/control/mpc/tube_mpc.py` (`TubeMPC`) robustifies the horizontal MPC two ways: (1) it
folds the disturbance-observer estimate `d_hat` into the prediction model (`a_tot = a + d` in the relative
dynamics) so the planner pre-compensates a steady wind -- the predictive analogue of the geometric
controller's `dist_ff`; (2) it **tightens the acceleration constraint** to `a_max - tube_factor*d_bound`,
reserving authority for the ancillary feedback that keeps the true trajectory inside a bounded tube
(`tube_radius`), giving guaranteed tracking under a bounded disturbance. **Result (station-keeping under
steady wind, `scripts/eval_tube_mpc.py`):**

| wind d (m/s^2) | plain MPC offset | DOB-MPC offset | improvement |
| --- | --- | --- | --- |
| 0.5 | 0.048 m | 0.0006 m | 80x |
| 1.0 | 0.095 m | 0.0012 m | 82x |
| 1.5 | 0.143 m | 0.0017 m | 82x |
| 2.0 | 0.190 m | 0.0023 m | 82x |

Plain MPC (no integral) settles with a standing offset that grows with wind; DOB-MPC drives it to ~0 --
**~80x tighter** -- within the actuator authority. Beyond ~2 m/s^2 (a_max=3.0) no controller can both
reject and track: the disturbance leaves the robustly-controllable set (the honest authority limit).
**Honest scope:** the full ArUco->EKF *vision* closed loop is estimation-limited (the project's standing
finding -- geometric stays the in-loop default), so DOB-MPC's win is tracking fidelity under wind shown
with clean state; closed-loop autopilot integration is deferred (would double-compensate vs the geometric
`dist_ff` and is estimation-limited anyway). Exposed via `drone_landing.control.mpc` (`TubeMPC`,
`TubeMPCConfig`). **Tests: 2 in tests/test_control.py** (DOB-MPC beats plain MPC under wind; zero-wind
parity). Built in parallel (CPU-only) while the recurrent-PPO LSTM trains on the GPU, one training at a time.

---

## Phase B / B2 -- Formal safety: Hamilton-Jacobi reachability safe-landing set + runtime shield

**Date:** 2026-05-24

**A verified safe-landing envelope + a shield that keeps any controller inside it.** New
`src/drone_landing/control/reachability.py` (`LandingReachability`) computes, for the vertical landing
channel (`h_dot=w, w_dot=a+d`, thrust-limited `a`, bounded disturbance `|d|<=d_max`), the **robust
backward-reachable safe set** by discrete-time HJ dynamic programming -- a differential-game
`exists a, forall d` update iterated to a fixed point (the maximal control-invariant funnel to a soft
touchdown `|w|<=w_land`). Computed on a 161x161 grid in ~1.5 s. **Validated** against the analytic
worst-case braking curve `h=(w^2-w_land^2)/(2(a_max-d_max))`: the grid set's lower boundary matches it
across all descent speeds within one discrete step. The internal target speed is tightened by one
disturbance step (`d_max*dt`) so the realized touchdown respects the public `w_land` guarantee.

A **runtime-assurance shield** (`safe_action`) passes a controller's command through only if it keeps the
state safe against every admissible disturbance, else overrides with the safest (max-braking) action --
a provable safety envelope around any policy (geometric/MPC/RL), complementing the swarm CBF (A3). **A/B
(`scripts/eval_reachability.py`, reckless dive controller under random wind, 200 trials):** unshielded
**100% hard landings (mean touchdown 5.25 m/s)** -> shielded **8% (0.40 m/s)** -- a 13x softer touchdown;
the residual few % are marginal grid/discretization boundary cases (honest for a grid reachability).
**Tests: tests/test_reachability.py 4/4** (membership, respects analytic braking boundary, shield
overrides unsafe + passes safe, closed-loop soft touchdown under adversarial disturbance). Exposed via
`drone_landing.control` (`LandingReachability`, `ReachabilityConfig`). Pure offline math/sim -- no GPU,
no hardware. (Built in parallel while the recurrent-PPO LSTM trains on the GPU, one training at a time.)

---

## Phase A / A5 -- K moving platforms + dynamic re-tasking

**Date:** 2026-05-24

**The swarm now recovers M drones onto K moving decks, re-tasking online as conditions change.** New
`src/drone_landing_swarm/multi_deck.py` (`MultiDeckCoordinator` / `MultiDeckConfig`): K decks (each sailing
a distinct heading so the fleet spreads out), per-deck `LandingScheduler` + `HoldingStack`, reusing the
no-cheats `SwarmSensing` and the non-bypassable global `SafetyFilter`. **Assignment** solved online:
**balanced Hungarian** initial allocation (each deck replicated to `ceil(M/K)` slots -> globally
min-distance balanced start) + a **decentralized auction** every `reassign_dt` (each not-yet-committed
drone picks `argmin distance + congestion*load`, switching only past a hysteresis margin). Committed drones
lock to their deck unless it **fouls**.

**Result (ship moderate):** nominal recovery **100% all-landed, load balanced** (9 drones -> 3/3/3, 12 ->
4/4/4), separation kept, **0 re-tasks** -- the Hungarian start is already optimal (honest/correct). The
re-tasking earns its keep in the scenario it exists for, a **fouled deck** closing mid-recovery
(`--foul-deck 2 --foul-time 4`):

| 9 drones, deck 2 fouls at t=4 s | all-landed | recovered | re-tasks |
| --- | --- | --- | --- |
| **re-tasking ON** (auction) | **100%** | 9 / 9 | ~3 |
| re-tasking OFF (static) | 0% | 6.2 / 9 | 0 |

So a closing deck strands ~3 drones unless dynamic re-tasking moves them onto the remaining decks -- a
clean, honest demonstration of the feature's value. **CLI:** new `swarm multi` (`--drones/--decks/--slots/
--foul-deck/--foul-time/--comms/--margin`), updated `swarm info`. **Tests: 22/22 swarm** (+2 MultiDeckTests:
balanced K-deck recovery, fouled-deck re-tasking ON vs OFF). 53/53 total. **Phase A swarm series
(A1,A2,A3,A5) complete** + B1 maritime. Next candidates: A4 (GNN-MARL), B3 (CNN perception), B2/B4.
Benchmark matrix (B6) HELD until you say.

---

## Phase B / B1 -- Maritime fidelity: real wave spectra (JONSWAP/PM) + RAOs + ship air-wake

**Date:** 2026-05-24

**The ship deck is now driven by a real wave spectrum, and the air-wake stresses the approach.** Two new
modules, both wired in **opt-in** so the validated 90% ship landing is untouched by default:

* `src/drone_landing/sim/platforms/wave_spectrum.py` -- a **JONSWAP / Pierson-Moskowitz** wave spectrum
  (Hs, Tp, gamma) passed through **response-amplitude operators** (heave/sway low-pass, pitch/roll
  second-order resonances) to generate the per-DOF sinusoid components. The validated `ShipDeckMotion`
  synthesis engine (`sum a_i sin(w_i t + phi_i)`) is reused unchanged -- this is exactly random-phase
  spectral synthesis -- so the spectral model is a richer *parameterization*, not a rewrite. Each DOF's
  magnitude is **normalized to the validated sum-of-sinusoids RMS** (so the ship-landing result is
  preserved) while the frequency content becomes physically correct. `sea_state(name, spectral=True)` /
  `--sea-model spectral` / preset `ship-spectral`. **Verified:** spectrum recovers Hs exactly
  (calm/moderate/rough 0.60/1.88/5.01 m vs targets 0.60/1.88/5.00); spectral RMS matches sinusoid within
  a few percent (moderate z 0.092->0.096, roll 4.49->4.69deg, pitch 3.00->2.96deg); full-pipeline
  success 92% sinusoid vs 83% spectral (richer transients, within noise on 12 eps).
* `src/drone_landing/sim/airwake.py` -- a **ship air-wake ("burble")**: a position-dependent turbulence
  field over the deck (spatial envelope strongest just above the landing spot, mean downdraft + heading
  deficit + OU turbulence modulated by the envelope). `--airwake` / preset `ship-airwake`; wired into
  `world._apply_wind` (ship only). **Verified (14 eps, full vision pipeline):** no-airwake **86%** ->
  +air-wake **50%** (measurably stresses the approach, the dominant real shipboard disturbance) ->
  +air-wake **+ DOB 86%** -- the **disturbance observer fully recovers it** (+36 pts). This is the DOB's
  strongest, most physically-motivated win (vs the original +9 pts in generic wind).

**CLI:** `--sea-model {sinusoid,spectral}`, `--airwake` on run/watch; presets `ship-spectral`,
`ship-airwake`; updated `drone list`. `scripts/eval_seakeeping.py` reproduces the spectrum/motion table
and the air-wake A/B. **Tests: tests/test_seakeeping.py 6/6** (Hs recovery, PM=JONSWAP(gamma=1), spectral
peak ~ Tp, spectral RMS ~ sinusoid, air-wake envelope decay, force localized to the deck). **51/51 total.**
The A1->A3->A2 swarm core + B1 maritime fidelity are done; next candidates A5 (K platforms), A4 (GNN-MARL),
B3 (CNN perception). Benchmark matrix (B6) stays HELD until you say.

---

## Phase A / A2 -- Cooperative consensus deck estimation (distributed Kalman-Consensus Filter)

**Date:** 2026-05-24

**The drones now cooperate on tracking the deck.** New `src/drone_landing_swarm/consensus.py`
(`ConsensusDeckEstimator` / `ConsensusConfig`) is a **Kalman-Consensus Filter** (Olfati-Saber, CDC 2007):
per drone, per step -- **predict** a constant-velocity deck model -> **measurement update** (only if the
drone currently sees the deck, with its own range-dependent `R`) -> **consensus update**
`x <- x + gamma * mean_{j in N_i}(x_j - x_i)` pulling toward neighbours' one-step-stale broadcast
estimates. Fully decentralized (neighbour exchange only). To make cooperation meaningful, the A1 sensing
model now makes **deck-measurement noise grow with range** and adds a **visibility cutoff**
(`deck_vis_range=6 m`): a far holding drone barely resolves the marker; past 6 m it gets **no** direct fix
and must rely on consensus. Opt-in via `--consensus` / `SwarmConfig(consensus=True)` (off = each drone's
flat A1 estimate, so all prior numbers are unchanged); wired into both runners; fused estimate feeds
scheduling/holding/guidance.

**Controlled 5-drone experiment** (`scripts/eval_consensus.py`):

| drone | range | sees deck | raw meas err | fused (consensus) err |
| --- | --- | --- | --- | --- |
| 0 | 1.1 m | yes | 0.180 m | 0.117 m |
| 1 | 2.5 m | yes | 0.293 m | 0.120 m |
| 2 | 4.1 m | yes | 0.410 m | 0.124 m |
| 3 | 7.8 m | **blind** | -- | **0.137 m** |
| 4 | 8.3 m | **blind** | -- | **0.137 m** |

Network mean **raw 0.295 -> fused 0.127 m (57% lower)** -- the fused error beats even the best single
drone's raw fix (0.180 m), and the **blind** drones get a usable estimate purely from neighbours (hits the
roadmap A2 metric). The sharper shared deck reference propagates to the traffic: full-coordinator A/B (6
drones, ship) lifts min-separation **0.86 -> 0.94 m** (and **0.89 -> 0.99 m** at a tight 4 m comms range,
where more drones are blind), all-landed staying 100%. **CLI:** `--consensus` on run/watch/verify, updated
`swarm info`. **Tests: 20/20 swarm** (+3 `ConsensusTests`: blind-drone recovery, fused < raw, coordinator
lands with consensus on). 45/45 total. Next per approved sequence: **B1 maritime fidelity** (the A1->A3->A2
core is complete: honest + safe + cooperative).

---

## Phase A / A3 -- Non-bypassable CBF safety layer + formal forward-invariance certificate

**Date:** 2026-05-24

**The CBF is now a structural chokepoint, not an incidental filter.** New
`src/drone_landing_swarm/safety.py` (`SafetyFilter` / `SafetySpec` / `verify_separation`) wraps the exact
CBF-QP so the coordinator has exactly **one** path from a desired velocity to the plant --
`SafetyFilter.filter(...)`. The MARL residual is added to the *desired* velocity **before** that call, so
**no learned command can reach the plant unfiltered**. Both the kinematic `SwarmCoordinator` and the
`MujocoSwarmCoordinator` route through it; the old per-runner `_cbf_multi` helper is removed.

**Formal guarantee.** With barrier `h = ||p_i - p_j||^2 - d_min^2`, single-integrator model, forward-Euler
step `dt`, the CBF condition gives `h+ >= (1 - alpha*dt) h`, so **`0 < alpha*dt < 1` and `h >= 0` =>
`h+ >= 0`**: the safe set is forward-invariant in discrete time (provable, not just continuous-time).
`SafetySpec` **asserts `alpha*dt < 1`** at construction (alpha=3, dt=0.05 -> 0.15). The filter records a
per-step **certificate** = worst predicted `h+` over active constraints (`>= 0` => provably collision-free
that step), plus activations / max-correction diagnostics, surfaced in every `run()` result.

**Verification harness (`swarm verify`, 6 drones, ship moderate, 25 seeds):**

| sensing | worst true min-sep | worst certificate | violations |
| --- | --- | --- | --- |
| perfect (exact inputs) | 0.734 m | **+0.049** (>= 0, matches the proof) | 0 / 25 |
| realistic (no-cheats) | 0.725 m | -0.175 (bounded by sensing noise) | 0 / 25 |
| realistic + margin 0.25 | 0.923 m | -- | 0 / 25 |

So with **exact** inputs the certificate is provably non-negative (the theory); under realistic no-cheats
sensing the swarm still has **zero true separation violations** across the sweep, residual policy active
underneath -- honest scope: the guarantee holds up to a bounded margin set by sensing/tracking error,
which `SafetySpec.margin` absorbs (and which `swarm verify` measures). **CLI:** new `swarm verify`
(`--seeds`, `--margin`), a `--margin` flag on `run`/`watch`, certificate line on `swarm run`, updated
`swarm info`. **Tests: 17/17 swarm** (+4 `SafetyLayerTests`: invariance precondition enforced,
pass-through when clear, certificate >= 0 with exact inputs over a head-on rollout, `verify_separation`
passes feasible). 42/42 total. Next per approved sequence: **A2 consensus deck estimation**.

---

## Phase A / A1 -- Swarm "cheat" fixed: decentralized onboard sensing (no-cheats audit-clean)

**Date:** 2026-05-24

**The one swarm cheat is gone.** The coordination loop previously read `world.drone_pos/vel` (ground
truth) + the exact deck pose. New `src/drone_landing_swarm/sensing.py` (`SwarmSensing` / `SensingConfig`)
converts the simulator's true states into the **onboard view** each drone actually has, and the loop now
runs **entirely on estimates**:
- **own state** -- per-drone noisy estimate (own EKF surrogate: `pos_noise=5cm`, `vel_noise=0.10 m/s`);
- **deck pose** -- per-drone noisy estimate (`deck_noise=6cm`), to be fused by the A2 consensus layer;
- **neighbours** -- only those within **comms range**, as **broadcast estimates** with **latency** (1
  step stale) + **dropout** (5%/step) + relative-sensing noise (`rel_noise=8cm`).

Scheduler readiness, holding-slot assignment, CBF neighbour/deck-keepout constraints, and the per-drone
landing guidance all consume these estimates. True `world` state is used **only** for physics, true
contact (touchdown), and the separation *metric* -- never a decision. Wired identically into **both**
the kinematic `SwarmCoordinator` and the `MujocoSwarmCoordinator` (`step()` builds a `sensing.sense(...)`
view; `_landing_velocity` refactored to take estimates). `SensingConfig.perfect()` reproduces the old
truth baseline for honest A/B.

**Result (4 drones, ship moderate, seeds 0-2):**

| sensing | engine | all-landed | min-sep | sep-ok |
| --- | --- | --- | --- | --- |
| perfect (=old truth) | MuJoCo physics | 100% | 0.921 m | 100% |
| **realistic (no-cheats)** | MuJoCo physics | **100%** | 0.932 m | 100% |
| realistic + comms 2.5 m | MuJoCo physics | 100% | 0.941 m | 100% |

At this drone count / spacing, the estimator noise sits well inside the CBF + holding margins, so the
honest path matches the baseline (kinematic A/B identical: 100% / sep kept across perfect, realistic,
comms-1.5). The genuinely hard regime (where partial observability finally bites) stays the dense,
short-comms MARL setting -- that's where A2 (consensus deck estimation) is meant to earn the numbers
back. **Tests: 13/13 swarm pass** (+4 new `SensingTests`: perfect==truth, realistic noisy-but-bounded,
comms-range limits neighbours, coordinator lands audit-clean). Docs: `docs/SWARM.md` "Onboard sensing"
section; roadmap `docs/ADVANCED_ROADMAP.md` A1. Next per approved sequence: **A3 formal CBF safety**.

---

## Phase 6 (c) — RL beats baseline (94%) + Phase 3 (b) — MuJoCo physics swarm

**Date:** 2026-05-24

**RL WIN — tuned residual policy beats the classical baseline.** After the first residual run lost to
the baseline (56% vs 86%), the tuned retrain (stronger control-cost `w_ctrl` 0.01→0.05 + smaller
residual authority `residual_a` 1.5→0.8, so the policy stays near the strong baseline and only deviates
when it helps) gives, on the **hard regime** (50 ep, domain-randomized fast rover + wind):

| controller | success | crashes |
| --- | --- | --- |
| classical baseline (supervisor+geometric) | 86% | 5 |
| **residual RL (tuned)** | **94%** | **2** |

RL adds **+8 points and halves crashes** — matching the literature's ~94%. Run it on the full vision
pipeline with **`drone run --controller rl`** (the trained policy as a first-class controller).

**Full-pipeline validation (the honest test):** the trained residual policy run on the **real
ArUco→EKF pipeline** (`drone run --controller rl`, `rl/policy.py ResidualPolicy` rebuilds the train-time
obs from the live EKF estimate) lands **100% (8/8) on the default ground scenario** — it transfers from
the estimator-noise surrogate to the full vision stack. On the **hardest fixed motion (fast rover, vmax
1.3, full pipeline)**: **RL 33% vs geometric 17% (12 ep)** — RL ~doubles the classical controller where
the vision-estimation fly-off bottleneck bites hardest. So RL is the better controller on the surrogate
(94 vs 86), the default full pipeline (100), *and* the hard full pipeline (33 vs 17).

**Ship RL (trained 2M on GPU) — RL does NOT beat the strong ship baseline:** eval (hard ship regime)
**RL 44% vs baseline 96%** (28 fly-offs). Clean, consistent finding: **RL adds value only where the
classical baseline is weak.** The ground baseline (86%) had headroom → RL helped (94%); the *ship*
baseline (supervisor + geometric + green-deck) is already **96%** (near-ceiling), so the residual's noise
just causes fly-offs. **Decision: geometric stays the ship default; RL (`--controller rl`) is for ground.**
This is the recurring project theme — the classical stack is excellent; RL is the tool for the hardest,
weak-baseline regimes (fast/random motion).

**VecNormalize + LR-annealing — tried, measured, OFF by default (honest negative).** Added VecNormalize
(obs+reward normalization, the usual top PPO tweak) + linear LR annealing + the deploy-time obs-stat
re-application. **Result: 36% vs the 94% non-normalized model** — VecNormalize's *reward* normalization
diluted our success-bonus-dominated shaped reward and re-introduced centring-disruption timeouts. So both
are now **optional flags off by default** (`drone train --normalize --anneal-lr`); the default reproduces
the 94% policy. Lesson logged: a general "best practice" can hurt a carefully shaped reward — always A/B.

**MuJoCo physics swarm (was kinematic + matplotlib → now real physics).** New `swarm/world.py`
(`SwarmMujocoWorld`: N validated X2 bodies stamped into one world + a servo-driven 6-DOF moving deck,
true contact) and `swarm/mujoco_runner.py` (`MujocoSwarmCoordinator`: the same scheduler + **CBF-QP**
avoidance + holding stack, but each drone is a real MuJoCo quad flown by the **geometric controller**
velocity→accel→thrust, landing on **distinct deck spots** so they don't stack/collide). **`swarm run
--engine mujoco`** (now the default) + **`swarm watch --engine mujoco`** (real 3-D viewer); `--engine
kinematic` keeps the fast point-mass version. **Result: 3–4 drones, 100% all-landed on a moving ship
deck, separation maintained, true physics** (~15–22 s). 9/9 swarm tests pass (incl. a MuJoCo physics
test). The single-drone sim remains untouched (separate module).

---

## Phase 5 (Tier 2) — Markerless deck fallback (accurate at altitude; gated to stay safe)

**Date:** 2026-05-24

**Built `perception/markerless.py MarkerlessDeckTracker`:** when the ArUco code can't be *decoded* but the
bright fiducial pad is still visible, segment the saturated-white pad (threshold 235 — isolates the pad
from the gray deck), take its centroid, and back-project through the nadir pinhole + rangefinder altitude
to a deck-centre relative-XY fix. Fused into the EKF (`update_markerless`, coarser noise r=0.15 m, gated).
`drone run --markerless`. Synthetic + MuJoCo validation: **~1 cm fix when the full pad is in view**.

**Honest A/B (fast rover, vmax 1.3, full pipeline):** naively fusing the centroid **hurt (42% → 25%,
more off-platform)** — at close range / under fast maneuvers the pad **clips the image edge**, so its
centroid is biased (no longer the deck centre) and the fix pulls the drone off-centre. **Fix: a
border-touch gate** — only fuse when the pad is *fully in frame*. With the gate, markerless ON returns to
**42% (= baseline, neutral)**: it now only contributes accurate fixes (decode-failure with the pad fully
visible) and never a biased one. **Net: a safe, accurate-at-altitude fallback; marginal in the current
scenarios** (the stabilized gimbal + nested centre marker already keep ArUco working in most cases). A
proper **feature/corner-based markerless pose** (solvePnP on detected pad corners, no decode) is the
robust close-range upgrade — future work. Kept opt-in, off by default. 34 tests pass.

---

## Phase 5 (Tier 2) — Wind-aware control: disturbance observer (works, +9 pts in strong wind)

**Date:** 2026-05-24

**Built `control/disturbance.py DisturbanceObserver`:** estimates the lumped external acceleration (wind
gusts + unmodeled aero) from the **IMU-vs-commanded-thrust acceleration residual** —
`a_expected = (thrust/m)·b_z + g`, `d_hat ← lowpass(a_measured − a_expected)` — and feeds `−d_hat`
forward in the controller (`GeometricController.compute(dist_ff=...)` subtracts it from the horizontal
accel command). It reacts to gusts faster than the controller's slow integral term, needs no new sensor,
and is exposed via **`drone run --dob`** (`AutopilotConfig.use_dob`). Unit test: `d_hat` converges to a
known constant disturbance.

**A/B under strong wind (mean ~1.2 N ≈ 9% of weight + 0.8 N gusts, 12 ep, full pipeline):** **DOB off 33%
→ DOB on 42%** (+9 pts), slightly tighter centring (0.170 vs 0.178 m). A genuine win — kept as an opt-in
flag (off by default so the established baseline numbers are unchanged; the gentle default wind doesn't
need it, but it helps materially in stiff wind). 32 tests pass.

---

## Phase 5 (Tier 2) — Fault tolerance (rotor-out): infrastructure built, honest limitation

**Date:** 2026-05-24

**Built:** a **fault-tolerant control allocation** (`control/allocation.py`) — with a rotor out, yaw is no
longer independently controllable, so it **sacrifices yaw** and solves the 3×3 system for the three
working rotors to hold collective thrust + roll + pitch (clipped to non-negative). Rotor-failure
**injection** in the world (`failed_rotor`/`fail_time` — the motor is physically zeroed at `fail_time`),
the autopilot engages FT allocation on detection, and **`drone run --fail-rotor K`** drives the demo.
2 allocator unit tests (`tests/test_control.py`): nominal allocation inverts the wrench exactly; FT
keeps the dead rotor at 0, thrusts non-negative, and tracks collective+roll+pitch when feasible.

**Honest result:** with the basic FT allocation + the existing geometric controller, the drone **does NOT
land (0%/6 ep, rotor 0 killed)** — it flies off. This is correct physics: a quad that loses a rotor is
severely **underactuated**, cannot hover level, and **spins rapidly about yaw**; the full-actuation
geometric controller can't stabilize it, and the allocation must clip infeasible commands. **Proper
rotor-out recovery needs a dedicated reduced-attitude controller** — *Mueller & D'Andrea (2014), "Stability
and control of a quadrocopter despite the complete loss of one, two, or three propellers"* — which gives
up yaw and controls the **direction of the primary thrust axis** while the body spins.

**Reduced-attitude controller implemented (`GeometricController.compute_rotor_out`):** desired thrust
vector → desired thrust-axis `n_des`; track only the thrust-axis direction (roll/pitch torque from the
reduced-attitude error `R^T (n × n_des)`), yaw torque = 0, 3-rotor allocation. After **4 tuning
iterations** the behaviour improved (fly-off → off-platform → controlled-spin descent with no crashes) but
**still does not reliably land (0%)**. Root cause identified: a **per-step** thrust-axis tracker cannot
hold the *average* thrust direction fixed under the fast yaw spin, so horizontal control drifts. The
correct fix is Mueller & D'Andrea's **LQR around the spinning equilibrium** (control the period-averaged
thrust direction, with the spin rate as a managed state) — a substantially larger formulation than
gain-tuning. **Status: allocation + injection + CLI + reduced-attitude controller + tests are in; reliable
rotor-out landing needs the full LQR design (a dedicated focused effort).** Kept honest — not claimed as
working. `--fail-rotor` remains a valid stress-test / research tool.

---

## Phase 6 (a) — RL foundation: Gymnasium env + PPO training pipeline

**Date:** 2026-05-24

**Goal:** Stand up the reinforcement-learning stack so a learned guidance policy can be trained (the
route to the hardest motion; literature ~94%). Per the agreed plan, RL training runs in the background
while the swarm is built in parallel.

**Built:**
- `src/drone_landing/rl/landing_env.py` — `LandingEnv` (Gymnasium): the policy outputs **high-level
  guidance** (horizontal accel + descent rate), executed by the **validated geometric attitude inner
  loop** on the **true MuJoCo dynamics**. So RL learns the outer tracking/descent law (the part the
  geometric/MPC/IBVS controllers occupy); the low-level control and contact physics are unchanged.
  - **Key decision — estimator-noise surrogate, not the camera.** Rendering ArUco every step is ~100×
    too slow for the millions of steps PPO needs. So the observation is the relative state corrupted by
    a **calibrated noise model** (≈3 cm position, noisy velocity, dropout) matching the measured
    ArUco→EKF error, plus the real IMU/AHRS stream. The learned policy is later evaluated on the **full
    vision pipeline** — the honest test (REALISM_CHARTER). Truth is exposed only via `info` (asymmetric
    critic / scoring). **Throughput: ~1771 env-steps/s on CPU** (no rendering) — RL is now tractable.
  - Obs (14): noisy rel_pos(3), rel_vel(3), tilt_xy(2), gyro(3), prev_action(3). Action (3): a_x, a_y,
    vz. Reward: shaped centre→descend→soft-land with terminal success/crash/out-of-bounds. **Per-episode
    domain randomization** (platform motion, sea state, wind) on the existing world (no MJCF recompile).
    Passes `gymnasium.utils.env_checker`.
- `src/drone_landing/rl/train.py` + **`drone train`** CLI — PPO (SB3), `SubprocVecEnv`, checkpoints +
  tensorboard to `runs/rl/<scenario>/`, auto GPU/CPU. `evaluate()` scores a checkpoint on the surrogate.
  **Smoke-trained 32 k steps end-to-end** (4 envs, ~1986 fps, model saved) — pipeline validated.
- **CUDA**: env had CPU-only torch (2.12.0+cpu); installing **torch 2.11.0+cu128** (CUDA 12.8, Blackwell
  / RTX 5060 sm_120) so training can use the GPU. (CPU training is also viable for this small MLP.)

**GPU + training kickoff:** installed **torch 2.11.0+cu128** into the project's Python 3.10 (`python -m
pip`, after a first attempt landed in a stray Python 3.14); **`torch.cuda.is_available()` → True, RTX
5060 Laptop GPU detected**. (Transient "_C DLL load failed" was just the mid-write install; resolved on
completion. Harmless torchvision==0.27 vs torch 2.11 pin warning — torchvision unused by RL.) GPU
smoke-trained, then **launched the real run on GPU in the background**: ground, PPO + curriculum, 2M
steps. For this small MLP the bottleneck is MuJoCo env-stepping (CPU workers), so GPU vs CPU wall-time is
similar here; GPU matters more for the recurrent (LSTM) policy.

**Method choices (researched the 2024–25 literature; see RESEARCH_NOTES.md):** PPO backbone (shown
superior/stable for landing) + **curriculum** (easy→hard difficulty schedule, the proven accelerator) +
**domain randomization**; **recurrent PPO (LSTM)** option added (`--algo recurrent_ppo`, `sb3-contrib`) as
the SOTA fix for the partial-observability POMDP (noisy/dropping vision); asymmetric privileged critic
(truth in `info`) and residual-on-MPC noted as next enhancements.

**Next:** monitor training; build the swarm (separate module) meanwhile.

**RL debugging — two real bugs found by evaluating the first run (honest):** the first 2M-step run reached
**0% success (all out_of_bounds)**. Evaluating + a hand-coded expert exposed: (1) a **reward-design bug** —
a *negative per-step shaping penalty accumulated* over the long episode to far exceed the −50 out-of-bounds
penalty, so the optimal policy was to **end the episode ASAP by flying out** (reward-accumulation
"suicide"); (2) a **sign/convention bug** — the observation mixed conventions (`rel_pos` was drone−platform
while `rel_vel` was platform−drone) and the potential's clearance had the wrong sign, so the altitude
shaping never fired. **Fixes:** switched to **potential-based shaping** (Ng et al. 1999 — `γ·Φ(s′)−Φ(s)`,
policy-invariant, *no* accumulating penalty, so no bail-out incentive) + a tiny alive bonus + dominant
terminal bonus; and made the whole env consistent with the EKF/controller convention (platform−drone).
**Validated**: a PD+velocity-feedforward expert now centres to ~0.05 m with **positive return** (good
behaviour scores positive, bad scores negative), confirming the env is landable and the reward gradient is
correct. Re-launched training with the fixed env. (This is exactly why honest evaluation + principled
reward design matter — the lesson is logged.)

**RL integration + first real result (honest).** Wired the trained policy into the full pipeline:
`control/policy.py ResidualPolicy` reproduces the train-time obs (platform-drone, 13-D) + residual on the
real EKF estimate; `AutopilotConfig.rl_policy_path` + **`drone run --controller rl`** (default checkpoint
`runs/rl/<scenario>/ppo_final.zip`, `--rl-policy` to override). Eval (50 ep, hard regime, surrogate):
**baseline 86%, residual-RL 56%** (28 success / 22 timeout / 0 crash). So the first residual policy is
*safer* (no crashes) but *worse* — its learned residual disrupts centring enough that the supervisor often
can't clear to descend (→ timeout). Beating an already-strong 86% baseline is the hard part. **Tuning
retrain in progress:** stronger control-cost (`w_ctrl` 0.01→0.05) + smaller residual authority
(`residual_a` 1.5→0.8) so the policy stays near the baseline (action≈0 == 86%) and only deviates when it
clearly helps. Honest stance: the **classical supervisor+geometric baseline (86% on hard) is the strong
performer**; RL is fully built + integrated and we're tuning it to *match or beat* it (not claiming a win
until it does).

**Direct RL still plateaued → switched to RESIDUAL RL (the research-backed, robust design).** Even with
the fixed reward, *direct* RL (policy outputs full guidance from scratch) only learned to approach +
descend but landed just off the deck (0% success, plateaued) — the precise centred touchdown is a hard
exploration problem. So the env now defaults to **`control_mode="residual"`**: the policy's action
*perturbs* a competent baseline — the **actual `LandingSupervisor` (commit/press/cut) + geometric PD**
guidance — so **action = 0 reproduces the proven controller**. Verified: the **zero-action baseline lands
100%** on the easy/default distribution, so RL starts competent and only learns refinements for the
**hard** regime (fast/random rover, rough seas, wind) where the baseline degrades. This matches the
literature's "residual RL on a model-based controller" (sample-efficient, safe, guaranteed landable).
`evaluate()` now reports **RL-residual vs the zero-action baseline on matched hard episodes** — RL's value
is the success gap on the hard cases. Residual-RL training (2M steps) running on the GPU.

**First residual run — another honest negative + fix (this is the 3rd RL bug caught by evaluating):** the
hard-regime eval gave **baseline 86%** (the supervisor+geometric guidance is genuinely strong even on
fast-rover + wind!) but **RL-residual only 4% (48/50 timeouts)**. Cause: I'd given RL residual authority
over **vertical velocity** *plus* an **alive bonus**, so the policy learned to **cancel the supervisor's
descent and hover** (timeout banks the alive bonus and dodges the crash penalty). **Fix:** the supervisor
now **fully owns the vertical** (descent/commit/cut); RL residuals **only the horizontal** accel (2-D
action) — where the baseline is actually weakest on fast rovers — and the **alive bonus is removed**. Now
action = 0 *is* the 86% baseline and the policy physically cannot prevent landing; it can only improve
centring. 3 RL-env unit tests added (env-checker, baseline lands on easy, flee-policy earns negative
return). Retraining (2M) on GPU. **Takeaway:** give a residual policy authority only where it can help and
not where it can sabotage the proven baseline — and keep evaluating honestly.

---

## Phase 3 (Tier 3) — MARL swarm: decentralized collision avoidance under limited comms

**Date:** 2026-05-24

**Goal:** a regime where the *classical* coordination breaks, then a learned policy that beats it.

**Hard regime found:** the classical scheduler + CBF holds 100% all-landed *and* keeps separation up to
~12 drones with full comms, but with a **finite communication range** (each drone only sees neighbours
within range) it degrades — drones don't see each other until too close and the reactive CBF can't avoid
in time. Swept it: **14 drones + 1.0 m comms → separation kept only 11/20 (55%)**, mean min-sep 0.56 m
(< the 0.7 m safety bound). Exposed via `SwarmConfig.comms_range` + `swarm run --comms`.

**MARL built (parameter-sharing PPO):** `marl_env.py SwarmMARLEnv` — an **ego-agent view**: one drone
runs the policy (the rest classical), the ego is randomized each episode so a single shared policy
generalizes, and at deployment it runs on **every** drone (CTDE-style decentralized execution from a
**local** observation: own state-rel-deck + nearest-K neighbours within comms). The action is a velocity
**residual** on the classical guidance (0 = classical), so it starts safe; the reward is dominated by a
**separation penalty** (the classical baseline's failure). `coordinator.py` got a `policy_residual` hook
+ `local_obs()`. `marl_train.py` trains (PPO) and evaluates **classical vs MARL-on-all** on the hard
regime.

**Result (14 drones, 1.0 m comms, 20 ep): MARL ties the classical, does not beat it** — both 100%
all-landed, **separation-kept 50% vs 50%** (min-sep 0.549 vs 0.552 m). **Honest analysis of why:** in this
regime the violations are **comms-physics-limited, not coordination-limited** — at 1.0 m sensing, 0.7 m
safety, and up to ~3 m/s closing, a drone first sees a neighbour only ~0.3 m inside the safety bound, too
late for *any* policy to avoid; and the learned residual sits *beneath* the CBF safety filter, bounding
its authority. So the failures aren't fixable by smarter coordination here. A MARL win would need a
*hard-but-physically-feasible* band (larger comms / lower speed where avoidance is possible yet the
reactive CBF is suboptimal) — a narrow window between "classical already solves it (full comms)" and
"physically unavoidable (very short comms)". **Status: full MARL infrastructure built + trained + honestly
evaluated** (multi-agent env, parameter-sharing PPO, CTDE decentralized deployment, `swarm marl`); the
learned *decentralized* policy **matches** the strong hand-coded coordinator (a legitimate result), but
does not beat it in the tested regime. Recurring theme confirmed: the classical stack is excellent; RL/MARL
earns wins only where the bottleneck is algorithmic (e.g. single-drone fast-rover), not physical.

---

## Phase 3 (start) — Swarm flight-deck recovery (separate module)

**Date:** 2026-05-24

By design, the swarm is a **completely separate module** (`src/drone_landing_swarm/`,
its own `swarm` CLI) that **reuses the single-drone autopilot as a black-box inner loop** and never
touches the single-drone sim. Design in [SWARM.md](SWARM.md): N drones recover onto one moving deck via
**optimal slot scheduling** + **CBF collision avoidance** + a deconflicted **holding stack**.

**Built (coordination layer, unit-tested — 8/8):**
- `scheduler.py` — `LandingScheduler`: clears the K readiest drones (K-lowest-cost = optimal for
  identical slots) with **anti-starvation** + **hysteresis**; `optimal_assignment()` exposes the
  **Hungarian** algorithm for the future M-drones × K-distinct-platforms case.
- `avoidance.py` — `cbf_safe_velocity`: an **exact CBF-QP** safety filter (control-barrier function for
  pairwise separation, solved by **Hildreth's dual QP** — minimal-intervention, no solver dependency),
  decentralized (needs only neighbours' broadcast state).
- `holding.py` — `HoldingStack`: ring + altitude-layer holding slots **relative to the deck** (the stack
  rides with the ship); waiting drones assigned to slots via the **Hungarian** algorithm; station-keeping
  guidance with deck-velocity feedforward.
- `coordinator.py` — `SwarmCoordinator`: the closed loop — advance the moving deck (reuses
  `ShipDeckMotion`/`RandomGroundMotion`), scheduler clears K, cleared drones land while the rest hold,
  **every** velocity passes the CBF filter (pairwise + deck keep-out column). Velocity-controlled drones
  for now (the coordination layer is what this validates); full per-drone autopilot+physics is the
  drop-in next step.
- `cli.py` + **`swarm` command** (separate entry point; `python -m drone_landing_swarm`): `swarm
  run | watch | info`. `watch` is a matplotlib 3-D animation (red=landing, blue=holding, green=down).

**Results (`swarm run`, kinematic coordination):** **100% all-landed with separation always maintained**
across 4 drones (ship moderate), **6 drones (rough sea)**, **8 drones with 2 slots**, and 5 drones
(ground) — min inter-drone separation ≥ the 0.7 m safety bound. The single-drone sim is untouched.

**Next:** integrate the full single-drone autopilot + MuJoCo physics per drone (multi-drone world);
later, MARL (MAPPO/CTDE) for the coordination policy + cooperative (consensus) deck estimation.

---

## Phase 1 refinements — ground-MPC gate + green-deck estimator (honest A/Bs)

**Date:** 2026-05-24

**ground-MPC confidence-gated engagement:** the MPC override now engages only once the estimate is
confident (`tracked` and EKF horizontal std < `mpc_confident_std`=0.20 m); during acquisition the gentle
geometric PD flies, so the MPC's aggressive intercept can't amplify early estimate error into a fly-off.
**25% → 30% (flow velocity) → 40% (gate).** Still estimation-limited — kept as the clean-state benchmark
/ RL base, not a standalone winner.

**Green-deck estimator — Kalman oscillator tried, reverted:** implemented a recursive Kalman-oscillator
(PLL-style) deck-heave estimator. **Measured it honestly: it underperformed** the windowed least-squares
fit — a fixed-frequency oscillator accumulates phase drift over an episode, so the phase-sensitive
*velocity* nowcast went anti-correlated in rough seas (corr −0.15). Reverted. **The real win was tuning
the LS fit**: a wider 8 s window + a "span ≥ one wave period" lock-gate (fitting < one period conflates
trend and sinusoid into a junk trend). Nowcast deck-velocity correlation **0.68 → 0.86 (moderate) / 0.82
(rough)**; amplitudes now match truth. 5 unit tests pass (one rewritten to check *tracking*, the
meaningful property, vs a too-strict single-instant value).

**Green-deck control still does NOT beat baseline** (rough sea, 12 ep): baseline **83% / 0.140 m/s**
impact vs green-deck **75% / 0.142 m/s**. Even with the far better nowcast, timing the commit doesn't
help — **why is now clear:** with the larger deck + gentle commit descent, the touchdown impact is
dominated by the *commanded descent rate*, not the deck heave, and the failures are acquisition/wind
fly-offs (`out_of_bounds`), not hard impacts — so better heave timing has almost nothing to convert.
**Decision:** keep the improved estimator (a better component; useful for RL observations / telemetry /
extreme-sea or tight-impact regimes), keep `--green-deck` optional (not default), and treat green-deck
*control* as a thoroughly-investigated honest negative in this configuration.

---

## Phase 5 polish — larger landing decks + ground-MPC velocity fix

**Date:** 2026-05-24

**Larger platforms (realism + margin):** enlarged both decks from 1.0 m to **1.8 m**
(`x2_landing_{ground,ship}.xml`). To keep perception **byte-identical**, the ArUco fiducial is now a
**visual-only pad kept at 1.0 m = `BoardSpec.deck_size`** (so the printed markers stay at their
calibrated 0.08 m), sitting on a larger plain collision `deck` geom (which `world.py` still keys
touchdown on; `deck_top_z` unchanged at 0.30). `success_radius` relaxed 0.30 → **0.40 m** (a touchdown
anywhere on the central pad counts; the supervisor's `align_radius` still aims for centre).

**Result (10 ep each, vision-only):** **ground 70% → 90%**, **ship 90%** (steady) — the bigger pad
converts edge-tipover crashes and on-deck-but-off-centre landings into successes; each lone remaining
failure is an early-acquisition fly-off, not a deck issue.

**Ground-MPC fix:** `ground-mpc` was landing only ~25% (6/8 fly-offs / `out_of_bounds`) — the documented
estimation-amplification problem (MPC's decisive commands amplify the EKF's spiky differentiated
velocity). Fix attempt: the MPC now consumes the **robust optical-flow relative velocity** (the same
clean signal IBVS uses), not the EKF velocity. **Result: 25% → 30%** — only marginal. The fly-offs are
*early* (during acquisition, before the flow/estimate stabilises), where the MPC's aggressive predictive
intercept overshoots on initial estimate error. The flow feed is kept (principled, doesn't hurt), but a
real fix needs **confidence-gated engagement** (fly geometric during APPROACH, hand to MPC only once the
estimate is confident/aligned) and/or MPC de-tuning. **Recorded as an open item**; this reinforces the
project's core finding that robust closed-loop landing is *estimation-limited*, not control-limited —
MPC remains the clean-state tracking benchmark (3.4× tighter) and the right tool for RL's residual.

**Realism — wind disturbance (default on):** added a steady breeze + **correlated Ornstein-Uhlenbeck
gusts** as a body force (`world._apply_wind`, `wind_mean`/`wind_gust_std`/`wind_gust_tau`). Calibrated
to a *light* breeze (mean ~0.23 N ≈ 1.7% of weight ≈ 1° steady tilt) so the world is realistically
disturbed without dominating — a first pass at 0.47 N dropped ground to 50% (random rover + wind is the
sensitive combination), so it was halved. Sensor noise (IMU white+bias-walk, AHRS, rangefinder
noise/dropout, baro drift, GPS) was already realistic. **Result under wind: ground 90%, ship 90%** —
no regression.

**Controller hardening:** geometric controller now uses a **leaky integrator** (`int_leak`, bleeds
stale bias when the platform's acceleration changes) + **conditional-integration anti-windup** (accept
the new integral only if it doesn't increase an already-saturated command, so it never winds up against
the accel/tilt limit). Confirmed neutral-to-positive (ground/ship hold 90% under wind).

---

## Tooling — Unified simulation CLI (`drone`)

**Date:** 2026-05-23

**Goal:** a clean and organized CLI to choose and launch whichever simulation easily — individually or in parallel.

**Built `src/drone_landing/cli.py`** — primary command **`drone`** (aliases `dl`, `drone-landing`; also
`python -m drone_landing`), installed via `pip install -e .`. A `SimSpec` dataclass + named **preset
registry** (`ground`, `ground-mpc`,
`ground-ibvs`, `ground-hard`, `ship-calm`, `ship`, `ship-rough`, `ship-green`, `ship-rough-green`) and a
single `build(spec)` factory that constructs the (world, autopilot) for any scenario×controller×sea×
green-deck combination. Subcommands:
- `list` — scenarios, controllers, sea states, and presets.
- `run [preset] [--scenario/--controller/--sea/--green-deck/--vmax…] [--episodes/--seed] [--json]` — one
  headless batch; prints a success/impact summary table (or a JSON line).
- `watch [preset] …` — live MuJoCo window (reuses the passive-viewer loop).
- `eval --seas … --episodes …` — the green-deck baseline-vs-green ablation table.
- `parallel <preset…> | --all [--workers N]` — runs several presets **concurrently**, each in its own
  process (`ProcessPoolExecutor`, own MuJoCo renderer), streaming `[done]` lines then an aggregated table.
- `info` — build status (worlds, optional deps, GPU/CPU device, trained RL policies).
- `train [--algo ppo|recurrent_ppo] [--no-curriculum] [--eval <ckpt>]` — RL training/evaluation.

**CLI completeness pass (2026-05-24):** verified the CLI exposes every implemented feature and closed two
gaps — added **`drone train --eval <ckpt>`** (RL-residual-vs-baseline eval, was only reachable via
`python -m drone_landing.rl.train`) and **`drone run --no-wind`** (ablate the default wind gusts). The
**`swarm`** command (separate module) covers `info | run | watch` with `--drones/--scenario/--sea/--slots`.
README, RESEARCH_NOTES, SWARM.md, and memory cross-checked against the implementation — all current.

**Performance fix:** the CLI reuses **one** `mujoco.Renderer` across all frames/episodes; the older
scripts created and closed a renderer *per frame* (a large GL-context overhead) — the main reason the
eval batches were slow. 5 unit tests still pass; README + this log updated. (Output is ASCII-only to
avoid Windows cp1252 console-encoding errors.)

---

## Phase 5 — Maritime scenario: 6-DOF ship deck + green-deck timing

**Date:** 2026-05-23

**Goal:** The second scenario — landing on a **ship deck at sea** that heaves, rolls, and pitches with
the waves. Build the 6-DOF world, then the maritime-specific control: timing the touchdown to a
low-motion ("green-deck") window, the way real shipboard recovery does.

**Built:**
- **6-DOF ship world** `assets/mujoco/worlds/x2_landing_ship.xml`: the validated X2 + gimbal +
  rangefinder over a **dynamic deck on six servo'd joints** (slide x/y/z + hinge yaw/pitch/roll), so
  it carries true contact velocity (friction drags a landed drone with the heaving/rolling deck).
  Buoyancy is modeled as **gravity compensation** (`gravcomp="1"`) so the servos track only the wave
  oscillation. Driven by `ShipDeckMotion`. Verified: deck heaves 0.14–0.47 m (mean 0.30), rolls ±8°,
  pitches ±5°, servos tracking to ~1.5 cm / ~0.3°.
- **Generalized `sim/world.py`** to drive whatever platform DOFs the loaded model exposes (3 for the
  ground rover, 6 for the ship) and to auto-select the motion model from the world name. `platform_truth`
  now reports the heaving deck-top height.
- **Sea-state presets** (`sim/platforms/ship.py sea_state`): `calm | moderate | rough` (rough ≈ doubles
  heave/roll: peak heave rate ~0.4 m/s, roll ±12°). `run_autopilot --ship [--sea rough]`.
- **Onboard green-deck predictor** `planning/deck_predictor.py` (`DeckMotionPredictor`): estimates the
  deck heave from the **relative-altitude signal the EKF already tracks** (no truth, no wave model) by a
  sliding-window **multi-sinusoid + linear-trend least-squares fit** (matching-pursuit frequency
  selection; the trend term removes the drone's own descent). Exposes the deck vertical-velocity
  nowcast/forecast and green-window logic. Refit is decimated (every 10 samples; phase still
  extrapolates each call). **Validated:** locks on; recovers period/amplitude; nowcasts deck vertical
  velocity to **~0.09 m/s RMS** — but the +1 s *forecast* error grows to ~0.17 m/s (deck-motion
  prediction is inherently short-horizon). 5 unit tests in `tests/test_deck_predictor.py`.
- **Green-deck commit timing** (supervisor + autopilot, `--green-deck`): two mechanisms designed around
  the short forecast horizon — (1) **perch** at the commit altitude and wait for a predicted low-motion
  window (with a `green_max_wait` fall-through so it never stalls in rough seas); (2) **heave-synchronized
  descent** — feed the deck-velocity *nowcast* forward into the descent command so the drone rides the
  heave and closes at a small constant *relative* rate, making impact velocity depend on the accurate
  nowcast rather than the unreliable forecast.

**Baseline (geometric controller, vision-only, no truth in loop), moderate sea, 12 ep:** **11/12 = 92%**
landing — the ship's predictable surge+oscillation is *easier* to track than the random ground rover
(70%). Mean touchdown error 0.090 m; the one failure was an early-acquisition fly-off (common to all
controllers). Mean contact vertical speed **0.113 m/s** (higher than ground's ~0.04 — the heave
signature the green-deck timing targets).

**Green-deck ablation (baseline vs `--green-deck`, rough sea, 10 ep; `drone eval --seas rough`):**

| sea | timing | success | mean impact \|vz\| | max \|vz\| |
| --- | --- | --- | --- | --- |
| rough | baseline | 60% | 0.124 m/s | 0.157 |
| rough | green-deck | 60% | 0.143 m/s | 0.204 |

**Honest finding:** the green-deck timing **as currently tuned does not beat the baseline** — same
success, slightly *worse* contact velocity in rough seas. Likely causes: in rough seas the heave-rate
**nowcast error (~0.09–0.13 m/s) is a large fraction of the heave itself**, so the velocity feedforward
injects about as much noise as it cancels; and the perch-and-wait keeps the drone longer in the
roll/pitch disturbance field. The mechanism is sound and literature-backed, but it needs (a) a better
heave-rate estimate (e.g. a phase-locked loop / Kalman oscillator instead of a windowed LS refit) and/or
(b) feedforward gain scheduling and lag compensation. **Recorded as an open Phase-5 item**; the baseline
geometric controller remains the maritime default. (Rough-sea baseline 60% vs moderate 92% confirms the
sea state, not the timing, is the dominant difficulty.)

**Realism (no cheats):** the deck's wave motion is prescribed (it drives only the plant; the deck is far
heavier than the drone). The controller's deck-motion estimate is reconstructed **onboard** from the
downward camera + rangefinder; the simulator's wave model is never read in the control loop.

---

## Phase 4 — IBVS controller + Phase 5 start — ship seakeeping motion

**Date:** 2026-05-23

**IBVS (`control/ibvs.py`, `--ibvs`):** image-based visual servoing guidance — commands horizontal
acceleration from the ArUco image position + the fiducial's **optical-flow velocity** (robust,
validated ~0.2 m/s with the gimbal), *bypassing the EKF's spiky differentiated velocity*. This is the
research-backed estimation-bottleneck fix and it **removes the APPROACH velocity-spike fly-offs**.
On realistic motion it isn't yet better than the tuned geometric controller (remaining failures are
touchdown tip-over and lost-track go-arounds — supervisor/touchdown robustness, common to all
controllers). It is kept as a controller in the suite {PID, geometric+supervisor, MPC, IBVS} for the
Phase-7 benchmark; the geometric controller remains the tuned default for the working demo.

**Ship seakeeping motion (`sim/platforms/ship.py`):** `ShipDeckMotion` — a wave-driven 6-DOF deck
(forward surge + heave/roll/pitch/sway as sums of sinusoids from a sea state; RAO-style time-domain
model). Validated: heave 0.13–0.47 m, roll ±8°, heave rate ≤0.25 m/s; `deck_motion_energy()`
forecasts the vertical-motion profile and shows clear low-motion ("green-deck") windows (|vz| down to
~0 between peaks of ~0.19 m/s) for the touchdown-timing planner. This is the first piece of the
maritime scenario; the 6-DOF deck world (MJCF) + MPC green-deck commit timing are next.

---

## Phase 4 — Nonlinear MPC (predictive horizontal tracking)

**Date:** 2026-05-23

**Goal:** The headline advanced controller — a CasADi optimal-control outer loop that plans to
*intercept the platform's predicted future trajectory* (no PD lag), feeding the geometric attitude
inner loop. Implemented to provide stronger optimization-based control.

**Built:**
- `control/mpc/nmpc.py` — `HorizontalMPC`: a receding-horizon QP (CasADi `qrqp`, 4 states, 2 controls,
  1 s horizon) over the relative dynamics with a constant-velocity platform; outputs the horizontal
  acceleration command. `control/geometric.py` accepts it via `a_xy_override`; the autopilot uses it
  when `use_mpc=True` (`run_autopilot --mpc`). `scripts/test_mpc.py` benchmarks it vs PD.

**Verified (state-level, perfect state — isolates the controller):** the MPC tracks the moving
platform **~3.4× tighter than PD**:

| Platform | PD steady-state err | **MPC** |
| --- | --- | --- |
| realistic (0.7 m/s) | 5.5 cm (peak 17.9) | **1.6 cm (peak 2.7)** |
| fast random (1.5 m/s) | 15.7 cm (peak 28.5) | **4.7 cm (peak 9.5)** |

**Closed-loop landing (vision-only):** fast random rover **20%**, realistic motion **~22%** — both
*worse* than the geometric controller (25% / 70%). This is the decisive, honest finding: with the
**noisy vision velocity estimate**, the MPC's decisive predictive commands *amplify* the estimate
noise (more fly-offs) than the gentler, integral-bounded geometric PD. A better controller does not
help — and can hurt — when the bottleneck is **estimation**, not control.

**Conclusion (now firmly established):** MPC is a genuine *control* win (3.4× tighter tracking with
clean state; the headline advanced controller; the right tool for the maritime green-deck timing).
But robust closed-loop landing is **estimation-limited**, so the path to the hardest cases is
**(a) IBVS / a better velocity sensor** (image-plane control, robust to velocity error) and/or
**(b) RL** (Phase 6; literature ~94%), which learns robustness to the noisy observations directly.
The MPC autopilot integration is kept (`--mpc`) but defaults off; the geometric controller remains
the more robust closed-loop choice for now.

---

## Phase 4 (investigation) — Estimator hardening: optical-flow velocity sensor

**Date:** 2026-05-23

**Goal:** Fix the vision-loss divergences (the main cause of the ~25% landing rate) by hardening the
velocity estimate, which is the diagnosed bottleneck (velocity differentiated from 33 Hz vision is
noisy and spikes to 1–3 m/s, which the controller's damping term turns into a fly-off).

**Built (kept as a validated module):**
- `perception/optical_flow.py` — a downward optical-flow velocity sensor (deck-masked dense flow,
  gyro-compensated, range-scaled), like real GPS-denied precision-landing drones. `scripts/test_flow.py`.
- `estimation/ekf.py` gained `update_velocity_xy` so the filter can fuse a direct velocity measurement.

**Verified in isolation:** flow velocity tracks ground truth to **~0.3 m/s RMS** across seeds — a
clean, direct measurement vs the 1–3 m/s spikes from differentiation.

**Result in the closed loop: it did NOT help (12% vs 25%) — and that is the key finding.** Wiring the
(good) velocity sensor into the loop did not raise the landing rate, because the dominant failure is
**not** velocity noise. It is a **control↔perception coupling**: to correct horizontal error the
drone tilts, which tilts the downward camera *off* the marker; vision is lost; the position estimate
then drifts and the controller chases it off the arena. No estimator change fixes an unobservable
state caused by the camera pointing away.

**Then added a stabilized gimbal camera (chosen direction).** `assets/.../x2_landing_ground.xml` now
carries the downward camera on a kinematic gimbal body the world holds at the drone's belly but
always **world-level (nadir)** (`world._drive_gimbal`); the autopilot reads it as an identity camera
frame. The gimbal view is clean (full fiducial, zero self-occlusion) and the camera no longer tilts
off the marker — it fixed individual tilt-loss failures (e.g. seed 6). It is the correct, realistic
architecture and is kept.

**But the batch rate stayed at 2/8 = 25%.** Why: the dominant failure (4/8 → out-of-bounds) is not
vision loss but a **velocity-estimate spike during the aggressive early approach** that flies the
drone off *before* vision-loss matters. Softening the controller gains stops the spike but then it
can't keep up with the fast (1.5 m/s, random) rover and drifts/times out — a fundamental
tracking-vs-stability tension for hand-tuned classical control on a fast random platform.

**Conclusion (well-established after extensive iteration):** the classical floor reliably tracks,
descends, and plants 4 feet within ~0.12 m at ~0.13 m/s **when it commits**, but robust *every-time*
landing on a fast random platform is beyond hand-tuned classical control + a vision-derived velocity
estimate.

### Resolution — literature review + realistic scenario (see [RESEARCH_NOTES.md](RESEARCH_NOTES.md))

A thorough web review of how real systems land on moving platforms found the core lesson: real
systems use **IBVS (image-based visual servoing)** — control in the image plane, which is *"less
sensitive to depth estimation"* and *"overcomes the lack of velocity feedback"* — exactly the
weakness of our PBVS (estimate-3D-state-then-track) design. Real demos also use **smooth, predictable
platform motion + velocity feed-forward**, and the highest robustness on the hardest motion comes
from **RL (~94%)**.

Applied: (1) made a **realistic vehicle-motion profile the default** scenario (0.7 m/s, gentle accel
— what real targets do), with the darting 1.5 m/s random rover kept as a stress test
(`--vmax/--amax`); (2) bounded the relative-velocity estimate and softened its initial convergence to
kill the spike-driven fly-offs.

**Result: landing success on the realistic scenario rose 25% → 70%** (10-episode batch, vision-only,
no truth), touchdowns ~0.19 m / 0.06 m/s. The remaining failures are init-transient velocity spikes
during APPROACH — the same root cause, and the reason the next steps are the *stronger methods*.

**Next (the advanced methods the project always planned):**
- **Phase 4 — nonlinear MPC (CasADi):** predictive tracking (intercept the predicted platform, no
  lag), optimal constrained commit, and the basis for the maritime green-deck timing. The headline
  advanced-control benchmark.
- **Phase 6 — RL:** robust learned policy for the hardest (fast/unpredictable) motion (literature
  ~94%); needs a CUDA PyTorch install.

---

## Phase 3 — Classical advanced control floor (baseline established)

**Date:** 2026-05-23

**Goal:** A clean, reusable control stack + a landing supervisor that lands on the vision-only
estimate — the classical baseline that MPC/RL must beat.

**Done — the full vision autopilot stack:**
- `control/allocation.py` — quad-X control allocation (mixer) computed from geometry (full rank).
- `control/geometric.py` — geometric SE(3) controller with **integral action** (zero steady-state
  lag tracking an accelerating platform), velocity-hold and level-press modes for touchdown.
- `planning/supervisor.py` — **landing supervisor FSM**: APPROACH → DESCEND → COMMIT → SECURED with
  GO_AROUND recovery; descends only while the platform is confidently tracked and centred.
- `autopilot.py` — integrated stack (ArUco → Kalman filter → supervisor → control); consumes only
  onboard signals, no truth in the loop.
- Perception upgraded to a **multi-scale fiducial**: grid board (approach) + a nested centre marker
  (DICT_5X5) that stays in the FOV to touchdown. Gear given shock-absorbing (damped) contact; deck a
  grippy landing pad.
- Scripts: `run_autopilot.py`, `watch_autopilot.py` (live), `trace_autopilot.py`.

**Verified (8-episode batch, vision-only, no truth):**
- The stack reliably **acquires, tracks, approaches, descends, and plants all four feet within
  ~0.13 m of deck centre** at ~0.13 m/s contact — flying on the estimate.
- **Landing success 2/8 = 25%**; successes are clean (0.10–0.13 m, 0.13 m/s). Failures are
  go-around loops / estimator divergence when vision is lost during a rover maneuver, and the final
  settle being thrown by sharp deck turns (motors-off drone tips/slides).

**Honest assessment:** the classical floor nails the hard 95% (tracking + descent + plant) but the
final settle on a *maneuvering* deck is fragile under hand-tuning — a textbook case for the stronger
methods. This is the baseline; **Phase 4 (nonlinear MPC)** brings predictive tracking (no lag) and a
faster, firmer, constrained commit, and **Phase 6 (RL/residual)** learns a robust touchdown.

**Key engineering notes:** PD lag on an accelerating platform needed integral action; the touchdown
vision blind-spot needed the nested centre marker; the EKF can diverge while coasting through a
vision gap (drives the go-around/divergence failures). Velocity-hold during commit (match platform
velocity, don't chase a coasting position estimate) was essential to land at all.

**Next (Phase 4):** CasADi nonlinear MPC tracking the predicted platform, benchmarked against this
classical floor.

---

## Phase 2 — Perception + estimation (core complete; robust closed loop -> Phase 3)

**Date:** 2026-05-23

**Goal:** Estimate the platform's relative state from sensors only (no truth in the loop): ArUco
vision + IMU + rangefinder fused in a Kalman filter, validated against ground truth, then fly the
controller on the estimate.

**Done:**
- `perception/` — camera model (intrinsics from MuJoCo `fovy`), **ArUco GridBoard** detector
  (`solvePnP`, SQPNP to avoid planar pose ambiguity), shared board geometry (`board.py`). A *grid*
  of markers keeps the deck detectable from the ~2 m start altitude down to ~0.24 m (a single
  marker leaves the frame far too early).
- Deck textured with the rendered board (`scripts/gen_aruco_deck.py`); shadow-free lighting so
  rotor shadows don't wash out the fiducial; downward camera repositioned.
- `estimation/ekf.py` — **`RelativeStateEKF`**, a Kalman filter on the platform-relative state
  `[r, v_rel]`. IMU-aided prediction (accelerometer + AHRS attitude as input), ArUco position
  updates with an innovation gate (rejects `solvePnP` ambiguity flips), rangefinder vertical
  updates. No GPS needed — the relative state is fully observable from vision.
- Closed-loop runner (`scripts/run_closed_loop.py`) — controller flies on the EKF estimate, **zero
  ground truth in the control path** (only a gear-contact sensor + camera/IMU/range).
- Validation/diagnostic scripts: `test_perception.py`, `test_ekf.py`.

**Verified:**
- Raw ArUco relative-position measurement (quality-gated): **~1 cm** RMS across the descent.
- Kalman filter vs truth with continuous vision: **~12 cm** rel-position RMS, ~25 cm/s velocity.
- Estimator **degrades during detection gaps** (drone low + off-centre → marker leaves frame →
  IMU dead-reckoning drifts). This is a coupled perception↔control problem.
- Closed loop on the estimate (rough Phase-1 controller): lands as a proof-of-concept but is **not
  yet robust** — when the estimate nudges the drone off-centre, detection degrades and the error
  compounds.

**Key fixes (charter-driven debugging):** grid board (vs single marker) for continuous detection;
shadow-free lighting; SQPNP + innovation gating for `solvePnP` robustness; IMU-aided filter; camera
mounting-offset correction. (Tried and rejected: image-centroid bearing — biased when the board is
partly out of frame.)

**Deferred to Phase 3 (by design):** robust closed-loop landing on the estimate. The plan always
placed the **landing supervisor FSM** + proper geometric controller in Phase 3 — exactly the piece
that keeps the target centred, holds/re-acquires when vision degrades, and aborts/goes-around. That
breaks the perception↔control instability seen here.

**Next (Phase 3):** geometric SE(3) controller + Kalman platform tracker + rendezvous guidance +
landing supervisor FSM → robust closed-loop landing on the vision-only estimate.

---

## Phase 1 — High-fidelity ground world (complete)

**Date:** 2026-05-23

**Goal:** An honest, landable ground-rover world: realistic platform motion, realistic sensors,
strict contact-based touchdown (no cheats), and proof a drone can land on the moving deck.

**Done:**
- `sim/platforms/` — `PlatformMotion` abstraction + `RandomGroundMotion`: a jerk-limited,
  bounded random-walk rover (OU desired-velocity, accel/jerk caps, yaw aligned to heading).
  Verified over 60 s: peak speed 1.26 ≤ 1.5 m/s, jerk capped at 2.0, stays in bounds.
- `sim/sensors/` — IMU (white noise + bias random-walk), rangefinder (noise/range-gate/dropout),
  barometer (drift), GPS (slow/noisy/dropout), AHRS attitude, and a `LatencyBuffer`. Magnitudes
  target MEMS/consumer hardware (documented in code).
- `sim/world.py` — `LandingWorld`: drives the platform, uses the X2 motor actuators, exposes
  **separate** sensor-derived vs ground-truth (eval/critic-only) observations, and detects
  success / crash / off-platform / out-of-bounds / timeout via **strict frictional contact**
  with a 1 s settle window (no weld lock).
- Truth-based geometric SE(3) cascade controller (`scripts/validate_landing.py`) — temporary
  physics-validation tool only — with a weight-on-gear touchdown.
- `scripts/watch_landing.py` — interactive real-time viewer that loops landings.

**Two important fixes (both flagged by the Realism Charter):**
1. **Mocap → dynamic platform.** The platform was a kinematic mocap body, which the physics
   engine treats as zero-velocity, so friction could not carry a landed drone with the moving
   deck (it slid off). Replaced with a heavy (65 kg) **dynamic rover** on slide(x,y)+hinge(yaw)
   joints driven by **position servos** — contact friction is now genuine.
2. **X2 rotor spin pattern.** The Menagerie X2 groups rotor spins by side, making the
   control-allocation matrix rank-3 (yaw uncontrollable). Corrected to a proper quad-X
   (diagonal rotors same spin) → full rank (cond ~50), hover yaw torque still balanced.

**Verified:** model compiles (`nq=10 nv=9 nu=7`); hover 0 drift; landing validation **7/8 = 88%**
with mean contact speed **0.065 m/s**, mean touchdown error **0.19 m**, level attitude. The single
failure touched at the deck edge and tipped (a controller-robustness case for Phase 3, not a
physics issue). Landed snapshot saved to `runs/phase1/landed_track.png`.

**Next (Phase 2):** ArUco marker on the deck + `solvePnP` 6-DOF pose, markerless fallback, and an
EKF fusing IMU + vision + range into the relative platform state — then switch controllers off
ground truth onto the estimator.

---

## Phase 0 — Foundation (complete)

**Date:** 2026-05-23

**Goal:** Stand up the project skeleton, adopt the real Skydio X2 drone, and prove a physically
sane landing world loads, hovers, and renders.

**Done:**
- Authored the master plan ([PROJECT_PLAN.md](PROJECT_PLAN.md)) and the no-cheats
  [REALISM_CHARTER.md](REALISM_CHARTER.md).
- Copied the validated Skydio X2 mesh/texture into a clean (no-spaces) asset path:
  `assets/mujoco/meshes/skydio_x2/`.
- Built the first high-fidelity world `assets/mujoco/worlds/x2_landing_ground.xml`:
  - Skydio X2 body with **preserved** validated dynamics (rotor masses, motor `gear`/`ctrlrange`,
    IMU/gyro/accel/quat sensors).
  - Added **landing gear** (4 legs + feet, frictional collision geoms).
  - Added a **downward camera** (`down`), a **rangefinder** (downward altitude sensor), and kept the
    `track` camera.
  - Added a **moving platform** (kinematic mocap deck) with a placeholder fiducial marker (real ArUco
    texture comes in Phase 2). Air density/viscosity enabled for passive aerodynamic drag.
- Added a sim loader (`src/drone_landing/sim/mjcf.py`) that resolves world paths independent of CWD.
- Added a smoke check (`scripts/check_world.py`).
- Upgraded `pyproject.toml` with realistic extras: `vision` (opencv-contrib for ArUco), `mpc` (casadi),
  `rl`, `viz`, `tune`, `track`, `dev` (ruff/mypy/pre-commit/pytest).
- Initialized git for checkpointing; added `.gitignore`.

**Verified (`scripts/check_world.py`):**
- Model compiles: `nq=7 nv=6 nu=4`, 4 thrust actuators, sensors `[gyro, accel, quat, rangefinder]`,
  cameras `[track, down]`.
- Drone mass **1.373 kg** (weight 13.47 N) → ideal hover **3.367 N/motor**; keyframe set to 3.368.
- Hover stability: altitude drift **0.009 m** over 3 s (1500 steps) — finite, stable.
- Both cameras render; the downward camera sees the deck + marker.

**Environment notes:**
- Installed & ready: mujoco 3.8.1, casadi 3.7.2, opencv-contrib 4.13, scipy, gymnasium, numpy 2.2.
- ⚠️ torch is the **CPU-only** build (`2.12.0+cpu`) — must install a CUDA wheel for the RTX 5060
  (Blackwell, needs CUDA 12.8+) before Phase 6 (RL). stable-baselines3 not yet installed.

**Next (Phase 1):** randomized rover platform with honest motion; refine downward-camera placement to
reduce self-occlusion; realistic sensor models (noise/bias/latency/rate); strict contact landing with
no weld lock; touchdown success detection.
