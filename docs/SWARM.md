# Swarm flight-deck recovery — design

A **separate** application module (`src/drone_landing_swarm/`) that coordinates **N drones recovering
onto one moving deck** with a limited number of landing slots — "air-traffic control for a drone
swarm landing on a carrier." It **reuses the validated single-drone autopilot as a black-box inner
loop** and never modifies the single-drone simulation (per the project constraint). It has its own
entry point, the **`swarm`** command (separate from `drone`).

## The problem

Many drones, one small moving deck. Only one (or a few) can occupy the deck at a time. The swarm must:
1. **Schedule** — decide the landing order / who is cleared to descend (decentralized).
2. **Hold** — the rest wait in a deconflicted holding stack (distinct positions + altitude layers),
   tracking the moving deck so the stack travels with the ship.
3. **Deconflict** — guarantee inter-drone separation at all times (collision avoidance).
4. **Land** — the cleared drone runs the ordinary single-drone landing; on touchdown (or abort) the
   next drone is cleared.

This maps directly to real carrier marshalling / helicopter recovery and to drone-delivery fleets.

## Architecture (layers)

```
            ┌─────────────────────── SwarmCoordinator ───────────────────────┐
            │  scheduler (slot assignment)   +   holding-stack manager         │
            └───────────────┬─────────────────────────────┬───────────────────┘
       cleared drone │                              holding drones │
                     ▼                                             ▼
        single-drone autopilot                     holding guidance (track deck offset)
                     │                                             │
                     └──────────────► collision avoidance (CBF) ◄──┘   (safety filter on every drone)
                                              │
                                              ▼
                                    multi-drone physics world
```

- **`scheduler.py`** — `LandingScheduler`: decentralized assignment of the single landing slot (and,
  later, K slots) to drones, e.g. by readiness/proximity with hysteresis so the clearance doesn't
  chatter. Auction / Hungarian for the K-slot / K-platform generalization.
- **`avoidance.py`** — `collision_avoidance`: a **control-barrier-function (CBF) safety filter** that
  minimally modifies each drone's desired horizontal velocity/accel to enforce pairwise separation
  (and a deck-keep-out for non-cleared drones). Reactive, decentralized, provably safe within model.
- **`holding.py`** — assigns each holding drone a distinct **stack slot** (ring position + altitude
  layer) that rides with the moving deck, and a guidance law to hold it.
- **`coordinator.py`** — `SwarmCoordinator`: ties it together; steps N drones, runs the scheduler,
  drives the cleared drone with the single-drone autopilot and the others with holding guidance, and
  applies the CBF filter to all. Cooperative deck estimation (consensus) is a later enhancement.
- **`world.py`** — builds an N-drone MuJoCo world (N × X2 + one moving deck) programmatically.
- **`cli.py`** — the `swarm` command (`swarm run`, `swarm watch`, `swarm info`).

## Build order (incremental, each piece testable in isolation)

1. **Coordination algorithms** (`scheduler`, `avoidance`, `holding`) — pure, unit-tested with a
   lightweight kinematic multi-agent model (fast, no rendering). ← *start here*
2. **Multi-drone world** + `coordinator` wiring to the real single-drone autopilot/physics.
3. **`swarm` CLI** + visualization (watch N drones recover).
4. Cooperative perception (consensus deck estimate); MARL for the coordination policy (stretch).

## Realism stance

Same no-cheats principle: each drone flies on its own onboard estimate; the coordinator exchanges only
what real drones could broadcast (own state estimate, intent/clearance) over a comms model. The
coordination layer may be validated first on a kinematic model, then run on the full physics + vision.

## Onboard sensing + comms (A1 — DONE, no-cheats audit)

`sensing.py` (`SwarmSensing` / `SensingConfig`) is the layer that makes the swarm audit-clean: the
coordination loop reads **zero ground truth**. Each step the simulator's true states are converted into
the **onboard view** every drone actually has:

- **own state** — a per-drone noisy estimate (own EKF, modelled as truth + calibrated noise:
  `pos_noise=5cm`, `vel_noise=0.10 m/s`);
- **deck pose** — a per-drone noisy estimate (`deck_noise=6cm`), later fused by the consensus layer (A2);
- **neighbours** — only those within **comms range**, received as **broadcast estimates** subject to
  **latency** (1 step stale) and **dropout** (5%/step), plus relative-sensing noise (`rel_noise=8cm`).

The scheduler readiness cost, holding-slot assignment, the CBF neighbour/deck-keepout constraints, and
the per-drone landing guidance **all run on these estimates**. True `world` state is used *only* for the
physics integration, true contact (touchdown via planted gear), and the separation *metric* — never for
a decision. Wired identically into **both** the kinematic `SwarmCoordinator` and the
`MujocoSwarmCoordinator`. `SensingConfig.perfect()` (zero noise, full comms) reproduces the old
truth baseline for honest A/B.

**Result (4 drones, ship moderate, MuJoCo physics, seeds 0–2):** perfect-sensing baseline and the
realistic no-cheats path both hold **100% all-landed, separation kept** (min-sep ≈ 0.92–0.94 m), and
stay 100% even with a finite 2.5 m comms range — at this drone count / spacing the estimator noise is
well within the CBF/holding margins. The honest hard regime (where partial observability finally bites)
remains the dense, short-comms MARL setting documented in `progress-state`.

## Safety layer (A3 — DONE, non-bypassable CBF + formal certificate)

`safety.py` (`SafetyFilter` / `SafetySpec` / `verify_separation`) makes the collision-avoidance guarantee
**structural** rather than incidental. The coordinator has exactly **one** path from a desired velocity
to the plant — `SafetyFilter.filter(...)` — and the MARL residual is added to the *desired* velocity
*before* that call, so **no learned command can ever reach the plant unfiltered**. Both runners route
through it (the old `_cbf_multi` is gone).

**The formal guarantee.** With the pairwise barrier `h = ||p_i - p_j||^2 - d_min^2`, a single-integrator
model, and a forward-Euler step `dt`, the CBF condition `2 d·(v_i - v_j) + α h ≥ 0` gives
`h⁺ ≥ (1 - α·dt) h`. So **if `0 < α·dt < 1` and `h ≥ 0` then `h⁺ ≥ 0`** — the safe set is
**forward-invariant in discrete time** (a provable statement, not just continuous-time). `SafetySpec`
*asserts* `α·dt < 1` at construction (with α=3, dt=0.05 → 0.15), so the precondition can't be violated
silently. The filter records a per-step **certificate**: the worst predicted `h⁺` over all active
constraints (`≥ 0` ⇒ provably collision-free that step).

**Scope / honesty.** The proof assumes the kinematic model + exact neighbour state. Under A1 (noisy,
stale, dropped broadcasts) and on the second-order MuJoCo quad, it holds **up to a bounded margin** set
by the sensing/tracking error; `SafetySpec.margin` tightens every `d_min` to absorb it. Verified
empirically with **`swarm verify`** (`verify_separation`), a seed sweep that asserts the *true* minimum
separation never drops below `d_min`:

| sensing | worst true min-sep | worst certificate | violations |
| --- | --- | --- | --- |
| perfect (exact inputs) | 0.734 m | **+0.049** (≥0, matches the proof) | 0 / 25 |
| realistic (no-cheats) | 0.725 m | −0.175 (bounded by noise) | 0 / 25 |
| realistic + margin 0.25 | 0.923 m | — | 0 / 25 |

i.e. with **exact** inputs the certificate is provably non-negative (the theory), and under realistic
no-cheats sensing the swarm still has **zero true separation violations** across the sweep — the residual
policy active underneath. Run it yourself: `swarm verify --drones 6 --scenario ship --seeds 25`
(`--margin 0.2` to trade conservatism for wider separation). Reported every `swarm run` as the "worst
one-step certificate" line.

**Sense-and-avoid of static obstacles (P3 swarm hook — DONE).** `SafetyFilter.filter(..., obstacles=...)`
folds **sensed static keep-out volumes** (the OSV superstructure) into the *same* exact CBF-QP as the
inter-drone/deck constraints — zero-velocity constraints with their own radius. `SwarmConfig.obstacles`
holds them as **deck-relative** `(dx, dy, radius)` offsets, and each drone places them at **its own deck
estimate** + offset (so the obstacle positions inherit that drone's deck-estimate noise — **no ground
truth**, just the A1 estimate + a known vessel map). Enable with `swarm ... --avoid` (auto-loads the
wheelhouse + bow). A/B (`swarm run --drones 5 --offshore --avoid`): **100% all-landed, separation kept,
worst obstacle clearance +0.25 m (kept clear every episode)** — the swarm skirts the superstructure and
still recovers. Additive (empty list = the original behaviour). **Multi-deck too:** `swarm multi --avoid`
puts per-vessel keep-outs at each drone's *assigned* deck estimate (`swarm multi --drones 6 --decks 2
--avoid` → 100% all-landed, kept clear).

## Cooperative consensus deck estimation (A2 — DONE, distributed Kalman-Consensus Filter)

`consensus.py` (`ConsensusDeckEstimator` / `ConsensusConfig`) makes the drones *cooperate* on tracking the
deck. Under the no-cheats sensing model the deck-measurement noise now **grows with range**, and past
`deck_vis_range` (6 m) a drone gets **no** direct fix at all — a far holding drone barely sees the marker,
while the cleared drone hovering over the deck has a sharp lock. A **Kalman-Consensus Filter**
(Olfati-Saber, CDC 2007) fuses these over the comms graph. Per drone, per step: **predict** a
constant-velocity deck model → **measurement update** (only if it currently sees the deck, with its own
range-dependent `R`) → **consensus update** `x ← x + γ·mean_{j∈N_i}(x_j − x_i)` pulling toward neighbours'
(one-step-stale) broadcast estimates. Fully decentralized (neighbour exchange only, no central node).
Enable with `--consensus` / `SwarmConfig(consensus=True)`; off by default (each drone uses its flat A1
estimate). Wired into both the kinematic and MuJoCo runners; the fused estimate feeds scheduling,
holding, and landing guidance.

**Result — controlled 5-drone experiment** (`scripts/eval_consensus.py`, deck_vis_range 6 m):

| drone | range | sees deck | raw measurement err | fused (consensus) err |
| --- | --- | --- | --- | --- |
| 0 | 1.1 m | yes | 0.180 m | 0.117 m |
| 1 | 2.5 m | yes | 0.293 m | 0.120 m |
| 2 | 4.1 m | yes | 0.410 m | 0.124 m |
| 3 | 7.8 m | **blind** | — (no view) | **0.137 m** |
| 4 | 8.3 m | **blind** | — (no view) | **0.137 m** |

Network mean: **raw 0.295 m → fused 0.127 m (57% lower)** — and the fused error beats even the *best*
single drone's raw fix (0.180 m), while the **blind** drones get a usable estimate purely from their
neighbours. That sharper shared deck reference propagates to the traffic: A/B on the full coordinator
(6 drones, ship) lifts the min-separation **0.86 → 0.94 m** (and **0.89 → 0.99 m** at a tight 4 m comms
range, where more drones are blind), all-landed staying 100%. Hits the roadmap's A2 metric: consensus
error < best single-drone estimate, and separation recovers toward the truth baseline under A1 noise.

## K moving platforms + dynamic re-tasking (A5 — DONE)

`multi_deck.py` (`MultiDeckCoordinator` / `MultiDeckConfig`) generalizes the recovery to **M drones onto K
moving decks**. The new problem is **assignment**, solved online:

- **Initial allocation** — a balanced **Hungarian** assignment (`optimal_assignment`, each deck replicated
  to `ceil(M/K)` virtual slots) gives the globally min-distance, load-balanced start.
- **Online re-tasking** — every `reassign_dt` a **decentralized auction**: each not-yet-committed drone
  picks the deck minimizing `distance + congestion·load` (on its own onboard estimates), switching only
  past a **hysteresis** margin. Decks sail on **distinct headings** so the fleet spreads out and the
  geometry genuinely changes.
- A drone **committed** to a deck (cleared for descent) is locked to it — unless that deck **fouls**.

Everything else is reused: per-deck `LandingScheduler` + `HoldingStack`, the no-cheats `SwarmSensing`
onboard view, and the non-bypassable global `SafetyFilter` (every drone avoids every other; deck keep-out
is around its *assigned* deck).

**Result (ship moderate):** nominal recovery is **100% all-landed, load balanced** (e.g. 9 drones → 3/3/3,
12 → 4/4/4), separation kept, with **zero re-tasks** — the balanced Hungarian start is already optimal
(the honest, correct behaviour). Re-tasking earns its keep under the scenario it exists for — a **fouled
deck** that closes mid-recovery (`swarm multi --foul-deck 2`):

| 9 drones, deck 2 fouls at t=4 s | all-landed | drones recovered | re-tasks |
| --- | --- | --- | --- |
| **re-tasking ON** (auction) | **100%** | 9 / 9 | ~3 |
| re-tasking OFF (static) | 0% | 6.2 / 9 | 0 |

i.e. when a deck closes, dynamic re-tasking moves its ~3 stranded drones onto the remaining decks and
**all 9 still land**; without it they are stranded. Run: `swarm multi --drones 9 --decks 3
[--foul-deck 2 --foul-time 4]`.

**Real-physics multi-deck (P4.1).** `multideck_world.py` + `multideck_runner.py` remove the kinematic
shortcut for this scenario: **K independent 6-DOF servo decks + N real X2 quadrotors** in one MuJoCo model,
flown by the validated geometric controller, landing via **true contact** on the assigned deck (distinct
on-deck spots so a second drone lands *beside* an already-landed one). The A5 *decision* layer — Hungarian
start, auction re-tasking, per-deck scheduler + holding stack, no-cheats sensing, the non-bypassable CBF —
is reused unchanged. **6 drones → 3 decks 6/6 (2/2/2), 4 drones → 2 decks 4/4 (2/2)**, balanced,
separation kept (min-sep 0.78–1.11 m), no collisions. Run: `swarm multi --engine mujoco --drones 6
--decks 3 [--watch]` (real MuJoCo viewer).

## GNN-based MARL policy (A4 — DONE, permutation-invariant + size-agnostic)

`marl_gnn.py` (`SwarmMARLGraphEnv`, `build_extractor_class`, `swarm gnn`) replaces the fixed-K MLP
observation with a **graph neural network** over the ego's neighbour graph: each in-range neighbour is
encoded by a shared MLP, the messages are **permutation-invariantly aggregated** (masked mean + max), and
combined with the ego encoding (a one-layer message-passing GNN over the star graph). It is **size-
agnostic** — a single trained policy runs at *any* number of drones. Trained with parameter-sharing PPO on
a **randomized-N** ego env (the ego is driven by the policy, the rest classical; N resampled per episode),
then deployed on every drone (CTDE).

**Cross-N generalization (the A4 success metric)** — the *one* policy, trained on N∈[6,10], deployed
unchanged at larger N vs the classical coordinator:

| swarm size | classical sep-kept | GNN sep-kept |
| --- | --- | --- |
| N=10 | 100% | 100% |
| N=14 | 46.7% | 46.7% |
| N=18 | 0% | 0% |

The policy generalizes across N with no retraining (permutation-invariant + size-agnostic) and **ties**
classical at every size. Honest: it does **not beat** classical — the dense/short-comms regime is
**comms-physics-limited** (you can't avoid what you can't see in time) and the learned residual sits under
the non-bypassable CBF (A3), so separation is already at the physical limit. The GNN's value is the SOTA
representation + cross-N generalization, not the separation metric — the recurring project theme that
RL/MARL wins only where the bottleneck is algorithmic, not physical. `swarm gnn --eval <zip> --sizes 10 14 18`.

## Learned active cooperative perception (A6 — DONE, attention fusion + VoI gating)

`active_perception.py` (`swarm active`) is the collaborative-perception research capstone (V2VNet /
Where2comm). The A2 consensus fuses neighbour deck fixes by **inverse-variance** weighting — optimal for
homogeneous Gaussian noise, but (a) fooled by a **confident outlier** (a reflection/wrong decode reported
with high confidence), and (b) blind to bandwidth. Two learned/active pieces fix that:

- **Attention fusion** (`AttentionFusion`, torch) — a permutation-invariant attention over the fix set
  whose output is a **convex combination of the actual fixes** (it cannot hallucinate a position). Its
  decisive input is each fix's **deviation from the group median** — a translation-invariant *agreement*
  feature inverse-variance weighting structurally can't use — so it learns to **reject confident outliers**.
  Trained supervised (~4 s CPU; `runs/active/fusion.pt`).
- **Value-of-information gating** (`select_broadcasters`) — under a bandwidth budget of B messages, pick the
  B broadcasters that most cut the fused variance (greedy by expected precision over the comms graph).

| regime | equal-mean | inverse-variance (consensus) | **learned** |
|---|---|---|---|
| heterogeneous (confident outliers) | 0.40 m | **0.47 m** (trusts them) | **0.16 m** (3×) |
| homogeneous (Gaussian) | 0.15 m | 0.12 m | **0.11 m** (tie) |

Learning **wins where the Gaussian assumption breaks and ties where it holds** — the same honest lesson as
the MARL, from the perception side. Value-of-information selection reaches near-full accuracy (0.127 m) at
**3 of 8** messages (fuse-all = 0.119 m) and beats random selection at every budget — the Where2comm
bandwidth story. Run: `swarm active` (train + A/B) or `swarm active --eval runs/active/fusion.pt`.

## Cooperative perception — real per-drone onboard vision (CP — DONE)

The realism upgrade over the modeled `SwarmSensing` deck estimate: each drone runs **real vision on its
own rendered downward camera** (`world.py` now stamps a `cam_i` on every drone + a bright landing pad on
the deck; `vision.py SwarmVision`). A drone detects the saturated-white pad (border-gated centroid) and
back-projects it through the pinhole + rangefinder altitude to a **world-frame deck fix** (validated
accurate to ~1–2 cm). A drone whose camera *cannot* see the pad — it has drifted out, the deck left the
field of view, or it is too tilted — returns **None (blind)** and must rely on the cooperative consensus.
The visual fixes feed the **A2 consensus filter**, so a blind drone lands on the deck location its
**neighbours** can see. Enable with `swarm run --vision` (MuJoCo engine only; per-drone rendering, small N).

**Cooperative-perception proof** (`scripts/eval_cooperative_perception.py` — 5 drones, real vision, 3
deliberately blind, wrong 3 m prior):

| drones | isolated (no sharing) | cooperative (shared real fixes) |
| --- | --- | --- |
| seeing the pad | 0.01 m | 0.26 m |
| **blind** (pad out of FOV) | **4.24 m** (stuck at prior) | **0.34 m** (recovered) |

Blind drones recover the deck from neighbours' *real visual* fixes — **4.24 m → 0.34 m**. (Sharing costs
the seeing drones a little — the consensus blends toward neighbours — but the net is a large win.)

**Fast-deck tracking (P2.1+P2.2).** The initial version failed on a translating deck (drones lost the pad,
0% landing). Three fixes: **(P2.1)** nadir-stabilized **gimbal cameras** (tilt-invariant fix — 0.008 m at
0deg and 20deg roll); **(P2.2)** a wider **120deg FOV** (the holding ring radius ~= altitude, so a 90deg
camera saw nothing) plus the consensus filter's **constant-velocity prediction** (extrapolates the deck
between vision frames + feeds the deck velocity to the guidance). Together these cut the **deck-estimate
error 1.99 m -> 0.22 m (9x)** and raised **vision landing on the fast 0.4 m/s deck from 0% to ~25-33%
all-landed**.

**Heterogeneous fleet (P2.4).** Above, a "blind" drone still *has* a camera (the pad is merely out of its
FOV). P2.4 adds a true **camera-less** class of drone that carries **no camera at all** — the realistic
asymmetric fleet (a few well-equipped sensor drones + cheap relay drones). `SwarmVision(camera_ids=...)`
renders only the listed drones; camera-less drones return `None` every frame and recover the deck purely
from neighbours' broadcast fixes via the A2 consensus filter. Enable with `swarm run --vision --cameras N`
(only the first `N` drones carry a camera). A/B (`scripts/eval_heterogeneous_perception.py`, 5 drones,
cameras {0,1}):

| drones | isolated (no sharing) | cooperative (shared fixes) |
|---|---|---|
| {0,1} **camera** | 0.01 m | 0.26 m (drift toward shared estimate) |
| {2,3,4} **camera-less** | **4.24 m** (stuck at prior) | **0.34 m** (recovered) |

The camera-less majority recovers the deck to 0.34 m purely from the two camera drones — a 12x gain — at a
small coupling cost to the camera drones (0.01 → 0.26 m), the same trade-off seen in the blind-recovery
case.

**Honest status.** The **perception** now tracks a moving deck (0.22 m) via shared vision + prediction, the
cooperative-recovery mechanism is proven (blind drones 4.24 -> 0.34 m), and it extends to a heterogeneous
fleet (camera-less drones 4.24 -> 0.34 m). But **full-swarm vision *landing* on a moving deck is only
partially reliable** (~1/3 land, high variance) — the modeled-sensing swarm (continuous estimate) lands
100%; vision-grade, intermittent, partial-coverage estimates + a translating deck cap the closed-loop
landing. Reliable full-swarm moving-deck vision landing (sequential-touchdown coordination tuned for
vision-grade estimates) is the hard open part.
