# Safety layer — sense-and-avoid + contingency failsafes (Phase 3)

A setpoint tracker that lands beautifully on a clear pad is not a *safe* field system. Real autopilots
(PX4/ArduPilot) add two things this project previously lacked: **sense-and-avoid** for static structure,
and a **contingency failsafe state machine**. Phase 3 adds both as a small, rigorous, estimate-driven
module (`src/drone_landing/safety/`). Everything here is **decision-only and additive** — it reads onboard
estimates/sensor returns (never ground truth, per the no-cheats charter) and does not touch the validated
ground/ship/offshore landing loops.

Run the demos:

```
drone safety                 # both demos
drone safety --demo avoid    # sense-and-avoid past the OSV superstructure
drone safety --demo contingency   # the failsafe FSM over a fault timeline
```

(or the standalone `scripts/eval_obstacle_avoidance.py` / `scripts/eval_contingency.py`.)

## 1. The obstacle field (P3.1) — `obstacles.py` + collidable geoms

The offshore-support-vessel carries a **wheelhouse, mast and raised bow** fore of the helideck
(`assets/mujoco/worlds/x2_landing_offshore.xml`). Two complementary representations:

* **Physical (true contact):** the wheelhouse, its base, the mast and the bow are **collidable geoms**
  (`contype=1`) — solid keep-out volumes. A drone that flies into them physically collides and the episode
  terminates `hit_structure` (a real crash), so the avoidance guard has genuine teeth rather than guarding a
  ghost. The clear-deck ground/ship worlds have no such geoms (the obstacle set is empty → no behaviour
  change). Verified: nominal offshore landing is **unchanged (~100% on 8 ep)** — the validated approach
  stays high until aligned over the pad, so it never goes near the fore structure (the guard is *latent*).
* **Software model for the planner:** `ObstacleField.offshore_osv()` mirrors the same geometry as simple
  vertical primitives — a box footprint (wheelhouse/bow), a thin circle (mast), each spanning `[z_min,
  z_max]` — for exact ray-casting (ray–circle, ray–box slab) and signed-distance queries; pure geometry,
  fast and unit-testable, and what the range sensor scans against in-loop.

## 2. Onboard obstacle sensing (P3.2) — `RangeSensor`

A real drone is not *handed* obstacle coordinates; it senses surfaces. `RangeSensor` models an **onboard
2-D scanning rangefinder** (a LiDAR slice at the drone's altitude): it casts `n_beams` rays around the
drone, returns the range to the nearest surface within `max_range` (with a blind zone, **Gaussian range
noise** and random **dropout**), and gives back **world-frame surface points** — *not* obstacle identities
or centres. That is the no-cheats contract: avoidance runs on what a sensor returns, exactly as the
controller runs on the EKF estimate. It models a **horizontal scan plane** (2.5-D): a beam at altitude `z`
only strikes obstacles whose vertical span contains `z` (a fixed 2-D LiDAR; a tilting/3-D scan is a future
refinement).

## 3. Higher-order CBF sense-and-avoid (P3.3 / P3.4) — `avoid.py`

The swarm's inter-drone filter is a **single-integrator** CBF (it bends a *velocity*). A real quadrotor is
acceleration-controlled, so a static-obstacle barrier `h(p) = ||p − p_o||² − R²` has **relative degree 2**
in the acceleration command — a velocity filter would act one integrator too late. `HOCBFAvoider`
implements the standard **higher-order CBF** (Xiao & Belta 2019) for relative degree 2:

```
psi0 = h
psi1 = h_dot + a1 * h
require   psi1_dot + a2 * psi1 >= 0
```

For a double integrator and a static obstacle this is linear in the acceleration `a`:

```
g . a  >=  -2||v||^2 - 2 a1 (p-p_o).v - a2 ( 2(p-p_o).v + a1 h ),     g = 2 (p - p_o)
```

solved as the minimal-intervention QP `min ||a − a_des||² s.t. g_i·a ≥ rhs_i` (plus an accel-magnitude cap)
with **Hildreth's dual** — the same exact small-QP method as the swarm filter, no external solver. The
sensed surface points are first clustered (`cluster_returns`) so one wall is not double-counted.

**Actuation latency (P3.4).** A command takes `latency` seconds to bite, so the barrier is evaluated at the
**look-ahead** state `p + v·latency` (where the vehicle *will* be) and the keep-out radius is inflated by
the drone radius + a margin — a sound, cheap robustification.

**Result** (`drone safety --demo avoid`, level transit past the superstructure, noisy onboard scan):

| mode | reached pad | collided | min clearance to surface |
|---|---|---|---|
| no-avoid | yes | **yes** | **−0.24 m** (inside a structure) |
| AVOID (HOCBF) | yes | **no** | **+0.32 m** |

The setpoint tracker flies into the wheelhouse; the CBF keeps clearance ≥ 0 throughout *and* still reaches
the pad.

**Honest scope.** A reactive CBF guarantees *safety* (forward-invariance of the clear set) but can
**deadlock** on an obstacle sitting directly between the drone and the goal — a known limitation; a side
must exist (real decks are approached from a clear sector, modelled by the offset approach). Global liveness
needs a planner on top; the CBF is the last-line safety filter, not the planner.

## 4. Contingency / failsafe FSM (P3.5) — `contingency.py`

A priority-ordered state machine that overrides guidance on a fault, mirroring PX4/ArduPilot failsafes:

```
ROTOR_OUT > LOW_BATTERY > GEOFENCE > LOST_COMMS > OBSTACLE_ABORT > NOMINAL
```

* **ROTOR_OUT** — a quad with a dead rotor is underactuated (see `docs/` rotor-out note). We do **not**
  attempt a precision landing; the contingency is a **controlled spinning descent** at a bounded sink rate
  (graceful degradation that bounds the impact). This is the agreed near-term handling; a full
  Mueller–D'Andrea spinning-LQR "land on 3 rotors" is a deferred stretch.
* **LOW_BATTERY** — below the energy reserve → return-to-launch, then land.
* **GEOFENCE** — outside the allowed cylinder/ceiling → steer back inside.
* **LOST_COMMS** — datalink stale beyond a timeout → loiter at a safe altitude.
* **OBSTACLE_ABORT** — a sensed obstacle inside the abort radius → climb-and-hold (break off the approach),
  with hysteresis so it does not chatter.

`drone safety --demo contingency` runs a scripted mission that injects each fault and prints the state the
FSM enters, demonstrating the priority ordering (e.g. a simultaneous rotor failure + low battery resolves to
ROTOR_OUT).

## Live-loop integration (`drone run/watch ... --avoid`)

The Phase 3 layer is exposed standalone via `drone safety` (15 tests in `tests/test_safety.py`) **and is
now also spliced into the live landing autopilot** behind the opt-in `--avoid` flag. In the loop it runs in
the **deck-relative frame** (obstacles are fixed offsets from the deck; the drone's deck-relative pose comes
from the EKF — no ground truth): the onboard `RangeSensor` scans the offshore superstructure, the
**higher-order CBF** bends the horizontal command to keep clear, and the **contingency FSM** breaks off the
approach (`AVOID_ABORT`, climb-and-hold) inside the abort radius.

It is a **latent guard**: with no obstacle within `avoid_engage` (1.6 m) it is a pure passthrough, so the
validated controller is untouched on a clear deck — **offshore `--avoid` lands 88 % = offshore baseline**.
In a nominal offshore landing the drone descends onto the pad clear of the fore superstructure, so the
guard rarely engages; it activates when the approach strays near the structure (the standalone
`drone safety --demo avoid` shows the full deflection: −0.24 m collision → +0.32 m clearance).

```
drone run offshore --avoid          # in-loop sense-and-avoid + obstacle-abort on the OSV superstructure
drone watch offshore --avoid        # watch it
```

The FSM's battery / lost-comms / geofence branches stay exercised **standalone** (`drone safety
--demo contingency`) — the sim carries no battery/datalink signal to drive them in-loop, so obstacle-abort
and rotor-out are the in-loop branches. The analogous **swarm** hook (feeding sensed obstacle points into
the swarm `SafetyFilter`, which already accepts static constraints) is the remaining small follow-on.
