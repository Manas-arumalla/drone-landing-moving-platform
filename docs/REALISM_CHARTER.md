# Realism Charter

The governing constraint of this project: **the simulation must reflect what really happens in the
real world — no cheats.** Every design decision obeys these rules. Reviewers and future contributors
should treat a violation of this charter as a bug.

| Rule | Concrete requirement |
| --- | --- |
| **No privileged state in the loop** | The controller/policy consumes only what real sensors provide: camera frames, IMU (accel+gyro), barometer, downward rangefinder, optional noisy GPS — and the EKF output derived from them. The platform's pose/velocity is **estimated from the camera**, never read from the simulator ground truth. |
| **Ground truth is training/eval-only** | Used only to (a) compute evaluation metrics and (b) train the RL *critic* (asymmetric actor-critic). The *actor* still consumes only sensor-derived observations. |
| **Real sensor models** | Noise, bias, drift, finite rate, latency, quantization; camera FOV limits, motion blur, marker occlusion/dropout. |
| **Real actuation** | Motor first-order lag, thrust/torque maps, saturation, control allocation. Use the Skydio X2's validated actuator model. |
| **Real aerodynamics** | Body & rotor drag, ground effect near touchdown, wind field + gusts (Dryden), ship air-wake turbulence. |
| **Real contact** | Touchdown is true frictional contact — the drone can bounce, slide, or tip over. No weld/teleport lock in the evaluation path. Success = genuinely resting on the deck through continued motion for a settle window. |
| **Honest platform motion** | The platform's own motion is prescribed from a validated model (bounded-accel random trajectory for the rover; wave-spectrum + RAO seakeeping for the ship). Legitimate because the ship/rover is far heavier than the drone; the drone↔deck *interaction* is always pure contact physics. |
| **Real timing** | Multi-rate cascade like a real flight controller: rate loop ~500 Hz–1 kHz, attitude ~250 Hz, position/MPC ~50–100 Hz, perception at camera rate (~30–60 Hz) with latency, estimator at IMU rate. |

## Consequences for existing code

- `envs/mujoco_env.py` currently feeds **ground-truth platform position** into the observation and offers a
  `lock_after_success` weld. Both are removed/re-routed: platform state must come through perception + EKF, and the
  evaluation path must be lock-free (success via real contact + friction + settle window).
- A separate **privileged** observation (ground truth) is exposed *only* to evaluation and the RL critic, never to the actor.
