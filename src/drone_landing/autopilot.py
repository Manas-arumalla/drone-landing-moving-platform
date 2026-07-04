"""Vision-based landing autopilot: the integrated, deployable closed-loop stack.

Wires perception (ArUco) -> estimation (Kalman filter) -> planning (landing supervisor) ->
control (geometric SE(3)). It consumes only onboard signals — camera images, IMU/AHRS, rangefinder,
and a gear-contact count — and emits four motor thrusts. No simulator truth enters the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from drone_landing.control import GeometricController, IBVSGuidance
from drone_landing.estimation import RelativeStateEKF, accel_world, quat_to_rotmat
from drone_landing.perception import ArucoDetector, CameraModel
from drone_landing.perception.aruco_detector import board_normal_world
from drone_landing.perception.optical_flow import FlowConfig, OpticalFlowVelocity
from drone_landing.planning import DeckMotionPredictor, LandingSupervisor


@dataclass(frozen=True)
class AutopilotConfig:
    cam_offset_body: tuple[float, float, float] = (0.0, 0.0, -0.09)
    cam_period: int = 3             # control steps between camera frames (~33 Hz at 100 Hz)
    track_timeout_steps: int = 15   # consider the target "tracked" within this many steps
    reproj_max: float = 3.0         # px  reject detections worse than this
    min_markers: int = 2
    init_min_markers: int = 4       # stricter gate for the first (initialising) detection
    init_reproj_max: float = 2.0
    range_max: float = 4.0          # m   use rangefinder below this
    hover_thrust: float = 3.37      # N per motor, used while searching (pre-init)
    use_mpc: bool = False           # use the predictive MPC for horizontal tracking (vs PD/integral)
    use_ibvs: bool = False          # use IBVS guidance (image position + optical-flow velocity)
    use_minsnap: bool = False       # flatness-based minimum-snap approach trajectory + feedforward
    rl_policy_path: str | None = None  # trained residual-RL policy (.zip); horizontal residual on geometric
    rl_algo: str = "ppo"            # ppo | recurrent_ppo (matches the trained checkpoint)
    failed_rotor: int | None = None    # fault-tolerance demo: engage 3-rotor allocation for this rotor
    fail_time: float = 3.0             # s   when fault detection engages FT allocation
    rotor_out_sink: float = 0.6        # m/s bounded descent for the rotor-out spinning-descent contingency
    rotor_out_mode: str = "descent"    # "descent" = Option-D bounded spinning descent (default, wind-robust);
                                       # "floquet" = averaged-precession controller that LANDS wind-off (~50%)
    use_dob: bool = False              # wind-aware: disturbance-observer feedforward (cancels gusts)
    use_markerless: bool = False       # markerless deck fallback when the ArUco code can't be decoded
    use_cnn_markerless: bool = False   # B3: learned CNN deck detector as the markerless fallback
    use_shield: bool = False           # B2: HJ reachability shield clamps descent to stay in the safe set
    use_avoid: bool = False            # P3: in-loop sense-and-avoid (HOCBF) + obstacle-abort contingency
    avoid_engage: float = 1.6          # m  nearest-obstacle distance at which the avoider engages (else
                                       #     passthrough -> the validated controller is untouched when clear)
    avoid_abort: float = 0.7           # m  nearer than this -> contingency climb-and-hold (break off)
    cnn_weights_path: str | None = None  # trained DeckCNN weights (default runs/cnn/deck_cnn.pt)
    use_green_deck: bool = False    # maritime: time the commit to a low-motion deck window
    commit_descent_time: float = 0.6  # s  descent duration the green-deck check must stay quiescent
    mpc_confident_std: float = 0.20   # m  engage MPC only when EKF horizontal std is below this


class VisionLandingAutopilot:
    def __init__(self, mass: float, inertia: np.ndarray, camera: CameraModel,
                 control_dt: float, config: AutopilotConfig | None = None,
                 detector: ArucoDetector | None = None, ekf: RelativeStateEKF | None = None,
                 supervisor: LandingSupervisor | None = None,
                 controller: GeometricController | None = None,
                 obstacle_field=None):
        self.cfg = config or AutopilotConfig()
        self.camera = camera
        self.control_dt = control_dt
        self.detector = detector or ArucoDetector(camera)
        self.ekf = ekf or RelativeStateEKF()
        if supervisor is None:
            from drone_landing.planning import SupervisorConfig
            supervisor = LandingSupervisor(SupervisorConfig(green_deck=self.cfg.use_green_deck))
        self.supervisor = supervisor
        self.controller = controller or GeometricController(mass, inertia, control_dt=control_dt)
        self.mpc = None
        if self.cfg.use_mpc:
            from drone_landing.control.mpc import HorizontalMPC
            self.mpc = HorizontalMPC()
        # Flatness-based minimum-snap approach planner (Mellinger & Kumar): plans the relative
        # position to a rendezvous with the platform and supplies acceleration feedforward through
        # the same a_xy_override path as the MPC (drop-in comparable; commit logic untouched).
        self.minsnap = None
        if self.cfg.use_minsnap:
            from drone_landing.planning.minsnap import MinSnapTracker
            self.minsnap = MinSnapTracker(control_dt=control_dt)
        # IBVS uses optical flow (robust velocity) instead of the EKF's differentiated velocity.
        # The MPC also consumes the flow velocity: its decisive predictive commands amplify the EKF's
        # spiky differentiated velocity into fly-offs, so it needs the same clean relative-velocity
        # term — and so does the min-snap tracker (any predictive tracker does).
        self.ibvs = IBVSGuidance() if self.cfg.use_ibvs else None
        self.flow = (OpticalFlowVelocity(camera, FlowConfig(gyro_comp=0.0))
                     if (self.cfg.use_ibvs or self.cfg.use_mpc or self.cfg.use_minsnap) else None)
        # Maritime green-deck predictor: estimates the deck heave from the relative-altitude signal
        self.deck = DeckMotionPredictor() if self.cfg.use_green_deck else None
        # Trained residual-RL policy (optional): horizontal residual on the geometric baseline, run on
        # the real EKF estimate (the honest full-pipeline test of the learned policy).
        self.rl = None
        if self.cfg.rl_policy_path:
            from drone_landing.rl.policy import ResidualPolicy
            self.rl = ResidualPolicy(self.cfg.rl_policy_path, algo=self.cfg.rl_algo)
        # Wind-aware disturbance observer (optional): estimates the wind accel and feeds it forward.
        self.dob = None
        if self.cfg.use_dob:
            from drone_landing.control.disturbance import DisturbanceObserver
            self.dob = DisturbanceObserver(mass, control_dt=control_dt)
        self._hover_thrust = mass * 9.81
        self._last_thrust = self._hover_thrust
        # Markerless fallback: locate the deck pad when ArUco can't be decoded (keeps tracking alive).
        self.markerless = None
        if self.cfg.use_cnn_markerless:                 # B3: learned detector (drop-in: .detect->.found/.rel_xy)
            from drone_landing.perception.cnn_detector import CNNDeckDetector
            self.markerless = CNNDeckDetector(camera, weights_path=self.cfg.cnn_weights_path)
        elif self.cfg.use_markerless:
            from drone_landing.perception.markerless import MarkerlessDeckTracker
            self.markerless = MarkerlessDeckTracker(camera)
        self.shield = None                              # B2: reachability safe-descent shield
        if self.cfg.use_shield:
            from drone_landing.control.reachability import LandingReachability
            self.shield = LandingReachability()
        # Rotor-out averaged/precession controller (lands the dead-rotor drone wind-off; opt-in).
        self.rotor_floquet = None
        if self.cfg.rotor_out_mode == "floquet" and self.cfg.failed_rotor is not None:
            from drone_landing.control.rotor_out_floquet import RotorOutFloquet
            self.rotor_floquet = RotorOutFloquet(mass, inertia, self.controller.alloc,
                                                 self.cfg.failed_rotor, control_dt=control_dt)
        # P3 in-loop sense-and-avoid: an onboard range sensor + higher-order CBF that keeps the drone clear
        # of static structure (e.g. the offshore superstructure), and an obstacle-abort contingency. Works
        # in the DECK-RELATIVE frame (obstacles are fixed offsets from the deck; the drone's deck-relative
        # pose comes from the EKF -> no ground truth). Latent guard: engages only near an obstacle.
        self.obstacle_field = obstacle_field
        self.avoid = self.range_sensor = self.contingency = None
        if self.cfg.use_avoid and obstacle_field is not None:
            from drone_landing.safety import (
                AvoidConfig,
                ContingencyConfig,
                ContingencySupervisor,
                HealthStatus,
                HOCBFAvoider,
                RangeSensor,
                RangeSensorConfig,
                cluster_returns,
            )
            self.range_sensor = RangeSensor(obstacle_field, RangeSensorConfig(max_range=5.0))
            self.avoid = HOCBFAvoider(AvoidConfig(a_max=4.0, drone_radius=0.25, margin=0.15))
            self.contingency = ContingencySupervisor(
                ContingencyConfig(obstacle_abort=self.cfg.avoid_abort,
                                  obstacle_clear=self.cfg.avoid_engage, abort_climb_alt=2.5))
            self._HealthStatus = HealthStatus
            self._cluster = cluster_returns
        self._avoid_rng = np.random.default_rng(0)
        self._nearest_obstacle = float("inf")
        self.cam_dt = self.cfg.cam_period * control_dt
        self.cam_offset = np.asarray(self.cfg.cam_offset_body, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.ekf.reset()
        self.supervisor.reset()
        self.controller.reset()
        self.controller.failed_rotor = None   # re-engaged after fail_time each episode (fault demo)
        if self.mpc is not None:
            self.mpc.reset()
        if self.minsnap is not None:
            self.minsnap.reset()
        if self.deck is not None:
            self.deck.reset()
        if self.rl is not None:
            self.rl.reset()
        if self.dob is not None:
            self.dob.reset()
        if self.contingency is not None:
            self.contingency.reset()
            self.avoid.reset()
        if self.rotor_floquet is not None:
            self.rotor_floquet.reset()
        self._nearest_obstacle = float("inf")
        self._last_thrust = self._hover_thrust
        self.prev_gray = None
        self.v_flow = np.zeros(2)   # latest optical-flow relative velocity (IBVS)
        self.deck_normal = None     # low-passed deck-surface normal from the ArUco PnP rotation
        self.k = 0
        self.last_good_k = -10_000
        self.state = "SEARCH"

    def wants_frame(self) -> bool:
        """True when the autopilot needs a camera frame this step (so the caller can render it)."""
        return self.k % self.cfg.cam_period == 0

    def step(self, image, sensors, support_feet: int) -> np.ndarray:
        cfg = self.cfg
        R_ahrs = quat_to_rotmat(sensors.attitude_quat)

        # Wind-aware DOB: estimate the external (wind) acceleration from the IMU vs the commanded thrust.
        dist_ff = None
        if self.dob is not None:
            dist_ff = self.dob.update(accel_world(sensors.accel, sensors.attitude_quat),
                                      self._last_thrust, R_ahrs[:, 2])

        # Fault-tolerance: once a rotor failure is detected, switch the allocator to 3-rotor mode.
        if cfg.failed_rotor is not None and self.k * self.control_dt >= cfg.fail_time:
            self.controller.failed_rotor = cfg.failed_rotor

        # The nested centre marker keeps the platform observable through touchdown, so the filter
        # runs continuously (no blind-commit freeze). Velocity is clamped below as a safety net.
        if self.ekf.initialized:
            self.ekf.predict(self.control_dt, accel_world(sensors.accel, sensors.attitude_quat))

        if image is not None:
            det = self.detector.detect(image)
            # accept a well-conditioned grid pose, OR the nested centre marker (n=1) at close range
            usable = det.found and det.reproj_error < cfg.reproj_max and (
                det.source == "center" or det.n_markers >= cfg.min_markers)
            if usable:
                # the gimbal camera is world-level (nadir), so its frame is identity, not the drone
                # attitude; the mount offset still rotates with the drone body
                rel = self.camera.opencv_to_world(det.tvec_cam, np.eye(3)) + R_ahrs @ self.cam_offset
                if not self.ekf.initialized:
                    # initialise only from the grid board (well constrained)
                    if det.source == "grid" and det.n_markers >= cfg.init_min_markers \
                            and det.reproj_error < cfg.init_reproj_max:
                        self.ekf.reset(r0=rel, v0=np.zeros(3))
                        self.last_good_k = self.k
                else:
                    if self.ekf.update_aruco(rel):
                        self.last_good_k = self.k
                # Deck-surface normal from the SAME PnP fix (grid only — the single centre marker's
                # rotation suffers the planar-pose ambiguity). Low-passed so it estimates the deck's
                # MEAN normal (rejects seaway wobble); consumed by the attitude-matched touchdown.
                if det.source == "grid":
                    n_w = board_normal_world(det.rvec_cam, self.camera)
                    if self.deck_normal is None:
                        self.deck_normal = n_w
                    else:
                        n = self.deck_normal + 0.05 * (n_w - self.deck_normal)
                        self.deck_normal = n / max(float(np.linalg.norm(n)), 1e-9)
                # IBVS: robust relative velocity from the fiducial's optical flow (nadir gimbal ->
                # identity frame, no gyro compensation), held between camera frames
                if self.flow is not None and self.prev_gray is not None:
                    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
                    bbox = self.flow.bbox_from_corners(det.corners)
                    v_meas, ok = self.flow.estimate(self.prev_gray, gray, bbox, sensors.range,
                                                    sensors.gyro, np.eye(3), self.cam_dt)
                    if ok:
                        self.v_flow = v_meas
            # Markerless fallback: if ArUco wasn't usable but the EKF is initialised and we have a range,
            # locate the bright deck pad and fuse its centroid as a coarse position fix — keeps the
            # platform observable (and `tracked` alive) through brief marker loss.
            if (self.markerless is not None and not usable and self.ekf.initialized
                    and sensors.range_valid and sensors.range < cfg.range_max):
                ml = self.markerless.detect(image, sensors.range)
                if ml.found:
                    rel_xy = ml.rel_xy + (R_ahrs @ self.cam_offset)[:2]
                    if self.ekf.update_markerless(rel_xy):
                        self.last_good_k = self.k
            if self.flow is not None:
                self.prev_gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)

        if self.ekf.initialized and sensors.range_valid and sensors.range < cfg.range_max:
            self.ekf.update_range(sensors.range)   # altitude stays observable through touchdown

        # Downward optical-flow: fuse the deck-relative horizontal velocity (flow_vel = drone − deck vel,
        # so rel-velocity = −flow_vel). It works in the <0.3 m close-range blind-zone where the ArUco is
        # too zoomed to decode, so it keeps the relative-VELOCITY estimate alive to touchdown. No ground
        # truth: the measurement is the modeled flow sensor (true rel-velocity + noise), like the IMU/range.
        # NOTE: flow is a velocity fix, NOT a position fix, so it deliberately does NOT refresh
        # `last_good_k` (the marker-tracking liveness signal) — keeping `tracked` alive on velocity alone
        # would mask genuine marker loss and change the validated nominal commit behavior.
        if self.ekf.initialized and getattr(sensors, "flow_valid", False):
            self.ekf.update_velocity_xy(-np.asarray(sensors.flow_vel, dtype=float))

        if self.ekf.initialized:  # bound the relative-velocity estimate (limits spike-driven fly-off)
            self.ekf.x[3:] = np.clip(self.ekf.x[3:], -1.2, 1.2)

        tracked = (self.k - self.last_good_k) <= cfg.track_timeout_steps
        pos_std = float(np.sqrt(np.mean(np.diag(self.ekf.P)[:2])))  # horizontal uncertainty

        # Maritime: track the deck heave from the relative-altitude signal. rel_pos[2] = z_deck -
        # z_drone, so its oscillation is the deck heave directly (the predictor's linear-trend term
        # absorbs the drone's own descent), and heave_rate() returns the deck's vertical velocity.
        # From it we get the deck-velocity feedforward and the green (low-motion) window go/no-go.
        green_light = True
        deck_vz = 0.0
        if self.deck is not None and self.ekf.initialized:
            self.deck.update(self.k * self.control_dt, float(self.ekf.rel_pos[2]))
            green_light = self.deck.in_green_window(cfg.commit_descent_time)
            deck_vz = self.deck.heave_rate(0.0)

        if self.ekf.initialized:
            cmd = self.supervisor.update(self.ekf.rel_pos, self.ekf.rel_vel, pos_std,
                                         tracked, support_feet, self.control_dt,
                                         green_light=green_light)
            self.state = cmd.state
            if cmd.cut:
                ctrl = np.zeros(4)
            elif self.controller.failed_rotor is not None:
                # ROTOR-OUT CONTINGENCY (Option D — controlled spinning descent). A quad with a dead rotor
                # is underactuated and spins about yaw; *chasing* the deck under that spin smears the thrust
                # vector around the rotation and flies the drone off (the old behaviour -> out_of_bounds).
                # The contingency instead **stops the horizontal chase** (zero horizontal target) and holds
                # the thrust axis vertical while descending at a BOUNDED sink rate, so the vehicle comes
                # down roughly in place with a bounded impact — graceful degradation, NOT a precision
                # landing. A true 3-rotor LANDING needs the Mueller–D'Andrea LQR-around-the-spin design
                # (deferred; see docs + rotor-out memory). The descent is shielded below if --shield is on.
                self.state = "ROTOR_OUT"
                if self.rotor_floquet is not None:
                    # Averaged-precession controller: steer the spin-averaged thrust axis to track the deck
                    # and centre-then-descend to an actual landing (works wind-off; see the module docstring).
                    sink = -cfg.rotor_out_sink
                    if self.shield is not None:
                        sink = max(sink, -self.shield.safe_descent_speed(max(-float(self.ekf.rel_pos[2]), 0.0)))
                    # downward optical-flow velocity (deck-relative) feeds the blind close-range phase where
                    # the ArUco is lost — the sensing upgrade that arrests the terminal blind-descent drift.
                    flow = sensors.flow_vel if getattr(sensors, "flow_valid", False) else None
                    ctrl = self.rotor_floquet.control(self.ekf.rel_pos, self.ekf.rel_vel, R_ahrs,
                                                      sensors.gyro, vz_des=sink, hold_xy=not tracked,
                                                      flow_vel=flow)
                else:
                    rel_descent = np.array([0.0, 0.0, float(self.ekf.rel_pos[2])])  # no horizontal chase
                    rel_vel_descent = np.array([0.0, 0.0, float(self.ekf.rel_vel[2])])
                    sink = -cfg.rotor_out_sink
                    if self.shield is not None:                                      # stay in the safe set
                        sink = max(sink, -self.shield.safe_descent_speed(max(-float(self.ekf.rel_pos[2]), 0.0)))
                    ctrl = self.controller.compute_rotor_out(rel_descent, rel_vel_descent, R_ahrs,
                                                             sensors.gyro, vz_des=sink)
            elif (not tracked) and self.state not in ("COMMIT", "SECURED"):
                # vision lost in flight: climb straight up to widen the FOV and re-acquire, rather
                # than chase a stale/diverging horizontal estimate (which would fly the drone away)
                ctrl = self.controller.compute(self.ekf.rel_pos, self.ekf.rel_vel, R_ahrs,
                                               sensors.gyro, vz_des=0.4, hold_level=True)
            else:
                # during the committed final descent (pre-contact), hold platform velocity rather
                # than chase the position estimate (the marker has usually left the FOV by now)
                vhold = self.state == "COMMIT" and not cmd.press
                # IBVS exception: while the fiducial is STILL tracked, keep closing the loop on the robust
                # IMAGE position through the commit descent instead of switching to open-loop velocity-hold.
                # On a fast *translating* deck the velocity-hold lets the deck slide out (~0.26 m drift,
                # landing just outside the success radius); image-position servoing holds centring. Falls
                # back to velocity-hold the moment the marker is lost (so no stale-position fly-off).
                ibvs_commit = self.ibvs is not None and vhold and tracked
                if ibvs_commit:
                    vhold = False
                # With IBVS, use the robust optical-flow velocity throughout the control path (incl.
                # the commit velocity-hold), not the EKF's spiky differentiated velocity.
                rel_vel_ctrl = self.ekf.rel_vel
                if self.ibvs is not None:
                    rel_vel_ctrl = np.array([self.v_flow[0], self.v_flow[1], self.ekf.rel_vel[2]])
                # Horizontal command: IBVS (image position + optical-flow velocity) or MPC (predictive)
                # override the geometric PD during the normal approach/descent.
                a_xy_override = None
                if not cmd.press and not vhold:
                    if self.rl is not None:
                        # trained residual policy on the real EKF estimate (geometric PD + learned residual)
                        a_xy_override = self.rl.horizontal_accel(self.ekf.rel_pos, self.ekf.rel_vel,
                                                                 R_ahrs, sensors.gyro)
                    elif self.ibvs is not None:
                        a_xy_override = self.ibvs.horizontal_accel(self.ekf.rel_pos[:2], self.v_flow)
                    elif self.mpc is not None and tracked and pos_std < cfg.mpc_confident_std:
                        # Confidence-gated MPC: engage the decisive predictive tracker only once the
                        # estimate is confident (tracked + low position std). During acquisition the
                        # estimate is still settling, and the MPC's aggressive intercept would amplify
                        # that early error into a fly-off — so we fly the gentle geometric PD until the
                        # estimate supports the MPC. Feed the robust optical-flow velocity, not the EKF's.
                        v_xy = self.v_flow if self.flow is not None else self.ekf.rel_vel[:2]
                        a_xy_override = self.mpc.compute(self.ekf.rel_pos[:2], v_xy)
                    elif self.minsnap is not None and tracked and pos_std < cfg.mpc_confident_std:
                        # Flatness/min-snap tracker: same confidence gate and flow velocity as the MPC
                        # (predictive trackers amplify a settling estimate into fly-offs otherwise).
                        v_xy = self.v_flow if self.flow is not None else self.ekf.rel_vel[:2]
                        a_xy_override = self.minsnap.compute(self.ekf.rel_pos[:2], v_xy)
                # Maritime heave-synchronized descent: near the deck, add the deck's vertical velocity
                # (nowcast) to the descent command so the drone *rides* the heave and closes at the
                # gentle commanded relative rate — making the touchdown impact independent of the
                # (poorly forecastable) deck phase. Uses the accurate nowcast, not a long forecast.
                vz_des = cmd.vz_des
                if self.deck is not None and -float(self.ekf.rel_pos[2]) < 0.8:
                    vz_des = cmd.vz_des + float(np.clip(deck_vz, -0.3, 0.3))
                # P3 in-loop sense-and-avoid (latent guard): work in the DECK-RELATIVE frame (obstacles are
                # fixed offsets from the deck; drone pose from the EKF -> no truth). When a sensed obstacle
                # (e.g. the offshore superstructure) is within `avoid_engage`, the higher-order CBF bends the
                # horizontal command to keep clear; the contingency FSM breaks off the approach
                # (climb-and-hold) inside `avoid_abort`. Far from any structure it is a pure passthrough, so
                # the validated landing controller is untouched on a clear deck.
                if self.avoid is not None and not cmd.press and not vhold:
                    drel = -self.ekf.rel_pos[:2]
                    dvel = -rel_vel_ctrl[:2]
                    alt = max(-float(self.ekf.rel_pos[2]), 0.0)
                    nearest = self.obstacle_field.nearest_surface_distance(drel, alt)
                    self._nearest_obstacle = float(nearest)
                    cont = self.contingency.assess(
                        self._HealthStatus(pos=np.array([drel[0], drel[1], alt]), nearest_obstacle=nearest,
                                           rotor_ok=self.controller.failed_rotor is None),
                        self.k * self.control_dt)
                    aborting = cont.state == "OBSTACLE_ABORT"
                    if nearest < cfg.avoid_engage:
                        hits = self._cluster(self.range_sensor.scan(drel, alt, self._avoid_rng))
                        g = self.controller.g
                        a_des = (np.zeros(2) if aborting else
                                 (a_xy_override if a_xy_override is not None
                                  else g.kp_xy * self.ekf.rel_pos[:2] + g.kd_xy * rel_vel_ctrl[:2]))
                        a_xy_override = self.avoid.filter(drel, dvel, a_des, hits)
                    if aborting:
                        self.state = "AVOID_ABORT"
                        vz_des = max(vz_des, 0.4)          # break off the descent: climb-and-hold clear
                # B2 runtime-assurance shield: never command a descent faster than the reachability set
                # allows at this altitude (so the drone can always brake to a soft touchdown).
                if self.shield is not None:
                    h_above = max(-float(self.ekf.rel_pos[2]), 0.0)
                    vz_des = max(vz_des, -self.shield.safe_descent_speed(h_above))
                # Attitude-matched touchdown (flatness terminal shaping, minsnap only): if the
                # measured mean deck normal is meaningfully tilted (> ~4 deg — inclined decks; a
                # ship's LPF'd mean normal stays level), hand it to the terminal controller so the
                # commit descent pre-tilts and the press pushes along the normal. Every other
                # controller path passes None and is untouched.
                press_normal = None
                if (self.minsnap is not None and self.deck_normal is not None
                        and float(self.deck_normal[2]) < np.cos(np.deg2rad(4.0))):
                    press_normal = self.deck_normal
                ctrl = self.controller.compute(self.ekf.rel_pos, rel_vel_ctrl, R_ahrs,
                                               sensors.gyro, vz_des=vz_des, press=cmd.press,
                                               velocity_hold=vhold, a_xy_override=a_xy_override,
                                               dist_ff=dist_ff, press_normal=press_normal)
        else:
            self.state = "SEARCH"
            ctrl = np.full(4, cfg.hover_thrust)

        self.k += 1
        self._last_thrust = float(np.sum(ctrl))   # for the DOB's expected-acceleration model next step
        return ctrl
