"""Landing supervisor finite-state machine.

Decides *when* it is safe to descend onto the moving platform, given only the estimator's relative
state and its uncertainty. The key robustness rule: **descend only while the platform is confidently
tracked, centred, and slow relative to the drone** — otherwise hold altitude or go around to
re-acquire. This breaks the perception<->control instability where a drifting estimate pushes the
drone off-centre, the marker leaves the camera frame, and the estimate degrades further.

States: APPROACH -> DESCEND -> COMMIT -> SECURED, with GO_AROUND as the recovery branch.
The FSM outputs a vertical-velocity command (+ touchdown press / motor cut); the geometric
controller always handles horizontal centring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupervisorConfig:
    approach_alt: float = 1.0       # m   hold altitude while centring
    go_around_alt: float = 1.5      # m   climb target on abort
    align_radius: float = 0.20      # m   horizontal error allowing descent (estimate-error aware)
    abort_radius: float = 0.55      # m   drift beyond this during descent -> go around
    descend_max_relspeed: float = 0.6   # m/s  (relative-velocity estimate is noisy; keep generous)
    confident_std: float = 0.30     # m   max estimator position std to be "confident"
    commit_altitude: float = 0.28   # m   below this + aligned -> COMMIT and finish (no abort)
    commit_descent_rate: float = -0.18  # m/s  gentle descent (centre marker keeps vision -> soft touch)
    bounce_clearance: float = 0.55  # m   if we float back above this after committing -> re-descend
    descend_dwell: float = 0.20     # s   must stay aligned+confident this long before descending
    reacquire_dwell: float = 0.30   # s   stable at go-around altitude before re-approaching
    # Maritime green-deck timing: when enabled, at the commit altitude the supervisor *perches* and
    # waits for a low-deck-motion ("green") window before the final drop, falling through after
    # green_max_wait so it never stalls in rough seas. The green verdict comes from the autopilot's
    # onboard deck-motion predictor (see planning/deck_predictor.py); ground landings leave it off.
    green_deck: bool = False
    green_max_wait: float = 5.0     # s   perch-and-wait cap before committing regardless
    kp_alt: float = 0.8             # altitude-hold gain -> vertical velocity command
    vz_descent_far: float = -0.45
    vz_descent_mid: float = -0.30
    vz_descent_near: float = -0.18
    vz_limit: float = 0.6


@dataclass
class SupervisorCommand:
    vz_des: float
    press: bool
    cut: bool
    state: str


class LandingSupervisor:
    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()
        self.reset()

    def reset(self) -> None:
        self.state = "APPROACH"
        self._dwell = 0.0
        self._green_wait = 0.0
        self._cut_latched = False

    def _alt_vz(self, target_clearance: float, clearance: float) -> float:
        # vz_des is the drone's vertical velocity command (positive = up). If the drone is above the
        # target clearance, command a descent; if below, climb.
        c = self.config
        return float(np.clip(c.kp_alt * (target_clearance - clearance), -c.vz_limit, c.vz_limit))

    def update(self, rel_pos: np.ndarray, rel_vel: np.ndarray, pos_std: float,
               tracked: bool, support_feet: int, dt: float,
               green_light: bool = True) -> SupervisorCommand:
        """``green_light`` is the maritime go/no-go for the final drop: True means the deck is in (or
        near) a low-motion window. It is ignored unless ``config.green_deck`` is set; ground landings
        leave it at the default True."""
        c = self.config
        clearance = -float(rel_pos[2])
        horiz = float(np.linalg.norm(rel_pos[:2]))
        relspd = float(np.linalg.norm(rel_vel[:2]))
        confident = tracked and (pos_std < c.confident_std)
        aligned = horiz < c.align_radius and relspd < c.descend_max_relspeed

        # all gear planted -> cut motors; the grippy, damped deck holds the settled drone
        if self.state == "SECURED" or (support_feet >= 3 and self.state in ("DESCEND", "COMMIT")):
            self.state = "SECURED"
            return SupervisorCommand(0.0, False, True, self.state)

        if self.state == "APPROACH":
            # hold approach altitude and centre; descend once stably aligned + confident
            vz = self._alt_vz(c.approach_alt, clearance)
            if confident and aligned:
                self._dwell += dt
                if self._dwell >= c.descend_dwell:
                    self.state = "DESCEND"
                    self._dwell = 0.0
            else:
                self._dwell = 0.0
            return SupervisorCommand(vz, False, False, self.state)

        if self.state == "DESCEND":
            # commit-and-finish once low + aligned: below this altitude we no longer abort, because
            # the marker leaves the camera FOV near touchdown (rangefinder + coasting estimate finish it)
            if clearance < c.commit_altitude and aligned:
                # Maritime: perch at the commit altitude and wait for a green (low-motion) deck window
                # before the final drop, so touchdown lands in the wave lull. Fall through after a cap
                # so we never stall in persistent rough seas.
                if c.green_deck and not green_light and self._green_wait < c.green_max_wait:
                    self._green_wait += dt
                    return SupervisorCommand(self._alt_vz(c.commit_altitude, clearance),
                                             False, False, self.state)
                self._green_wait = 0.0
                self.state = "COMMIT"
                return SupervisorCommand(c.commit_descent_rate, False, False, self.state)
            # otherwise abort if track lost / drifted off / too fast (still high enough to go around)
            if (not confident) or horiz > c.abort_radius or relspd > 1.5 * c.descend_max_relspeed:
                self.state = "GO_AROUND"
                self._dwell = 0.0
                return SupervisorCommand(self._alt_vz(c.go_around_alt, clearance), False, False, self.state)
            if clearance > 0.6:
                vz = c.vz_descent_far
            elif clearance > 0.30:
                vz = c.vz_descent_mid
            else:
                vz = c.vz_descent_near
            return SupervisorCommand(vz, False, False, self.state)

        if self.state == "COMMIT":
            # committed: gentle sink, then press the gear down to plant all four feet firmly before
            # the cut (handled above at feet>=3). Pressing avoids a bounce on first contact.
            if support_feet >= 1:
                return SupervisorCommand(0.0, True, False, self.state)   # weight-on-gear plant
            if clearance > c.bounce_clearance:    # floated way back up -> re-descend
                self.state = "DESCEND"
                return SupervisorCommand(c.vz_descent_near, False, False, self.state)
            return SupervisorCommand(c.commit_descent_rate, False, False, self.state)

        if self.state == "GO_AROUND":
            vz = self._alt_vz(c.go_around_alt, clearance)
            at_alt = abs(clearance - c.go_around_alt) < 0.25
            if confident and aligned and at_alt:
                self._dwell += dt
                if self._dwell >= c.reacquire_dwell:
                    self.state = "APPROACH"
                    self._dwell = 0.0
            else:
                self._dwell = 0.0
            return SupervisorCommand(vz, False, False, self.state)

        return SupervisorCommand(0.0, False, False, self.state)
