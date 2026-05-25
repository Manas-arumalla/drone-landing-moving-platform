"""Online green-deck predictor for maritime landing.

A ship deck heaves, rolls, and pitches with the sea. Touching down while the deck is moving fast
vertically means a hard, off-axis impact; real shipboard recovery waits for a low-motion
("green-deck") *quiescent window* — the brief lull between wave peaks. This module estimates the
deck's dominant heave oscillation **from onboard signals only** (the relative deck altitude the EKF
already tracks from the downward camera + rangefinder) and forecasts those windows so the supervisor
can time its commit. No simulator truth and no wave model are used — per docs/REALISM_CHARTER.md.

Method: over a sliding window of recent ``(t, deck_altitude)`` samples we fit

    z(t) ≈ a0 + a1·t + Σ Aᵢ·sin(ωᵢ t) + Bᵢ·cos(ωᵢ t)

The linear ``a0 + a1·t`` term absorbs the drone's own slow descent (and mean-height drift), so the
sinusoids isolate the wave-driven heave. The component frequencies ``ωᵢ`` are picked greedily
(matching pursuit) over a grid of plausible wave periods, then all coefficients are refit jointly.
Re-fitting over a *fresh* window each time re-anchors the phase continuously (a fixed-frequency
recursive oscillator instead accumulates phase drift over a long episode — measured to perform
*worse* here, so we keep this windowed fit). The fit predicts the heave displacement and **velocity**
at near-future times, from which we locate the next quiescent window. Forecasts are short-horizon by
nature (the velocity degrades fastest), so the controller relies on the *nowcast* for its feedforward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeckPredictorConfig:
    window: float = 8.0             # s    sliding-window length of history used for the fit
    min_samples: int = 40           # need at least this many samples before predicting
    min_span: float = 6.0           # s    window must span this long (~>= one wave period) to lock,
                                    #      else trend and sinusoid are not separable and the fit is junk
    min_period: float = 2.5         # s    shortest wave period considered
    max_period: float = 7.0         # s    longest wave period considered (< window, so it's resolvable)
    n_periods: int = 60             # grid resolution for the period search
    n_components: int = 2           # number of sinusoidal components (sea states are multi-modal)
    refit_stride: int = 10          # refit every N samples (phase still extrapolates each call)
    max_fit_residual: float = 0.06  # m    RMS fit residual above which we declare "not locked"
    min_amplitude: float = 0.02     # m    heave amplitude below which the deck is "calm" (always green)
    quiescent_rate: float = 0.08    # m/s  deck vertical speed defining a green-deck window
    horizon: float = 4.0            # s    how far ahead to search for a window


class DeckMotionPredictor:
    """Estimates the deck heave oscillation online and forecasts green-deck windows."""

    def __init__(self, config: DeckPredictorConfig | None = None):
        self.cfg = config or DeckPredictorConfig()
        self.reset()

    def reset(self) -> None:
        self._t: list[float] = []
        self._z: list[float] = []
        self._now = 0.0
        self.locked = False
        self.omega = 0.0               # dominant component angular frequency (reporting)
        self.amplitude = 0.0           # total heave amplitude (sum of components)
        self.residual = float("inf")
        self._components: list[tuple[float, float, float]] = []   # (omega, A, B) per component
        self._a0 = self._a1 = 0.0
        self._t0_local = 0.0
        self._since_fit = 0

    def update(self, t: float, deck_altitude: float) -> None:
        """Add a sample of the deck's world altitude (e.g. ``ekf.rel_pos[2]`` = z_deck - z_drone).

        Only the *oscillatory* component matters; the linear-trend term removes the drone's descent.
        """
        self._now = t
        self._t.append(float(t))
        self._z.append(float(deck_altitude))
        cutoff = t - self.cfg.window
        while len(self._t) > 2 and self._t[0] < cutoff:
            self._t.pop(0)
            self._z.pop(0)
        # Refit periodically (the sea state changes slowly); between refits the locked model's phase
        # still extrapolates through ``self._now``, so predictions stay current every call.
        self._since_fit += 1
        if not self.locked or self._since_fit >= self.cfg.refit_stride:
            self._since_fit = 0
            self._fit()

    # ------------------------------------------------------------------- fitting
    def _fit(self) -> None:
        """Fit a linear trend + several sinusoids. Frequencies are picked greedily (matching
        pursuit) on the detrended signal, then all coefficients are refit jointly for accuracy."""
        c = self.cfg
        n = len(self._t)
        # Need enough samples and enough *time span* (>= one period) so the trend and the sinusoid are
        # separable; fitting < one period conflates them into a spurious trend.
        if n < c.min_samples or (self._t[-1] - self._t[0]) < c.min_span:
            self.locked = False
            return
        t = np.asarray(self._t) - self._t[0]   # local time origin for conditioning
        z = np.asarray(self._z)
        periods = np.linspace(c.min_period, c.max_period, c.n_periods)

        # 1) greedily select component frequencies on the running residual
        trend, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), z, rcond=None)
        resid = z - np.column_stack([np.ones_like(t), t]) @ trend
        omegas: list[float] = []
        for _ in range(c.n_components):
            best = None
            for T in periods:
                w = 2 * np.pi / T
                if any(abs(w - wj) < 1e-6 for wj in omegas):
                    continue
                Phi = np.column_stack([np.sin(w * t), np.cos(w * t)])
                coef, *_ = np.linalg.lstsq(Phi, resid, rcond=None)
                r = resid - Phi @ coef
                rms = float(np.sqrt(np.mean(r**2)))
                if best is None or rms < best[0]:
                    best = (rms, w, r)
            if best is None:
                break
            omegas.append(best[1])
            resid = best[2]

        # 2) joint refit: trend + all selected sinusoids together
        cols = [np.ones_like(t), t]
        for w in omegas:
            cols += [np.sin(w * t), np.cos(w * t)]
        Phi = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(Phi, z, rcond=None)
        self.residual = float(np.sqrt(np.mean((Phi @ coef - z) ** 2)))
        self._a0, self._a1 = float(coef[0]), float(coef[1])
        self._components = [(w, float(coef[2 + 2 * i]), float(coef[3 + 2 * i]))
                            for i, w in enumerate(omegas)]
        amps = [float(np.hypot(A, B)) for (_, A, B) in self._components]
        self.amplitude = float(sum(amps))
        self.omega = self._components[int(np.argmax(amps))][0] if amps else 0.0
        self._t0_local = self._t[0]
        self.locked = self.residual < c.max_fit_residual and self.amplitude > 1e-4

    # ---------------------------------------------------------------- prediction
    def heave_rate(self, dt_ahead: float) -> float:
        """Predicted deck vertical *velocity* (m/s) ``dt_ahead`` seconds from the latest sample."""
        if not self.locked:
            return 0.0
        tau = (self._now - self._t0_local) + dt_ahead
        # d/dt [A sin(w t) + B cos(w t)] = A w cos(w t) - B w sin(w t), summed over components
        return float(sum(A * w * np.cos(w * tau) - B * w * np.sin(w * tau)
                         for (w, A, B) in self._components))

    def heave_offset(self, dt_ahead: float) -> float:
        """Predicted deck heave displacement (m) about its mean, ``dt_ahead`` s ahead."""
        if not self.locked:
            return 0.0
        tau = (self._now - self._t0_local) + dt_ahead
        return float(sum(A * np.sin(w * tau) + B * np.cos(w * tau)
                         for (w, A, B) in self._components))

    def is_calm(self) -> bool:
        """True when the deck heave is negligible (flat water) — every instant is a green deck."""
        return (not self.locked) or self.amplitude < self.cfg.min_amplitude

    def in_green_window(self, descent_time: float, rate_limit: float | None = None) -> bool:
        """True if a touchdown started now and lasting ``descent_time`` stays within a green window."""
        if self.is_calm():
            return True
        limit = self.cfg.quiescent_rate if rate_limit is None else rate_limit
        samples = np.linspace(0.0, max(descent_time, 1e-3), 12)
        return bool(np.all([abs(self.heave_rate(dt)) <= limit for dt in samples]))

    def time_to_green(self, descent_time: float, rate_limit: float | None = None) -> float | None:
        """Soonest lead time (s, within the horizon) at which committing yields a green-deck touchdown.

        0.0 if now is already green, a positive lead time if one is coming up, or ``None`` if no
        qualifying window is found within the horizon (caller should keep holding/centring).
        """
        if self.is_calm():
            return 0.0
        limit = self.cfg.quiescent_rate if rate_limit is None else rate_limit
        for lead in np.linspace(0.0, self.cfg.horizon, 40):
            samples = np.linspace(lead, lead + max(descent_time, 1e-3), 10)
            if np.all([abs(self.heave_rate(dt)) <= limit for dt in samples]):
                return float(lead)
        return None
