"""Two-tank plant physics.

Pure functions. SI units throughout: metres, cubic metres per second, pascals,
seconds. No sockets, no sleeping, no printing, no global state. Everything in
this module is deterministic and callable from a test.

Topology
--------
    PMP101 --> FCV101 --> [ Tank 1 ] --> orifice --> [ Tank 2 ] --> drain
                             |                          |
                           LT101                      LT102

A centrifugal pump feeds tank 1 through a control valve. Tank 1 drains by
gravity into tank 2, which drains to atmosphere. LIC101 holds tank 1 level by
stroking FCV101. Tank 2 is uncontrolled, so a fault in tank 1 propagates into
it with a lag, which is what makes it useful for diagnosis later.

Pump and valve are solved together
----------------------------------
Valve flow depends on the pressure the pump develops, and pump pressure
depends on flow, so the two form an algebraic loop. With a quadratic pump
curve it closes analytically::

    p_pump = dp0 (1 - (q/qmax)^2)
    q      = k sqrt(p_pump - rho g h1)          k = Cv * phi(z)

    =>  q = k sqrt( (dp0 - rho g h1) / (1 + k^2 dp0 / qmax^2) )

No iteration and no stiffness, which keeps the real-time runner honest at a
50 ms step.

Units live at the boundary
--------------------------
The state and every internal quantity are SI. `measure()` is the only place
that converts to the engineering units declared in `pyplant.tags`, and it is
the only function in this module that knows a tag name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pyplant.tags import TAGS

G = 9.81  # m/s^2
RHO = 998.0  # kg/m^3, water at 20 C

# State vector layout. x = [h1, h2, z]
H1, H2, Z = 0, 1, 2
N_STATES = 3
STATE_NAMES = ("h1_m", "h2_m", "valve_pos")


@dataclass(frozen=True)
class Params:
    """Plant geometry and hydraulics.

    The last three fields are nominal-value fault knobs. Step 9 mutates them
    to inject faults; nothing else should touch them.
    """

    # geometry
    area_tank1: float = 0.020  # m^2, ~160 mm diameter
    area_tank2: float = 0.020  # m^2
    height_max: float = 0.600  # m, overflow lip

    # pump curve
    pump_dp0: float = 60_000.0  # Pa at zero flow (~6 m head)
    pump_qmax: float = 4.17e-4  # m^3/s at zero pressure (25 L/min)

    # control valve
    valve_cv: float = 2.429e-6  # m^3/(s * sqrt(Pa)) fully open
    valve_rangeability: float = 30.0  # equal-percentage R
    valve_seat: float = 0.005  # below this fraction the valve shuts hard
    tau_valve: float = 2.0  # s, actuator first-order lag

    # gravity drains, each as an effective discharge coefficient times area
    cda_12: float = 6.90e-5  # m^2, tank 1 -> tank 2
    cda_2out: float = 9.00e-5  # m^2, tank 2 -> drain

    # fault knobs, nominal values are the healthy plant
    pump_gain: float = 1.0  # < 1 is impeller wear
    cda_12_gain: float = 1.0  # < 1 is fouling of the interconnect
    cda_leak: float = 0.0  # m^2, hole in tank 1

    def healthy(self) -> "Params":
        """Same plant with every fault knob back at its nominal value."""
        return replace(self, pump_gain=1.0, cda_12_gain=1.0, cda_leak=0.0)


@dataclass(frozen=True)
class Inputs:
    """What the PLC sends down to the plant."""

    valve_cmd: float = 0.0  # 0..1, demanded stem position
    pump_on: bool = False


@dataclass(frozen=True)
class Flows:
    """Algebraic quantities at a given state. All SI."""

    q_in: float  # through FCV101 into tank 1, m^3/s
    q_12: float  # tank 1 -> tank 2
    q_leak: float  # tank 1 -> floor
    q_out: float  # tank 2 -> drain
    p_pump: float  # pump discharge gauge pressure, Pa


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------

def valve_characteristic(z: float, p: Params) -> float:
    """Inherent equal-percentage characteristic, 0..1.

    Equal percentage under a pump curve gives a roughly linear installed
    characteristic, which is the usual reason for choosing it.
    """
    z = min(max(z, 0.0), 1.0)
    if z < p.valve_seat:
        return 0.0
    return min(p.valve_rangeability ** (z - 1.0), 1.0)


def _orifice(cda: float, head: float) -> float:
    """Torricelli discharge through an effective area, m^3/s."""
    if cda <= 0.0 or head <= 0.0:
        return 0.0
    return cda * math.sqrt(2.0 * G * head)


def flows(x: np.ndarray, u: Inputs, p: Params) -> Flows:
    """Solve the pump/valve loop and the gravity drains at this state."""
    h1 = max(float(x[H1]), 0.0)
    h2 = max(float(x[H2]), 0.0)

    dp0 = p.pump_dp0 * p.pump_gain
    k = p.valve_cv * valve_characteristic(float(x[Z]), p)

    if not u.pump_on or dp0 <= 0.0:
        q_in, p_pump = 0.0, 0.0
    elif k <= 0.0:
        q_in, p_pump = 0.0, dp0  # shut valve, pump deadheads
    else:
        static = RHO * G * h1  # backpressure from the level above the valve
        avail = dp0 - static
        if avail <= 0.0:
            q_in, p_pump = 0.0, dp0
        else:
            alpha = dp0 / (p.pump_qmax ** 2)
            q_in = k * math.sqrt(avail / (1.0 + k * k * alpha))
            p_pump = dp0 * (1.0 - (q_in / p.pump_qmax) ** 2)

    return Flows(
        q_in=q_in,
        q_12=_orifice(p.cda_12 * p.cda_12_gain, h1),
        q_leak=_orifice(p.cda_leak, h1),
        q_out=_orifice(p.cda_2out, h2),
        p_pump=p_pump,
    )


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def derivatives(x: np.ndarray, u: Inputs, p: Params) -> np.ndarray:
    """dx/dt. Constrained at the tank limits so integration cannot leave them."""
    f = flows(x, u, p)

    dh1 = (f.q_in - f.q_12 - f.q_leak) / p.area_tank1
    dh2 = (f.q_12 - f.q_out) / p.area_tank2
    dz = (min(max(u.valve_cmd, 0.0), 1.0) - float(x[Z])) / p.tau_valve

    # Overflow spills, an empty tank cannot drain further.
    if x[H1] >= p.height_max and dh1 > 0.0:
        dh1 = 0.0
    if x[H1] <= 0.0 and dh1 < 0.0:
        dh1 = 0.0
    if x[H2] >= p.height_max and dh2 > 0.0:
        dh2 = 0.0
    if x[H2] <= 0.0 and dh2 < 0.0:
        dh2 = 0.0

    return np.array([dh1, dh2, dz], dtype=float)


def step(x: np.ndarray, u: Inputs, p: Params, dt: float) -> np.ndarray:
    """Advance the state by dt with classical RK4."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    k1 = derivatives(x, u, p)
    k2 = derivatives(x + 0.5 * dt * k1, u, p)
    k3 = derivatives(x + 0.5 * dt * k2, u, p)
    k4 = derivatives(x + dt * k3, u, p)
    nxt = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    nxt[H1] = min(max(nxt[H1], 0.0), p.height_max)
    nxt[H2] = min(max(nxt[H2], 0.0), p.height_max)
    nxt[Z] = min(max(nxt[Z], 0.0), 1.0)
    return nxt


def initial_state(h1: float = 0.0, h2: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([h1, h2, z], dtype=float)


# ---------------------------------------------------------------------------
# Steady state, for tuning and for tests
# ---------------------------------------------------------------------------

def steady_state(u: Inputs, p: Params, tol: float = 1e-9) -> np.ndarray:
    """Equilibrium state at a fixed valve demand.

    Inflow falls with level and gravity outflow rises with it, so the residual
    is monotone and plain bisection is safe.
    """
    z = min(max(u.valve_cmd, 0.0), 1.0)

    def residual(h1: float) -> float:
        f = flows(np.array([h1, 0.0, z]), u, p)
        return f.q_in - f.q_12 - f.q_leak

    lo, hi = 0.0, p.height_max
    if residual(hi) > 0.0:  # floods
        h1 = p.height_max
    elif residual(lo) <= 0.0:  # cannot fill
        h1 = 0.0
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        h1 = 0.5 * (lo + hi)

    q12 = _orifice(p.cda_12 * p.cda_12_gain, h1)
    if p.cda_2out <= 0.0:
        h2 = p.height_max
    else:
        h2 = min((q12 / p.cda_2out) ** 2 / (2.0 * G), p.height_max)
    return np.array([h1, h2, z], dtype=float)


# ---------------------------------------------------------------------------
# Boundary: SI -> engineering units
# ---------------------------------------------------------------------------

M3S_TO_LPM = 60_000.0


def measure(x: np.ndarray, u: Inputs, p: Params) -> dict[str, float | bool]:
    """Instrument readings in the units declared in `pyplant.tags`.

    Sensor faults are not applied here. Bias, drift and noise belong to the
    fault layer in step 9, which wraps this function rather than editing it.
    """
    f = flows(x, u, p)
    return {
        "LT101": float(x[H1]) * 1000.0,
        "LT102": float(x[H2]) * 1000.0,
        "FT101": f.q_in * M3S_TO_LPM,
        "FT102": f.q_out * M3S_TO_LPM,
        "PT101": f.p_pump / 1000.0,
        "ZT101": float(x[Z]) * 100.0,
        "PMP101_RUN": bool(u.pump_on),
    }


def clip_to_tag_ranges(m: dict[str, float | bool]) -> dict[str, float | bool]:
    """Clamp readings into declared instrument spans.

    A real transmitter pegs at its span; the tag codec would clamp anyway. Doing
    it here keeps the plant's own logs identical to what the PLC will read.
    """
    out: dict[str, float | bool] = {}
    for name, value in m.items():
        t = TAGS[name]
        out[name] = value if isinstance(value, bool) else t.clamp(float(value))
    return out
