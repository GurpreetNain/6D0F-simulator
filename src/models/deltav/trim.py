"""
Reusable trim-point construction for the Interceptor: given a target flight
condition (hover, head-on cruise, ...), returns the (state, control_vector)
pair that -- if the aero/rotor coefficients matched the real airframe
exactly -- would hold that condition in equilibrium.

This does not run the sim or check anything; it just does the algebra
FINAL_CASES_WITH_PITCH_TRIM.pdf's Sec 5 works out by hand, so scripts that
want to fly a known case (fly_case3.py, InterceptorTrimController, future
cases) don't each re-derive it.
"""
import numpy as np

from . import rotors


def solve_rotor_speed_for_thrust(T_target, u, v, w, lam, p, n_lo=5.0, n_hi=1000.0, tol=1e-5, max_iter=80):
    """Bisects for the rotor speed n (rev/s, same on all 4 rotors) that
    makes rotors.rotor_forces_moments() produce T_target newtons per rotor
    at this fixed body velocity and tilt angle. All 4 rotors see the same
    V_N here (no body rotation, so no rotor-offset velocity), so solving
    one scalar n and broadcasting it to all 4 is exact, not an approximation.
    """
    lam_arr = np.full(4, lam)

    def thrust_at(n):
        rot = rotors.rotor_forces_moments(np.full(4, n), lam_arr, p, u=u, v=v, w=w)
        return float(rot["T"][0])

    f_lo, f_hi = thrust_at(n_lo) - T_target, thrust_at(n_hi) - T_target
    if f_lo > 0 or f_hi < 0:
        raise RuntimeError(
            f"Target thrust {T_target:.2f} N/rotor is outside the "
            f"[{thrust_at(n_lo):.2f}, {thrust_at(n_hi):.2f}] N bracket "
            f"for n in [{n_lo}, {n_hi}] rev/s -- widen the search range.")

    for _ in range(max_iter):
        n_mid = 0.5 * (n_lo + n_hi)
        f_mid = thrust_at(n_mid) - T_target
        if abs(f_mid) < tol:
            return n_mid
        if (f_mid > 0) == (f_lo > 0):
            n_lo, f_lo = n_mid, f_mid
        else:
            n_hi, f_hi = n_mid, f_mid
    return 0.5 * (n_lo + n_hi)


def solve_delta_e_for_moment(x, n, lam, params, interceptor, target_SMy=0.0,
                              delta_lo=None, delta_hi=None, tol=1e-3, max_iter=40):
    """Bisects for the symmetric elevon deflection delta_e (both elevons
    equal, no aileron differential) that zeros the pitching moment SMy at
    the GIVEN state x and rotor command (n, lam) -- i.e. "what elevon
    balances the moment right now," not a steady trim (position, velocity,
    and rotor speed are whatever they already are, not solved for).

    Used to continuously re-trim the elevator every timestep in
    fly_dynamics.py's closed-loop moment-hold mode: elevon deflection
    isn't a free input the way rotor speed/tilt are -- it's whatever
    balances the moment at the angle of attack the vehicle happens to be
    at that instant (elevons.py: My_delta = Cm_delta_e*eta_e*delta_e*
    qbar_wing*S*c_bar, and Cm_delta_e > 0 makes SMy monotonic in delta_e
    over the elevons' travel range, which is what makes bisection safe
    here -- no Newton/Jacobian needed for a single monotonic unknown).

    Returns (delta_e, report, converged). If SMy doesn't change sign
    across [-delta_max, +delta_max] (elevons alone can't null the moment
    at this condition -- e.g. deep stall or an extreme tilt), clamps to
    whichever end gets closer, returns that, and sets converged=False --
    since this is meant to show you what happens, not stop the run, but
    the caller needs to know the elevons saturated instead of actually
    balancing the moment.
    """
    lam = np.asarray(lam, dtype=float)
    n = np.asarray(n, dtype=float)
    delta_lo = -params.delta_max if delta_lo is None else delta_lo
    delta_hi = params.delta_max if delta_hi is None else delta_hi

    def smy_at(delta_e):
        u_in = np.array([delta_e, delta_e, lam[0], lam[1], lam[2], lam[3], n[0], n[1], n[2], n[3]])
        _, report = interceptor._compute(x, u_in)
        return report["SMy"] - target_SMy, report

    f_lo, report_lo = smy_at(delta_lo)
    f_hi, report_hi = smy_at(delta_hi)
    if abs(f_lo) < tol:
        return delta_lo, report_lo, True
    if abs(f_hi) < tol:
        return delta_hi, report_hi, True
    if (f_lo > 0) == (f_hi > 0):
        # Can't bracket a root -- elevons alone can't null the moment here.
        # Return whichever end gets closer instead of failing the run;
        # the caller is expected to flag this (see fly_dynamics.py's
        # elevon-saturated warning), not treat it as a normal solve.
        saturated = (delta_lo, report_lo) if abs(f_lo) < abs(f_hi) else (delta_hi, report_hi)
        return saturated[0], saturated[1], False

    report_mid = report_lo
    for _ in range(max_iter):
        delta_mid = 0.5 * (delta_lo + delta_hi)
        f_mid, report_mid = smy_at(delta_mid)
        if abs(f_mid) < tol:
            return delta_mid, report_mid, True
        if (f_mid > 0) == (f_lo > 0):
            delta_lo, f_lo = delta_mid, f_mid
        else:
            delta_hi, f_hi = delta_mid, f_mid
    return 0.5 * (delta_lo + delta_hi), report_mid, False


def case1_hover_trim(p):
    """Case 1 -- Hover (FINAL_CASES Sec 5). Rotors vertical, elevons
    neutral; front/aft thrust split lift-neutral via l_aft/l_front so
    Sigma Mx = Sigma My = 0. Returns (x0, control_vector)."""
    W = p.mass * p.g
    ratio = p.l_aft / p.l_front       # = Tf/Ta
    Ta = W / (2.0 * (ratio + 1.0))
    Tf = ratio * Ta

    CT_hover = p.CT_table_val[0]      # J=0 row -- static thrust
    n_front = np.sqrt(Tf / (CT_hover * p.rho * p.rotor_diameter ** 4))
    n_aft = np.sqrt(Ta / (CT_hover * p.rho * p.rotor_diameter ** 4))

    x0 = np.zeros(12)
    control_vector = np.array([
        0.0, 0.0,                          # delta1, delta2
        0.0, 0.0, 0.0, 0.0,                # lam1..4 -- rotors vertical
        n_front, n_aft, n_front, n_aft,    # n1..4, rotor order [FR,AL,FL,AR]
    ])
    return x0, control_vector


def _build_hover_state(theta, delta_e, s, base_n):
    """Hover candidate: zero velocity/rates (that's what "hover" means --
    Va=0, so alpha is 0 by construction, NOT affected by theta), rotors
    vertical, front/aft split fixed at case1_hover_trim's ratio and
    scaled uniformly by s. theta is a free pitch-attitude unknown: at
    zero velocity the wing's OWN body-axis force is theta-independent
    (project_to_body only uses alpha/beta), so the only thing theta can
    do is rotate gravity's split between body x/z -- which is exactly
    the lever needed to cancel the wing's residual horizontal force
    (see solve_hover_trim's docstring)."""
    n = s * base_n
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, theta, 0.0, 0.0, 0.0, 0.0])
    control_vector = np.array([
        delta_e, delta_e,
        0.0, 0.0, 0.0, 0.0,
        n[0], n[1], n[2], n[3],
    ])
    return x0, control_vector


def _hover_residual(x_vars, params, interceptor, base_n):
    """x_vars = [theta, delta_e, s]. Returns ([SFx, SFz, SMy], report)
    from the real Interceptor._compute(), same pattern as
    _cruise_residual."""
    theta, delta_e, s = x_vars
    x0, u_in = _build_hover_state(theta, delta_e, s, base_n)
    _, report = interceptor._compute(x0, u_in)
    residual = np.array([report["SFx"], report["SFz"], report["SMy"]])
    return residual, report


def solve_hover_trim(params, theta0=0.0, delta_e0=np.radians(-7.0), s0=1.0,
                      tol=1e-6, max_iter=30, fd_step=1e-6):
    """Real hover trim, solved against the FULL Interceptor._compute()
    pipeline -- unlike case1_hover_trim(), which only zeros rotor-thrust
    force/moment by hand and ignores the wing/downwash entirely.

    At hover the wing sits in extreme downwash-driven alpha_wing (~-85
    deg even at exact rest -- the rotor wash overwhelms the near-zero
    forward airspeed), which produces a real, nonzero HORIZONTAL body-
    axis force even though vertical thrust balances weight (confirmed
    numerically: SFx~4.4 N left over at case1_hover_trim's own trim
    point). Held open-loop, that unbalanced SFx slowly accelerates the
    vehicle forward, and the resulting velocity/attitude coupling is
    what actually produces the altitude loss and pitch drift seen in
    long fly_dynamics.py runs at case1_hover_trim's controls, not a
    single dramatic failure.

    Design: keep case1_hover_trim's front/aft thrust-split RATIO (it
    already zeros the pure rotor-lever moment) and solve 3 unknowns --
    theta (pitch attitude), delta_e (elevon), s (overall thrust-scale
    factor applied uniformly to the front/aft split) -- against 3
    equations: SFx=0, SFz=0, SMy=0. theta is the new piece case1 doesn't
    have: since Va=0 the wing's body-axis force is exactly the same
    regardless of theta, so tilting slightly off level is the ONLY way
    to bring gravity's own horizontal component in to cancel the wing's
    otherwise-unbalanceable residual force -- there is no rotor tilt or
    elevon effect that can do this at zero airspeed.

    Plain Newton-Raphson, central-finite-difference Jacobian (3x3), same
    pattern as solve_cruise_trim. Returns (x0, control_vector, report).
    """
    from .interceptor import Interceptor
    interceptor = Interceptor(params=params)

    _, base_u = case1_hover_trim(params)
    base_n = base_u[6:10]   # [n_front, n_aft, n_front, n_aft], case1's pure lever-arm split

    x_vars = np.array([theta0, delta_e0, s0], dtype=float)
    steps = np.full(3, fd_step)   # theta, delta_e (rad) and s (dimensionless, ~O(1)) are all similar scale

    report = None
    for _ in range(max_iter):
        residual, report = _hover_residual(x_vars, params, interceptor, base_n)
        if np.linalg.norm(residual) < tol:
            break

        J_mat = np.zeros((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = steps[j]
            r_plus, _ = _hover_residual(x_vars + dx, params, interceptor, base_n)
            r_minus, _ = _hover_residual(x_vars - dx, params, interceptor, base_n)
            J_mat[:, j] = (r_plus - r_minus) / (2.0 * steps[j])

        delta = np.linalg.solve(J_mat, -residual)
        x_vars = x_vars + delta
    else:
        raise RuntimeError(
            f"solve_hover_trim did not converge in {max_iter} iterations "
            f"(final residual norm {np.linalg.norm(residual):.4g}, tol {tol}).")

    theta, delta_e, s = x_vars
    x0, control_vector = _build_hover_state(theta, delta_e, s, base_n)
    return x0, control_vector, report


def case3_head_on_trim(p):
    """Case 3 -- Head-on interception at 50 m/s (FINAL_CASES Sec 5).
    alpha=3.3deg, gamma=4.574deg dive, delta_e=6.5deg, rotor tilt=86.7deg,
    rotor speed solved (not given directly by the paper) from the real
    prop CT(J) table for the paper's 22.75 N total / 5.69 N-per-rotor
    thrust target. Returns (x0, control_vector).

    NOTE: 86.7deg is the paper's own literal number (90deg-alpha), kept
    exactly as given for direct comparison against the paper. It does NOT
    actually zero the perpendicular thrust component under this codebase's
    locked n_hat_i=[sin(lam),0,-cos(lam)] convention -- that requires
    lam=90deg+alpha instead (verified by dot product with the velocity
    direction: 90-alpha gives 0.9934, 90+alpha gives exactly 1.0). This is
    an inconsistency in the source paper between its "90-alpha" tilt
    formula and its own "T_perp=0" claim for this case, not a transcription
    error here. solve_cruise_trim()/head_on_intercept_trim() below use the
    corrected 90+alpha rule; this function stays a literal paper
    transcription on purpose."""
    Va = 50.0
    alpha = np.radians(3.3)
    gamma = np.radians(4.574)
    delta_e = np.radians(6.5)
    lam_trim = np.radians(86.7)
    T_target_per_rotor = 22.75 / 4.0

    u0 = Va * np.cos(alpha)
    w0 = Va * np.sin(alpha)
    v0 = 0.0
    theta0 = alpha - gamma            # pitch attitude that puts the dive at gamma

    n_trim = solve_rotor_speed_for_thrust(T_target_per_rotor, u0, v0, w0, lam_trim, p)

    x0 = np.array([0.0, 0.0, 0.0, u0, v0, w0, 0.0, theta0, 0.0, 0.0, 0.0, 0.0])
    control_vector = np.array([
        delta_e, delta_e,
        lam_trim, lam_trim, lam_trim, lam_trim,
        n_trim, n_trim, n_trim, n_trim,
    ])
    return x0, control_vector


def _build_cruise_state(alpha, delta_e, n, Va, gamma):
    """Shared state/control builder for the cruise-trim solver: rotors
    tilted exactly along the flight-path velocity (lam=pi/2+alpha -- see
    the note on case3_head_on_trim above for why it's +alpha, not -alpha),
    symmetric across all 4 rotors (delta_a=0, no roll/yaw needed for
    wings-level no-sideslip cruise)."""
    lam = (np.pi / 2.0) + alpha
    u0 = Va * np.cos(alpha)
    w0 = Va * np.sin(alpha)
    theta0 = alpha - gamma
    x0 = np.array([0.0, 0.0, 0.0, u0, 0.0, w0, 0.0, theta0, 0.0, 0.0, 0.0, 0.0])
    control_vector = np.array([
        delta_e, delta_e,
        lam, lam, lam, lam,
        n, n, n, n,
    ])
    return x0, control_vector


def _cruise_residual(x_vars, Va, gamma, params, interceptor):
    """x_vars = [alpha, delta_e, n]. Returns ([SFx, SFz, SMy], report)
    from the REAL Interceptor._compute() at the state/control this implies
    -- not a hand-derived closed form -- so the solved trim is automatically
    self-consistent with whatever the simulator actually computes,
    including the downwash-alpha coupling. Raises RuntimeError if alpha
    exceeds params.alpha_stall or any rotor's J leaves the CT/CQ table's
    tabulated range -- a Newton step that wanders there should fail loudly,
    not silently extrapolate."""
    alpha, delta_e, n = x_vars
    if abs(alpha) > params.alpha_stall:
        raise RuntimeError(
            f"alpha={np.degrees(alpha):.2f} deg exceeds the {np.degrees(params.alpha_stall):.1f} deg "
            f"stall cap during solve -- try a different initial guess.")

    x0, u_in = _build_cruise_state(alpha, delta_e, n, Va, gamma)
    _, report = interceptor._compute(x0, u_in)

    J = report["rotor_J"]
    J_lo, J_hi = min(params.CT_table_J), max(params.CT_table_J)
    if np.any(J < J_lo) or np.any(J > J_hi):
        raise RuntimeError(
            f"rotor advance ratio J={J} left the tabulated CT/CQ range "
            f"[{J_lo}, {J_hi}] during solve -- try a different initial guess.")

    residual = np.array([report["SFx"], report["SFz"], report["SMy"]])
    return residual, report


def solve_cruise_trim(Va, gamma, params, alpha0=np.radians(3.0),
                       delta_e0=np.radians(5.0), n0=None,
                       tol=1e-6, max_iter=30, fd_step=1e-6):
    """Solves for (alpha, delta_e, n) -- angle of attack, elevon
    deflection, common rotor speed -- that trims the FULL Interceptor
    pipeline (rotors + wings + elevons + gravity, including downwash) at
    airspeed Va and flight-path angle gamma (positive = descending, same
    convention as case3_head_on_trim). Rotor tilt is not a free unknown:
    it's fixed by _build_cruise_state's lam=pi/2+alpha rule (thrust exactly
    along the flight path, zero lift from the rotors -- the paper's own
    Sec 3 "wing below cap: rotors carry no lift" design law).

    Plain Newton-Raphson with a central-finite-difference Jacobian (3x3,
    cheap) -- no scipy dependency. This is a well-conditioned, near-linear
    system in the cruise regime, so this converges in a handful of
    iterations from a reasonable initial guess.

    Returns (x0, control_vector, report) -- report is the FULL dict
    Interceptor._compute() returns at the converged solution (forces,
    moments, Lclean, D, per-rotor T/Q/J, alpha_wing, everything), so the
    caller gets forces/moments for free.
    """
    from .interceptor import Interceptor

    p = params
    interceptor = Interceptor(params=p)

    if n0 is None:
        # Seed from a physically-grounded guess instead of a flat constant:
        # target thrust ~ W*sin(gamma)/4 per rotor, at the initial (alpha0, lam0).
        W = p.mass * p.g
        lam0 = (np.pi / 2.0) + alpha0
        u0 = Va * np.cos(alpha0)
        w0 = Va * np.sin(alpha0)
        T_guess = max(W * np.sin(gamma) / 4.0, 1.0)
        n0 = solve_rotor_speed_for_thrust(T_guess, u0, 0.0, w0, lam0, p)

    x_vars = np.array([alpha0, delta_e0, n0], dtype=float)
    steps = np.array([fd_step, fd_step, fd_step * 1e5])   # alpha,delta_e in rad; n in rev/s -- different scales

    report = None
    for _ in range(max_iter):
        residual, report = _cruise_residual(x_vars, Va, gamma, p, interceptor)
        if np.linalg.norm(residual) < tol:
            break

        # Central-difference Jacobian, one column per unknown.
        J_mat = np.zeros((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = steps[j]
            r_plus, _ = _cruise_residual(x_vars + dx, Va, gamma, p, interceptor)
            r_minus, _ = _cruise_residual(x_vars - dx, Va, gamma, p, interceptor)
            J_mat[:, j] = (r_plus - r_minus) / (2.0 * steps[j])

        delta = np.linalg.solve(J_mat, -residual)
        x_vars = x_vars + delta
    else:
        raise RuntimeError(
            f"solve_cruise_trim did not converge in {max_iter} iterations "
            f"(final residual norm {np.linalg.norm(residual):.4g}, tol {tol}).")

    alpha, delta_e, n = x_vars
    x0, control_vector = _build_cruise_state(alpha, delta_e, n, Va, gamma)
    return x0, control_vector, report


def head_on_intercept_trim(D, H, Va, params, **kwargs):
    """Case-3-style geometry: horizontal distance D, altitude offset H ->
    gamma = atan2(H, D) (same sign convention already used by
    case3_head_on_trim, where positive gamma is a descent toward a target
    below) -> solve_cruise_trim. Lets you run "the same kind of case as
    Case 3" at any distance/altitude instead of only the paper's fixed
    D=2500 m, H=200 m. Returns (x0, control_vector, report)."""
    gamma = np.arctan2(H, D)
    return solve_cruise_trim(Va, gamma, params, **kwargs)


def solve_trim_from_condition(condition, params, **kwargs):
    """General entry point: given a target-flight-condition dict, solves
    for and returns the matching equilibrium via the real-physics solvers
    above (solve_hover_trim / solve_cruise_trim) -- one place to call
    regardless of whether the condition is hover or cruise, instead of
    the caller picking which solver function to use.

    condition keys:
        V_target   : airspeed, m/s. 0 (or < 0.5) routes to solve_hover_trim;
                     otherwise routes to solve_cruise_trim.
        gamma_deg  : flight-path angle, deg (positive = descending, same
                     convention as elsewhere in this file). Ignored for
                     hover (V_target ~ 0), since hover has no flight path.
        n_z        : load factor. Only 1.0 (the default, standard 1g
                     level/climbing/descending flight -- what
                     solve_cruise_trim already solves) is supported.
        turn_radius: turn radius, m. Only float('inf') (the default,
                     straight-line flight) is supported.

    n_z != 1.0 or a finite turn_radius are DELIBERATELY NOT implemented --
    STILL OPEN, not silently approximated. A steady turn or pull-up isn't
    a simple extension of this solver's existing 3 equations: it requires
    solving for a nonzero, SUSTAINED body rate (p/q/r) as part of the
    equilibrium itself (the vehicle is genuinely rotating in steady
    state, e.g. q = g*(n_z-1)*cos(gamma)/Va for a symmetric pull-up, or a
    banked coordinated turn's r = g*tan(phi)/Va), not just zero-rate
    force/moment balance -- a real, separate derivation (Stevens & Lewis-
    style steady-state trim), not a parameter tweak. Raises
    NotImplementedError rather than quietly ignoring the request.

    Returns (x0, control_vector, report).
    """
    if condition.get("n_z", 1.0) != 1.0:
        raise NotImplementedError(
            "solve_trim_from_condition: n_z != 1.0 (load-factor trim) is not "
            "implemented -- see this function's docstring for why it's not a "
            "simple extension. Use solve_cruise_trim/solve_hover_trim for "
            "standard 1g straight-line flight.")
    if condition.get("turn_radius", float("inf")) != float("inf"):
        raise NotImplementedError(
            "solve_trim_from_condition: finite turn_radius (turning-flight "
            "trim) is not implemented -- see this function's docstring for "
            "why it's not a simple extension. Use solve_cruise_trim/"
            "solve_hover_trim for standard straight-line flight.")

    V_target = condition["V_target"]
    if V_target < 0.5:
        return solve_hover_trim(params, **kwargs)
    gamma = np.radians(condition.get("gamma_deg", 0.0))
    return solve_cruise_trim(V_target, gamma, params, **kwargs)


def hover_solved_trim(p):
    """TRIM_CASES-compatible wrapper around solve_hover_trim (drops the
    report, keeping the (x0, control_vector) shape the other entries use)."""
    x0, control_vector, _ = solve_hover_trim(p)
    return x0, control_vector


TRIM_CASES = {
    "hover": case1_hover_trim,
    "hover_solved": hover_solved_trim,
    "head_on": case3_head_on_trim,
}
