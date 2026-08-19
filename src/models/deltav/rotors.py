"""
Rotor forces, moments, reaction torque, and gyroscopic coupling.

Implements Full_6DOF_RigidBody_Derivation.pdf Ch 10, cross-checked
against 6DOF.pdf Sec 5.3/6.3/7.3 and the input->state trace in
Interceptor_Input_State_Dependency_Map.pdf Sec 4.6-4.7.

Rotor layout, viewed from above (6DOF.pdf Sec 2.1):
    1: front-right, CCW  (spin sign s = +1)
    2: aft-left,    CCW  (spin sign s = +1)
    3: front-left,  CW   (spin sign s = -1)
    4: aft-right,   CW   (spin sign s = -1)

Tilt convention: lambda = 0   -> hover, thrust along -z_b (up)
                 lambda = 90deg -> cruise, thrust along +x_b (forward)
                 n_hat_i = [sin(lambda_i), 0, -cos(lambda_i)]^T

Moment-arm sign convention (r_i x F_T,i under the locked body frame
x=forward, y=right, z=down, cross-checked by three independent physical
checks -- a right rotor thrusting harder rolls right side up, a front
rotor thrusting harder pitches nose up, and a right rotor's forward-tilt
thrust yaws the nose away from that side, like a twin losing its right
engine):
    x_i = +l_front  for the front pair (ahead of CG -> positive, since
                     +x_b is forward)
    x_i = -l_aft    for the aft pair (behind CG -> negative)
    Mx = -y_i*T_i*cos(lam_i),  My = x_i*T_i*cos(lam_i),  Mz = -y_i*T_i*sin(lam_i)
"""
import numpy as np

_BLADE_LIFT_SLOPE = 2.0 * np.pi   # a, 2-D thin-airfoil lift-curve slope (oblique-flow
                                    # doc Sec 7.1) -- a theoretical constant, not a
                                    # measured/vehicle-specific parameter.


def hub_velocity(u, v, w, p_rate, q_rate, r_rate, x_i, y_i):
    """
    Full 3-vector hub velocity per rotor (oblique-flow doc Eq. 1):
    V_hub,i = V_cg + Omega x r_i, r_i = (x_i, y_i, 0).

    local_axial_inflow() below only needs the projection onto the shaft
    (n_hat has no y-component, so v/r never entered it) -- that's enough
    for the baseline CT/CQ table lookup. The oblique-flow correction
    needs the full vector: its y-component (v + r*x_i) is what lets the
    in-plane flow direction point out of the x-z plane during sideslip
    or yaw rate, which the scalar projection can't represent.
    """
    Vx = u - r_rate * y_i
    Vy = v + r_rate * x_i
    Vz = w + p_rate * y_i - q_rate * x_i
    return np.stack([Vx, Vy, Vz], axis=-1)   # (4,3)


def local_axial_inflow(u, v, w, p_rate, q_rate, r_rate, lam, x_i, y_i):
    """
    V_ax,i : component of rotor i's LOCAL HUB velocity along its own thrust
    axis n_hat_i = [sin(lam_i), 0, -cos(lam_i)]. The hub isn't at the CG, so
    it sees the CG velocity (u,v,w) plus the rigid-body rotation carrying it
    around: V_hub,i = [u,v,w] + omega x r_i, r_i = (x_i, y_i, 0). Projected
    onto n_hat_i (its y-component never contributes, since n_hat has no y
    part -- tilt is single-axis about body y only):
        V_ax,i = (u - r*y_i)*sin(lam_i) - (w + p*y_i - q*x_i)*cos(lam_i)

    In an aggressive coordinated turn the front/aft and left/right rotors
    see materially different effective inflow purely from the p*y_i, q*x_i,
    r*y_i cross terms -- dropping them (as an earlier version of this
    function did, using only the CG's u,v,w) biases the thrust map used by
    the control allocator.
    """
    lam = np.asarray(lam, dtype=float)
    sin_l, cos_l = np.sin(lam), np.cos(lam)
    return (u - r_rate * y_i) * sin_l - (w + p_rate * y_i - q_rate * x_i) * cos_l  # (4,)


def advance_ratio(V_N, n, D, params):
    """J = V_N / (n D), Selig 2014 Eq. (25). n is floored at params.n_reg
    (rev/s) to prevent J diverging at low/zero commanded rotor speed --
    same regularization pattern as params.Va_reg for the wing rate terms.
    """
    n_safe = np.maximum(np.asarray(n, dtype=float), params.n_reg)
    return V_N / (n_safe * D)


def lookup_CT_CQ(J, params):
    """
    CT(J), CQ(J) via linear interpolation over the tables in params.py
    (Selig 2014 Sec IV.D: thrust/torque coefficients from lookup tables
    on advance ratio). np.interp clamps outside the table range, so J
    values beyond the tabulated envelope hold at the endpoint value
    rather than extrapolating -- reverse-flow propeller aerodynamics
    (J beyond where CT/CQ -> 0) is not modeled.
    """
    CT = np.interp(J, params.CT_table_J, params.CT_table_val)
    CQ = np.interp(J, params.CQ_table_J, params.CQ_table_val)
    return CT, CQ


def rotor_forces_moments(n, lam, params, u=0.0, v=0.0, w=0.0, p=0.0, q=0.0, r=0.0):
    """
    Vectorized over the 4 rotors.

    Parameters
    ----------
    n   : array-like, shape (4,), rotor speed in rev/s (NOT rad/s)
    lam : array-like, shape (4,), tilt angle in radians
    params : InterceptorParams
    u, v, w : vehicle CG body velocity (m/s) -- drives the per-rotor advance
              ratio J so CT/CQ vary with flight condition instead of being
              fixed constants. Default 0.0 reproduces hover-at-rest.
    p, q, r : vehicle body rates (rad/s) -- the hub is offset from the CG,
              so body rotation adds a local velocity component at each
              rotor (see local_axial_inflow). Default 0.0 (no rotation)
              reproduces the old CG-velocity-only behavior if omitted.

    Returns
    -------
    dict of length-4 arrays: T, Q, FTx, FTz, MTx, MTy, MTz, MQx, MQz
    (baseline, axial-table-driven, unchanged by tilt in form), plus
    FNx, FNy, FNz, MNx, MNy, MNz (oblique-flow normal-force / P-factor
    correction, Rotor_Block_Oblique_Flow_Extension.pdf Sec 7-8 -- zero
    whenever the disc sees purely axial flow, e.g. trimmed cruise or
    hover at rest, and only nonzero when hub velocity has an in-plane
    component from transition, sideslip, or body rate), plus J, CT, CQ,
    alpha_i (disc incidence, rad), mu_i (in-plane/tip-speed validity
    ratio -- reliable roughly below 0.3-0.4, see module docstring) for
    reporting/validation.
    """
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    x_i = params.rotor_x_arm                 # (4,), +l_front / -l_aft
    y_i = params.rotor_y_arm                 # (4,), asymmetric front/aft H-frame arms
    s_i = params.rotor_spin_sign             # (4,), +1 CCW / -1 CW

    V_ax = local_axial_inflow(u, v, w, p, q, r, lam, x_i, y_i)
    J = advance_ratio(V_ax, n, D=params.rotor_diameter, params=params)
    CT, CQ = lookup_CT_CQ(J, params)

    T = CT * params.rho * n**2 * params.rotor_diameter**4                # Doc4 10.2, now J-dependent CT
    Q = CQ * params.rho * n**2 * params.rotor_diameter**5                # Doc4 10.2, now J-dependent CQ

    sin_l = np.sin(lam)
    cos_l = np.cos(lam)

    FTx = T * sin_l                          # 6DOF.pdf Sec 6.3
    FTz = -T * cos_l

    MTy = x_i * T * cos_l                    # r_i x F_T,i, y-component
    MTx = -y_i * T * cos_l                   # r_i x F_T,i, x-component (roll)
    MTz = -y_i * T * sin_l                   # r_i x F_T,i, z-component (yaw)

    MQx = -s_i * Q * sin_l                   # -s_i*Q_i*n_hat_i, x-component
    MQz = s_i * Q * cos_l                    # -s_i*Q_i*n_hat_i, z-component

    # ---- oblique-flow correction (Sec 7): normal force + P-factor moment,
    # from the in-plane component of the hub velocity the axial table can't
    # see. Vanishes identically whenever the disc sees pure axial flow.
    n_hat = np.stack([sin_l, np.zeros_like(lam), -cos_l], axis=-1)        # (4,3)
    V_hub = hub_velocity(u, v, w, p, q, r, x_i, y_i)                      # (4,3)
    V_perp_vec = V_hub - V_ax[:, None] * n_hat                           # (4,3), Eq. 3
    V_perp_mag = np.linalg.norm(V_perp_vec, axis=-1)                     # (4,)
    V_perp_safe = np.maximum(V_perp_mag, 1e-9)
    d_hat = V_perp_vec / V_perp_safe[:, None]                            # (4,3), undefined at
                                                                           # V_perp=0 but alpha_i=0
                                                                           # there too, so it's
                                                                           # multiplied out below.

    # Eq. 6, disc incidence. Forced to exactly 0 (not atan2's own value)
    # when V_perp is negligible: atan2(0, V_ax) returns pi, not 0, if V_ax
    # happens to be negative (reverse axial flow) -- but Sec 5 defines the
    # trigger for a nonzero correction as V_perp,i != 0, not V_ax's sign,
    # so pure-axial reverse flow must give alpha_i=0 like pure-axial
    # forward flow does, not a spurious 180 deg incidence.
    alpha_i = np.where(V_perp_mag < 1e-9, 0.0, np.arctan2(V_perp_mag, V_ax))

    # Magnitude-based ramp, separate from alpha_i itself: alpha_i=atan2(V_perp,V_ax)
    # is an ANGLE, so it can sit at a full 90 deg even when V_perp is only a
    # tiny fraction of a m/s (whenever V_ax happens to be ~0 too, e.g. at
    # hover) -- atan2(tiny_positive, 0) is exactly 90 deg regardless of how
    # tiny "tiny_positive" is. That let a sub-mm/s numerical perturbation in
    # V_perp flip d_hat's direction and hand the oblique correction a
    # near-full-strength force pointed in a numerically-arbitrary direction
    # -- verified by direct finite-difference linearization at a hover trim
    # point: d(u_dot)/du came out to ~1600/s, an aerodynamically impossible
    # value, traced to exactly this. Scaling PN/NP by V_perp_mag's own
    # magnitude (not its angle) below the same Va_reg floor used everywhere
    # else forces the whole correction to zero out smoothly as V_perp -> 0,
    # regardless of what alpha_i's angle happens to be doing there.
    perp_ramp = np.clip(V_perp_mag / params.Va_reg, 0.0, 1.0)
    Omega_i = 2.0 * np.pi * np.maximum(n, params.n_reg)                   # rad/s, same n_reg floor as advance_ratio
    R = params.rotor_diameter / 2.0
    mu_i = V_perp_mag / (Omega_i * R)                                    # Eq. 5, validity check only -- not enforced here

    V_hub_mag = np.linalg.norm(V_hub, axis=-1)
    V_hub_safe = np.maximum(V_hub_mag, params.Va_reg)                     # prevents q_i=0 at rest (hover-at-rest
                                                                           # start state), same floor pattern as
                                                                           # lifting_body.py's 1/Va terms
    q_i = 0.5 * params.rho * V_hub_safe**2                                # Eq. 10, disc dynamic pressure
    A = params.rotor_disk_area
    sigma = params.blade_solidity
    Cd = params.blade_Cd

    # J floored ONLY inside the 1/J, pi/J singular terms below (Sec 7.5:
    # this linear model "cannot be reliably applied near hover" and
    # explicitly blows up as J->0) -- the real, unfloored J is still used
    # everywhere else (Cl_bar's leading J/pi*T term, which itself -> 0 as
    # J->0 and needs no floor).
    J_sign = np.where(J >= 0.0, 1.0, -1.0)
    J_safe = J_sign * np.maximum(np.abs(J), params.J_reg)

    Cl_bar = (3.0 * J / (2.0 * np.pi)) * (
        (2.0 / (sigma * q_i * A)) * (J / np.pi) * T + Cd)                 # Eq. 11 (corrected form, Sec 7.2)

    PN = (sigma * q_i * A / 2.0) * (
        Cl_bar
        + (_BLADE_LIFT_SLOPE * J / (2.0 * np.pi)) * np.log(1.0 + (np.pi / J_safe) ** 2)
        + (np.pi / J_safe) * Cd
    ) * alpha_i * perp_ramp                                                # Eq. 12, normal force magnitude

    NP = -(sigma * q_i * A * R / 2.0) * (
        (2.0 * np.pi / (3.0 * J_safe)) * Cl_bar
        + (_BLADE_LIFT_SLOPE / 2.0) * (1.0 - (J_safe / np.pi) ** 2 * np.log(1.0 + (np.pi / J_safe) ** 2))
        - (np.pi / J_safe) * Cd
    ) * alpha_i * perp_ramp                                                # Eq. 13, P-factor moment magnitude

    FN = PN[:, None] * d_hat                                              # (4,3), F_N,i = P_N,i * d_hat_i
    MN = s_i[:, None] * NP[:, None] * d_hat                               # (4,3), M_N,i = s_i * N_P,i * d_hat_i

    return {"T": T, "Q": Q, "FTx": FTx, "FTz": FTz,
            "MTx": MTx, "MTy": MTy, "MTz": MTz, "MQx": MQx, "MQz": MQz,
            "FNx": FN[:, 0], "FNy": FN[:, 1], "FNz": FN[:, 2],
            "MNx": MN[:, 0], "MNy": MN[:, 1], "MNz": MN[:, 2],
            "J": J, "CT": CT, "CQ": CQ, "alpha_i": alpha_i, "mu_i": mu_i}


def gyroscopic_moment(omega, n, lam, params):
    """
    M_gyro = -omega x h_rot,   h_rot = sum_i Ip * (2*pi*n_i) * s_i * n_hat_i

    s_i (spin sign) matters here: n_hat_i is the SHAFT direction, common to
    both rotors of a tilt pair, but a CCW and a CW rotor spinning about the
    same shaft direction carry angular momentum in OPPOSITE senses. Without
    s_i, a CCW/CW pair would wrongly add instead of partially cancelling.

    omega : array-like (3,) = [p, q, r]
    """
    omega = np.asarray(omega, dtype=float)
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    n_hat = np.stack([np.sin(lam), np.zeros_like(lam), -np.cos(lam)], axis=-1)  # (4,3)
    Omega_i = 2.0 * np.pi * n
    s_i = params.rotor_spin_sign
    h_rot = np.sum(params.Ip * (Omega_i * s_i)[:, None] * n_hat, axis=0)       # (3,)
    return -np.cross(omega, h_rot)


def front_pair_thrust(T):
    """Tf = T1 + T3 (front-right + front-left). 6DOF.pdf Sec 5.3 / Doc4 Sec 12.2.

    NOTE: array index 0 = rotor 1, index 2 = rotor 3 (see module docstring
    for the rotor-index layout).
    """
    T = np.asarray(T)
    return T[0] + T[2]