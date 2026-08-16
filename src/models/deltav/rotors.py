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

Moment-arm sign convention (verified against FINAL_CASES Case 1 hover
trim: Tf/Ta = l_aft/l_front = 0.805, reproduced exactly by the check
in validate_case1_hover.py):
    x_i = +l_front  for the front pair (ahead of CG -> positive, since
                     +x_b is forward)
    x_i = -l_aft    for the aft pair (behind CG -> negative)
"""
import numpy as np


def local_axial_inflow(u, v, w, lam):
    """
    V_N,i : component of the vehicle's body velocity along rotor i's own
    thrust axis n_hat_i = [sin(lam_i), 0, -cos(lam_i)] (Selig 2014 calls
    this the "local relative flow velocity" component driving the
    propeller's advance ratio, Nomenclature / Sec IV.D). Reuses the same
    n_hat convention as gyroscopic_moment() below -- no new geometry.

    NOTE: this uses only the CG's body velocity, not the extra local
    velocity a rotor picks up from being offset from the CG during body
    rotation (omega x r_i). That coupling is a further-fidelity item, not
    included here -- flag if you need it for high-rate maneuvers.
    """
    lam = np.asarray(lam, dtype=float)
    n_hat = np.stack([np.sin(lam), np.zeros_like(lam), -np.cos(lam)], axis=-1)  # (4,3)
    V_body = np.array([u, v, w])
    return n_hat @ V_body  # (4,)


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


def rotor_forces_moments(n, lam, params, u=0.0, v=0.0, w=0.0):
    """
    Vectorized over the 4 rotors.

    Parameters
    ----------
    n   : array-like, shape (4,), rotor speed in rev/s (NOT rad/s)
    lam : array-like, shape (4,), tilt angle in radians
    params : InterceptorParams
    u, v, w : vehicle body velocity (m/s) -- drives the per-rotor advance
              ratio J so CT/CQ vary with flight condition instead of
              being fixed constants (Selig 2014 Sec IV.D). Default 0.0
              reproduces the old hover-only behavior if omitted.

    Returns
    -------
    dict of length-4 arrays: T, Q, FTx, FTz, MTx, MTy, MTz, MQx, MQz,
    plus J, CT, CQ (the advance-ratio-dependent coefficients actually
    used) for reporting/validation.
    """
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    V_N = local_axial_inflow(u, v, w, lam)
    J = advance_ratio(V_N, n, D=params.rotor_diameter, params=params)
    CT, CQ = lookup_CT_CQ(J, params)

    T = CT * params.rho * n**2 * params.rotor_diameter**4                # Doc4 10.2, now J-dependent CT
    Q = CQ * params.rho * n**2 * params.rotor_diameter**5                # Doc4 10.2, now J-dependent CQ

    sin_l = np.sin(lam)
    cos_l = np.cos(lam)

    FTx = T * sin_l                          # 6DOF.pdf Sec 6.3
    FTz = -T * cos_l

    x_i = params.rotor_x_arm                 # (4,), +l_front / -l_aft
    y_i = params.rotor_y_arm                 # (4,), asymmetric front/aft H-frame arms
    s_i = params.rotor_spin_sign             # (4,), +1 CCW / -1 CW

    MTy = x_i * T * cos_l                    # 6DOF.pdf Sec 7.3 (zi=0 specialisation, Doc4 10.2)
    MTx = y_i * T * cos_l
    MTz = y_i * T * sin_l

    MQx = s_i * Q * sin_l                    # 6DOF.pdf Sec 7.3 / Doc4 10.5
    MQz = -s_i * Q * cos_l

    return {"T": T, "Q": Q, "FTx": FTx, "FTz": FTz,
            "MTx": MTx, "MTy": MTy, "MTz": MTz, "MQx": MQx, "MQz": MQz,
            "J": J, "CT": CT, "CQ": CQ}


def gyroscopic_moment(omega, n, lam, params):
    """
    M_gyro = -omega x h_rot,   h_rot = sum_i Ip * (2*pi*n_i) * n_hat_i
    (Doc4 Sec 10.6; 6DOF.pdf Sec 7.3)

    omega : array-like (3,) = [p, q, r]
    """
    omega = np.asarray(omega, dtype=float)
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)

    n_hat = np.stack([np.sin(lam), np.zeros_like(lam), -np.cos(lam)], axis=-1)  # (4,3)
    Omega_i = 2.0 * np.pi * n
    h_rot = np.sum(params.Ip * Omega_i[:, None] * n_hat, axis=0)               # (3,)
    return -np.cross(omega, h_rot)


def front_pair_thrust(T):
    """Tf = T1 + T3 (front-right + front-left). 6DOF.pdf Sec 5.3 / Doc4 Sec 12.2.

    NOTE: array index 0 = rotor 1, index 2 = rotor 3 (see module docstring
    for the rotor-index layout).
    """
    T = np.asarray(T)
    return T[0] + T[2]