"""
Wing (lifting-body) aerodynamics.
Covers airdata (Va, alpha, beta, qbar), the rotor-downwash correction
the stall-blended lift curve and parabolic drag polar, and the
CLEAN aerodynamic forces/moments 
"""
import numpy as np


def airdata(u, v, w, rho):
   
    Va = float(np.sqrt(u**2 + v**2 + w**2)) # Total Airspeed (Va)
    if Va > 1e-6:
        beta = float(np.arcsin(np.clip(v / Va, -1.0, 1.0))) #sideslip angle
    else:
        beta = 0.0
    alpha = float(np.arctan2(w, u)) # angle of attack
    qbar = 0.5 * rho * Va**2 #Dynamic Pressure
    return Va, alpha, beta, qbar


def downwash(Tf, Va, qbar, params):
    """
    Front-rotor slipstream correction to the wing's effective alpha and
    dynamic pressure. 

        wi        = sqrt(Tf / (2 rho Adisk))                  -- induced velocity from front-rotor thrust
        epsilon   = k_eps * wi / Va                           -- effective alpha shift from the downwash
        qbar_wing = qbar * (1 + k_q * Tf / (qbar * Adisk))    -- dynamic pressure seen by the wing after downwash

    NOTE: at hover, Va is near zero while wi (from the rotor thrust) is
    not, so epsilon = k_eps*wi/Va would blow up without a limit. We cap
    it at +/-85 deg, which is fine physically since at that point the
    wing's loading is dominated by qbar_wing anyway, not by alpha.
    qbar_wing has no such blow-up problem and needs no cap.
    """
    Adisk = params.rotor_disk_area
    Tf = max(float(Tf), 0.0)                                     # no negative thrust
    wi = np.sqrt(Tf / (2.0 * params.rho * Adisk))                # induced velocity (momentum theory)
    eps_cap = np.radians(85.0)                                   # largest allowed downwash angle (85 deg), so the math below can't blow up when the drone is barely moving
    eps = np.clip(params.k_eps * wi / max(Va, params.Va_reg), -eps_cap, eps_cap)  # downwash angle, kept within the +/-85 deg limit above
    if qbar > 1e-9:
        qbar_wing = qbar * (1.0 + params.k_q * Tf / (qbar * Adisk))  # normal flight: forward-flight air pressure plus extra push from the rotor wash
    else:
        qbar_wing = params.k_q * Tf / Adisk                          # hovering (near-zero airspeed): pressure on the wing comes only from the rotor wash
    return eps, qbar_wing


def stall_blend_sigma(alpha, alpha_s, M):
    """
    Smoothly blends between pre-stall and post-stall lift behavior:
    close to 0 well before the stall angle, close to 1 well past it.

    The exponent is clipped just to avoid overflow at extreme alpha --
    doesn't affect the blend near the actual stall region.
    """
    z1 = np.clip(-M * (alpha - alpha_s), -50.0, 50.0)   # upper stall break
    z2 = np.clip(M * (alpha + alpha_s), -50.0, 50.0)    # lower stall break
    num = 1.0 + np.exp(z1) + np.exp(z2)
    den = (1.0 + np.exp(z1)) * (1.0 + np.exp(z2))
    return num / den   # 0 pre-stall, 1 post-stall


def aero_coefficients(alpha_wing, p_rate, q_rate, r_rate, beta, Va, params):
    """
    Computes the wing's force/moment coefficients (CL, CD, Cm, Cl, Cn) at
    the current angle of attack. These are "clean" -- they don't include
    the elevons, which are added separately (see elevons.py).

    CL/CD: uses a lookup table from real wind-tunnel/XFLR5 data if one is
    given (params.polar_table_path), and a simple backup formula outside
    the table's range so the simulator never runs out of data.

    Cm/Cl/Cn: always use the simple formula, even if a table is given --
    the table's Cm isn't reliable enough to mix in safely.
    """
    Va_s = max(Va, params.Va_reg)                        # floor for the 1/Va rate-damping terms
    CL, CD = _lift_drag_coefficients(alpha_wing, params)

    Cm = (params.Cm0                                                    # baseline pitch moment at zero angle of attack
          + params.Cm_alpha * alpha_wing                                # pitch stiffness: moment change per degree of angle of attack
          + params.Cm_q * (params.c_bar / (2.0 * Va_s)) * q_rate)       # pitch damping: resists pitch rate -- clean: no delta_e

    Cl = (params.Cl_beta * beta                                         # roll moment caused by sideslip
          + params.Cl_p * (params.b / (2.0 * Va_s)) * p_rate)           # roll damping: resists roll rate -- clean: no delta_a

    Cn = (params.Cn_beta * beta                                         # yaw moment caused by sideslip (weathercock effect)
          + params.Cn_r * (params.b / (2.0 * Va_s)) * r_rate)           # yaw damping: resists yaw rate

    return CL, CD, Cm, Cl, Cn


def _parametric_CL_CD(alpha_wing, params):
    """The pure flat-plate-blended CL/CD model (unchanged from before the
    polar-table integration) -- used everywhere if no table is
    configured, and as the fallback partner in the table blend below."""
    sigma = stall_blend_sigma(alpha_wing, params.alpha_stall, params.stall_blend_M)
    CL_lin = params.CL0 + params.CL_alpha * alpha_wing   # attached-flow (pre-stall) lift
    CL = (1.0 - sigma) * CL_lin + \
         sigma * 2.0 * np.sign(alpha_wing) * np.sin(alpha_wing)**2 * np.cos(alpha_wing)  # post-stall flat-plate lift

    # Uses the BLENDED CL here, not the raw unbounded CL_lin -- CL_lin
    # grows without limit as alpha increases, which caused a ~450 N
    # phantom force in hover testing. The blended CL saturates near
    # +/-90 deg like a real wing does, so this stays physical everywhere.
    CD = params.CDp + (CL**2) / (np.pi * params.oswald_e * params.AR)  # parasite + induced drag
    return CL, CD


# The next two functions blend real measured data (the optional XFLR5
# table) with the backup formula above. The table only covers a limited
# angle-of-attack range, so _table_trust_weight works out how much to
# trust it at the current angle (1 = fully inside the table's range,
# 0 = fully outside it), and _lift_drag_coefficients uses that number to
# mix the table's CL/CD with the backup formula's CL/CD -- so instead of
# a sudden jump when the angle crosses the table's edge, the sim eases
# smoothly from "real data" to "backup formula".
def _table_trust_weight(alpha_wing, table, blend_width_rad):
    """
    1.0 anywhere INSIDE the table's own alpha range (real computed data
    is trusted fully, never discounted near its own edges), tapering by
    cosine ease down to 0.0 over blend_width_rad beyond whichever edge
    alpha_wing has crossed. Continuous at the boundary: right at the
    edge, weight=1 either way you approach it.
    """
    if table.alpha_min <= alpha_wing <= table.alpha_max:
        return 1.0   # inside table range, fully trusted
    d_outside = (table.alpha_min - alpha_wing) if alpha_wing < table.alpha_min \
        else (alpha_wing - table.alpha_max)   # distance past whichever edge
    if d_outside >= blend_width_rad:
        return 0.0   # past the taper, pure parametric model
    t = 1.0 - d_outside / blend_width_rad  # 1 at the edge, 0 at blend_width_rad beyond it
    return 0.5 - 0.5 * np.cos(np.pi * t)   # smooth (cosine) ease in/out, not a hard cutoff


def _lift_drag_coefficients(alpha_wing, params):
    if not params.polar_table_path:
        return _parametric_CL_CD(alpha_wing, params)   # no table configured, parametric everywhere

    from . import polar_table
    table = polar_table.load_xflr5_polar(params.polar_table_path)

    CL_param, CD_param = _parametric_CL_CD(alpha_wing, params)   # fallback, always computed
    weight = _table_trust_weight(alpha_wing, table, np.radians(params.polar_blend_deg))
    if weight <= 0.0:
        return CL_param, CD_param   # fully outside table + blend zone
    CL_tab, CD_tab, _Cm_tab = table.interp(alpha_wing)   # Cm from the table is unused, see docstring
    if weight >= 1.0:
        return float(CL_tab), float(CD_tab)   # fully inside table range
    CL = weight * CL_tab + (1.0 - weight) * CL_param   # blend zone
    CD = weight * CD_tab + (1.0 - weight) * CD_param
    return float(CL), float(CD)


def clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, q_rate, Va, beta, params):
    """
    Turns the coefficients into actual forces (Newtons) and moments
    (Newton-meters).

    My_aero (pitch moment) is built from the lift force acting at the
    neutral point, plus pitch damping -- it does NOT also add
    Cm_alpha*alpha separately, because that would double-count the same
    physical effect (the NP lever arm on Lclean already captures it).
    """
    Va_s = max(Va, params.Va_reg)
    Lclean = qbar_wing * params.S * (CL + params.CL_q * (params.c_bar / (2.0*Va_s)) * q_rate)  # lift + pitch damping
    D = qbar_wing * params.S * CD
    Y = qbar_wing * params.S * params.CY_beta * beta   # sideforce

    Mx_aero = qbar_wing * params.S * params.b * Cl   # roll moment
    My_aero = -(params.xnp - params.xcg) * Lclean + \
              qbar_wing * params.S * params.c_bar * params.Cm_q * (params.c_bar / (2.0*Va_s)) * q_rate  # NP lever + damping, see docstring
    Mz_aero = qbar_wing * params.S * params.b * Cn   # yaw moment

    return Lclean, D, Y, Mx_aero, My_aero, Mz_aero


def project_to_body(Lclean, D, Y, alpha_wing, beta):
    """
    Rotates the wind-axis forces (L, D, Y) into the body-axis forces
    (Fx, Fy, Fz) the rest of the sim uses, accounting for both alpha
    and sideslip (beta):

        Fx = L sin(a) - D cos(a) cos(b) - Y cos(a) sin(b)
        Fy = Y cos(b) - D sin(b)
        Fz = -L cos(a) - D sin(a) cos(b) - Y sin(a) sin(b)
    """
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)   # wind-to-body, alpha rotation
    sb, cb = np.sin(beta), np.cos(beta)               # wind-to-body, beta rotation
    Fx = Lclean * sa - D * ca * cb - Y * ca * sb
    Fy = Y * cb - D * sb
    Fz = -Lclean * ca - D * sa * cb - Y * sa * sb
    return Fx, Fy, Fz