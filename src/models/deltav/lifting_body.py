"""
Wing (lifting-body) aerodynamics.

Covers airdata (Va, alpha, beta, qbar), the rotor-downwash correction
(Full_6DOF_RigidBody_Derivation.pdf Ch 12; 6DOF.pdf Sec 5.3), the
stall-blended lift curve and parabolic drag polar (Doc4 Ch 8), and the
CLEAN aerodynamic forces/moments (Doc4 Ch 8-9) -- "clean" meaning the
elevon contribution is deliberately excluded here. See the note below
for why, and see elevons.py for where it is added back in.

--------------------------------------------------------------------
IMPORTANT -- elevon double-count resolved (read this before editing)
--------------------------------------------------------------------
A literal reading of Full_6DOF_RigidBody_Derivation.pdf Ch 13.3-13.4
folds delta_e into Cm (Sec 9.2's own Cm formula includes
"+ Cm_delta_e * eta * delta_e") AND separately lists an identical
My_delta = Cm_delta_e * eta * delta_e * qbar * S * c_bar term in the
"Elevons" block, then Sec 13.4 sums Sigma_My = My_aero + My_delta --
so the elevon pitch moment enters Sigma_My twice.

The same pattern reproduces for roll: Sec 9.3's
    Cl = Cl_beta*beta + Cl_p*(b/2Va)*p + Cl_delta_a*delta_a,
    Cl_delta_a := CL_delta_e * eta * y_e / b,
contributes qbar*S*b*Cl_delta_a*delta_a = qbar*S*CL_delta_e*eta*y_e*delta_a
via Mx_aero -- exactly HALF of the explicit
Mx_delta = 2*qbar*S*CL_delta_e*eta*y_e*delta_a term listed right next to
it. That factor-of-2 mismatch is the fingerprint of the same bug.

The correct, single-count convention -- stated explicitly in
6DOF.pdf Sec 7.2 ("Elevon pitch term is carried once ... Cm is the
clean-wing coefficient with no elevon term"), listed in that document's
own "Summary of corrections", and the ONLY convention consistent with
the numeric moment ledger in FINAL_CASES_WITH_PITCH_TRIM.pdf Sec 5
Case 3 ("moment: -0.013(36.0) + 4.14(0.113) = 0") -- is:

    Cl and Cm computed here are CLEAN: no delta_a, no delta_e term.
    The elevon's roll/pitch moment is added exactly ONCE, in
    elevons.py, as Mx_delta / My_delta.

The same rule applies to lift: Lclean (no delta_e) is used for the NP
lever arm and the wing force block; the elevon's own Delta_L is added
once, also in elevons.py. This mirrors 6DOF.pdf Sec 6.1's explicit use
of the symbol "Lclean" in its force-projection equations.
"""
import numpy as np


def airdata(u, v, w, rho):
   
    Va = float(np.sqrt(u**2 + v**2 + w**2)) # Total Airspeed (Va)
    if Va > 1e-6:
        beta = float(np.arcsin(np.clip(v / Va, -1.0, 1.0)))
    else:
        beta = 0.0
    alpha = float(np.arctan2(w, u))
    qbar = 0.5 * rho * Va**2 #Dynamic Pressure
    return Va, alpha, beta, qbar


def downwash(Tf, Va, qbar, params):
    """
    Front-rotor slipstream correction to the wing's effective alpha and
    dynamic pressure. Doc4 Ch 12 / 6DOF.pdf Sec 5.3.

        wi        = sqrt(Tf / (2 rho Adisk))
        epsilon   = k_eps * wi / Va
        qbar_wing = qbar * (1 + k_q * Tf / (qbar * Adisk))

    IMPLEMENTATION NOTE: epsilon = k_eps*wi/Va is singular as Va -> 0
    (at hover, wi is O(10 m/s) while Va -> 0, so a naive Va floor of
    1e-3 m/s still sends epsilon to O(1e4) rad and overflows the
    downstream stall-blend exponential). Doc4 Sec 12.4 flags this as an
    algebraic loop that must be broken "using the commanded Tf or a
    one-step delay" but does not specify a numeric regularisation. We
    clip epsilon to +/-85 deg: the induced-flow angle cannot physically
    exceed a near-reversed wake, and this bound is only ever active in
    the hover/very-low-speed regime where the downwash-dominated
    q_bar_wing (not alpha_wing) carries the wing loading anyway (Doc4
    Sec 12.3 physical check). qbar_wing itself has a well-defined,
    non-singular hover limit (k_q*Tf/Adisk) and needs no clipping.
    """
    Adisk = params.rotor_disk_area
    Tf = max(float(Tf), 0.0)
    wi = np.sqrt(Tf / (2.0 * params.rho * Adisk))
    eps_cap = np.radians(85.0)
    eps = np.clip(params.k_eps * wi / max(Va, params.Va_reg), -eps_cap, eps_cap)
    if qbar > 1e-9:
        qbar_wing = qbar * (1.0 + params.k_q * Tf / (qbar * Adisk))
    else:
        qbar_wing = params.k_q * Tf / Adisk
    return eps, qbar_wing


def stall_blend_sigma(alpha, alpha_s, M):
    """Beard-McLain blending function sigma(alpha). Doc4 Sec 8.2.

    Defensive clip: the exponentials overflow for |M*alpha| beyond a
    few hundred (double precision), which is far outside anything
    aerodynamically meaningful. Clipping the exponent argument (not
    alpha itself) keeps sigma numerically well-defined without
    distorting the blend anywhere near the actual stall region.
    """
    z1 = np.clip(-M * (alpha - alpha_s), -50.0, 50.0)
    z2 = np.clip(M * (alpha + alpha_s), -50.0, 50.0)
    num = 1.0 + np.exp(z1) + np.exp(z2)
    den = (1.0 + np.exp(z1)) * (1.0 + np.exp(z2))
    return num / den


def aero_coefficients(alpha_wing, p_rate, q_rate, r_rate, beta, Va, params):
    """
    CLEAN (elevon-free) non-dimensional coefficients CL, CD, Cm, Cl, Cn,
    evaluated at alpha_wing (the downwash-corrected angle of attack).
    Doc4 Ch 8-9; see module docstring for why Cm/Cl exclude delta_e/delta_a.

    CL/CD source: if params.polar_table_path is set, an XFLR5 alpha-sweep
    is trusted inside its own coverage (see _table_trust_weight below for
    the blend at its edges) and the parametric flat-plate-blended model
    is used only outside it -- exactly the behavior you asked for
    ("alpha values can be found from here but blended function helps
    with simulator to not crash if it goes above stall"). If
    polar_table_path is None, behavior is unchanged from before: pure
    parametric model everywhere.

    Cm/Cl/Cn stay parametric regardless of polar_table_path. The table
    DOES contain a Cm(alpha) column, but it is deliberately not wired
    into the pitch moment here -- XFLR5's Cm reference point (about the
    CG as configured in your XFLR5 project, presumably, but not
    independently confirmed) would need to be reconciled with the
    NP-lever formula already used in My_aero (Doc4 Sec 9.2's "either
    form, not both" equivalence) before mixing them, or you risk exactly
    the kind of double-count the elevon terms had. Flagging this as an
    open decision rather than guessing.
    """
    Va_s = max(Va, params.Va_reg)
    CL, CD = _lift_drag_coefficients(alpha_wing, params)

    Cm = params.Cm0 + params.Cm_alpha * alpha_wing + \
         params.Cm_q * (params.c_bar / (2.0 * Va_s)) * q_rate           # clean: no delta_e
    Cl = params.Cl_beta * beta + params.Cl_p * (params.b / (2.0 * Va_s)) * p_rate   # clean: no delta_a
    Cn = params.Cn_beta * beta + params.Cn_r * (params.b / (2.0 * Va_s)) * r_rate

    return CL, CD, Cm, Cl, Cn


def _parametric_CL_CD(alpha_wing, params):
    """The pure flat-plate-blended CL/CD model (unchanged from before the
    polar-table integration) -- used everywhere if no table is
    configured, and as the fallback partner in the table blend below."""
    sigma = stall_blend_sigma(alpha_wing, params.alpha_stall, params.stall_blend_M)
    CL_lin = params.CL0 + params.CL_alpha * alpha_wing
    CL = (1.0 - sigma) * CL_lin + \
         sigma * 2.0 * np.sign(alpha_wing) * np.sin(alpha_wing)**2 * np.cos(alpha_wing)

    # CORRECTION: Doc4 Sec 8.3 writes CD = CDp + CL_lin^2/(pi*e*AR), using
    # the UNBLENDED linear CL_lin. That formula is an attached-flow
    # induced-drag model and is only valid pre-stall -- consistent with
    # 6DOF.pdf's own Assumption 5 ("Pre-stall operation, envelope capped
    # near alpha_s ~ 13 deg"). But alpha_wing legitimately reaches
    # 80-90 deg in hover once the downwash correction (Ch 12) is applied
    # (the wing genuinely sits in near-vertical rotor wash there), and
    # CL_lin is unbounded in alpha, so CL_lin^2 blows up (verified: it
    # produced CD ~ 9.5 and a ~450 N phantom force in a hover integration
    # test -- nonphysical, since a flat plate's CD cannot exceed ~2). We
    # use the BLENDED CL (which itself saturates towards 0 near +/-90 deg,
    # exactly like a real post-stall lift curve) in the induced-drag term
    # instead. This keeps the formula identical to the source pre-stall
    # (sigma ~ 0) and only changes behaviour in the post-stall/hover
    # regime the unblended formula was never valid for.
    CD = params.CDp + (CL**2) / (np.pi * params.oswald_e * params.AR)
    return CL, CD


def _table_trust_weight(alpha_wing, table, blend_width_rad):
    """
    1.0 anywhere INSIDE the table's own alpha range (real computed data
    is trusted fully, never discounted near its own edges), tapering by
    cosine ease down to 0.0 over blend_width_rad beyond whichever edge
    alpha_wing has crossed. Continuous at the boundary: right at the
    edge, weight=1 either way you approach it.
    """
    if table.alpha_min <= alpha_wing <= table.alpha_max:
        return 1.0
    d_outside = (table.alpha_min - alpha_wing) if alpha_wing < table.alpha_min \
        else (alpha_wing - table.alpha_max)
    if d_outside >= blend_width_rad:
        return 0.0
    t = 1.0 - d_outside / blend_width_rad  # 1 at the edge, 0 at blend_width_rad beyond it
    return 0.5 - 0.5 * np.cos(np.pi * t)   # cosine ease, matches Selig 2014 Eq.(10) style


def _lift_drag_coefficients(alpha_wing, params):
    if not params.polar_table_path:
        return _parametric_CL_CD(alpha_wing, params)

    from . import polar_table
    table = polar_table.load_xflr5_polar(params.polar_table_path)

    CL_param, CD_param = _parametric_CL_CD(alpha_wing, params)
    weight = _table_trust_weight(alpha_wing, table, np.radians(params.polar_blend_deg))
    if weight <= 0.0:
        return CL_param, CD_param
    CL_tab, CD_tab, _Cm_tab = table.interp(alpha_wing)
    if weight >= 1.0:
        return float(CL_tab), float(CD_tab)
    CL = weight * CL_tab + (1.0 - weight) * CL_param
    CD = weight * CD_tab + (1.0 - weight) * CD_param
    return float(CL), float(CD)


def clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, q_rate, Va, beta, params):
    """
    Dimensional CLEAN wing forces (wind-axis L, D, Y) and moments
    (body-axis Mx, My, Mz). Doc4 Sec 8.6, 9.4.

    NOTE on My_aero: the source assembles this as the neutral-point
    lever term on Lclean PLUS the pitch-damping term only
    (Full_6DOF Sec 13.3: "My,aero = -(xnp-xcg)Lclean + qbar*S*c_bar*
    Cmq*(c_bar/2Va)*q"). It deliberately does NOT also multiply the full
    Cm (which contains Cm_alpha*alpha) by qbar*S*c_bar -- that would
    double-count the static-stiffness term, since Lclean's NP lever arm
    already represents it (Full_6DOF Sec 9.2: the neutral-point lever
    form and the Cm_alpha coefficient form are "identical ... either may
    be used", not both at once). We follow the source's explicit
    Ch 13.3 assembly here.
    """
    Va_s = max(Va, params.Va_reg)
    Lclean = qbar_wing * params.S * (CL + params.CL_q * (params.c_bar / (2.0*Va_s)) * q_rate)
    D = qbar_wing * params.S * CD
    Y = qbar_wing * params.S * params.CY_beta * beta

    Mx_aero = qbar_wing * params.S * params.b * Cl
    My_aero = -(params.xnp - params.xcg) * Lclean + \
              qbar_wing * params.S * params.c_bar * params.Cm_q * (params.c_bar / (2.0*Va_s)) * q_rate
    Mz_aero = qbar_wing * params.S * params.b * Cn

    return Lclean, D, Y, Mx_aero, My_aero, Mz_aero


def project_to_body(Lclean, D, Y, alpha_wing, beta):
    """
    Full-sideslip wind-to-body projection, Doc4 Ch 8.7 (with the
    corrected sign on the Fz sideslip-coupling term):

        Fx = L sin(a) - D cos(a) cos(b) - Y cos(a) sin(b)
        Fy = Y cos(b) - D sin(b)
        Fz = -L cos(a) - D sin(a) cos(b) - Y sin(a) sin(b)
    """
    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)
    sb, cb = np.sin(beta), np.cos(beta)
    Fx = Lclean * sa - D * ca * cb - Y * ca * sb
    Fy = Y * cb - D * sb
    Fz = -Lclean * ca - D * sa * cb - Y * sa * sb
    return Fx, Fy, Fz