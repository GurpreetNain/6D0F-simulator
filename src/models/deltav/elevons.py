import numpy as np # Import the numpy library for math operations


def split(delta1, delta2):
    """delta_e (pitch, symmetric), delta_a (roll, differential).
    """
    delta_e = 0.5 * (delta1 + delta2) # direct input delta1, delta 2 - pitch
    delta_a = 0.5 * (delta2 - delta1) # direct input delta1, delta 2 - roll
    return delta_e, delta_a           # Return the calculated pitch and roll inputs


def efficiency(delta_rad, params):
    """
    DATCOM large-deflection efficiency eta(delta)
    """
    deg = np.degrees(np.abs(delta_rad)) # Convert the absolute deflection angle from radians to degrees
    return float(np.interp(deg, params.eta_table_deg, params.eta_table_val)) # Look up and interpolate the efficiency factor from the DATCOM table


def apply_stall_constraint(delta_e, alpha_wing, params):
    """
    delta_e >= 0 once the wing is at/above the stall cap: down-elevon
    would push the near-stall aft wing past separation.
    Doc4 Sec 11.6; FINAL_CASES Sec 2 ("Stall constraint").
    Hard operating limit enforced here, not part of the continuous
    dynamics -- mirrors how it is used in the trim cases ("Wing at cap: delta_e = 0").
    """
    if alpha_wing >= params.alpha_stall:    # Check if the wing's angle of attack has reached or exceeded the stall limit
        return max(delta_e, 0.0)            # If stalling, prevent trailing-edge down (negative) elevon which worsens separation; clamp to 0 or positive
    return delta_e                          # If not stalling, return the requested pitch elevon deflection as-is


def elevon_forces_moments(delta_e, delta_a, alpha_wing, qbar_wing, params):
    """
    Elevon force (in the slipstream-corrected qbar_wing, alpha_wing) and
    moment, added exactly once to the vehicle's Sigma_F / Sigma_M.
    

        Delta_L   = -CL_delta_e * eta(delta_e) * delta_e * qbar_wing * S
        Delta_De  =  qbar_wing * S * k_e * (eta(delta_e)*delta_e)^2
        Fx_delta  =  Delta_L*sin(a) - Delta_De*cos(a)
        Fz_delta  = -Delta_L*cos(a) - Delta_De*sin(a)
        Mx_delta  =  2*qbar_wing*S*CL_delta_e*eta(delta_a)*delta_a*y_elevon
        My_delta  =  Cm_delta_e*eta(delta_e)*delta_e*qbar_wing*S*c_bar
        Mz_delta  =  0   (no rudder; finless airframe)
    """
    eta_e = efficiency(delta_e, params) # Calculate the aerodynamic efficiency loss for the pitch deflection
    eta_a = efficiency(delta_a, params) # Calculate the aerodynamic efficiency loss for the roll deflection

    dL = -params.CL_delta_e * eta_e * delta_e * qbar_wing * params.S   # lift loss due to elevons deflection
    dDe = qbar_wing * params.S * params.k_e * (eta_e * delta_e) ** 2   # drag (airbrake) caused due to large deflection of elevons

    sa, ca = np.sin(alpha_wing), np.cos(alpha_wing)          # Calculate sin and cos of wing AOA to project forces into the body frame
    Fx = dL * sa - dDe * ca                                  # Total force in x due to elevons
    Fz = -dL * ca - dDe * sa                                 # Total force in z due to elevons

    Mx = 2.0 * qbar_wing * params.S * params.CL_delta_e * eta_a * delta_a * params.y_elevon   # Moment in x due to elevons
    My = params.Cm_delta_e * eta_e * delta_e * qbar_wing * params.S * params.c_bar            # Moment in y due to elevons
    Mz = 0.0                                                                                  # Moment in z due to elevons - no vertical fin

    return {"delta_L": dL, "delta_De": dDe, "Fx": Fx, "Fy": 0.0, "Fz": Fz,      # Pack the local forces (L, D) and body-axis forces (X, Y, Z) into a dictionary
            "Mx": Mx, "My": My, "Mz": Mz, "eta_e": eta_e, "eta_a": eta_a}       # Pack body-axis moments (X, Y, Z) and efficiencies into the same dictionary to return