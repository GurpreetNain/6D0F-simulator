"""
6-DOF nonlinear rigid-body model of the winged quad-tiltrotor
interceptor. Computes x_dot = f(x, u): a 12-state, direct-rotor-speed
-input flight model (no rotor motor-lag state). This is the top-level
vehicle class for the deltav package -- it wires together rotors.py,
lifting_body.py, elevons.py, and gravity.py (Levels 1-6) into the full
force/moment build-up and 12-state EOM (Levels 7-8), then integrates it.

State  x  (12,): [xe, ye, h, u, v, w, phi, theta, psi, p, q, r]
Input  u  (10,): [delta1, delta2, lam1, lam2, lam3, lam4, n1, n2, n3, n4]
    delta1, delta2 : elevon deflections, rad (left, right)
    lam1..4        : rotor tilt, rad (0 = hover/up, pi/2 = cruise/fwd)
    n1..4          : rotor speed, rev/s

Still-air assumption: relative velocity = body velocity, no wind/gust
modeled. To add gusts, feed a wind vector into airdata()'s u,v,w before
calling _dynamics -- that's an extension beyond what's implemented here.

Interface note: get_state_vector() / set_control_vector() /
state_update(dt) match the unrelated Quadcopter class in
src/models/Quadcopter.py by naming convention only -- the two are not
interchangeable. That class is a mass/inertia box-frame quad with a
13-element quaternion state and 4-element [thrust, tx, ty, tz] control;
this one is the winged interceptor with a 12-element Euler-angle state
and a 10-element elevon/tilt/rpm control. Any controller/allocator
built for the old 4-element control vector needs to output delta1,
delta2, lam(1:4), n(1:4) directly to drive this class instead.
"""
import numpy as np

from .params import InterceptorParams
from . import rotors, lifting_body, elevons, gravity


class Interceptor:
    STATE_NAMES = ["xe", "ye", "h", "u", "v", "w",
                   "phi", "theta", "psi", "p", "q", "r"]   # 12-element state x
    INPUT_NAMES = ["delta1", "delta2",
                   "lam1", "lam2", "lam3", "lam4",
                   "n1", "n2", "n3", "n4"]                 # 10-element control u

    def __init__(self, params: InterceptorParams = None, id=1):
        self._id = id
        self.p = params if params is not None else InterceptorParams()
        self._state_vector = np.zeros(12)
        self._control_vector = np.zeros(10)   # [d1, d2, lam1..4, n1..4]

    # ---- kept-identical interface -----------------------------------------
    def get_state_vector(self):
        return self._state_vector

    def set_state_vector(self, x):
        self._state_vector = np.asarray(x, dtype=float).copy()

    def set_control_vector(self, control_vector):
        cv = np.asarray(control_vector, dtype=float)
        if cv.shape != (10,):
            raise ValueError(
                f"Interceptor control vector must have 10 elements "
                f"{self.INPUT_NAMES}, got shape {cv.shape}")
        self._control_vector = cv

    def state_update(self, dt):
        """Fixed-step RK4 integration of x_dot = f(x, u)."""
        x = self._state_vector
        u_in = self._control_vector
        k1 = self._dynamics(x, u_in)                            # slope at the start of the step
        k2 = self._dynamics(x + 0.5 * dt * k1, u_in)             # slope at the midpoint, using k1
        k3 = self._dynamics(x + 0.5 * dt * k2, u_in)             # slope at the midpoint, using k2
        k4 = self._dynamics(x + dt * k3, u_in)                   # slope at the end, using k3
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)    # weighted average of the four slopes
        x_next[6] = _wrap_pi(x_next[6])   # phi
        x_next[8] = _wrap_pi(x_next[8])   # psi
        self._state_vector = x_next

    def _dynamics(self, x, u_in):
        """Thin wrapper used by the RK4 integrator -- just the state
        derivative, no report overhead on the hot path."""
        xdot, _ = self._compute(x, u_in)
        return xdot

    # ---- core physics: x_dot = f(x, u), plus every intermediate ------------
    def _compute(self, x, u_in):
        """
        Single source of truth for the force/moment build-up. Returns
        (xdot, report) so the RK4 integrator and get_submission_report()
        can never drift apart -- both read from exactly this computation,
        never a second copy of the physics.
        """
        p = self.p
        xe, ye, h, u, v, w, phi, theta, psi, p_rate, q_rate, r_rate = x

        delta1, delta2 = u_in[0], u_in[1]
        lam = u_in[2:6]
        n = u_in[6:10]

        # 1. Rotors --------------------------------------------------------
        rot = rotors.rotor_forces_moments(n, lam, p, u=u, v=v, w=w,
                                           p=p_rate, q=q_rate, r=r_rate)
        Tf = rotors.front_pair_thrust(rot["T"])
        lam_f = lam[0]   # front pair (rotors 1, 3) share one tilt actuator
        Mgyro = rotors.gyroscopic_moment(
            np.array([p_rate, q_rate, r_rate]), n, lam, p)

        # 2. Airdata + downwash coupling ------------------------------------
        Va, alpha, beta, qbar = lifting_body.airdata(u, v, w, p)
        eps, qbar_wing = lifting_body.downwash(Tf, lam_f, Va, qbar, p)
        alpha_wing = alpha - eps

        # 3. Elevon input split + stall interlock ----------------------------
        delta_e, delta_a = elevons.split(delta1, delta2)
        delta_e = elevons.apply_stall_constraint(delta_e, alpha_wing, p)

        # 4. Clean wing aerodynamics (elevon-free, see lifting_body.py) -------
        CL, CD, Cm, Cl, Cn = lifting_body.aero_coefficients(
            alpha_wing, p_rate, q_rate, r_rate, beta, Va, p)
        Lclean, D, Y, Mx_aero, My_aero, Mz_aero = lifting_body.clean_forces_moments(
            CL, CD, Cl, Cn, qbar_wing, q_rate, Va, beta, p)
        Fx_aero, Fy_aero, Fz_aero = lifting_body.project_to_body(
            Lclean, D, Y, alpha_wing, beta)

        # 5. Elevon forces/moments (added exactly once) -----------------------
        ev = elevons.elevon_forces_moments(delta_e, delta_a, alpha_wing, qbar_wing, p)

        # 6. Gravity ----------------------------------------------------------
        Fg = gravity.gravity_forces(phi, theta, p.mass, p.g)

        # 7. Sum forces and moments --------------------------------------------
        # Oblique-flow normal-force / P-factor terms (FNx/y/z, MNx/y/z) are
        # zero whenever every rotor sees purely axial flow (trimmed hover or
        # cruise) and only engage during transition, sideslip, or body rate
        # -- see rotors.py's oblique-flow correction block.
        SFx = Fg[0] + Fx_aero + ev["Fx"] + np.sum(rot["FTx"] + rot["FNx"])
        SFy = Fg[1] + Fy_aero + ev["Fy"] + np.sum(rot["FNy"])
        SFz = Fg[2] + Fz_aero + ev["Fz"] + np.sum(rot["FTz"] + rot["FNz"])

        SMx = Mx_aero + ev["Mx"] + np.sum(rot["MTx"] + rot["MQx"] + rot["MNx"]) + Mgyro[0]
        SMy = My_aero + ev["My"] + np.sum(rot["MTy"] + rot["MNy"]) + Mgyro[1]
        SMz = Mz_aero + ev["Mz"] + np.sum(rot["MTz"] + rot["MQz"] + rot["MNz"]) + Mgyro[2]

        # 8. State derivatives ---------------------------------------------------
        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)
        ctheta_safe = ctheta if abs(ctheta) > 1e-6 else np.sign(ctheta or 1.0) * 1e-6  # avoid divide-by-zero at theta = +-90 deg

        xdot = np.empty(12)

        # navigation: body velocity (u,v,w) rotated into earth-frame position rates
        xdot[0] = (u * ctheta * cpsi                              # xe_dot (north)
                   + v * (sphi * stheta * cpsi - cphi * spsi)
                   + w * (cphi * stheta * cpsi + sphi * spsi))
        xdot[1] = (u * ctheta * spsi                              # ye_dot (east)
                   + v * (sphi * stheta * spsi + cphi * cpsi)
                   + w * (cphi * stheta * spsi - sphi * cpsi))
        xdot[2] = u * stheta - v * sphi * ctheta - w * cphi * ctheta   # h_dot (climb rate, positive up)

        # translational dynamics: Newton's law in the rotating body frame,
        # -q*w+r*v etc. are the Coriolis terms from being in a rotating frame
        xdot[3] = SFx / p.mass - q_rate * w + r_rate * v   # u_dot
        xdot[4] = SFy / p.mass - r_rate * u + p_rate * w   # v_dot
        xdot[5] = SFz / p.mass - p_rate * v + q_rate * u   # w_dot

        # attitude kinematics: how body rates p,q,r translate into Euler angle rates
        xdot[6] = p_rate + (q_rate * sphi + r_rate * cphi) * (stheta / ctheta_safe)   # phi_dot
        xdot[7] = q_rate * cphi - r_rate * sphi                                       # theta_dot
        xdot[8] = (q_rate * sphi + r_rate * cphi) / ctheta_safe                       # psi_dot

        # rotational dynamics: Euler's equations, with Ixz product-of-inertia coupling
        Ixx, Iyy, Izz, Ixz = p.Ixx, p.Iyy, p.Izz, p.Ixz
        Gamma = Ixx * Izz - Ixz ** 2   # shared denominator for the coupled roll/yaw equations
        xdot[9] = (Ixz * (Ixx - Iyy + Izz) * p_rate * q_rate
                   + (Iyy * Izz - Izz ** 2 - Ixz ** 2) * q_rate * r_rate
                   + Izz * SMx + Ixz * SMz) / Gamma        # p_dot (roll acceleration)
        xdot[10] = ((Izz - Ixx) * p_rate * r_rate
                    + Ixz * (r_rate ** 2 - p_rate ** 2)
                    + SMy) / Iyy                           # q_dot (pitch acceleration)
        xdot[11] = ((Ixz ** 2 + Ixx ** 2 - Ixx * Iyy) * p_rate * q_rate
                    + Ixz * (Iyy - Ixx - Izz) * q_rate * r_rate
                    + Ixz * SMx + Ixx * SMz) / Gamma        # r_dot (yaw acceleration)

        report = {
            "SFx": SFx, "SFy": SFy, "SFz": SFz,
            "SMx": SMx, "SMy": SMy, "SMz": SMz,
            "rotor_T": rot["T"], "rotor_Q": rot["Q"],
            "rotor_J": rot["J"], "rotor_CT": rot["CT"], "rotor_CQ": rot["CQ"],
            "rotor_alpha_i": rot["alpha_i"], "rotor_mu_i": rot["mu_i"],
            "rotor_FN": np.stack([rot["FNx"], rot["FNy"], rot["FNz"]], axis=-1),
            "rotor_MN": np.stack([rot["MNx"], rot["MNy"], rot["MNz"]], axis=-1),
            "Tf_front_pair": Tf,
            "Mgyro": Mgyro,
            "Lclean": Lclean, "D": D, "Y": Y,
            "Mx_aero": Mx_aero, "My_aero": My_aero, "Mz_aero": Mz_aero,
            "elevon": ev,
            "Va": Va, "alpha": alpha, "alpha_wing": alpha_wing,
            "beta": beta, "qbar": qbar, "qbar_wing": qbar_wing,
            "delta_e": delta_e, "delta_a": delta_a,
        }
        return xdot, report

    # ---- submission-format report -------------------------------------------
    def get_submission_report(self):
        """
        The F, M, and per-rotor torque summary for reporting/documentation
        -- re-runs the exact same computation state_update() uses (via
        _compute), so this can never disagree with the simulated dynamics.

        Returns a dict:
          forces  : {Fx, Fy, Fz}         body-axis, N, Sigma F (includes gravity)
          moments : {Mx, My, Mz}         body-axis, N m, Sigma M (includes gyroscopic)
          rotor_torques : {Q1..Q4}       N m, reaction torque per rotor
          rotor_thrusts : {T1..T4}       N, thrust per rotor
        """
        xdot, r = self._compute(self._state_vector, self._control_vector)
        return {
            "forces": {"Fx": r["SFx"], "Fy": r["SFy"], "Fz": r["SFz"]},
            "moments": {"Mx": r["SMx"], "My": r["SMy"], "Mz": r["SMz"]},
            "rotor_thrusts": {f"T{i+1}": r["rotor_T"][i] for i in range(4)},
            "rotor_torques": {f"Q{i+1}": r["rotor_Q"][i] for i in range(4)},
            "rotor_advance_ratio": {f"J{i+1}": r["rotor_J"][i] for i in range(4)},
        }

    def get_force_moment_breakdown(self):
        """Full intermediate breakdown (all terms, not just totals) --
        useful for debugging or cross-checking against hand-worked trim
        numbers."""
        _, report = self._compute(self._state_vector, self._control_vector)
        return report


def _wrap_pi(angle):
    """Keeps an angle in (-pi, pi] after each integration step."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
