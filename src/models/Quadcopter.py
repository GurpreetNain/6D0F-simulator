"""
6-DOF nonlinear rigid-body model of the winged quad-tiltrotor
interceptor. Implements x_dot = f(x, u) exactly as assembled in
Full_6DOF_RigidBody_Derivation.pdf Ch 13, using the 12-STATE, direct-
rotor-speed-input formulation of 6DOF.pdf / Interceptor_Input_State_
Dependency_Map.pdf (no rotor motor-lag state -- that is Doc4's separate
13-state variant, not used here per your selection).

State  x  (12,): [xe, ye, h, u, v, w, phi, theta, psi, p, q, r]
Input  u  (10,): [delta1, delta2, lam1, lam2, lam3, lam4, n1, n2, n3, n4]
    delta1, delta2 : elevon deflections, rad (left, right)
    lam1..4        : rotor tilt, rad (0 = hover/up, pi/2 = cruise/fwd)
    n1..4          : rotor speed, rev/s

Still-air assumption (6DOF.pdf Sec 3, item 3): "no wind/gust in this
version (relative velocity = body velocity)". Wind is therefore NOT
wired in here, unlike the previous Quadcopter.py's Environment coupling.
If you want gusts, feed a wind vector into airdata()/u,v,w before
calling _dynamics -- that is a model extension beyond the source docs,
not part of the validated equations, and should be flagged as such in
any published results.

--------------------------------------------------------------------
Interface note (why this isn't a numeric drop-in for the old file)
--------------------------------------------------------------------
The method names get_state_vector() / set_control_vector() /
state_update(dt) are kept identical to the previous Quadcopter.py so
the rest of your pipeline (loggers, GUI, etc.) needs minimal rewiring.
But the STATE is now 12 Euler-angle elements (not 13 quaternion
elements) and the CONTROL is now 10 raw elevon/tilt/rpm elements (not
4 [thrust, tx, ty, tz]) -- this is unavoidable: the old model's
[thrust,3 torques] abstraction has no way to represent tilt angle or
elevon deflection, and the new vehicle's forces genuinely depend on
those. Any controller/allocator that fed the old 4-element control
vector must be replaced by one that outputs delta1, delta2, lam(1:4),
n(1:4) directly, or by a trim/allocation layer built on
Full_6DOF_RigidBody_Derivation.pdf Ch 14.
"""
import numpy as np

from .params import InterceptorParams
from . import rotors, lifting_body, elevons, gravity


class Quadcopter:
    STATE_NAMES = ["xe", "ye", "h", "u", "v", "w",
                   "phi", "theta", "psi", "p", "q", "r"]
    INPUT_NAMES = ["delta1", "delta2",
                   "lam1", "lam2", "lam3", "lam4",
                   "n1", "n2", "n3", "n4"]

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
        """Fixed-step RK4 integration of x_dot = f(x, u), Doc4 Sec 14.5."""
        x = self._state_vector
        u_in = self._control_vector
        k1 = self._dynamics(x, u_in)
        k2 = self._dynamics(x + 0.5 * dt * k1, u_in)
        k3 = self._dynamics(x + 0.5 * dt * k2, u_in)
        k4 = self._dynamics(x + dt * k3, u_in)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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
        rot = rotors.rotor_forces_moments(n, lam, p, u=u, v=v, w=w)
        Tf = rotors.front_pair_thrust(rot["T"])
        Mgyro = rotors.gyroscopic_moment(
            np.array([p_rate, q_rate, r_rate]), n, lam, p)

        # 2. Airdata + downwash coupling ------------------------------------
        Va, alpha, beta, qbar = lifting_body.airdata(u, v, w, p.rho)
        eps, qbar_wing = lifting_body.downwash(Tf, Va, qbar, p)
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
        SFx = Fg[0] + Fx_aero + ev["Fx"] + np.sum(rot["FTx"])
        SFy = Fg[1] + Fy_aero + ev["Fy"]
        SFz = Fg[2] + Fz_aero + ev["Fz"] + np.sum(rot["FTz"])

        SMx = Mx_aero + ev["Mx"] + np.sum(rot["MTx"] + rot["MQx"]) + Mgyro[0]
        SMy = My_aero + ev["My"] + np.sum(rot["MTy"]) + Mgyro[1]
        SMz = Mz_aero + ev["Mz"] + np.sum(rot["MTz"] + rot["MQz"]) + Mgyro[2]

        # 8. State derivatives (Doc4 Sec 13.5) -----------------------------------
        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)
        ctheta_safe = ctheta if abs(ctheta) > 1e-6 else np.sign(ctheta or 1.0) * 1e-6

        xdot = np.empty(12)

        # navigation (no inputs)
        xdot[0] = (u * ctheta * cpsi
                   + v * (sphi * stheta * cpsi - cphi * spsi)
                   + w * (cphi * stheta * cpsi + sphi * spsi))
        xdot[1] = (u * ctheta * spsi
                   + v * (sphi * stheta * spsi + cphi * cpsi)
                   + w * (cphi * stheta * spsi - sphi * cpsi))
        xdot[2] = u * stheta - v * sphi * ctheta - w * cphi * ctheta

        # translational dynamics
        xdot[3] = SFx / p.mass - q_rate * w + r_rate * v
        xdot[4] = SFy / p.mass - r_rate * u + p_rate * w
        xdot[5] = SFz / p.mass - p_rate * v + q_rate * u

        # attitude kinematics (no inputs) -- singularity guarded at |theta|=90deg
        xdot[6] = p_rate + (q_rate * sphi + r_rate * cphi) * (stheta / ctheta_safe)
        xdot[7] = q_rate * cphi - r_rate * sphi
        xdot[8] = (q_rate * sphi + r_rate * cphi) / ctheta_safe

        # rotational dynamics, Ixz != 0 coupled
        Ixx, Iyy, Izz, Ixz = p.Ixx, p.Iyy, p.Izz, p.Ixz
        Gamma = Ixx * Izz - Ixz ** 2
        xdot[9] = (Ixz * (Ixx - Iyy + Izz) * p_rate * q_rate
                   + (Iyy * Izz - Izz ** 2 - Ixz ** 2) * q_rate * r_rate
                   + Izz * SMx + Ixz * SMz) / Gamma
        xdot[10] = ((Izz - Ixx) * p_rate * r_rate
                    + Ixz * (r_rate ** 2 - p_rate ** 2)
                    + SMy) / Iyy
        xdot[11] = ((Ixz ** 2 + Ixx ** 2 - Ixx * Iyy) * p_rate * q_rate
                    + Ixz * (Iyy - Ixx - Izz) * q_rate * r_rate
                    + Ixz * SMx + Ixx * SMz) / Gamma

        report = {
            "SFx": SFx, "SFy": SFy, "SFz": SFz,
            "SMx": SMx, "SMy": SMy, "SMz": SMz,
            "rotor_T": rot["T"], "rotor_Q": rot["Q"],
            "rotor_J": rot["J"], "rotor_CT": rot["CT"], "rotor_CQ": rot["CQ"],
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
          rotor_torques : {Q1..Q4}       N m, reaction torque per rotor (Doc4 Sec 10.5)
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
        """Full intermediate breakdown (all terms, not just totals) for
        debugging / reproducing the FINAL_CASES ledger checks."""
        _, report = self._compute(self._state_vector, self._control_vector)
        return report


def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi