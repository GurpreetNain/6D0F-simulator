"""
Open-loop ROTOR dynamics test for the Interceptor: no rotor-speed/tilt
controller, no assumption your rotor inputs balance to zero -- but
elevons are NOT a free input either. Elevon deflection isn't something
you choose; it's whatever balances the pitching moment (SMy=0) at the
angle of attack the vehicle actually has, so it's re-solved every
timestep (trim.py's solve_delta_e_for_moment(), the same zero-moment
condition trim.py's steady-trim solver uses for delta_e, just re-solved
continuously here instead of once). You give it a starting
position/velocity/attitude and HOLD a fixed rotor speed and tilt --
optionally stepping to a second rotor command partway through (a
doublet-style test) -- and this integrates the real 12-state EOM forward
in time (state_update(), the same RK4 every other script uses), plots it,
logs the full time history to CSV, and flags (not blocks) anywhere the
run left the validity range of the underlying models.

Usage (from src/models/):
    python -m deltav.fly_dynamics --n 200 --lam 0 --t 10

Easier than typing flags every time: every default below is read from
dynamics_input.py -- edit the values in that file and just run
`python -m deltav.fly_dynamics` with no flags at all. Any flag you DO pass
on the command line overrides that one value from the file; everything
else still comes from the file.

Initial condition -- everything below defaults to a dead-stop hover
start (the original behavior); only override what you actually want to
test (e.g. --u0 30 to start already at 30 m/s forward instead of at rest):
    --x --y --z              start position, m (z is altitude, positive = up)
    --u0 --v0 --w0           start body-axis velocity, m/s (default 0,0,0)
    --phi0 --theta0 --psi0   start attitude, deg (default 0,0,0)
    --p0 --q0 --r0           start body rates, deg/s (default 0,0,0)

Held ROTOR command, applied from t=0 (or until --step_t, if given).
Elevons are not set here -- see the module docstring above:
    --n --lam                rotor speed (rev/s), tilt (deg) -- broadcast
                              to all 4 rotors
    --n1..--n4, --lam1..--lam4
                              per-rotor overrides, take priority over the
                              broadcast values

Optional step input -- hold one rotor command, then switch to a second
one partway through the run:
    --step_t                 time, s, to switch (default None = no step,
                              rotor command held constant for the whole run)
    --step_n --step_lam      post-step broadcast values
    --step_n1..--step_n4, --step_lam1..--step_lam4
                              post-step per-rotor overrides

    --t                       duration, s (default 10)
    --dt                      timestep, s (default 0.01)
    --out                     output file basename, default "dynamics_test"
                              (writes <out>.png and <out>.csv)
"""
import argparse
import csv
import os
import sys
import numpy as np

from .params import InterceptorParams
from .interceptor import Interceptor
from . import trim
from . import dynamics_input as di

# Project root, so the animation log below can reach src.common/src.logging
# the same way scripts/run_interceptor_simulation.py does -- needed because
# fly_dynamics.py is normally run as `python -m deltav.fly_dynamics` from
# src/models/, where "src" itself is never on sys.path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.common.utils import euler_to_quat
from src.logging.data_logger import DataLogger

MU_VALID_LIMIT = 0.35   # oblique-flow normal-force/P-factor theory (rotors.py)
                          # is only reliable roughly below mu~0.3-0.4 (rotor-block
                          # doc Sec 7.5) -- flagged after the run, not enforced;
                          # this tool is meant to show you what happens even
                          # outside a model's validity range, not hide it.


def _rotor_command_from_args(n_b, lam_b, n1, n2, n3, n4, lam1, lam2, lam3, lam4):
    """Per-rotor overrides win over the broadcast value; returns
    (n[4], lam_deg[4])."""
    n = [n1 if n1 is not None else n_b,
         n2 if n2 is not None else n_b,
         n3 if n3 is not None else n_b,
         n4 if n4 is not None else n_b]
    lam_deg = [lam1 if lam1 is not None else lam_b,
               lam2 if lam2 is not None else lam_b,
               lam3 if lam3 is not None else lam_b,
               lam4 if lam4 is not None else lam_b]
    return n, lam_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # ---- initial condition (defaults from dynamics_input.py) --------------
    parser.add_argument("--x", type=float, default=di.X0, help="start xe, m")
    parser.add_argument("--y", type=float, default=di.Y0, help="start ye, m")
    parser.add_argument("--z", type=float, default=di.Z0, help="start altitude, m (positive = up)")
    parser.add_argument("--u0", type=float, default=di.U0, help="start body-x velocity, m/s")
    parser.add_argument("--v0", type=float, default=di.V0, help="start body-y velocity, m/s")
    parser.add_argument("--w0", type=float, default=di.W0, help="start body-z velocity, m/s")
    parser.add_argument("--phi0", type=float, default=di.PHI0, help="start roll, deg")
    parser.add_argument("--theta0", type=float, default=di.THETA0, help="start pitch, deg")
    parser.add_argument("--psi0", type=float, default=di.PSI0, help="start yaw, deg")
    parser.add_argument("--p0", type=float, default=di.P0, help="start roll rate, deg/s")
    parser.add_argument("--q0", type=float, default=di.Q0, help="start pitch rate, deg/s")
    parser.add_argument("--r0", type=float, default=di.R0, help="start yaw rate, deg/s")

    # ---- held rotor command, phase 1 (from t=0) ----------------------------
    parser.add_argument("--n", type=float, default=di.N, help="rotor speed, rev/s, all 4 rotors")
    parser.add_argument("--lam", type=float, default=di.LAM, help="rotor tilt, deg, all 4 rotors")
    parser.add_argument("--n1", type=float, default=di.N1)
    parser.add_argument("--n2", type=float, default=di.N2)
    parser.add_argument("--n3", type=float, default=di.N3)
    parser.add_argument("--n4", type=float, default=di.N4)
    parser.add_argument("--lam1", type=float, default=di.LAM1)
    parser.add_argument("--lam2", type=float, default=di.LAM2)
    parser.add_argument("--lam3", type=float, default=di.LAM3)
    parser.add_argument("--lam4", type=float, default=di.LAM4)

    # ---- optional step input, phase 2 (from --step_t onward) --------------
    parser.add_argument("--step_t", type=float, default=di.STEP_T, help="time, s, to switch to the step_* rotor command")
    parser.add_argument("--step_n", type=float, default=di.STEP_N)
    parser.add_argument("--step_lam", type=float, default=di.STEP_LAM)
    parser.add_argument("--step_n1", type=float, default=di.STEP_N1)
    parser.add_argument("--step_n2", type=float, default=di.STEP_N2)
    parser.add_argument("--step_n3", type=float, default=di.STEP_N3)
    parser.add_argument("--step_n4", type=float, default=di.STEP_N4)
    parser.add_argument("--step_lam1", type=float, default=di.STEP_LAM1)
    parser.add_argument("--step_lam2", type=float, default=di.STEP_LAM2)
    parser.add_argument("--step_lam3", type=float, default=di.STEP_LAM3)
    parser.add_argument("--step_lam4", type=float, default=di.STEP_LAM4)

    parser.add_argument("--t", type=float, default=di.T, help="duration, s")
    parser.add_argument("--dt", type=float, default=di.DT, help="timestep, s")
    parser.add_argument("--out", type=str, default=di.OUT, help="output file basename")
    parser.add_argument("--show", action="store_true",
                         help="don't save the PNG/CSV to disk -- pop up an interactive "
                              "matplotlib window instead, and reuse a single fixed "
                              "animation-log filename (logs/live_interceptor.csv, "
                              "overwritten every run) instead of a new file per run")
    args = parser.parse_args()

    n, lam_deg = _rotor_command_from_args(
        args.n, args.lam, args.n1, args.n2, args.n3, args.n4,
        args.lam1, args.lam2, args.lam3, args.lam4)
    lam_rad = np.radians(lam_deg)

    stepping = args.step_t is not None
    if stepping:
        n_s, lam_s_deg = _rotor_command_from_args(
            args.step_n, args.step_lam, args.step_n1, args.step_n2, args.step_n3, args.step_n4,
            args.step_lam1, args.step_lam2, args.step_lam3, args.step_lam4)
        lam_s_rad = np.radians(lam_s_deg)

    p = InterceptorParams()
    vehicle = Interceptor(params=p)

    x0 = np.array([
        args.x, args.y, args.z,
        args.u0, args.v0, args.w0,
        np.radians(args.phi0), np.radians(args.theta0), np.radians(args.psi0),
        np.radians(args.p0), np.radians(args.q0), np.radians(args.r0),
    ])
    vehicle.set_state_vector(x0)

    print(f"Start position: xe={args.x:.1f}  ye={args.y:.1f}  z(alt)={args.z:.1f} m")
    print(f"Start velocity (body): u={args.u0:.1f}  v={args.v0:.1f}  w={args.w0:.1f} m/s")
    print(f"Start attitude: phi={args.phi0:.1f}  theta={args.theta0:.1f}  psi={args.psi0:.1f} deg")
    print(f"Start rates:    p={args.p0:.1f}  q={args.q0:.1f}  r={args.r0:.1f} deg/s")
    print(f"\nHeld rotor command from t=0 (no controller):")
    print(f"  rotor speed n = {n} rev/s")
    print(f"  rotor tilt  lam = {lam_deg} deg")
    print(f"  elevons: auto-solved every step to zero the pitching moment (SMy=0)")
    if stepping:
        print(f"\nStep at t={args.step_t:.2f}s -> new held rotor command:")
        print(f"  rotor speed n = {n_s} rev/s")
        print(f"  rotor tilt  lam = {lam_s_deg} deg")
    print(f"\nDuration: {args.t:.1f} s at dt={args.dt} s\n")

    n_steps = int(args.t / args.dt)
    t_log = np.zeros(n_steps + 1)
    x_log = np.zeros((n_steps + 1, 12))
    L_log = np.zeros(n_steps + 1)
    D_log = np.zeros(n_steps + 1)
    Va_log = np.zeros(n_steps + 1)
    alpha_log = np.zeros(n_steps + 1)
    alpha_wing_log = np.zeros(n_steps + 1)
    beta_log = np.zeros(n_steps + 1)
    eps_log = np.zeros(n_steps + 1)
    delta_e_log = np.zeros(n_steps + 1)
    elevon_converged_log = np.ones(n_steps + 1, dtype=bool)
    rotor_T_log = np.zeros((n_steps + 1, 4))
    rotor_J_log = np.zeros((n_steps + 1, 4))
    rotor_alpha_i_log = np.zeros((n_steps + 1, 4))
    rotor_mu_i_log = np.zeros((n_steps + 1, 4))

    def _log(i, report):
        L_log[i], D_log[i], Va_log[i] = report["Lclean"], report["D"], report["Va"]
        alpha_log[i], alpha_wing_log[i], beta_log[i] = report["alpha"], report["alpha_wing"], report["beta"]
        eps_log[i] = report["alpha"] - report["alpha_wing"]   # downwash angle, derived (alpha_wing = alpha - eps)
        rotor_T_log[i] = report["rotor_T"]
        rotor_J_log[i] = report["rotor_J"]
        rotor_alpha_i_log[i] = report["rotor_alpha_i"]
        rotor_mu_i_log[i] = report["rotor_mu_i"]

    def _active_rotor_command(t_start):
        if stepping and t_start >= args.step_t:
            return n_s, lam_s_rad
        return n, lam_rad

    # t=0: solve the elevon that balances the moment at the starting state,
    # given the phase-1 rotor command, then set that as the initial control.
    n_now, lam_now = _active_rotor_command(0.0)
    delta_e0, report0, conv0 = trim.solve_delta_e_for_moment(x0, n_now, lam_now, p, vehicle)
    u0_in = np.array([delta_e0, delta_e0, lam_now[0], lam_now[1], lam_now[2], lam_now[3],
                       n_now[0], n_now[1], n_now[2], n_now[3]])
    vehicle.set_control_vector(u0_in)

    x_log[0] = x0
    delta_e_log[0] = delta_e0
    elevon_converged_log[0] = conv0
    _log(0, report0)

    for i in range(1, n_steps + 1):
        t_start = (i - 1) * args.dt
        n_now, lam_now = _active_rotor_command(t_start)
        x_current = vehicle.get_state_vector()
        delta_e, _, conv = trim.solve_delta_e_for_moment(x_current, n_now, lam_now, p, vehicle)
        u_in_now = np.array([delta_e, delta_e, lam_now[0], lam_now[1], lam_now[2], lam_now[3],
                              n_now[0], n_now[1], n_now[2], n_now[3]])
        vehicle.set_control_vector(u_in_now)
        vehicle.state_update(args.dt)

        x_log[i] = vehicle.get_state_vector()
        t_log[i] = i * args.dt
        delta_e_log[i] = delta_e
        elevon_converged_log[i] = conv
        report = vehicle.get_force_moment_breakdown()   # re-evaluates _compute() at the NEW state, for logging
        _log(i, report)

    xe, ye, h = x_log[:, 0], x_log[:, 1], x_log[:, 2]
    u_h, v_h, w_h = x_log[:, 3], x_log[:, 4], x_log[:, 5]
    phi, theta, psi = np.degrees(x_log[:, 6]), np.degrees(x_log[:, 7]), np.degrees(x_log[:, 8])
    p_h, q_h, r_h = np.degrees(x_log[:, 9]), np.degrees(x_log[:, 10]), np.degrees(x_log[:, 11])
    alpha_deg, alpha_wing_deg, beta_deg = np.degrees(alpha_log), np.degrees(alpha_wing_log), np.degrees(beta_log)
    delta_e_deg = np.degrees(delta_e_log)

    print(f"{'t':>6} {'h(alt)':>9} {'Va':>7} {'Lift':>8} {'Drag':>8} {'alpha':>7} {'a_wing':>7} {'delta_e':>8} {'theta':>7}")
    stride = max(1, int(1.0 / args.dt))
    for i in range(0, n_steps + 1, stride):
        print(f"{t_log[i]:6.1f} {h[i]:9.2f} {Va_log[i]:7.2f} {L_log[i]:8.2f} {D_log[i]:8.2f} "
              f"{alpha_deg[i]:7.2f} {alpha_wing_deg[i]:7.2f} {delta_e_deg[i]:8.2f} {theta[i]:7.2f}")

    print(f"\nFinal: h={h[-1]:.2f} m  Va={Va_log[-1]:.2f} m/s  "
          f"phi={phi[-1]:.2f} theta={theta[-1]:.2f} psi={psi[-1]:.2f} deg  delta_e={delta_e_deg[-1]:.2f} deg")

    # ---- validity warnings: flagged, never enforced -- this tool shows you
    # what the model does even outside where it's trustworthy, but you
    # should know when that's happening. ------------------------------------
    warnings = []
    sat_hits = ~elevon_converged_log
    if np.any(sat_hits):
        first_t = t_log[np.argmax(sat_hits)]
        warnings.append(
            f"elevons saturated (hit the +/-{np.degrees(p.delta_max):.0f} deg travel limit without "
            f"nulling the pitching moment) for {np.sum(sat_hits)}/{n_steps + 1} steps (first at "
            f"t={first_t:.2f}s) -- the moment-hold assumption broke down there; alpha/theta after "
            f"that point are not a trimmed result.")
    stall_hits = np.abs(alpha_wing_log) > p.alpha_stall
    if np.any(stall_hits):
        first_t = t_log[np.argmax(stall_hits)]
        warnings.append(
            f"alpha_wing exceeded the {np.degrees(p.alpha_stall):.1f} deg stall cap for "
            f"{np.sum(stall_hits)}/{n_steps + 1} steps (first at t={first_t:.2f}s, "
            f"max |alpha_wing|={np.max(np.abs(alpha_wing_deg)):.1f} deg) -- CL/CD beyond this "
            f"point are the blended flat-plate fallback, not real polar data.")
    J_lo, J_hi = min(p.CT_table_J), max(p.CT_table_J)
    J_out = (rotor_J_log < J_lo) | (rotor_J_log > J_hi)
    if np.any(J_out):
        n_hits = int(np.sum(np.any(J_out, axis=1)))
        warnings.append(
            f"rotor advance ratio J left the tabulated CT/CQ range [{J_lo:.3f}, {J_hi:.3f}] "
            f"for {n_hits}/{n_steps + 1} steps (min J={np.min(rotor_J_log):.3f}, "
            f"max J={np.max(rotor_J_log):.3f}) -- CT/CQ clamp to the table's endpoint out there, "
            f"not a real extrapolation.")
    mu_out = rotor_mu_i_log > MU_VALID_LIMIT
    if np.any(mu_out):
        n_hits = int(np.sum(np.any(mu_out, axis=1)))
        warnings.append(
            f"oblique-flow validity ratio mu_i exceeded {MU_VALID_LIMIT} for {n_hits}/{n_steps + 1} "
            f"steps (max mu_i={np.max(rotor_mu_i_log):.3f}) -- the normal-force/P-factor correction "
            f"is linear blade-element theory, only reliable roughly below mu~0.3-0.4 "
            f"(rotor-block doc Sec 7.5); treat it as indicative, not quantitative, out there.")
    if warnings:
        print("\nValidity warnings (model still ran, but check these before trusting the result):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nNo validity warnings -- elevons stayed within travel limits, and alpha, rotor J, "
              "and rotor mu all stayed inside their tabulated/theoretical ranges.")

    # ---- CSV export (skipped in --show mode) ---------------------------------
    if not args.show:
        csv_path = f"{args.out}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["t", "xe", "ye", "h", "u", "v", "w",
                      "phi_deg", "theta_deg", "psi_deg", "p_deg_s", "q_deg_s", "r_deg_s",
                      "Lift_N", "Drag_N", "Va", "alpha_deg", "alpha_wing_deg", "beta_deg", "eps_deg",
                      "delta_e_deg", "elevon_converged"]
            header += [f"T{i+1}_N" for i in range(4)] + [f"J{i+1}" for i in range(4)] \
                      + [f"alpha_i{i+1}_deg" for i in range(4)] + [f"mu_i{i+1}" for i in range(4)]
            writer.writerow(header)
            for i in range(n_steps + 1):
                row = [t_log[i], xe[i], ye[i], h[i], u_h[i], v_h[i], w_h[i],
                       phi[i], theta[i], psi[i], p_h[i], q_h[i], r_h[i],
                       L_log[i], D_log[i], Va_log[i], alpha_deg[i], alpha_wing_deg[i], beta_deg[i],
                       np.degrees(eps_log[i]), delta_e_deg[i], bool(elevon_converged_log[i])]
                row += list(rotor_T_log[i]) + list(rotor_J_log[i]) \
                       + list(np.degrees(rotor_alpha_i_log[i])) + list(rotor_mu_i_log[i])
                writer.writerow(row)
        print(f"Saved time history: {csv_path}")

    # ---- animation log (13-column quaternion format AnimationWindow/
    # DataLogger already understand -- InterceptorAgent.get_state_for_logger's
    # exact adapter, just applied here post-hoc to x_log instead of live).
    # --show reuses one fixed filename (overwritten every run) instead of a
    # new file per run, so repeated runs don't pile up in logs/. ---------------
    logger = DataLogger(drone_ids=[1])
    for i in range(n_steps + 1):
        xe_i, ye_i, h_i, u_i, v_i, w_i, phi_i, theta_i, psi_i, p_i, q_i, r_i = x_log[i]
        qw, qx, qy, qz = euler_to_quat(phi_i, theta_i, psi_i)
        row = np.array([xe_i, ye_i, -h_i, u_i, v_i, w_i, qw, qx, qy, qz, p_i, q_i, r_i])
        logger.log_step(t_log[i], {1: row})
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    anim_filename = "live_interceptor.csv" if args.show else f"{args.out}_interceptor.csv"
    anim_path = os.path.join(logs_dir, anim_filename)
    logger.export_to_csv(anim_path)
    anim_rel = os.path.relpath(anim_path, _PROJECT_ROOT)
    print(f"{'Animation log (overwritten each run)' if args.show else 'Saved animation log'}: {anim_rel}")
    print(f"To watch it in 3D: from the project root, run")
    print(f"    python scripts/run_animation.py {anim_rel}")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(13, 13))

        ax = axes[0, 0]
        ax.plot(t_log, L_log, label="Lift (Lclean, N)")
        ax.plot(t_log, D_log, label="Drag (N)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("N"); ax.legend(); ax.grid(True)
        ax.set_title("Lift & Drag")

        ax = axes[0, 1]
        ax.plot(t_log, phi, label="phi (roll)")
        ax.plot(t_log, theta, label="theta (pitch)")
        ax.plot(t_log, psi, label="psi (yaw)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg"); ax.legend(); ax.grid(True)
        ax.set_title("Euler angles")

        ax = axes[1, 0]
        ax.plot(t_log, p_h, label="p (roll rate)")
        ax.plot(t_log, q_h, label="q (pitch rate)")
        ax.plot(t_log, r_h, label="r (yaw rate)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg/s"); ax.legend(); ax.grid(True)
        ax.set_title("Body rates")

        ax = axes[1, 1]
        ax.plot(t_log, h, label="altitude (h, m)")
        ax.plot(t_log, Va_log, label="airspeed (Va, m/s)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("m  /  m/s"); ax.legend(); ax.grid(True)
        ax.set_title("Altitude & airspeed (context)")

        ax = axes[2, 0]
        ax.plot(t_log, alpha_deg, label="alpha (kinematic)")
        ax.plot(t_log, alpha_wing_deg, label="alpha_wing (after downwash)")
        ax.plot(t_log, beta_deg, label="beta (sideslip)")
        ax.axhline(np.degrees(p.alpha_stall), color="red", linestyle="--", linewidth=1, label="stall cap")
        ax.axhline(-np.degrees(p.alpha_stall), color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg"); ax.legend(fontsize=8); ax.grid(True)
        ax.set_title("Angle of attack & sideslip")

        ax = axes[2, 1]
        ax.plot(t_log, delta_e_deg, label="delta_e (auto-solved)", color="tab:purple")
        if np.any(sat_hits):
            ax.plot(t_log[sat_hits], delta_e_deg[sat_hits], "x", color="red", markersize=4, label="saturated")
        ax.axhline(np.degrees(p.delta_max), color="gray", linestyle="--", linewidth=1, label="travel limit")
        ax.axhline(-np.degrees(p.delta_max), color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg"); ax.legend(fontsize=8); ax.grid(True)
        ax.set_title("Elevon deflection (auto-solved for SMy=0)")

        fig.suptitle(
            "Open-loop rotor dynamics, auto-trimmed elevons: "
            f"n={n} rev/s, lam={lam_deg} deg"
            + (f"  ->  step at t={args.step_t:.1f}s" if stepping else ""))
        fig.tight_layout()
        if args.show:
            print("\nClose the plot window to finish.")
            plt.show()
        else:
            out_path = f"{args.out}.png"
            fig.savefig(out_path, dpi=130)
            print(f"Saved plot: {out_path}")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
