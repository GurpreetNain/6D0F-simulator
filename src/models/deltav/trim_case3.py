"""
General head-on-intercept trim solver, CLI wrapper around
trim.head_on_intercept_trim(). Unlike case3_head_on_trim() (which just
plugs in the paper's own worked Case 3 numbers), this actually SOLVES for
the angle of attack, elevon deflection, and rotor speed/tilt that trim the
full Interceptor pipeline at whatever distance/altitude/airspeed you give
it -- so it works for "a case like Case 3" at any geometry, not just the
paper's one example.

Usage:
    python -m deltav.trim_case3 --D 2500 --H 200 --Va 50 [--fly 15]

    --D, --H, --Va : intercept geometry (m, m, m/s) -- gamma = atan2(H, D)
    --fly SECONDS  : optional. If given, also integrates the solved trim
                     forward open-loop for that many seconds (same pattern
                     as fly_case3.py) and saves a trajectory + alpha/theta
                     plot, to show whether the solved point actually holds
                     over time, not just at the instant it was solved for.

Run from src/models/ so `deltav` resolves as a package.
"""
import argparse
import numpy as np

from .params import InterceptorParams
from .interceptor import Interceptor
from .trim import head_on_intercept_trim


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--D", type=float, default=2500.0, help="horizontal distance to target, m (default 2500, Case 3)")
    parser.add_argument("--H", type=float, default=200.0, help="altitude offset to target, m (default 200, Case 3)")
    parser.add_argument("--Va", type=float, default=50.0, help="airspeed, m/s (default 50, Case 3)")
    parser.add_argument("--fly", type=float, default=None, help="also integrate open-loop for this many seconds and plot")
    args = parser.parse_args()

    p = InterceptorParams()
    gamma = np.arctan2(args.H, args.D)
    print(f"Geometry: D={args.D:.0f} m, H={args.H:.0f} m -> gamma={np.degrees(gamma):.3f} deg dive, Va={args.Va:.1f} m/s")

    x0, u_in, report = head_on_intercept_trim(args.D, args.H, args.Va, p)

    alpha = report["alpha"]
    delta_e = u_in[0]
    lam = u_in[2]
    n = u_in[6]

    print("\nSolved trim:")
    print(f"  alpha   = {np.degrees(alpha):.3f} deg")
    print(f"  delta_e = {np.degrees(delta_e):.3f} deg")
    print(f"  tilt    = {np.degrees(lam):.3f} deg")
    print(f"  n       = {n:.2f} rev/s ({n*60:.0f} RPM)")

    print("\nForces and moments at the solved trim (all should be ~0):")
    print(f"  SFx,SFy,SFz = {report['SFx']:+.6f}, {report['SFy']:+.6f}, {report['SFz']:+.6f} N")
    print(f"  SMx,SMy,SMz = {report['SMx']:+.6f}, {report['SMy']:+.6f}, {report['SMz']:+.6f} N m")

    print("\nFull breakdown:")
    print(f"  alpha_wing = {np.degrees(report['alpha_wing']):.3f} deg   (before downwash: {np.degrees(alpha):.3f} deg)")
    print(f"  Lclean = {report['Lclean']:.2f} N   D = {report['D']:.2f} N   Y = {report['Y']:.2f} N")
    print(f"  elevon: delta_L = {report['elevon']['delta_L']:.2f} N   My = {report['elevon']['My']:.3f} N m")
    print(f"  per-rotor: T = {report['rotor_T']}")
    print(f"             Q = {report['rotor_Q']}")
    print(f"             J = {report['rotor_J']}")

    if args.fly is not None:
        vehicle = Interceptor(params=p)
        vehicle.set_state_vector(x0)
        vehicle.set_control_vector(u_in)

        dt = 0.01
        n_steps = int(args.fly / dt)
        t_log = np.zeros(n_steps + 1)
        x_log = np.zeros((n_steps + 1, 12))
        alpha_log = np.zeros(n_steps + 1)
        x_log[0] = x0
        alpha_log[0] = np.degrees(np.arctan2(x0[5], x0[3]))

        for i in range(1, n_steps + 1):
            vehicle.state_update(dt)
            x_log[i] = vehicle.get_state_vector()
            t_log[i] = i * dt
            alpha_log[i] = np.degrees(np.arctan2(x_log[i][5], x_log[i][3]))

        xe, ye, h = x_log[:, 0], x_log[:, 1], x_log[:, 2]
        theta_hist = np.degrees(x_log[:, 7])
        Va_hist = np.linalg.norm(x_log[:, 3:6], axis=1)

        print(f"\nIntegrated {args.fly:.0f} s open-loop, controls held fixed at the solved trim:")
        print(f"{'t':>6} {'xe':>10} {'ye':>8} {'h':>9} {'Va':>7} {'alpha':>7} {'theta':>7}")
        stride = max(1, int(2.0 / dt))
        for i in range(0, n_steps + 1, stride):
            print(f"{t_log[i]:6.1f} {xe[i]:10.2f} {ye[i]:8.3f} {h[i]:9.2f} "
                  f"{Va_hist[i]:7.2f} {alpha_log[i]:7.2f} {theta_hist[i]:7.2f}")

        print(f"\nAlpha drift over {args.fly:.0f}s: {alpha_log[0]:.3f} deg -> {alpha_log[-1]:.3f} deg")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12, 5))
            ax3d = fig.add_subplot(1, 2, 1, projection="3d")
            ax3d.plot(xe, ye, -h, "b-")
            ax3d.set_xlabel("xe (m, north)")
            ax3d.set_ylabel("ye (m, east)")
            ax3d.set_zlabel("altitude (m, up)")
            ax3d.set_title(f"Solved trim trajectory ({args.fly:.0f} s)")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(t_log, alpha_log, label="alpha (deg)")
            ax2.plot(t_log, theta_hist, label="theta (deg)")
            ax2.axhline(np.degrees(alpha), color="gray", linestyle="--", linewidth=1, label="solved alpha target")
            ax2.set_xlabel("time (s)")
            ax2.legend()
            ax2.set_title("Attitude vs. time")

            fig.tight_layout()
            out_path = "trim_case3_flight_test.png"
            fig.savefig(out_path, dpi=130)
            print(f"Saved plot: {out_path}")
        except ImportError:
            print("(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
