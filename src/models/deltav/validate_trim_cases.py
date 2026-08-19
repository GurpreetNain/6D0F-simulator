"""
Static sanity checks of the force/moment build-up against
FINAL_CASES_WITH_PITCH_TRIM.pdf's own worked numbers. These do NOT run
the time integrator -- they evaluate rotors.py / lifting_body.py /
elevons.py / gravity.py at hand-specified trim points and compare
against the paper's closed-form results. This is the fastest way to
catch a sign error or a unit mismatch before trusting the full
simulation.

Run with:  python -m deltav.validate_trim_cases
(from src/models/, so `deltav` resolves as a package).
"""
import numpy as np

from deltav.params import InterceptorParams
from deltav import rotors, lifting_body, elevons, gravity
from deltav.interceptor import Interceptor
from deltav.trim import case3_head_on_trim


def case1_hover():
    print("=" * 70)
    print("CASE 1 -- Hover (FINAL_CASES Sec 5, Case 1)")
    print("=" * 70)
    p = InterceptorParams()
    W = p.mass * p.g
    print(f"W = {W:.2f} N   (paper: 29.43 N)")

    # Paper's closed-form split: 2Tf + 2Ta = W, Tf*l_front = Ta*l_aft
    ratio = p.l_aft / p.l_front              # = Tf/Ta
    Ta = W / (2.0 * (ratio + 1.0))
    Tf = ratio * Ta
    print(f"Tf = {Tf:.2f} N/rotor  (paper: 6.56 N, {100*Tf/W:.0f}% of W -> paper 22%)")
    print(f"Ta = {Ta:.2f} N/rotor  (paper: 8.15 N, {100*Ta/W:.0f}% of W -> paper 28%)")
    print(f"Tf/Ta = {Tf/Ta:.3f}          (paper: 0.805)")

    # Drive rotors.py with rotor speeds solved to hit these thrusts via CT
    # (only meaningful once CT is a real measured value -- see
    # params.py NEEDS_MEASUREMENT notes -- but the geometric/sign checks
    # below do not depend on CT being correct, only self-consistent).
    # Drive rotors.py with rotor speeds solved to hit these thrusts. At
    # hover (V_N=0) J=0 for every rotor, so CT/CQ come from the J=0 row
    # of the table -- which was set to match the old fixed constants
    # exactly, so this reproduces the previous (pre-advance-ratio) check.
    CT_hover = p.CT_table_val[0]
    n_front = np.sqrt(Tf / (CT_hover * p.rho * p.rotor_diameter ** 4))
    n_aft = np.sqrt(Ta / (CT_hover * p.rho * p.rotor_diameter ** 4))
    n = np.array([n_front, n_aft, n_front, n_aft])   # [1,2,3,4] = [FR,AL,FL,AR]
    lam = np.zeros(4)

    rot = rotors.rotor_forces_moments(n, lam, p)
    Fg = gravity.gravity_forces(0.0, 0.0, p.mass, p.g)
    SFz = np.sum(rot["FTz"]) + Fg[2]
    SMy = np.sum(rot["MTy"])
    SMx = np.sum(rot["MTx"])

    print(f"\nrotors.py + gravity.py check at this trim point:")
    print(f"  Sigma Fz = {SFz:+.4f} N     (expect ~0)")
    print(f"  Sigma My = {SMy:+.4f} N m   (expect ~0)")
    print(f"  Sigma Mx = {SMx:+.4f} N m   (expect ~0, symmetric front/aft speeds)")
    ok = abs(SFz) < 1e-6 and abs(SMy) < 1e-6 and abs(SMx) < 1e-6
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def case3_head_on():
    print()
    print("=" * 70)
    print("CASE 3 -- Head-on interception, 50 m/s (FINAL_CASES Sec 5, Case 3)")
    print("=" * 70)
    p = InterceptorParams()

    # Paper's numbers for this case (Sec 5, Case 3), for comparison only:
    delta_e = np.radians(6.5)
    alpha = np.radians(3.3)
    Lclean_paper = 36.0
    dLe_paper = -6.7

    # Build the SAME trim point fly_case3.py flies (deltav.trim), and read
    # the numbers straight out of Interceptor._compute() -- this runs the
    # real front-pair thrust and tilt angle through the actual downwash
    # gate, rather than manually zeroing Tf the way this check used to.
    # (That manual zeroing was exactly why this static check never caught
    # the missing tilt gate that fly_case3.py's dynamic run did.)
    x0, u_in = case3_head_on_trim(p)
    vehicle = Interceptor(params=p)
    _, report = vehicle._compute(x0, u_in)

    alpha_wing = report["alpha_wing"]
    Lclean = report["Lclean"]
    dLe = report["elevon"]["delta_L"]

    print(f"alpha_wing = {np.degrees(alpha_wing):.2f} deg (paper alpha: {np.degrees(alpha):.1f} deg)")
    print(f"Lclean     = {Lclean:.1f} N        (paper: {Lclean_paper} N)")
    print(f"Delta_L    = {dLe:.1f} N        (paper Delta_Le: {dLe_paper} N)")

    # Paper's moment ledger (Sec 5, Case 3): -0.013*Lclean_paper + 4.14*delta_e[rad] == 0
    ledger_paper = -0.013 * Lclean_paper + 4.14 * delta_e
    print(f"\nPaper's own ledger check: -0.013*Lclean + 4.14*delta_e = {ledger_paper:+.4f} (paper claims ~0)")

    # Our single-count moment: My_aero (clean, NP-lever term only) + My_delta
    SMy = report["My_aero"] + report["elevon"]["My"]
    print(f"Our Sigma_My = My_aero + My_delta = {SMy:+.4f} N m (expect ~0 at trim)")
    ok = abs(SMy) < 0.5   # loose tolerance: our CL_alpha/Cm_q etc are NEEDS_MEASUREMENT placeholders
    print(f"  -> {'PASS (within placeholder-coefficient tolerance)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = case1_hover()
    ok3 = case3_head_on()
    print()
    print("=" * 70)
    print(f"Overall: {'ALL CHECKS PASSED' if (ok1 and ok3) else 'CHECK FAILURES ABOVE'}")
    print("=" * 70)