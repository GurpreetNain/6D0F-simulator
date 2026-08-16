"""
Central parameter set for the winged quad-tiltrotor interceptor.

Consolidated from:
  - 6DOF.pdf                               (Secs 2, 3, 5)
  - Full_6DOF_RigidBody_Derivation.pdf     (Ch 13, Appendix C)
  - FINAL_CASES_WITH_PITCH_TRIM.pdf        (Sec 1, 2, 4)

Every field below is commented with its source. Where the source
documents only give a qualitative "typical" value or leave a
coefficient blank (Appendix C.3 marks several with "--"), the field is
tagged NEEDS_MEASUREMENT. These are placeholders sized to be physically
plausible (so the simulator runs and hovers), NOT validated numbers.
For a publication-grade result, every NEEDS_MEASUREMENT field must be
replaced with real CFD / BEMT / wind-tunnel / motor-bench data.

--------------------------------------------------------------------
KNOWN CROSS-DOCUMENT INCONSISTENCY -- chord length
--------------------------------------------------------------------
6DOF.pdf's vehicle table lists "chord = 0.40 m", while
FINAL_CASES_WITH_PITCH_TRIM.pdf's config table gives "mean chord
c_bar = 0.291 m" directly, and S/b = 0.156/0.60 = 0.26 m is a third,
different number. We use c_bar = 0.291 m here because it is the only
one of the three that reproduces FINAL_CASES' own arithmetic checks
exactly (q*S*c_bar = 1531*0.156*0.291 = 69.6 N*m, matching Sec 1
verbatim). 0.40 m is most likely a root/plan chord rather than the
mean aerodynamic chord used in the coefficient normalisation. Resolve
this against your actual planform before publication.
"""
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class InterceptorParams:
    # ---- environment ----------------------------------------------------
    rho: float = 1.225                      # kg/m^3, sea level (Appendix C.2)
    g: float = 9.81                         # m/s^2

    # ---- mass & geometry (FINAL_CASES Sec 1, Appendix C.2) --------------
    mass: float = 3.0                       # kg
    S: float = 0.156                        # m^2, wing reference area
    b: float = 0.60                         # m, span
    c_bar: float = 0.291                    # m, MAC -- see chord note above
    AR: float = 2.30

    xcg: float = 0.198                      # m, from nose (selected CG, FINAL_CASES Sec 4)
    xnp: float = 0.211                      # m, from nose
    x_front_pivot: float = 0.034            # m, from nose
    x_aft_pivot: float = 0.330              # m, from nose
    l_front: float = 0.164                  # m = xcg - x_front_pivot
    l_aft: float = 0.132                    # m = x_aft_pivot - xcg

    # Rotor hub geometry -- H-frame layout confirmed by CAD/top-view sketch:
    # front pair 164 mm ahead of CG, aft pair 132 mm behind (matches
    # l_front/l_aft above); front pair spans +/-218.5 mm, aft pair spans
    # +/-451.5 mm -- the front and aft crossbars are DIFFERENT widths (an
    # H-shape, not a simple rectangular frame), so a single symmetric
    # "rotor_lateral_arm" cannot represent this -- replaced with explicit
    # per-rotor y stations below.
    rotor_y_front: float = 0.2185           # m, +/- half-span of the FRONT crossbar (rotors 1, 3)
    rotor_y_aft: float = 0.4515             # m, +/- half-span of the AFT crossbar (rotors 2, 4)

    y_elevon: float = 0.110                 # m, elevon lateral centroid arm (FINAL_CASES/App. C)
    x_elevon_chordwise: float = 0.268       # m, chordwise centroid from LE (informational only;
                                             # NOT used in the body-axis EOM, which take moment
                                             # arms as xcg/xnp/l_front/l_aft/y_elevon directly)

    # ---- inertia tensor ---------------------------------------------------
    # NEEDS_MEASUREMENT: Appendix C.3 lists Ixx, Iyy, Izz, Ixz with NO
    # numeric value ("typical" placeholder only). Values below are a
    # flat-plate estimate using the known span (0.60 m) and an assumed
    # ~0.36-0.40 m fuselage length (x_aft_pivot - x_front_pivot region):
    #   Ixx (roll)  ~ m*b^2/12   -- resists the SPANWISE mass distribution
    #   Iyy (pitch) ~ m*L^2/12   -- resists the shorter FORE-AFT distribution
    #   Izz (yaw)   ~ Ixx + Iyy  (thin-body/perpendicular-axis approx.)
    # Since span > fuselage length here, Ixx > Iyy is the physically
    # correct ordering -- an earlier draft of this file had them
    # backwards (Ixx < Iyy), which was verified to produce a spuriously
    # fast (~20-40 rad/s growth rate) open-loop pitch instability in a
    # hover integration test. These are still coarse guesses and MUST be
    # replaced with a real mass-property estimate (CAD inertia or
    # bifilar/trifilar swing test) before any trim/control result is
    # treated as final -- real hardware (motors, battery, structure)
    # concentrated away from the CG typically increases these further.
    Ixx: float = 0.045                      # kg m^2   NEEDS_MEASUREMENT
    Iyy: float = 0.020                      # kg m^2   NEEDS_MEASUREMENT
    Izz: float = 0.060                      # kg m^2   NEEDS_MEASUREMENT
    Ixz: float = 0.003                      # kg m^2   NEEDS_MEASUREMENT

    # ---- lift / drag / moment coefficients ---------------------------------
    CL0: float = 0.0                        # symmetric section (Doc4 Sec 8.2)
    CL_alpha: float = 5.0                   # /rad, "typical" at AR=2.3 (Doc4 Sec 8.2)  NEEDS_MEASUREMENT
    CL_q: float = 3.0                       # /rad, Appendix C.3 "typical"              NEEDS_MEASUREMENT
    CDp: float = 0.02                       # parasite drag, Appendix C.3 "typical"     NEEDS_MEASUREMENT
    oswald_e: float = 0.8                   # Appendix C.3 "typical"                    NEEDS_MEASUREMENT

    Cm0: float = 0.0                        # Appendix C.3 "~0"
    Cm_alpha: float = None                  # computed in __post_init__ from the static-margin
                                             # identity Cm_alpha = -CL_alpha*(xnp-xcg)/c_bar
                                             # (Full_6DOF Sec 9.2) unless overridden explicitly
    Cm_q: float = -10.0                     # /rad, Appendix C.3 "typical"               NEEDS_MEASUREMENT

    # NEEDS_MEASUREMENT: lateral-directional derivatives are explicitly
    # left blank ("--") or only qualitatively described ("small",
    # "weak without a tail") in Appendix C.3 / Full_6DOF Sec 9.3.
    CY_beta: float = -0.10                  # /rad   NEEDS_MEASUREMENT
    Cl_beta: float = -0.05                  # /rad   NEEDS_MEASUREMENT
    Cl_p: float = -0.40                     # /rad   NEEDS_MEASUREMENT
    Cn_beta: float = 0.02                   # /rad, small & weakly stabilising (Sec 9.3)  NEEDS_MEASUREMENT
    Cn_r: float = -0.05                     # /rad, weak, no tail (Sec 9.3)               NEEDS_MEASUREMENT

    alpha_stall: float = np.radians(13.0)   # FINAL_CASES Sec 1 -- the AoA CAP the vehicle
                                             # is operated within (Cases 1-6); kept separate
                                             # from the polar table's own coverage below.
    stall_blend_M: float = 50.0             # Beard-McLain blend sharpness -- not given
                                             # numerically anywhere in the sources; 50 is the
                                             # standard literature default. NEEDS_TUNING --
                                             # only used when polar_table_path is None
                                             # (pure parametric fallback model).

    # --- XFLR5 polar table (replaces the parametric CL(alpha)/CD(alpha)
    # model within the table's own coverage) ------------------------------
    # Set to None to use the pure parametric stall-blend model everywhere
    # (the original behavior). When set, lifting_body.py trusts the table
    # for alpha inside its range and blends smoothly into the SAME
    # flat-plate extension used before (stall_blend_sigma) for alpha
    # outside the table -- e.g. a typical XFLR5 sweep only covers about
    # -23 to +24 deg, but hover's downwash-corrected alpha_wing routinely
    # reaches 80-90 deg, so the blend-to-flat-plate fallback is mandatory,
    # not optional, or the simulator has no defined aero data out there.
    polar_table_path: str = None            # e.g. "config/aero/wing_polar_xflr5.csv"
    polar_blend_deg: float = 5.0            # width of the cosine taper straddling each
                                             # edge of the table's alpha range, blending
                                             # table values -> flat-plate values. Not a
                                             # source concept -- an engineering choice,
                                             # same category as Va_reg/n_reg. NEEDS_TUNING

    # ---- elevon -----------------------------------------------------------
    CL_delta_e: float = 0.246               # /rad, XFLR5-calibrated (FINAL_CASES Sec 2 eq. 3)
    Cm_delta_e: float = 0.059               # /rad, at xcg = 198 mm (FINAL_CASES Sec 2 / Sec 5)
    k_e: float = 0.5                        # elevon drag factor, Appendix C.3 "typical"  NEEDS_MEASUREMENT
    eta_table_deg: tuple = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    eta_table_val: tuple = (1.00, 1.00, 0.87, 0.75, 0.66, 0.60)   # DATCOM, Doc4 Sec 11.5
    delta_max: float = np.radians(25.0)     # mechanical travel limit

    # ---- rotor / propulsion -------------------------------------------------
    # NEEDS_MEASUREMENT: CT, CQ, rotor diameter D and polar inertia Ip
    # are all "--" (no numeric value) in Appendix C.3. Placeholders below
    # are only sized so 4 rotors can roughly hover 3 kg near a plausible
    # rev/s for an 8-inch prop -- replace with real motor/prop bench data.
    rotor_diameter: float = 0.2032          # m (8 in)                      NEEDS_MEASUREMENT

    # CT(J), CQ(J) as interpolated tables over advance ratio J = V_N/(nD),
    # replacing the earlier fixed CT/CQ constants (Selig 2014, Sec IV.D,
    # Eqs. 23-25: thrust/torque coefficients come from lookup tables on
    # advance ratio, not fixed values -- a fixed CT/CQ is only valid at
    # the single flight condition it was tuned for). CT(0)/CQ(0) below
    # match the old constants exactly so hover trim (Case 1) is
    # unaffected; the decline with J is a generic RC-prop shape (CT
    # falls faster than CQ, both roughly -> 0 near J~1.1-1.2) and MUST
    # be replaced with real manufacturer or BEMT (PROPID-style) data --
    # this shape only exists so the model is qualitatively right, not
    # quantitatively validated.                                          NEEDS_MEASUREMENT
    CT_table_J:  tuple = (0.0,  0.2,  0.4,   0.6,   0.8,  1.0,   1.2)
    CT_table_val: tuple = (0.11, 0.10, 0.085, 0.065, 0.04, 0.015, 0.0)
    CQ_table_J:  tuple = (0.0,   0.2,   0.4,   0.6,   0.8,   1.0,  1.2)
    CQ_table_val: tuple = (0.010, 0.0098, 0.0093, 0.0085, 0.0074, 0.006, 0.0045)
    n_reg: float = 5.0                      # rev/s floor for the J = V_N/(nD)
                                             # denominator -- same regularization
                                             # pattern as Va_reg, prevents J from
                                             # diverging at low commanded rpm.
                                             # NEEDS_MEASUREMENT / ENGINEERING CHOICE
    Ip: float = 2.0e-5                      # kg m^2, rotor polar inertia    NEEDS_MEASUREMENT
    k_eps: float = 0.75                     # downwash angle factor, range 0.5-1.0 (Doc4 Sec 12.2)  NEEDS_TUNING
    k_q: float = 0.75                       # dyn. pressure augmentation, range 0.5-1.0             NEEDS_TUNING

    # NOT specified anywhere in the source documents. The rate-damping
    # terms (CLq*c_bar/(2Va)*q, Cmq*c_bar/(2Va)*q, Clp*b/(2Va)*p,
    # Cnr*b/(2Va)*r) are classical non-dimensional stability derivatives
    # that are only meaningful for Va away from zero; none of the three
    # PDFs discuss their Va->0 (hover) limit. A naive 1e-3 m/s floor
    # still amplifies a tiny hover body-rate by a factor of ~150 and was
    # verified to produce a runaway pitch moment in an open-loop hover
    # integration test. Va_reg is the floor used in EVERY 1/Va rate-term
    # in lifting_body.py (not in Va itself, which is left un-floored for
    # airdata/downwash). Below this speed, rate damping smoothly stops
    # growing rather than diverging -- physically defensible (a hovering
    # vehicle has no meaningful "forward-flight" rate derivative) but is
    # an implementation choice beyond what the sources specify, and
    # should be revisited once real low-speed/hover rate-damping data
    # (e.g. from a whirl-tower or CFD sweep) is available.
    Va_reg: float = 2.0                     # m/s   NEEDS_MEASUREMENT / ENGINEERING CHOICE

    rotor_x_arm: np.ndarray = field(
        default_factory=lambda: np.array([0.164, -0.132, 0.164, -0.132]))
    rotor_spin_sign: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, -1.0, -1.0]))

    def __post_init__(self):
        if self.Cm_alpha is None:
            # Static-margin identity, Full_6DOF Sec 9.2:
            #   Cm_alpha = -CL_alpha * (xnp - xcg) / c_bar
            self.Cm_alpha = -self.CL_alpha * (self.xnp - self.xcg) / self.c_bar

        self.rotor_disk_area = np.pi * (self.rotor_diameter / 2.0) ** 2

        # Rotor layout (H-frame, confirmed CAD geometry -- 6DOF.pdf Sec 2.1's
        # rotor-index/spin convention, with the ACTUAL per-crossbar spans):
        #   1 front-right (CCW, +y_front), 2 aft-left  (CCW, -y_aft),
        #   3 front-left  (CW,  -y_front), 4 aft-right (CW,  +y_aft)
        # Front and aft crossbars are different widths -- do not collapse
        # this back into a single symmetric arm.
        yf, ya = self.rotor_y_front, self.rotor_y_aft
        self.rotor_y_arm = np.array([yf, -ya, -yf, ya])

    @classmethod
    def from_json(cls, path):
        """Load overrides from a JSON file; unspecified fields keep the
        documented defaults above."""
        with open(path, "r") as f:
            overrides = json.load(f)
        # numpy arrays can't come straight from JSON lists for dataclass
        # fields with default_factory -- coerce known array fields
        for key in ("rotor_x_arm", "rotor_spin_sign"):
            if key in overrides:
                overrides[key] = np.asarray(overrides[key], dtype=float)
        return cls(**overrides)

    def to_json(self, path):
        # Only serialise genuine dataclass fields -- rotor_disk_area and
        # rotor_y_arm are DERIVED in __post_init__, not real parameters,
        # and must not be round-tripped through from_json (which passes
        # every key straight to the constructor).
        from dataclasses import fields
        field_names = {f.name for f in fields(self)}
        d = {}
        for k in field_names:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        with open(path, "w") as f:
            json.dump(d, f, indent=2)