"""
Edit this file to set up a dynamics-test run without typing a long command
line -- every value here is exactly one of fly_dynamics.py's CLI flags (see
that file's own docstring for the full physical explanation of each one).
Change a number, save, then just run:

    python -m deltav.fly_dynamics

with no flags at all, and it uses everything below. Command-line flags, if
you do pass any, override whatever is set here on a per-flag basis (e.g.
`python -m deltav.fly_dynamics --t 20` uses this file for everything except
duration). Nothing here is read by any other script -- only fly_dynamics.py
imports this module.

NOTE: elevon deflection is NOT set here. It isn't a free input the way
rotor speed/tilt are -- it's whatever balances the pitching moment (SMy=0)
at the angle of attack the vehicle actually has at each instant, solved
automatically every timestep (trim.py's solve_delta_e_for_moment(), same
zero-moment condition trim.py's steady-trim solver uses for delta_e, just
re-solved continuously here instead of once). You only choose rotor
speed/tilt below; the elevons react to whatever those produce.
"""

# ---- initial position, m ---------------------------------------------------
X0 = 0.0
Y0 = 0.0
Z0 = 0.0            # altitude, positive = up

# ---- initial body-axis velocity, m/s ---------------------------------------
U0 = 0.0
V0 = 0.0
W0 = 0.0

# ---- initial attitude, deg --------------------------------------------------
PHI0 = 0.0
THETA0 = 8.4398   # trim.solve_hover_trim()'s solved pitch attitude -- see below
PSI0 = 0.0

# ---- initial body rates, deg/s -----------------------------------------------
P0 = 0.0
Q0 = 0.0
R0 = 0.0

# ---- held rotor command, applied from t=0 (broadcast to all 4 rotors unless
# overridden per-rotor below). Elevons are NOT set here -- see the module
# docstring above; they're auto-solved every step from these. -------------------
N = 107.0           # rotor speed, rev/s -- only used where a per-rotor override below is None
LAM = 0.0           # rotor tilt, deg (0 = hover/up, 90 = cruise/forward)

# ---- per-rotor overrides -- set any of these to a number to override the
# broadcast values above for just that rotor; leave as None to use the
# broadcast value. Rotor order: [1 front-right, 2 aft-left, 3 front-left,
# 4 aft-right] (matches STATE_NAMES/INPUT_NAMES in interceptor.py).
# Defaults below are trim.solve_hover_trim()'s solved front/aft thrust
# split (front pair 1,3 lighter-loaded than aft pair 2,4 -- H-frame lever
# arms, not a typo) -- together with THETA0 above, this is the real,
# Newton-Raphson-solved hover trim (SFx=SFz=SMy~1e-9 at t=0), not a guess.
# Set to None (or edit) to fly a different rotor command instead. --------------
N1 = 103.390
N2 = 115.243
N3 = 103.390
N4 = 115.243
LAM1 = None
LAM2 = None
LAM3 = None
LAM4 = None

# ---- optional step input: hold the rotor command above, then switch to a
# second one partway through the run (a doublet-style test). STEP_T = None
# means no step -- the command above is held for the entire run. Elevons
# keep auto-solving against whichever rotor command is active. -----------------
STEP_T = None         # time, s, to switch
STEP_N = 0.0
STEP_LAM = 0.0
STEP_N1 = None
STEP_N2 = None
STEP_N3 = None
STEP_N4 = None
STEP_LAM1 = None
STEP_LAM2 = None
STEP_LAM3 = None
STEP_LAM4 = None

# ---- run settings -------------------------------------------------------------
T = 10.0             # duration, s
DT = 0.01            # timestep, s
OUT = "dynamics_test"   # output file basename -> writes <out>.png and <out>.csv
