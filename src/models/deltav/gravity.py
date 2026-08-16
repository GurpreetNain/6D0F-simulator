"""
Gravity resolved into body axes. Doc4 Ch 7; 6DOF.pdf Sec 6.4.

    Fg,x = -m g sin(theta)
    Fg,y =  m g sin(phi) cos(theta)
    Fg,z =  m g cos(phi) cos(theta)

level flight puts all weight on +z_b (belly-down); pitching forward spills weight into +x_b; banking spills it into y_b -- the very component that balances a coordinated turn.
"""
import numpy as np


def gravity_forces(phi, theta, mass, g):
    Fx = -mass * g * np.sin(theta)
    Fy = mass * g * np.sin(phi) * np.cos(theta)
    Fz = mass * g * np.cos(phi) * np.cos(theta)
    return np.array([Fx, Fy, Fz])