import numpy as np
from src.controllers.base_controller import BaseController
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController

class AccelerationController(BaseController):
    """
    Accepts direct acceleration setpoints [Ax, Ay, Az, Yaw].
    Bypasses spatial position error tracking entirely.
    """
    def __init__(self, att_gains, rate_gains, mass, gravity, alpha_filter=0.85):
        # We instantiate PositionController solely to use its internal quaternion generator math
        self.pos_controller = PositionController(np.zeros(3), np.zeros(3), mass, gravity, alpha_filter, 0.0)
        self.att_controller = AttitudeController(att_gains['Kp'])
        self.rate_controller = RateController(rate_gains['Kp'], rate_gains['Ki'], rate_gains['Kd'])
        
    def compute_control(self, state_vector, setpoint, dt):
        q_current = state_vector[6:10]
        rates = state_vector[10:13]
        
        # Parse Setpoint [Ax, Ay, Az, Yaw]
        accel_cmd = setpoint[0:3]
        yaw_cmd = setpoint[3]
        
        # 1. Direct to Thrust Vectorization (Bypassing PID position error)
        self.pos_controller._dt = dt
        self.pos_controller._compute_thrust_cmd(accel_cmd)
        q_des, q_des_dot = self.pos_controller.compute_desired_quats(yaw_cmd)
        
        # 2. Attitude Loop
        w_des = self.att_controller.compute_desired_rates(q_current, q_des, q_des_dot)
        
        # 3. Rate Loop
        torque_cmd = self.rate_controller.compute_torque_commands(rates, w_des, dt)
        
        thrust_mag = np.linalg.norm(self.pos_controller._thrust_vector)
        return np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])