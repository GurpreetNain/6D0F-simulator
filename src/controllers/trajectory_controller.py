import numpy as np
from src.controllers.base_controller import BaseController
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController
from src.common.utils import quat2rot

class TrajectoryController(BaseController):
    """
    Standard PID cascade expecting spatial setpoints [X, Y, Z, Yaw].
    """
    def __init__(self, pos_gains, att_gains, rate_gains, mass, gravity, alpha_filter=0.85):
        self.pos_controller = PositionController(pos_gains['Kp'], pos_gains['Kd'], mass, gravity, alpha_filter, 0.0)
        self.att_controller = AttitudeController(att_gains['Kp'])
        self.rate_controller = RateController(rate_gains['Kp'], rate_gains['Ki'], rate_gains['Kd'])
        
    def compute_control(self, state_vector, setpoint, dt):
        # Extract states
        pos = state_vector[0:3]
        body_vel = state_vector[3:6]
        q_current = state_vector[6:10]
        rates = state_vector[10:13]
        
        # Parse Setpoint [X, Y, Z, Yaw]
        x_des = setpoint[0:3]
        yaw_cmd = setpoint[3]
        
        # Geometry setup
        R_bg = quat2rot(q_current)
        global_vel = R_bg @ body_vel
        accel_ff = np.zeros(3)
        v_des = np.zeros(3)

        # 1. Position Loop
        self.pos_controller._dt = dt # Update dynamically
        accel_cmd = self.pos_controller._compute_acceleration_cmds(accel_ff, x_des, pos, v_des, global_vel)
        self.pos_controller._compute_thrust_cmd(accel_cmd)
        q_des, q_des_dot = self.pos_controller.compute_desired_quats(yaw_cmd)
        
        # 2. Attitude Loop
        w_des = self.att_controller.compute_desired_rates(q_current, q_des, q_des_dot)
        
        # 3. Rate Loop
        torque_cmd = self.rate_controller.compute_torque_commands(rates, w_des, dt)
        
        thrust_mag = np.linalg.norm(self.pos_controller._thrust_vector)
        return np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])