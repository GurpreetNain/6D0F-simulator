import numpy as np
from src.controllers.base_controller import BaseController
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController

class GeneratedAccelerationController(BaseController):
    def __init__(self, pos_gains, att_gains, rate_gains, mass, gravity):
        self.mass = mass
        self.gravity = gravity
        self.pos_controller = PositionController(pos_gains['Kp'], pos_gains['Kd'], mass, gravity, 0.85, 0.0)
        self.att_controller = AttitudeController(att_gains['Kp'])
        self.rate_controller = RateController(rate_gains['Kp'], rate_gains['Ki'], rate_gains['Kd'])
        
        self.t = 0.0
        self.id = 0
        self.swarm = []

    def update_context(self, t, agent_id, swarm):
        self.t = t
        self.id = agent_id
        self.swarm = swarm

    def compute_control(self, state, setpoint, dt):
        t = self.t
        id = self.id
        drones = {}
        for a in self.swarm:
            s = a.get_state()
            drones[a.id] = {'x': s[0], 'y': s[1], 'z': s[2], 'vx': s[3], 'vy': s[4], 'vz': s[5]}
            
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]

        try:
            target_ax = float(0.0)
            target_ay = float(0.0)
            target_az = float(0.0)
        except Exception as e:
            target_ax, target_ay, target_az = 0.0, 0.0, 0.0
            
        accel_cmd = np.array([target_ax, target_ay, target_az])

        self.pos_controller._dt = dt
        self.pos_controller._compute_thrust_cmd(accel_cmd)
        q_des, q_des_dot = self.pos_controller.compute_desired_quats(0.0)

        w_des = self.att_controller.compute_desired_rates(state[6:10], q_des, q_des_dot)
        torque_cmd = self.rate_controller.compute_torque_commands(state[10:13], w_des, dt)

        thrust_mag = np.linalg.norm(self.pos_controller._thrust_vector)
        return np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
