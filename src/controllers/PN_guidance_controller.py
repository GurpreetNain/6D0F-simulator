import numpy as np
from src.controllers.base_controller import BaseController
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController
from src.sensors.camera_sensor import CameraSensor

class GeneratedAccelerationController(BaseController):
    def __init__(self, pos_gains, att_gains, rate_gains, mass, gravity):
        self.mass = mass
        self.gravity = gravity
        self.pos_controller = PositionController(pos_gains['Kp'], pos_gains['Kd'], mass, gravity, 0.85, 0.0)
        self.att_controller = AttitudeController(att_gains['Kp'])
        self.rate_controller = RateController(rate_gains['Kp'], rate_gains['Ki'], rate_gains['Kd'])
        # self.camera = CameraSensor(30, 0.01, 5000, )
        
        self.t = 0.0
        self.id = 0
        self.swarm = []
        
        # State tracking for the discrete derivative
        self.q_e_prev = np.array([1.0, 0.0, 0.0, 0.0])
        self.has_prev = False

    @staticmethod
    def quat_mult(qa, qb):
        w1, x1, y1, z1 = qa
        w2, x2, y2, z2 = qb
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def quat_inv(q):
        return np.array([q[0], -q[1], -q[2], -q[3]]) / (np.sum(q**2) + 1e-8)

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

        # ---------------------------------------------------------------------
        # 1. Proportional Navigation (PN) Law for delta_a
        # ---------------------------------------------------------------------
        a_cmd = np.zeros(3)
        target_id = 1  
        
        if self.id != target_id and target_id in drones:
            p_own = np.array([x, y, z])
            v_own = np.array([vx, vy, vz])
            
            # Add a -2.0m altitude offset so Drone 2 flies strictly above Drone 1
            p_target = np.array([drones[target_id]['x'], drones[target_id]['y'], drones[target_id]['z'] - 2.0])
            v_target = np.array([drones[target_id]['vx'], drones[target_id]['vy'], drones[target_id]['vz']])
            
            r = p_target - p_own
            v_rel = v_target - v_own
            dist = np.linalg.norm(r)
            
            if dist > 0.1:
                los_vec = r / dist
                V_c = -np.dot(r, v_rel) / dist 
                omega_los = np.cross(r, v_rel) / (dist**2)
                
                N_gain = 4.0 
                a_cmd = N_gain * abs(V_c) * np.cross(omega_los, los_vec)
                a_cmd += 1.5 * r  # Increased proportional tracking anchor

        # ---------------------------------------------------------------------
        # 2. Required Propulsive Acceleration and Thrust Magnitude
        # ---------------------------------------------------------------------
        if isinstance(self.gravity, np.ndarray):
            g_vec = self.gravity
        else:
            g_vec = np.array([0.0, 0.0, self.gravity])
            
        F_des = self.mass * (a_cmd - g_vec)
        thrust_mag = np.linalg.norm(F_des)
        
        # ---------------------------------------------------------------------
        # 3. Target Attitude Construction (q_tilde)
        # ---------------------------------------------------------------------
        if thrust_mag > 1e-6:
            z_b = -F_des / thrust_mag
        else:
            z_b = np.array([0.0, 0.0, -1.0])
            
        qw, qx, qy, qz = state[6:10]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        y_b = np.cross(z_b, x_c)
        norm_y = np.linalg.norm(y_b)
        if norm_y > 1e-6:
            y_b = y_b / norm_y
        else:
            y_b = np.array([0.0, 1.0, 0.0])
            
        x_b = np.cross(y_b, z_b)
        R_des = np.column_stack((x_b, y_b, z_b))
        
        tr = np.trace(R_des)
        if tr > 0:
            S = 2.0 * np.sqrt(tr + 1.0)
            qw_t = 0.25 * S
            qx_t = (R_des[2, 1] - R_des[1, 2]) / S
            qy_t = (R_des[0, 2] - R_des[2, 0]) / S
            qz_t = (R_des[1, 0] - R_des[0, 1]) / S
        elif R_des[0, 0] > R_des[1, 1] and R_des[0, 0] > R_des[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R_des[0, 0] - R_des[1, 1] - R_des[2, 2])
            qw_t = (R_des[2, 1] - R_des[1, 2]) / S
            qx_t = 0.25 * S
            qy_t = (R_des[0, 1] + R_des[1, 0]) / S
            qz_t = (R_des[0, 2] + R_des[2, 0]) / S
        elif R_des[1, 1] > R_des[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R_des[1, 1] - R_des[0, 0] - R_des[2, 2])
            qw_t = (R_des[0, 2] - R_des[2, 0]) / S
            qx_t = (R_des[0, 1] + R_des[1, 0]) / S
            qy_t = 0.25 * S
            qz_t = (R_des[1, 2] + R_des[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R_des[2, 2] - R_des[0, 0] - R_des[1, 1])
            qw_t = (R_des[1, 0] - R_des[0, 1]) / S
            qx_t = (R_des[0, 2] + R_des[2, 0]) / S
            qy_t = (R_des[1, 2] + R_des[2, 1]) / S
            qz_t = 0.25 * S
            
        q_tilde = np.array([qw_t, qx_t, qy_t, qz_t])
        q_tilde = q_tilde / np.linalg.norm(q_tilde)
        
        # ---------------------------------------------------------------------
        # 4. Error Quaternion and Predictive Derivative
        # ---------------------------------------------------------------------
        q_curr = np.array([qw, qx, qy, qz])
        q_inv = self.quat_inv(q_curr)
        q_e = self.quat_mult(q_inv, q_tilde)
        
        if q_e[0] < 0:
            q_e = -q_e
            
        # Predictive Difference: Command the error towards the Identity Quaternion
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Attitude time constant (tau). Lower = more aggressive rotation.
        tau = 0.15 
        q_e_dot = (q_identity - q_e) / tau
        
        # ---------------------------------------------------------------------
        # 5. Inverse Kinematic Mapping to Angular Rates
        # ---------------------------------------------------------------------
        w_cmd_quat = -2.0 * self.quat_mult(q_e_dot, self.quat_inv(q_e))
        w_des = w_cmd_quat[1:4]
        
        torque_cmd = self.rate_controller.compute_torque_commands(state[10:13], w_des, dt)

        return np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])