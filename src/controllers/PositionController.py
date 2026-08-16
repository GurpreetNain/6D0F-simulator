import numpy as np
from src.common.utils import rot2quat, quat_multiply

class PositionController:
    def __init__(self, Kp, Kd, mass, gravity, alpha, time_step):
        self._Kp = Kp
        self._Kd = Kd
        self._mass = mass  # Keep mass positive for cleaner scalar scaling
        self._gravity = gravity  # Gravity vector points down (+Z) in NED
        self._thrust_vector = -mass * gravity
        self._R_desired = np.eye(3)
        self._q_desired_prev = np.array([1, 0, 0, 0])
        self._w_desired = np.array([0.0, 0.0, 0.0])
        self._alpha_filter = alpha
        self._dt = time_step
    
    def _compute_acceleration_cmds(self, accel_ff, x_desired=None, x_current=None, v_desired=None, v_current=None):
        controller_flag = 0
        if x_desired is None:
            controller_flag += 1
        if v_desired is None:
            controller_flag += 2
            
        match controller_flag:
            case 0:
                accel_cmd = accel_ff + self._Kp * (x_desired - x_current) + self._Kd * (v_desired - v_current)
            case 1:
                accel_cmd = accel_ff + self._Kd * (v_desired - v_current)
            case 2:
                accel_cmd = accel_ff + self._Kp * (x_desired - x_current)
            case 3:
                accel_cmd = accel_ff
        
        # Enforce a maximum tilt angle
        # 5.0 m/s^2 limits the pitch/roll to approx 27°
        accel_cmd[0] = np.clip(accel_cmd[0], -5.0, 5.0)
        accel_cmd[1] = np.clip(accel_cmd[1], -5.0, 5.0)
        
        # Cap vertical acceleration to prevent extreme thrust saturation
        accel_cmd[2] = np.clip(accel_cmd[2], -15.0, 15.0)
        
        return accel_cmd
    
    def _compute_thrust_cmd(self, accel_cmd):
        # F = m * (a - g) in Z-Down NED framework
        self._thrust_vector = self._mass * (accel_cmd - self._gravity)
    
    def compute_desired_quats(self, yaw_cmd):
        zb_desired = -self._thrust_vector / np.linalg.norm(self._thrust_vector)
        x_psi = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd), 0.0])
        yb_desired = np.cross(zb_desired, x_psi) / np.linalg.norm(np.cross(zb_desired, x_psi))
        xb_desired = np.cross(yb_desired, zb_desired)
        
        R_body_to_global = np.column_stack((xb_desired, yb_desired, zb_desired))
        
        # FIX: rot2quat mathematically expects a Body -> Global matrix
        q_desired = rot2quat(R_body_to_global)
        
        # Keep R_desired as Global -> Body strictly for the angular velocity matrix (M) calculation
        R_desired = R_body_to_global.T
        
        # --- DOUBLE COVER ALIGNMENT BLOCK ---
        dot_product = np.dot(q_desired, self._q_desired_prev)
        if dot_product < 0.0:
            q_desired = -q_desired
        # ------------------------------------
        
        # FIX 1 & 2: Safe Kinematics Derivative Extraction
        # Check if this is the first iteration step to prevent initial delta spikes
        if np.array_equal(self._R_desired, np.eye(3)) and np.all(self._w_desired == 0.0):
            w_desired_raw = np.array([0.0, 0.0, 0.0])
        else:
            R_desired_dot = (R_desired - self._R_desired) / self._dt
            # Correct order for Global -> Body tracking matrix: M = -R_dot @ R^T
            M = -R_desired_dot @ np.transpose(R_desired)
            
            w_desired_raw = np.array([
                0.5 * (M[2][1] - M[1][2]), 
                0.5 * (M[0][2] - M[2][0]), 
                0.5 * (M[1][0] - M[0][1])
            ])
        
        # 3. Filter noise and track kinematics state
        w_desired_filtered = self._alpha_filter * self._w_desired + (1.0 - self._alpha_filter) * w_desired_raw
        
        q_desired_dot = 0.5 * quat_multiply(
            q_desired, 
            np.array([0.0, w_desired_filtered[0], w_desired_filtered[1], w_desired_filtered[2]])
        )
        
        # State updates
        self._R_desired = R_desired
        self._w_desired = w_desired_filtered
        self._q_desired_prev = q_desired  
        
        return q_desired, q_desired_dot

if __name__ == "__main__":
    import numpy as np
    import json
    import sys
    import os
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # Force Python to look in the parent (root) directory first
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, parent_dir)

    from src.models.Quadcopter import Quadcopter
    from src.controllers.ControlAllocator import ControlAllocator
    from src.controllers.RateController import RateController
    from src.controllers.AttitudeController import AttitudeController
    from utils import quat2rot

    np.set_printoptions(precision=4, suppress=True)

    # Obtain physical constants
    with open(os.path.join(parent_dir, "Environment.json"), "r") as f:
        env_data = json.load(f)
    with open(os.path.join(parent_dir, "Quadcopter.json"), "r") as f:
        quad_data = json.load(f)

    dt = env_data["simulation"]["time_step"]
    t_final = 10.0  # Position takes significantly longer to settle
    time = np.arange(0, t_final, dt)

    Lx = quad_data["arm_length"] * np.cos((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    Ly = quad_data["arm_length"] * np.sin((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    
    # Instantiate Plant
    drone = Quadcopter(
        mass=quad_data["mass"], 
        J=quad_data["inertia_matrix"], 
        Lx=Lx, 
        Ly=Ly, 
        id=1
    )

    hover_thrust = drone._mass * np.linalg.norm(drone._env.g)
    allocator = ControlAllocator(dX=Ly, dY=Lx, nT2D=0.05, T_motor_max=3*hover_thrust)

    # --- CASCADED INNER LOOPS (Hardcode your previously tuned gains here) ---
    rate_Kp = np.array([4.0, 4.0, 8.0])
    rate_Ki = np.array([0.0, 0.0, 0.0])
    rate_Kd = np.array([0.0, 0.0, 0.0])
    rate_controller = RateController(Kp_rate=rate_Kp, Ki_rate=rate_Ki, Kd_rate=rate_Kd)

    att_Kp = np.array([18.0, 18.0, 18.0])
    att_controller = AttitudeController(Kp_att=att_Kp)
    # ------------------------------------------------------------------------

    # Outer Loop: Position Controller
    pos_controller = PositionController(
        Kp=np.zeros(3), 
        Kd=np.zeros(3), 
        mass=drone._mass, 
        gravity=drone._env.g, 
        alpha=0.85, 
        time_step=dt
    )

    # UI Setup
    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(8, 9))
    plt.subplots_adjust(bottom=0.35, hspace=0.3)

    line_in, = ax_in.plot(time, np.zeros_like(time), 'b-', label="Step Input (m)")
    ax_in.set_ylabel("Displacement (m)")
    ax_in.grid(True)
    ax_in.legend(loc="upper right")

    line_out, = ax_out.plot(time, np.zeros_like(time), 'g-', label="Step Response")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Displacement (m)")
    ax_out.grid(True)
    ax_out.legend(loc="upper right")

    # Sliders and Buttons
    ax_axis = plt.axes([0.15, 0.20, 0.6, 0.03])
    ax_step = plt.axes([0.15, 0.15, 0.6, 0.03])
    ax_kp   = plt.axes([0.15, 0.10, 0.6, 0.03])
    ax_kd   = plt.axes([0.15, 0.05, 0.6, 0.03])
    ax_run  = plt.axes([0.80, 0.05, 0.15, 0.18])

    s_axis = Slider(ax_axis, 'Axis (0=X, 1=Y, 2=Z)', 0, 2, valinit=0, valstep=1)
    s_step = Slider(ax_step, 'Step Size', 0.01, 1.0, valinit=0.1)
    s_kp   = Slider(ax_kp, 'Kp Pos', 0.0, 10.0, valinit=2.0)
    s_kd   = Slider(ax_kd, 'Kd Pos', 0.0, 10.0, valinit=1.5)
    
    btn_run = Button(ax_run, 'Run\nSimulation', color='lightblue', hovercolor='skyblue')

    def update_ui(val):
        """Updates slider visuals and quantization instantly during drag."""
        step = s_step.val
        s_kp.valstep = step
        s_kd.valstep = step
        
        def quantize(v, s): return round(v / s) * s
        
        kp_val = quantize(s_kp.val, step)
        kd_val = quantize(s_kd.val, step)
        axis_idx = int(s_axis.val)
        axis_names = ["X (Forward)", "Y (Right)", "Z (Down)"]

        s_axis.valtext.set_text(f'{axis_names[axis_idx]}')
        s_kp.valtext.set_text(f'{kp_val:.3f}')
        s_kd.valtext.set_text(f'{kd_val:.3f}')
        fig.canvas.draw_idle()

    def run_sim(event=None):
        """Executes the heavy 10-second physics loop only when the button is clicked."""
        step = s_step.val
        def quantize(v, s): return round(v / s) * s
            
        kp_val = quantize(s_kp.val, step)
        kd_val = quantize(s_kd.val, step)
        axis_idx = int(s_axis.val)
        axis_names = ["X (Forward)", "Y (Right)", "Z (Down)"]

        # Reinitialize controller gains symmetrically
        pos_controller._Kp = np.array([kp_val, kp_val, kp_val])
        pos_controller._Kd = np.array([kd_val, kd_val, kd_val])
        rate_controller.reset()
        
        # Reset physics state and turbulence filters for a clean run
        drone._state_vector = np.zeros(13)
        drone._state_vector[6] = 1.0 
        drone._env._X = np.zeros(6) 
        
        # Target Step Setup: If testing Z, must fly upwards (-10m) to avoid floor crash
        target_step_val = 10.0 if axis_idx != 2 else -10.0
        x_des = np.zeros(3)
        x_des[axis_idx] = target_step_val
        v_des = np.zeros(3)
        accel_ff = np.zeros(3)
        yaw_cmd = 0.0
        
        input_history = np.full(len(time), target_step_val)
        response_history = np.zeros(len(time))

        # Run the full 3-loop cascaded simulation
        for i in range(len(time)):
            state_vector = drone.get_state_vector()
            pos = state_vector[0:3]
            body_vel = state_vector[3:6]
            q_current = state_vector[6:10]
            rates = state_vector[10:13]

            # Translate velocity to global frame for position controller
            R_bg = quat2rot(q_current)
            global_vel = R_bg @ body_vel

            # 1. Outer Loop (Position)
            accel_cmd = pos_controller._compute_acceleration_cmds(accel_ff, x_des, pos, v_des, global_vel)
            pos_controller._compute_thrust_cmd(accel_cmd)
            q_des, q_des_dot = pos_controller.compute_desired_quats(yaw_cmd)

            # 2. Middle Loop (Attitude)
            w_des = att_controller.compute_desired_rates(q_current, q_des, q_des_dot)

            # 3. Inner Loop (Rate)
            torque_cmd = rate_controller.compute_torque_commands(rates, w_des, dt)

            # 4. Allocator & Plant Integration
            thrust_mag = np.linalg.norm(pos_controller._thrust_vector)
            control_vector = np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
            motor_cmds = allocator.get_control_cmds(control_vector)
            
            drone.set_control_vector(motor_cmds)
            drone.state_update(dt)
            
            response_history[i] = drone.get_state_vector()[axis_idx]
        
        # Update plotting data
        line_in.set_ydata(input_history)
        line_out.set_ydata(response_history)
        
        line_out.set_label(f"{axis_names[axis_idx]} Displacement")
        ax_out.legend(loc="upper right")

        # Dynamically adjust graph limits
        margin_out = 0.1 * abs(np.max(response_history) - np.min(response_history)) + 1.0
        ax_out.set_ylim(np.min(response_history) - margin_out, np.max(response_history) + margin_out)
        
        margin_in = 2.0
        ax_in.set_ylim(target_step_val - margin_in, target_step_val + margin_in)
        
        # Calculate max overshoot (handling negative target for Z-axis appropriately)
        if axis_idx == 2:
            max_resp = np.min(response_history)  # Minimum is highest altitude in NED
        else:
            max_resp = np.max(response_history)
            
        overshoot = ((max_resp - target_step_val) / target_step_val) * 100.0
        overshoot = max(0.0, overshoot)
        
        fig.suptitle(f"{axis_names[axis_idx]} Axis | Max Overshoot: {overshoot:.2f}%", fontsize=14, fontweight='bold')
        fig.canvas.draw()

    # Bind events
    s_axis.on_changed(update_ui)
    s_step.on_changed(update_ui)
    s_kp.on_changed(update_ui)
    s_kd.on_changed(update_ui)
    btn_run.on_clicked(run_sim)

    # Initialize
    update_ui(0)
    run_sim()
    plt.show()