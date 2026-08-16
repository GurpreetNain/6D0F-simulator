import numpy as np
from src.common.utils import quat_conjugate, quat_multiply

class AttitudeController:
    def __init__(self, Kp_att):
        """
        Inputs:
            Kp_att: List or array of 3 gains [Kp_roll, Kp_pitch, Kp_yaw]
        """
        self._Kp = np.array(Kp_att)

    def compute_desired_rates(self, current_quaternion, desired_quaternion, desired_quaternion_dot):
        """
        Computes the target angular velocities [p_des, q_des, r_des]^T.
        Inputs:
            current_quaternion: array of [w, x, y, z] active body orientation
            desired_quaternion: target tracking orientation array of [w, x, y, z] (q_d)
            desired_quaternion_dot: time derivative array of [w, x, y, z] (q_d_dot)
        """
        # --- 1. Compute Analytical Trajectory Feedforward (\omega_d) ---
        # Formulate via the exact compact identity: [0, \omega_ff] = 2 * conj(q_d) \otimes q_d_dot
        qd_inv = quat_conjugate(desired_quaternion)
        omega_ff_raw = quat_multiply(qd_inv, desired_quaternion_dot)
        omega_ff = 2.0 * omega_ff_raw[1:4]

        # --- 2. Compute Feedback Correction Error (\omega_desired_error) ---
        # FIX: Reverse the order to get the true error from Current -> Desired body frame
        q_inv = quat_conjugate(current_quaternion)
        q_e = quat_multiply(desired_quaternion, q_inv)  
        
        q_e_scalar = q_e[0]
        q_e_vector = q_e[1:4]
        
        norm_q_e_vec = np.linalg.norm(q_e_vector)
        
        # Guard against zero-error alignment to prevent division-by-zero or NaN outputs
        if norm_q_e_vec < 1e-6:
            omega_desired_error = np.zeros(3)
        else:
            # Shortest angular error via atan2 mapping (fully uncompressed linear error)
            theta_e = 2.0 * np.arctan2(norm_q_e_vec, np.abs(q_e_scalar))
            u_e = q_e_vector / norm_q_e_vec
            
            # Enforce shortest-path direction matching sign of scalar component
            shortest_path_multiplier = 1.0 if q_e_scalar >= 0.0 else -1.0
            
            omega_desired_error = shortest_path_multiplier * self._Kp * theta_e * u_e
        
        # --- 3. The Unification ---
        # Combine the moving trajectory feedforward term with your feedback loop
        desired_rates = omega_ff + omega_desired_error
        
        return desired_rates
    
if __name__ == "__main__":
    import numpy as np
    import json
    import sys
    import os
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Force Python to look in the parent (root) directory first
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, parent_dir)

    from src.models.Quadcopter import Quadcopter
    from src.controllers.ControlAllocator import ControlAllocator
    from src.controllers.RateController import RateController

    np.set_printoptions(precision=4, suppress=True)

    # Obtain physical constants from Quadcopter.json
    with open(os.path.join(parent_dir, "Environment.json"), "r") as f:
        env_data = json.load(f)
    with open(os.path.join(parent_dir, "Quadcopter.json"), "r") as f:
        quad_data = json.load(f)

    dt = env_data["simulation"]["time_step"]
    t_final = 5.0  # Attitude takes slightly longer to settle than rates
    time = np.arange(0, t_final, dt)

    Lx = quad_data["arm_length"] * np.cos((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    Ly = quad_data["arm_length"] * np.sin((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    
    # Use Quadcopter class
    drone = Quadcopter(
        mass=quad_data["mass"], 
        J=quad_data["inertia_matrix"], 
        Lx=Lx, 
        Ly=Ly, 
        id=1
    )

    hover_thrust = drone._mass * np.linalg.norm(drone._env.g)
    allocator = ControlAllocator(dX=Ly, dY=Lx, nT2D=0.05, T_motor_max=3*hover_thrust)

    # INNER LOOP: Hardcode the RateController with your previously tuned gains
    rate_Kp = np.array([4.0, 4.0, 5.0])
    rate_Ki = np.array([0.0, 0.0, 0.0])
    rate_Kd = np.array([0.0, 0.0, 0.0])
    rate_controller = RateController(Kp_rate=rate_Kp, Ki_rate=rate_Ki, Kd_rate=rate_Kd)

    # OUTER LOOP: Instantiate AttitudeController
    att_controller = AttitudeController(np.zeros(3))

    # Helper function to extract Euler angles for plotting
    def quat2euler(q):
        w, x, y, z = q
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return np.array([roll, pitch, yaw])

    # Graphs displayed vertically stacked
    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(8, 9))
    plt.subplots_adjust(bottom=0.4, hspace=0.3)

    line_in, = ax_in.plot(time, np.zeros_like(time), 'b-', label="Step Input (rad)")
    ax_in.set_ylabel("Angle (rad)")
    ax_in.grid(True)
    ax_in.legend(loc="upper right")

    line_out, = ax_out.plot(time, np.zeros_like(time), 'g-', label="Step Response")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Angle (rad)")
    ax_out.grid(True)
    ax_out.legend(loc="upper right")

    from matplotlib.widgets import Slider, Button

    # 5 sliders: Axis selection, discrete step size, and 3 controller gains
    # Widths reduced to 0.6 to make room for the Run Button
    ax_axis    = plt.axes([0.15, 0.25, 0.6, 0.03])
    ax_step    = plt.axes([0.15, 0.20, 0.6, 0.03])
    ax_kp_roll = plt.axes([0.15, 0.15, 0.6, 0.03])
    ax_kp_pit  = plt.axes([0.15, 0.10, 0.6, 0.03])
    ax_kp_yaw  = plt.axes([0.15, 0.05, 0.6, 0.03])

    # Simulation Execution Button Axis
    ax_run     = plt.axes([0.80, 0.05, 0.15, 0.23])

    s_axis    = Slider(ax_axis, 'Axis (0=R, 1=P, 2=Y)', 0, 2, valinit=0, valstep=1)
    s_step    = Slider(ax_step, 'Step Size', 0.01, 1.0, valinit=0.1)
    s_kp_roll = Slider(ax_kp_roll, 'Kp Roll', 0.0, 20.0, valinit=8.0)
    s_kp_pit  = Slider(ax_kp_pit, 'Kp Pitch', 0.0, 20.0, valinit=8.0)
    s_kp_yaw  = Slider(ax_kp_yaw, 'Kp Yaw', 0.0, 20.0, valinit=4.0)
    
    btn_run   = Button(ax_run, 'Run\nSimulation', color='lightblue', hovercolor='skyblue')

    def update_ui(val):
        """Updates slider visuals and quantization instantly during drag."""
        step = s_step.val
        
        s_kp_roll.valstep = step
        s_kp_pit.valstep = step
        s_kp_yaw.valstep = step
        
        def quantize(v, s):
            return round(v / s) * s
        
        kp_r = quantize(s_kp_roll.val, step)
        kp_p = quantize(s_kp_pit.val, step)
        kp_y = quantize(s_kp_yaw.val, step)
        
        axis_idx = int(s_axis.val)
        axis_names = ["Roll", "Pitch", "Yaw"]

        s_axis.valtext.set_text(f'{axis_names[axis_idx]}')
        s_kp_roll.valtext.set_text(f'{kp_r:.3f}')
        s_kp_pit.valtext.set_text(f'{kp_p:.3f}')
        s_kp_yaw.valtext.set_text(f'{kp_y:.3f}')
        fig.canvas.draw_idle()

    def run_sim(event=None):
        """Executes the heavy 5-second physics loop only when the button is clicked."""
        step = s_step.val
        
        def quantize(v, s):
            return round(v / s) * s
            
        kp_r = quantize(s_kp_roll.val, step)
        kp_p = quantize(s_kp_pit.val, step)
        kp_y = quantize(s_kp_yaw.val, step)
        
        axis_idx = int(s_axis.val)
        axis_names = ["Roll", "Pitch", "Yaw"]

        # Reinitialize controller gains
        att_controller._Kp = np.array([kp_r, kp_p, kp_y])
        rate_controller.reset()
        
        # Reset physics state for fresh simulation run
        drone._state_vector = np.zeros(13)
        drone._state_vector[6] = 1.0 
        
        target_step_val = 1.0
        q_des = np.array([np.cos(target_step_val / 2.0), 0.0, 0.0, 0.0])
        q_des[axis_idx + 1] = np.sin(target_step_val / 2.0)
        q_des_dot = np.zeros(4)
        
        input_history = np.full(len(time), target_step_val)
        response_history = np.zeros(len(time))

        # Run the cascaded simulation
        for i in range(len(time)):
            q_current = drone.get_state_vector()[6:10]
            
            w_des = att_controller.compute_desired_rates(q_current, q_des, q_des_dot)
            
            current_rates = drone.get_state_vector()[10:13]
            torque_cmd = rate_controller.compute_torque_commands(current_rates, w_des, dt)
            
            control_vector = np.array([hover_thrust, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
            motor_cmds = allocator.get_control_cmds(control_vector)
            
            drone.set_control_vector(motor_cmds)
            drone.state_update(dt)
            
            euler_angles = quat2euler(drone.get_state_vector()[6:10])
            response_history[i] = euler_angles[axis_idx]
        
        # Update plotting data
        line_in.set_ydata(input_history)
        line_out.set_ydata(response_history)
        
        line_out.set_label(f"{axis_names[axis_idx]} Angle")
        ax_out.legend(loc="upper right")

        margin_out = 0.1 * (np.max(response_history) - np.min(response_history)) + 0.1
        ax_out.set_ylim(np.min(response_history) - margin_out, np.max(response_history) + margin_out)
        
        margin_in = 0.1
        ax_in.set_ylim(target_step_val - margin_in, target_step_val + margin_in)
        
        max_resp = np.max(response_history)
        overshoot = ((max_resp - target_step_val) / target_step_val) * 100.0
        overshoot = max(0.0, overshoot)
        
        fig.suptitle(f"{axis_names[axis_idx]} Axis | Max Overshoot: {overshoot:.2f}%", fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()

    # Bind fast UI updates to continuous slider dragging
    s_axis.on_changed(update_ui)
    s_step.on_changed(update_ui)
    s_kp_roll.on_changed(update_ui)
    s_kp_pit.on_changed(update_ui)
    s_kp_yaw.on_changed(update_ui)

    # Bind heavy physics simulation strictly to the Run Button click
    btn_run.on_clicked(run_sim)

    # Initialize the plot with default values
    update_ui(0)
    run_sim()
    plt.show()