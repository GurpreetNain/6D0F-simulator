import numpy as np

class RateController:
    def __init__(self, Kp_rate, Ki_rate, Kd_rate): #, max_torque):
        # Array bounds: [roll_rate, pitch_rate, yaw_rate]
        self._Kp = Kp_rate
        self._Ki = Ki_rate
        self._Kd = Kd_rate
        # self._max_torque = np.array(max_torque)
        
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral_error.fill(0.0)
        self._prev_error.fill(0.0)

    def compute_torque_commands(self, local_angular_velocity, desired_angular_velocity, dt):
        """
        Computes desired body torques.
        Inputs:
            local_angular_velocity: Current [p, q, r] from quadcopter state_vector[10:13]
            desired_angular_velocity: Target [p, q, r] from Attitude Controller
        """
        # 1. Calculate rate error
        error = desired_angular_velocity - local_angular_velocity
        
        # 2. Update error integral term with Anti-Windup bounds
        self._integral_error = np.clip(self._integral_error + error * dt, -5.0, 5.0)
        
        # 3. Calculate error derivative term
        derivative = (error - self._prev_error) / dt if dt > 0.0 else np.zeros(3)
        self._prev_error = error.copy()
        
        # 4. Standard PID calculation
        torque_cmd = (self._Kp * error) + (self._Ki * self._integral_error) + (self._Kd * derivative)
        
        # 5. Apply symmetric hardware saturation limits
        # torque_cmd = np.clip(torque_cmd, -self._max_torque, self._max_torque)
        
        return torque_cmd

if __name__ == "__main__":
    import numpy as np
    import json
    import sys
    import os
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # 4. Use util.py in the project root
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, parent_dir)

    from src.models.Quadcopter import Quadcopter
    from src.controllers.ControlAllocator import ControlAllocator

    np.set_printoptions(precision=4, suppress=True)

    # 3. Obtain physical constants from Quadcopter.json
    with open(os.path.join(parent_dir, "Environment.json"), "r") as f:
        env_data = json.load(f)
    with open(os.path.join(parent_dir, "Quadcopter.json"), "r") as f:
        quad_data = json.load(f)

    dt = env_data["simulation"]["time_step"]
    t_final = 0.5       
    time = np.arange(0, t_final, dt)

    Lx = quad_data["arm_length"] * np.cos((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    Ly = quad_data["arm_length"] * np.sin((np.pi/180.0)*(quad_data["nose_spread_angle"]/2)) * 2.0
    
    # 3. Use Quadcopter class
    drone = Quadcopter(
        mass=quad_data["mass"], 
        J=quad_data["inertia_matrix"], 
        Lx=Lx, 
        Ly=Ly, 
        id=1
    )

    hover_thrust = drone._mass * np.linalg.norm(drone._env.g)
    allocator = ControlAllocator(dX=Ly, dY=Lx, nT2D=0.05, T_motor_max=3*hover_thrust)

    rate_controller = RateController(np.zeros(3), np.zeros(3), np.zeros(3))

    # 5. Graphs displayed vertically stacked
    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(8, 9))
    plt.subplots_adjust(bottom=0.4, hspace=0.3)

    line_in, = ax_in.plot(time, np.zeros_like(time), 'b-', label="Step Input")
    ax_in.set_ylabel("Rate (rad/s)")
    ax_in.grid(True)
    ax_in.legend(loc="upper right")

    line_out, = ax_out.plot(time, np.zeros_like(time), 'g-', label="Step Response")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Rate (rad/s)")
    ax_out.grid(True)
    ax_out.legend(loc="upper right")

    # 1. 5 sliders: 3 for controller gains, 1 for step size, 1 for axis selection
    ax_axis = plt.axes([0.15, 0.25, 0.7, 0.03])
    ax_step = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_kp   = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_ki   = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_kd   = plt.axes([0.15, 0.05, 0.7, 0.03])

    s_axis = Slider(ax_axis, 'Axis (0=R, 1=P, 2=Y)', 0, 2, valinit=0, valstep=1)
    s_step = Slider(ax_step, 'Step Size', 0.01, 1.0, valinit=0.01)
    s_kp   = Slider(ax_kp, 'Kp', 0.0, 5.0, valinit=1.0)
    s_ki   = Slider(ax_ki, 'Ki', 0.0, 5.0, valinit=0.0)
    s_kd   = Slider(ax_kd, 'Kd', 0.0, 100.0, valinit=16.0)

    def update(val):
        step = s_step.val
        
        # Dynamically set the 'valstep' property of the UI sliders
        s_kp.valstep = step
        s_ki.valstep = step
        s_kd.valstep = step
        
        # Quantize current values in case the step size was just changed
        def quantize(v, s):
            return round(v / s) * s
        
        kp_val = quantize(s_kp.val, step)
        ki_val = quantize(s_ki.val, step)
        kd_val = quantize(s_kd.val, step)
        
        axis_idx = int(s_axis.val)
        axis_names = ["Roll", "Pitch", "Yaw"]

        # Update slider display text
        s_axis.valtext.set_text(f'{axis_names[axis_idx]}')
        s_kp.valtext.set_text(f'{kp_val:.3f}')
        s_ki.valtext.set_text(f'{ki_val:.3f}')
        s_kd.valtext.set_text(f'{kd_val:.3f}')

        rate_controller._Kp = np.array([kp_val, kp_val, kp_val])
        rate_controller._Ki = np.array([ki_val, ki_val, ki_val])
        rate_controller._Kd = np.array([kd_val, kd_val, kd_val])
        rate_controller.reset()
        
        # Reset physics state for fresh simulation run
        drone._state_vector = np.zeros(13)
        drone._state_vector[6] = 1.0 
        
        target_step_val = 1.0
        target_rates = np.zeros(3)
        target_rates[axis_idx] = target_step_val
        
        input_history = np.full(len(time), target_step_val)
        response_history = np.zeros(len(time))

        # 2. Run the simulation again
        for i in range(len(time)):
            current_rates = drone.get_state_vector()[10:13]
            
            torque_cmd = rate_controller.compute_torque_commands(current_rates, target_rates, dt)
            control_vector = np.array([hover_thrust, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
            motor_cmds = allocator.get_control_cmds(control_vector)
            
            drone.set_control_vector(motor_cmds)
            drone.state_update(dt)
            
            response_history[i] = drone.get_state_vector()[10 + axis_idx]
        
        # Update plotting data
        line_in.set_ydata(input_history)
        line_out.set_ydata(response_history)
        
        # Update label dynamically based on active axis
        line_out.set_label(f"{axis_names[axis_idx]} Response")
        ax_out.legend(loc="upper right")

        # 5. Dynamically adjust graph range based on value
        margin_out = 0.1 * (np.max(response_history) - np.min(response_history)) + 0.1
        ax_out.set_ylim(np.min(response_history) - margin_out, np.max(response_history) + margin_out)
        
        margin_in = 0.1
        ax_in.set_ylim(target_step_val - margin_in, target_step_val + margin_in)
        
        # 6. Set Title to Max Overshoot
        max_resp = np.max(response_history)
        overshoot = ((max_resp - target_step_val) / target_step_val) * 100.0
        overshoot = max(0.0, overshoot)  # Prevent negative overshoot display
        
        fig.suptitle(f"{axis_names[axis_idx]} Axis | Max Overshoot: {overshoot:.2f}%", fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()

    # Bind the update function to slider changes
    s_axis.on_changed(update)
    s_step.on_changed(update)
    s_kp.on_changed(update)
    s_ki.on_changed(update)
    s_kd.on_changed(update)

    # Initialize the plot with default values
    update(0)
    plt.show()