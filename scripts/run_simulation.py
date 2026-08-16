import json
import numpy as np
import os
import sys

# Add the project root to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import from the new src/ package architecture
from src.common.utils import quat2rot
from src.models.Quadcopter import Quadcopter
from src.controllers.PositionController import PositionController
from src.controllers.AttitudeController import AttitudeController
from src.controllers.RateController import RateController
from src.controllers.ControlAllocator import ControlAllocator
from src.logging.data_logger import DataLogger

# 1. Load Configurations from the config/ directory
config_dir = os.path.join(parent_dir, "config")

with open(os.path.join(config_dir, "Environment.json"), "r") as env_file:    
    env_data = json.load(env_file)

with open(os.path.join(config_dir, "Quadcopter.json"), "r") as quadcopter_file:    
    quadcopter_data = json.load(quadcopter_file)

time_step = env_data["simulation"]["time_step"]
t_final = 60.0
sim_time = np.arange(0, t_final, step=time_step)

# 2. Instantiate the Physical Plant
drone = Quadcopter(
    mass = quadcopter_data["mass"], 
    J    = quadcopter_data["inertia_matrix"],
    Ly   = quadcopter_data["arm_length"] * np.sin((np.pi/180.0)*(quadcopter_data["nose_spread_angle"]/2)) * 2.0,
    Lx   = quadcopter_data["arm_length"] * np.cos((np.pi/180.0)*(quadcopter_data["nose_spread_angle"]/2)) * 2.0,
    id   = 1
)

# Scalable swarm initialization
swarm = [drone]

# 3. Controller Gains Setup
Kp_pos = np.array([3.5, 3.5, 3.5])
Kd_pos = np.array([2.3, 2.3, 2.3])
Kp_att = np.array([18.0, 18.0, 18.0])
Kp_rate = np.array([4.0, 4.0, 5.0])
Ki_rate = np.array([0.0, 0.0, 0.0])
Kd_rate = np.array([0.0, 0.0, 0.0])
alpha_filter = 0.85 

# 4. Initialize Controller Instances
pos_controller = PositionController(Kp_pos, Kd_pos, drone._mass, drone._env.g, alpha_filter, time_step)
att_controller = AttitudeController(Kp_att)
rate_controller = RateController(Kp_rate=Kp_rate, Ki_rate=Ki_rate, Kd_rate=Kd_rate)

hover_thrust = np.linalg.norm(drone._mass * drone._env.g)
control_allocator = ControlAllocator(dX=drone._Ly, dY=drone._Lx, nT2D=0.05, T_motor_max=3*hover_thrust)

# 5. Initialize the Decoupled Telemetry Logger
# Assuming drone._id is the property holding the ID
logger = DataLogger(drone_ids=[d._id for d in swarm])

# Target Waypoint
accel_ff = np.array([0.0, 0.0, 0.0])
x_des = np.array([1.0, -1.0, -1.0])
v_des = np.array([0.0, 0.0, 0.0])

print("--- Starting Headless Simulation ---")

# 6. Main Execution Loop
for t in sim_time:
    
    # --- A. PHYSICS & CONTROL PHASE ---
    for d in swarm:
        state_vector = d.get_state_vector()
        
        q_current = state_vector[6:10]
        body_vel_current = state_vector[3:6]

        # Translate velocity to global frame
        R_body_to_global = quat2rot(q_current)
        global_vel_current = R_body_to_global @ body_vel_current

        # Outer Loop: Position
        accel_cmd = pos_controller._compute_acceleration_cmds(
            accel_ff=accel_ff,
            x_desired=x_des,
            x_current=state_vector[0:3],
            v_desired=v_des,
            v_current=global_vel_current
        )
        pos_controller._compute_thrust_cmd(accel_cmd)
        q_des, q_des_dot = pos_controller.compute_desired_quats(0.0)
        
        # Middle Loop: Attitude
        w_des = att_controller.compute_desired_rates(q_current, q_des, q_des_dot)

        # Inner Loop: Rate
        torque_cmd = rate_controller.compute_torque_commands(state_vector[10:13], w_des, dt=time_step)
        
        # Allocation & Physics Integration
        thrust_mag = np.linalg.norm(pos_controller._thrust_vector)
        control_vector = np.array([thrust_mag, torque_cmd[0], torque_cmd[1], torque_cmd[2]])
        
        d.set_control_vector(control_allocator.get_control_cmds(control_vector))
        d.state_update(time_step)
        
    # --- B. TELEMETRY & LOGGING PHASE ---
    # The main loop pulls the data dynamically based on the current swarm size
    step_data = {d._id: d.get_state_vector() for d in swarm}
    logger.log_step(t, step_data)

# 7. Teardown & Export
log_dir = os.path.join(parent_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "flight_log.csv")

logger.export_to_csv(log_path)
print(f"Simulation complete. Telemetry safely written to {log_path}")