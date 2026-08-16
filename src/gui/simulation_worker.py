import numpy as np
import json
import os
from PyQt6.QtCore import QThread, pyqtSignal

from src.models.Environment import Environment
from src.models.DroneAgent import DroneAgent
from src.logging.data_logger import DataLogger

class SimulationWorker(QThread):
    progress_update = pyqtSignal(int)
    simulation_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, t_final, dt, swarm_configs, trajectory_file=None):
        super().__init__()
        self.t_final = t_final
        self.dt = dt
        self.swarm_configs = swarm_configs
        self.trajectory_file = trajectory_file
        self.is_running = True 

    def run(self):
        try:
            # 1. Load Trajectory CSV (if provided)
            trajectory_matrix = None
            if self.trajectory_file and os.path.exists(self.trajectory_file):
                trajectory_matrix = np.loadtxt(self.trajectory_file, delimiter=',', skiprows=1)

            # 2. Load Configurations (Environment & Quadcopter)
            current_dir = os.path.dirname(__file__)
            config_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'config'))
            
            with open(os.path.join(config_dir, "Environment.json"), "r") as env_file:    
                env_data = json.load(env_file)
            with open(os.path.join(config_dir, "Quadcopter.json"), "r") as quad_file:    
                quad_data = json.load(quad_file)
            with open(os.path.join(config_dir, "Camera.json"), "r") as camera_file:    
                camera_data = json.load(camera_file)

            # 3. Build the Swarm Environment
            environment = Environment(seed=1, wingspan_meters=quad_data["arm_length"] * 2) 

            pos_gains = {'Kp': np.array([3.5, 3.5, 3.5]), 'Kd': np.array([2.3, 2.3, 2.3])}
            att_gains = {'Kp': np.array([18.0, 18.0, 18.0])}
            rate_gains = {'Kp': np.array([4.0, 4.0, 5.0]), 'Ki': np.zeros(3), 'Kd': np.zeros(3)}

            swarm = []
            drone_ids = []
            
            for config in self.swarm_configs:
                agent_id = config["id"]
                start_pos = config["start_pos"]
                CtrlClass = config["controller_class"]
                
                controller_instance = CtrlClass(pos_gains, att_gains, rate_gains, quad_data["mass"], environment.g)
                agent = DroneAgent(agent_id=agent_id, quad_config=quad_data, camera_config=camera_data, controller_strategy=controller_instance)
                agent.plant._state_vector[0:3] = start_pos
                
                swarm.append(agent)
                drone_ids.append(agent_id)

            logger = DataLogger(drone_ids=drone_ids)

            # 4. Main Physics Loop
            sim_time = np.arange(0, self.t_final, step=self.dt)
            total_steps = len(sim_time)

            for i, t in enumerate(sim_time):
                if not self.is_running:
                    self.simulation_finished.emit("Simulation canceled by user.")
                    return
                
                # --- Map CSV Setpoints (If loaded) ---
                global_setpoint = np.zeros(11) 

                if trajectory_matrix is not None:
                    t_col = trajectory_matrix[:, 0]
                    num_cols = trajectory_matrix.shape[1]
                    
                    for j in range(4):
                        global_setpoint[j] = np.interp(t, t_col, trajectory_matrix[:, j+1])
                    
                    if num_cols >= 8:
                        for j in range(4, 7):
                            global_setpoint[j] = np.interp(t, t_col, trajectory_matrix[:, j+1])
                            
                    if num_cols >= 12:
                        for j in range(7, 11):
                            global_setpoint[j] = np.interp(t, t_col, trajectory_matrix[:, j+1])
                
                # Step the swarm
                for agent in swarm:
                    try:
                        # Feed necessary states to any controller built from our GUI Generator
                        if hasattr(agent.controller, 'update_context'):
                            agent.controller.update_context(t, agent.id, swarm)
                            
                        # Step the plant
                        agent.update_step(self.dt, global_setpoint)
                    except Exception as e:
                        raise RuntimeError(f"Math Error during execution (t={t:.2f}):\n{e}")
                
                # Log telemetry
                step_data = {a.id: a.get_state() for a in swarm}
                logger.log_step(t, step_data)
                
                # Update progress bar
                if i % 500 == 0:
                    progress_percentage = int((i / total_steps) * 100)
                    self.progress_update.emit(progress_percentage)
                    
            # 5. Teardown & Export
            log_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'logs'))
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "gui_flight_log.csv")
            
            logger.export_to_csv(log_path)
            
            self.progress_update.emit(100)
            self.simulation_finished.emit(f"Simulation complete! Log saved to:\n{log_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"Engine Fault:\n{str(e)}")

    def stop(self):
        self.is_running = False