import numpy as np
from src.models.Quadcopter import Quadcopter
from src.controllers.ControlAllocator import ControlAllocator
from src.sensors.camera_sensor import CameraSensor
from src.actuators.gimbal import Gimbal  # Imported Gimbal class

class DroneAgent:
    def __init__(self, agent_id, quad_config, camera_config, controller_strategy):
        self.id = agent_id
        
        arm_len = quad_config["arm_length"]
        spread = quad_config["nose_spread_angle"]
        
        Lx = arm_len * np.cos((np.pi/180.0)*(spread/2)) * 2.0
        Ly = arm_len * np.sin((np.pi/180.0)*(spread/2)) * 2.0
        
        self.plant = Quadcopter(
            mass=quad_config["mass"],
            J=quad_config["inertia_matrix"],
            Lx=Lx,
            Ly=Ly,
            id=self.id
        )
        
        hover_thrust = np.linalg.norm(self.plant._mass * self.plant._env.g)
        self.allocator = ControlAllocator(
            dX=Ly,
            dY=Lx,
            nT2D=0.05,
            T_motor_max=3 * hover_thrust
        )
        
        self.controller = controller_strategy

        # Camera Initialization
        update_rate_hz = camera_config["update_rate_hz"]
        min_range = camera_config["min_range"]
        max_range = camera_config["max_range"]
        fov_horizontal = camera_config["fov_horizontal"]
        fov_vertical = camera_config["fov_vertical"]
        focal_length = camera_config["focal_length"]
        image_width = focal_length * (2 * np.tan((fov_horizontal/2) * (np.pi/180)))
        image_height = focal_length * (2 * np.tan((fov_vertical/2) * (np.pi/180)))
    
        self.camera = CameraSensor(update_rate_hz=update_rate_hz,
                                   min_range=min_range,
                                   max_range=max_range,
                                   fov_horizontal=fov_horizontal,
                                   fov_vertical=fov_vertical,
                                   focal_length=focal_length,
                                   image_width=image_width,
                                   image_height=image_height)

        # Initialize the Gimbal and pass the camera into its sensor list
        self.gimbal = Gimbal(sensor_list=[self.camera])

    def get_state(self):
        return self.plant.get_state_vector()
        
    def update_sensors(self, current_time, swarm):
        """
        Pulls the current state and processes the environment through the gimbal pipeline.
        """
        current_state = self.get_state()
        return self.gimbal.update(current_time, current_state, swarm)

    def update_step(self, dt, setpoint):
        current_state = self.plant.get_state_vector()
        control_effort = self.controller.compute_control(current_state, setpoint, dt)
        final_control_vector = self.allocator.get_control_cmds(control_effort)
        self.plant.set_control_vector(final_control_vector)
        self.plant.state_update(dt)