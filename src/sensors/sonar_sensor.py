import numpy as np
from src.sensors.base_sensor import ExteroceptiveSensor

class SonarSensor(ExteroceptiveSensor):
    def __init__(self, update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, 
                 beam_angle, speed_of_sound=343.0, **kwargs):
        super().__init__(update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, **kwargs)
        self.beam_angle = beam_angle
        self.speed_of_sound = speed_of_sound

    def update(self, current_time, drone_state, environment):
        if not self.is_ready(current_time):
            return None
        self.last_update_time = current_time
        
        # Raycasting logic goes here using environment
        distance = self.max_range 
        return {"type": "sonar", "distance": distance}