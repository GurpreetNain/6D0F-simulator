import numpy as np
from src.sensors.base_sensor import ExteroceptiveSensor

class RadarSensor(ExteroceptiveSensor):
    def __init__(self, update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, 
                 doppler_resolution, rcs_threshold, **kwargs):
        super().__init__(update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, **kwargs)
        self.doppler_resolution = doppler_resolution
        self.rcs_threshold = rcs_threshold

    def update(self, current_time, drone_state, environment):
        if not self.is_ready(current_time):
            return None
        self.last_update_time = current_time
        
        targets = [] 
        return {"type": "radar", "targets": targets}