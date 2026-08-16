import numpy as np

class BaseSensor:
    """The absolute base class for all sensors, handling only timing logic."""
    def __init__(self, update_rate_hz):
        self.update_rate_hz = update_rate_hz
        self.last_update_time = 0.0

    def is_ready(self, current_time):
        if self.update_rate_hz <= 0:
            return True
        return (current_time - self.last_update_time) >= (1.0 / self.update_rate_hz)


class ExteroceptiveSensor(BaseSensor):
    """
    Sensors that look outward. 
    They require FOV, ranges, physical offsets, and an Environment to query.
    """
    def __init__(self, update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical,
                 position_offset=None, rotation_offset=None):
        super().__init__(update_rate_hz)
        self.min_range = min_range
        self.max_range = max_range
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        
        self.position_offset = np.array(position_offset) if position_offset is not None else np.zeros(3)
        self.rotation_offset = np.array(rotation_offset) if rotation_offset is not None else np.eye(3)

    def update(self, current_time, drone_state, environment):
        """Must accept both the drone's state AND the external environment."""
        raise NotImplementedError("Exteroceptive sensors must implement this update method.")


class ProprioceptiveSensor(BaseSensor):
    """
    Sensors that look inward.
    They only require the drone's own kinematic/dynamic state.
    """
    def __init__(self, update_rate_hz, noise_density=0.0):
        super().__init__(update_rate_hz)
        self.noise_density = noise_density

    def update(self, current_time, drone_state):
        """Only requires the drone's internal state (no environment needed)."""
        raise NotImplementedError("Proprioceptive sensors must implement this update method.")