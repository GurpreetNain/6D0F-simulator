import numpy as np
from src.common.utils import quat2rot

class Gimbal:
    def __init__(self, sensor_list=None):
        """
        Initializes the Gimbal with a list of exteroceptive sensors.
        """
        self.sensor_list = sensor_list if sensor_list is not None else []
        self.alpha_azi = 0.0  # Azimuthal angle[cite: 1]
        self.alpha_ele = 0.0  # Elevation angle[cite: 1]

    def set_gimbal_angles(self, azimuth, elevation):
        self.alpha_azi = azimuth
        self.alpha_ele = elevation

    def update(self, current_time, drone_state, environment):
        """
        Calculates the inertial-to-gimbal rotation matrix and triggers sensor updates.
        """
        # 1. Body to Gimbal_1 (Azimuthal rotation around Z)[cite: 1]
        cos_az = np.cos(self.alpha_azi)
        sin_az = np.sin(self.alpha_azi)
        R_b_g1 = np.array([
            [ cos_az, sin_az, 0.0],
            [-sin_az, cos_az, 0.0],
            [    0.0,    0.0, 1.0]
        ])

        # 2. Gimbal_1 to Gimbal (Elevation rotation around Y)[cite: 1]
        cos_el = np.cos(self.alpha_ele)
        sin_el = np.sin(self.alpha_ele)
        R_g1_g = np.array([
            [cos_el, 0.0, -sin_el],
            [   0.0, 1.0,     0.0],
            [sin_el, 0.0,  cos_el]
        ])

        # 3. Combined Body to Gimbal rotation[cite: 1]
        R_b_g = R_b_g1 @ R_g1_g

        # 4. Inertial to Body mapping using drone's attitude[cite: 1]
        quat_att = drone_state[6:10]
        R_b_i = quat2rot(quat_att) # Fetched from src.common.utils[cite: 1, 4]
        R_i_b = R_b_i.T 

        # 5. Full transformation from Inertial to Gimbal frame
        R_i_g = R_b_g @ R_i_b 

        # 6. Dispatch the rotation matrix to all mounted sensors
        results = []
        for sensor in self.sensor_list:
            reading = sensor.update(current_time, drone_state, environment, R_i_g)
            if reading is not None:
                results.append(reading)
                
        return results