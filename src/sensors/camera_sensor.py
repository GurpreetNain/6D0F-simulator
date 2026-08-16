import numpy as np
from src.sensors.base_sensor import ExteroceptiveSensor

class CameraSensor(ExteroceptiveSensor):
    def __init__(self, update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, 
                 focal_length, image_width, image_height, **kwargs):
        super().__init__(update_rate_hz, min_range, max_range, fov_horizontal, fov_vertical, **kwargs)
        self.focal_length = focal_length
        self.image_width = image_width
        self.image_height = image_height
        
        # Constant rotation mapping Gimbal axes to Camera axes
        self.R_g_c = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])

    def update(self, current_time, drone_state, environment, R_i_g=None):
        if not self.is_ready(current_time):
            return None
        self.last_update_time = current_time
        
        # Fallback to Identity if camera is hard-mounted (no gimbal)
        if R_i_g is None:
            R_i_g = np.eye(3)
        
        detections = [] 
        own_pos = drone_state[0:3]
        
        for drone in environment:
            # Ignore self-detection using the drone ID suffix in state vector
            if drone.id == drone_state[-1]:
                continue
                
            env_drone_state = drone.get_state()
            
            # v^i: The relative vector in the inertial frame
            v_i = env_drone_state[0:3] - own_pos
            
            dist = np.linalg.norm(v_i)
            
            # Distance filter (Min/Max Range)
            if self.min_range <= dist <= self.max_range:
                # Map inertial vector to raw camera frame: v^c_raw = R_g^c * R_i^g * v^i
                v_c_raw = self.R_g_c @ (R_i_g @ v_i) 
                
                Xc, Yc, Zc = v_c_raw[0], v_c_raw[1], v_c_raw[2]
                
                # Check if the target is in front of the camera (Depth > 0)
                if Zc > 0:
                    # Pinhole projection mapping
                    x_img = self.focal_length * (Xc / Zc)
                    y_img = self.focal_length * (Yc / Zc)
                    
                    # Visual Cone Filter: Check if coordinates fall within the image frame bounds
                    if (abs(x_img) <= self.image_width / 2.0) and (abs(y_img) <= self.image_height / 2.0):
                        
                        # Pack requested format: [x_coordinate, y_coordinate, depth_as_focal_length]
                        v_c_projected = np.array([x_img, y_img, self.focal_length])
                        
                        detections.append({
                            "id": drone.id,
                            "distance": dist,
                            "v_c": v_c_projected
                        })
            
        return {"type": "camera", "detections": detections}