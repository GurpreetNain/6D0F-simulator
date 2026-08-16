import numpy as np

class ControlAllocator:
    def __init__(self, dX, dY, nT2D, T_motor_max):
        self._motor_cmds = np.zeros(4)
        self._nT2D = nT2D
        self._T_motor_max = T_motor_max
        
        # Corrected X-Configuration Mapping:
        # Motor 1: Front-Left  (| +Roll | +Pitch | +Yaw |) -> Front motors positive for +Pitch
        # Motor 2: Front-Right (| -Roll | +Pitch | -Yaw |)
        # Motor 3: Rear-Right  (| -Roll | -Pitch | +Yaw |) -> Rear motors negative for +Pitch
        # Motor 4: Rear-Left   (| +Roll | -Pitch | -Yaw |)
        self._control_allocation_matrix = np.array(
            [[   1.0   ,    1.0   ,    1.0   ,    1.0   ],  # Collective Thrust
             [  dX/2.0 ,  -dX/2.0 ,  -dX/2.0 ,   dX/2.0 ],  # Roll Torque (Left vs Right)
             [  dY/2.0 ,   dY/2.0 ,  -dY/2.0 ,  -dY/2.0 ],  # FIX: Inverted Pitch Torque (Front vs Rear)
             [  nT2D   ,   -nT2D  ,   nT2D   ,   -nT2D  ]] # Yaw Torque (CW vs CCW pairs)
        )
        
        # Pre-calculate and cache the inverse allocation matrix on startup
        self._inverse_allocation_matrix = np.linalg.inv(self._control_allocation_matrix)

    def _set_motor_cmds(self, control_cmds):
        # Extremely fast dot product lookups using the cached static matrix
        self._motor_cmds = self._inverse_allocation_matrix @ control_cmds
        self._motor_cmds = np.clip(self._motor_cmds, 0.0, self._T_motor_max)
    
    def get_control_cmds(self, control_cmds):
        self._set_motor_cmds(control_cmds)
        return self._control_allocation_matrix @ self._motor_cmds