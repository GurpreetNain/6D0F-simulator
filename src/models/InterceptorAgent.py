import numpy as np
from src.models.deltav import Interceptor
from src.common.utils import euler_to_quat


class InterceptorAgent:
    """Wraps deltav.Interceptor with a control law, mirroring DroneAgent's
    shape (plant + controller + update_step) so the same headless-runner
    pattern used for the box-frame quad works for the winged interceptor
    too. No sensors/gimbal yet -- DroneAgent's camera handling doesn't
    apply here."""

    def __init__(self, agent_id, controller_strategy, params=None):
        self.id = agent_id
        self.plant = Interceptor(params=params, id=agent_id)
        self.controller = controller_strategy

    def get_state(self):
        return self.plant.get_state_vector()

    def get_state_for_logger(self):
        """Adapts the 12-state Euler state into the 13-column
        [PosX,PosY,PosZ, VelX,VelY,VelZ, Qw,Qx,Qy,Qz, RateP,RateQ,RateR]
        shape DataLogger/AnimationWindow already expect from Quadcopter --
        so the Interceptor can be logged and played back with unmodified
        infrastructure."""
        xe, ye, h, u, v, w, phi, theta, psi, p, q, r = self.plant.get_state_vector()
        qw, qx, qy, qz = euler_to_quat(phi, theta, psi)
        return np.array([
            xe, ye, -h,          # PosZ is NED down-positive; h is climb-positive
            u, v, w,             # both already body-frame velocities
            qw, qx, qy, qz,
            p, q, r,
        ])

    def update_step(self, dt, setpoint):
        state = self.plant.get_state_vector()
        control_vector = self.controller.compute_control(state, setpoint, dt)
        self.plant.set_control_vector(control_vector)
        self.plant.state_update(dt)
