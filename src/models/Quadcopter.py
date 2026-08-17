import numpy as np
from src.common.utils import quat2rot
from src.models.Environment import Environment

class Quadcopter:
    def __init__(self, mass, J, Lx, Ly, id):
        self._id = id
        self._mass = mass
        self._J = J
        self._Lx = Lx
        self._Ly = Ly
        '''
        _state_vector[0:3] = global_position
        _state_vector[3:6] = local_linear_velocity
        _state_vector[6:10] = quaternion_attitude
        _state_vector[10:13] = local_angular_velocity
        '''
        self._state_vector = np.zeros(13)
        self._state_vector[6] = 1
        '''
        _control_vector = [thrust, roll_torque, pitch_torque, yaw_torque]
        '''
        self._control_vector = np.zeros(4)
        self._env = Environment(wingspan_meters=Ly, seed=id)

    def _calculate_dynamics(self, state_vector, wind_speed, wind_rot):
        global_position = state_vector[0:3]
        local_linear_velocity = state_vector[3:6]
        quaternion_attitude = state_vector[6:10]
        local_angular_velocity = state_vector[10:13]
        wx = local_angular_velocity[0]
        wy = local_angular_velocity[1]
        wz = local_angular_velocity[2]

        h_meters = max(-global_position[2], 0.1)  # Assuming standard NED where Z is down/negative altitude
        V_mps = np.linalg.norm(local_linear_velocity)
        W20_kts = self._env.W20_knots

        # Pull the vectors directly (they are already in the vehicle's local axis system)
        # wind_speed_vector = self._env.get_wind_speed_vector()
        # wind_rotation_vector = self._env.get_wind_rotation_vector()
        wind_speed_vector = wind_speed
        wind_rotation_vector = wind_rot

        # Compute relative velocities straight away
        v_airspeed_body = local_linear_velocity - wind_speed_vector
        w_airRotation_body = local_angular_velocity - wind_rotation_vector
        translational_drag_coeff = self._env.translational_drag_coeff
        rotational_drag_coeff = self._env.rotational_drag_coeff

        F_drag_body = np.array([
            -0.5 * self._env.air_density * translational_drag_coeff[0] * v_airspeed_body[0] * abs(v_airspeed_body[0]),
            -0.5 * self._env.air_density * translational_drag_coeff[1] * v_airspeed_body[1] * abs(v_airspeed_body[1]),
            -0.5 * self._env.air_density * translational_drag_coeff[2] * v_airspeed_body[2] * abs(v_airspeed_body[2])
        ])

        Torque_drag = np.array([
            -0.5 * self._env.air_density * rotational_drag_coeff[0] * w_airRotation_body[0] * abs(w_airRotation_body[0]),
            -0.5 * self._env.air_density * rotational_drag_coeff[1] * w_airRotation_body[1] * abs(w_airRotation_body[1]),
            -0.5 * self._env.air_density * rotational_drag_coeff[2] * w_airRotation_body[2] * abs(w_airRotation_body[2])
        ])

        state_derivate          = np.zeros(13)
        # FIX: quat2rot is Body -> Global.
        # Global Velocity = Body_to_Global @ Body_Velocity
        state_derivate[0:3] = quat2rot(quaternion_attitude) @ local_linear_velocity

        # FIX: Gravity in Body = Global_to_Body @ Global_Gravity
        # Global_to_Body is the transpose of Body_to_Global
        gravity_body = np.transpose(quat2rot(quaternion_attitude)) @ (self._mass * np.array(self._env.g))

        state_derivate[3:6] = (1 / self._mass) * (np.array([0, 0, -self._control_vector[0]]) +
                                                  gravity_body -
                                                  self._mass * np.cross(local_angular_velocity, local_linear_velocity) +
                                                  F_drag_body)
        if (state_vector[2] >= 0.0) and (state_derivate[2] > 0.0):
            state_derivate[2] = 0.0
            # Also kill local linear velocity values along the normal axis to halt physical movement down through the plane
            local_linear_velocity[2] = 0.0
            state_derivate[5] = 0.0
        # FIX: Corrected Omega matrix to be perfectly skew-symmetric
        Omega = np.array([[  0, -wx, -wy, -wz],
                        [ wx,   0,  wz, -wy],
                        [ wy, -wz,   0,  wx],  # Changed -wy to wy
                        [ wz,  wy, -wx,   0]])

        state_derivate[6:10] = 0.5 * (Omega @ quaternion_attitude)
        state_derivate[10:13]   = np.linalg.inv(self._J) @ (self._control_vector[1:] -
                                                            np.cross(local_angular_velocity, self._J @ local_angular_velocity) +
                                                            Torque_drag)
        return state_derivate

    def state_update(self, dt):
        # 1. Lock down the noise realization vector for this step frame
        self._env.sample_noise(dt)

        # --- STEP 1: Baseline Horizon (t = 0) ---
        # Store the true, clean baseline filter memory before any sub-stepping mutations
        X_env_baseline = self._env._X.copy()

        wind_speed_0 = self._env.get_wind_speed_vector()
        wind_rot_0 = self._env.get_wind_rotation_vector()
        k1 = self._calculate_dynamics(self._state_vector, wind_speed_0, wind_rot_0)

        # --- STEP 2: First Midpoint Pass (t = +dt/2) ---
        pos_mid1 = self._state_vector[0:3] + k1[0:3] * (dt / 2.0)
        vel_mid1 = self._state_vector[3:6] + k1[3:6] * (dt / 2.0)

        # Advance the environment to the midpoint using the baseline state copy
        self._env._X = X_env_baseline.copy()
        w_speed_mid1, w_rot_mid1 = self._env.update_turbulence(
            dt / 2.0, h_meters=max(-pos_mid1[2], 0.1), V_mps=np.linalg.norm(vel_mid1 - wind_speed_0)
        )
        k2 = self._calculate_dynamics(self._state_vector + k1 * (dt / 2.0), w_speed_mid1, w_rot_mid1)

        # --- STEP 3: Second Midpoint Pass (t = +dt/2) ---
        pos_mid2 = self._state_vector[0:3] + k2[0:3] * (dt / 2.0)
        vel_mid2 = self._state_vector[3:6] + k2[3:6] * (dt / 2.0)

        # RESET the environment back to baseline before running the k3 midpoint update pass!
        self._env._X = X_env_baseline.copy()
        w_speed_mid2, w_rot_mid2 = self._env.update_turbulence(
            dt / 2.0, h_meters=max(-pos_mid2[2], 0.1), V_mps=np.linalg.norm(vel_mid2 - w_speed_mid1)
        )
        k3 = self._calculate_dynamics(self._state_vector + k2 * (dt / 2.0), w_speed_mid2, w_rot_mid2)

        # --- STEP 4: Final Boundary Pass (t = +dt) ---
        pos_end = self._state_vector[0:3] + k3[0:3] * dt
        vel_end = self._state_vector[3:6] + k3[3:6] * dt

        # RESET the environment back to baseline before computing the final k4 boundary step pass!
        self._env._X = X_env_baseline.copy()
        w_speed_end, w_rot_end = self._env.update_turbulence(
            dt, h_meters=max(-pos_end[2], 0.1), V_mps=np.linalg.norm(vel_end - w_speed_mid2)
        )
        k4 = self._calculate_dynamics(self._state_vector + k3 * dt, w_speed_end, w_rot_end)

        # --- STEP 5: Rigid Body Final Update ---
        self._state_vector += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- STEP 6: Commit Final Environment Progression ---
        # Now that the vehicle step is finished, permanently commit the step-forward filter memory
        # to the instance so the next loop frame starts from the correct progression state.
        self._env._X = X_env_baseline.copy()
        # Finalize the true persistent step-forward calculation
        self._env.update_turbulence(dt, h_meters=max(-self._state_vector[2], 0.1), V_mps=np.linalg.norm(self._state_vector[3:6] - w_speed_end))

    def get_state_vector(self):
        return self._state_vector

    def set_control_vector(self, control_vector):
        self._control_vector = control_vector
