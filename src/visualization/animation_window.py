import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.common.utils import quat2rot

class AnimationWindow:
    def __init__(self, data_matrix, Lx, Ly, dt):
        """
        Initializes the 3D visualization window.
        Inputs:
            data_matrix: The loaded numpy array from the telemetry CSV.
            Lx, Ly: Drone span dimensions for rendering the cross-frame.
            dt: The simulation time step (used to calculate playback speed).
        """
        self.data = data_matrix
        self.dt = dt
        
        # Local Drone Geometry (X-Configuration)
        # Defined as [X_coords, Y_coords, Z_coords] for two crossing arms
        self.arm1_local = np.array([
            [Lx/2, -Lx/2],
            [Ly/2, -Ly/2],
            [0, 0]
        ])
        
        self.arm2_local = np.array([
            [Lx/2, -Lx/2],
            [-Ly/2, Ly/2],
            [0, 0]
        ])

        # Setup Figure and 3D Axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize rendering objects
        self.line_arm1, = self.ax.plot([], [], [], 'b-', linewidth=4, label="Front-Right / Back-Left")
        self.line_arm2, = self.ax.plot([], [], [], 'r-', linewidth=4, label="Front-Left / Back-Right")
        self.path, = self.ax.plot([], [], [], 'g--', alpha=0.4, linewidth=1.5, label="Trajectory")
        
        self.ax.set_xlabel('X (North, m)')
        self.ax.set_ylabel('Y (East, m)')
        self.ax.set_zlabel('Altitude (-Z Down, m)')
        self.ax.set_title('Quadcopter Flight Playback')
        self.ax.legend(loc="upper left")

    def _update_frame(self, frame_idx):
        """Calculates geometry for a single frame of the animation."""
        # Map columns based on DataLogger export format
        # Col 0: Time | Cols 1-3: Pos | Cols 4-6: Vel | Cols 7-10: Quat
        x = self.data[frame_idx, 1]
        y = self.data[frame_idx, 2]
        z = self.data[frame_idx, 3]
        
        qw = self.data[frame_idx, 7]
        qx = self.data[frame_idx, 8]
        qy = self.data[frame_idx, 9]
        qz = self.data[frame_idx, 10]
        
        # 1. Rotate Local Frame
        R = quat2rot(np.array([qw, qx, qy, qz]))
        arm1_rotated = R @ self.arm1_local
        arm2_rotated = R @ self.arm2_local
        
        # 2. Translate to Global Position (Invert Z for visual altitude)
        arm1_global = arm1_rotated + np.array([[x], [y], [-z]])
        arm2_global = arm2_rotated + np.array([[x], [y], [-z]])
        
        # 3. Apply to Matplotlib lines
        self.line_arm1.set_data(arm1_global[0, :], arm1_global[1, :])
        self.line_arm1.set_3d_properties(arm1_global[2, :])
        
        self.line_arm2.set_data(arm2_global[0, :], arm2_global[1, :])
        self.line_arm2.set_3d_properties(arm2_global[2, :])
        
        # 4. Update the trailing trajectory ribbon
        self.path.set_data(self.data[:frame_idx, 1], self.data[:frame_idx, 2])
        self.path.set_3d_properties(-self.data[:frame_idx, 3])
        
        # 5. Dynamic "Chase Camera" limits
        margin = 3.0
        self.ax.set_xlim(x - margin, x + margin)
        self.ax.set_ylim(y - margin, y + margin)
        self.ax.set_zlim(-z - margin, -z + margin)
        
        return self.line_arm1, self.line_arm2, self.path

    def play(self):
        """Starts the Matplotlib event loop and handles downsampling."""
        # Calculate how many simulation steps fit into ~33ms (30 FPS)
        target_frame_time = 0.033 
        skip_steps = max(1, int(target_frame_time / self.dt))
        
        # Generate an array of indices to render
        frame_indices = np.arange(0, len(self.data), skip_steps)
        
        self.ani = animation.FuncAnimation(
            self.fig, 
            self._update_frame, 
            frames=frame_indices,
            interval=target_frame_time * 1000, 
            blit=False
        )
        
        plt.show()