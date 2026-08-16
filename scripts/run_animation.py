import os
import sys
import json
import numpy as np

# Force Python to look in the project root directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.visualization.animation_window import AnimationWindow

if __name__ == "__main__":
    
    # 1. Verify the log file exists
    log_path = os.path.join(parent_dir, "logs", "flight_log.csv")
    if not os.path.exists(log_path):
        print(f"Error: Could not find telemetry log at {log_path}")
        print("Run 'python scripts/run_simulation.py' first to generate data.")
        sys.exit(1)

    # 2. Load the CSV data skipping the header row
    print(f"Loading telemetry from {log_path}...")
    telemetry_data = np.loadtxt(log_path, delimiter=",", skiprows=1)

    # 3. Load configurations for rendering dimensions and playback timing
    config_dir = os.path.join(parent_dir, "config")
    
    with open(os.path.join(config_dir, "Environment.json"), "r") as env_file:    
        env_data = json.load(env_file)
        
    with open(os.path.join(config_dir, "Quadcopter.json"), "r") as quad_file:    
        quad_data = json.load(quad_file)

    dt = env_data["simulation"]["time_step"]
    arm_len = quad_data["arm_length"]
    spread = quad_data["nose_spread_angle"]

    Lx = arm_len * np.cos((np.pi/180.0)*(spread/2)) * 2.0
    Ly = arm_len * np.sin((np.pi/180.0)*(spread/2)) * 2.0

    # 4. Initialize and play
    print("Starting visualizer (Close the window to exit)...")
    visualizer = AnimationWindow(telemetry_data, Lx, Ly, dt)
    visualizer.play()