import numpy as np

class DataLogger:
    def __init__(self, drone_ids):
        """
        Initializes the logging dictionaries based on the active drones.
        """
        self.drone_ids = drone_ids
        self.timestamps = []
        
        # Create a dictionary to store a list of state vectors for each drone
        self.history = {d_id: [] for d_id in drone_ids}

    def log_step(self, time, step_data):
        """
        Captures the snapshot of the simulation at the current time step.
        Inputs:
            time: The current simulation time (float)
            step_data: Dictionary mapping {drone_id: state_vector}
        """
        self.timestamps.append(time)
        
        for d_id, state_vector in step_data.items():
            # CRITICAL: You must use .copy()
            # Otherwise, Python just appends a reference to the active array, 
            # and your entire log will just show the drone's final resting state.
            self.history[d_id].append(state_vector.copy())

    def export_to_csv(self, filename="flight_log.csv"):
        """
        Flattens the in-memory lists into a 2D matrix and saves to disk.
        """
        print(f"Exporting flight log to {filename}...")
        
        # Convert timestamps to a column vector: shape (N, 1)
        time_col = np.array(self.timestamps).reshape(-1, 1)
        
        # We will build one massive 2D array containing time + all drone states
        export_matrix = time_col
        header_labels = ["Time"]
        
        state_labels = [
            "PosX", "PosY", "PosZ", 
            "VelX", "VelY", "VelZ", 
            "Qw", "Qx", "Qy", "Qz", 
            "RateP", "RateQ", "RateR"
        ]

        for d_id in self.drone_ids:
            # Convert the list of states into a 2D NumPy array: shape (N, 13)
            drone_matrix = np.array(self.history[d_id])
            
            # Concatenate horizontally alongside the time vector
            export_matrix = np.hstack((export_matrix, drone_matrix))
            
            # Generate column headers for this specific drone
            for label in state_labels:
                header_labels.append(f"D{d_id}_{label}")

        # Write to a clean CSV file
        header_string = ",".join(header_labels)
        np.savetxt(filename, export_matrix, delimiter=",", header=header_string, comments='')
        print("Export complete.")