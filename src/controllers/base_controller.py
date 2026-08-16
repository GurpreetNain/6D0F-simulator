from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Abstract interface for all flight control laws.
    """
    
    @abstractmethod
    def compute_control(self, state_vector, setpoint, dt):
        """
        Calculates the required physical control vector based on current states.
        
        Inputs:
            state_vector: np.array of the drone's current 13-element state.
            setpoint: The target goal (format depends on the specific controller).
            dt: Time step for integral/derivative calculations.
            
        Returns:
            np.array: [thrust_mag, tau_x, tau_y, tau_z] 
                      (To be fed directly into the ControlAllocator)
        """
        pass