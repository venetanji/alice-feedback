"""
PID controller implementation for motor position optimization
"""
import numpy as np

class PIDController:
    """
    PID Controller implementation with adaptive gain for motor control.
    Used for precise motor movements during optimization.
    """
    def __init__(self, kp=0.2, ki=0.05, kd=0.1, setpoint=0, motor_gain=1.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.motor_gain = motor_gain  # Motor-specific gain multiplier
        self.max_adjustment = 0.1  # Default max adjustment size
        self.error_history = []  # Track error over time
        
    def update(self, measurement, dt=0.1):
        """
        Update the PID controller with a new measurement and calculate adjustment.
        
        Args:
            measurement: Current measured value
            dt: Time elapsed since last update (in seconds)
            
        Returns:
            float: Adjustment value to be applied
        """
        # Calculate error
        error = self.setpoint - measurement
        
        # Store error history (last 5 errors)
        self.error_history.append(error)
        if len(self.error_history) > 5:
            self.error_history.pop(0)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)  # Prevent integral windup
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply motor-specific gain - this helps motors with limited range
        output *= self.motor_gain
        
        # Limit output to valid range
        output = np.clip(output, -self.max_adjustment, self.max_adjustment)
        
        return output
    
    def adjust_gain(self, response_factor):
        """
        Adjusts the gain based on motor's responsiveness.
        
        Args:
            response_factor: Factor between 0.5 and 2.0 indicating motor responsiveness
                            Values > 1 for more responsive motors (reduce gain)
                            Values < 1 for less responsive motors (increase gain)
        """
        self.motor_gain = np.clip(self.motor_gain * (1.0 / response_factor), 0.5, 2.0)
        
    def set_max_adjustment(self, max_val):
        """Sets the maximum adjustment size for this controller"""
        self.max_adjustment = max_val
        
    def get_error_trend(self):
        """
        Analyzes recent error history to determine if errors are:
        - Decreasing (convergence)
        - Oscillating
        - Stuck (not changing much)
        - Increasing (divergence)
        
        Returns: string indicating trend
        """
        if len(self.error_history) < 3:
            return "insufficient_data"
            
        # Check for oscillation
        signs = [np.sign(e) for e in self.error_history]
        if len(set(signs[-3:])) > 1:  # If signs are changing
            return "oscillating"
            
        # Check for convergence/divergence
        abs_errors = [abs(e) for e in self.error_history]
        if abs_errors[-1] < abs_errors[0] * 0.8:
            return "converging"
        elif abs_errors[-1] > abs_errors[0] * 1.2:
            return "diverging"
            
        # If errors are similar, we're stuck
        if max(abs_errors) - min(abs_errors) < 0.05 * max(abs_errors):
            return "stuck"
            
        return "normal"