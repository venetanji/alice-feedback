"""
Motor Controller module for controlling servos based on facial expressions
"""
import numpy as np
import time
from .maestro import MaestroController

class MotorController:
    def __init__(self, port="COM6", baudrate=9600, yaml_file="config/motor_ranges.yaml"):
        """
        Initialize the motor controller.
        
        Args:
            port (str): Serial port for the Maestro controller
            baudrate (int): Serial baudrate
            yaml_file (str): Path to YAML file with motor range calibration
        """
        self.motor_positions = None  # Will be initialized after channels are determined
        self.active_channels = []  # List of active channel numbers
        self.connected = False  # Connection status
        
        try:
            # Initialize the Maestro controller
            self.maestro = MaestroController(port=port, baudrate=baudrate)
            
            # Load motor ranges from YAML file if specified
            if yaml_file:
                self.active_channels = self.maestro.load_motor_ranges_from_yaml(yaml_file)
                self.active_channels.sort()  # Sort channels for consistency
                print(f"Active channels from YAML: {self.active_channels}")
            
            # Initialize motor positions array
            if self.active_channels:
                self.motor_positions = np.zeros(len(self.active_channels))
            else:
                # no motor ranges found run in simulation mode
                print("No motor ranges found in YAML, running in simulation mode")
                raise ValueError("No active channels found in YAML file")
            
            
            # Center all servos at startup
            self.center_all_motors()
            self.connected = True
            print(f"Successfully connected to Maestro controller on {port}")
            print(f"Number of active motors: {len(self.active_channels)}")
            
        except Exception as e:
            # Fallback to dummy controller if hardware not available
            self.connected = False
            print(f"Could not connect to Maestro controller: {e}")
            print("Operating in simulation mode (no actual motor control)")
            
            # Initialize defaults for simulation mode
            if not self.active_channels:
                self.active_channels = list(range(6))
                self.motor_positions = np.zeros(len(self.active_channels))
    
    def get_num_motors(self):
        """Get the number of active motors.""" 
        return len(self.active_channels)
    
    def get_active_channels(self):
        """Get the list of active channel numbers."""
        return self.active_channels.copy()
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'maestro') and hasattr(self, 'connected') and self.connected:
            self.center_all_motors()  # Return to neutral position
            self.maestro.close()
    
    def center_all_motors(self):
        """Center all motors to their neutral positions."""
        if hasattr(self, 'maestro') and self.connected:
            for channel in self.active_channels:
                self.maestro.set_servo_normalized(channel, 0.0)  # 0.0 is neutral position
            
        # Reset all motor positions to zero (neutral)
        if self.motor_positions is not None:
            self.motor_positions = np.zeros(len(self.active_channels))
    
    def adjust_motors(self, predicted_positions):
        """
        Adjust motor positions based on predicted values.
        
        Args:
            predicted_positions (np.ndarray): Array of positions from -1.0 to 1.0
        """
        # Ensure the array size matches our active motors
        if len(predicted_positions) != len(self.active_channels):
            print(f"Warning: Predicted positions array length ({len(predicted_positions)}) " +
                  f"doesn't match number of active channels ({len(self.active_channels)})")
            # If more predictions than channels, truncate; if fewer, use what we have
            predicted_positions = predicted_positions[:len(self.active_channels)]
        
        # Store normalized positions internally
        self.motor_positions = np.clip(predicted_positions, -1.0, 1.0)
        
        if hasattr(self, 'maestro') and self.connected:
            # Send commands to the actual servo controller
            for i, channel in enumerate(self.active_channels):
                if i < len(self.motor_positions):
                    self.maestro.set_servo_normalized(channel, self.motor_positions[i])
        
    def get_motor_positions(self):
        """Get the current motor positions."""
        return self.motor_positions
    
    def read_actual_positions(self):
        """
        Read the actual positions from the servo controller.
        
        Returns:
            np.ndarray: Array of positions from -1.0 to 1.0
        """
        if hasattr(self, 'maestro') and self.connected:
            positions = []
            for channel in self.active_channels:
                try:
                    # Get the position in quarter-microseconds
                    qus_position = self.maestro.get_position(channel)
                    
                    # Convert to normalized position (-1.0 to 1.0)
                    cfg = self.maestro.channels[channel]
                    if qus_position <= cfg['neutral_qus']:
                        # Map from min...neutral to -1.0...0.0
                        normalized = -1.0 + ((qus_position - cfg['min_qus']) / 
                                           (cfg['neutral_qus'] - cfg['min_qus']))
                    else:
                        # Map from neutral...max to 0.0...1.0
                        normalized = (qus_position - cfg['neutral_qus']) / \
                                   (cfg['max_qus'] - cfg['neutral_qus'])
                    
                    positions.append(normalized)
                except Exception as e:
                    print(f"Error reading position for channel {channel}: {e}")
                    # Use last known position for this channel if available
                    idx = self.active_channels.index(channel)
                    if idx < len(self.motor_positions):
                        positions.append(self.motor_positions[idx])
                    else:
                        positions.append(0.0)  # Default to neutral
            
            return np.array(positions)
        else:
            # Return the simulated positions if not connected
            return self.motor_positions