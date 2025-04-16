import serial
import time
import yaml  # Add PyYAML for YAML file generation

class MaestroController:
    """
    A class to control servos using the Pololu Maestro servo controller.
    
    This implements the compact protocol as described in the Pololu Maestro documentation:
    https://www.pololu.com/docs/0J40/5.e
    """
    
    def __init__(self, port='COM6', device_number=0x0C, baudrate=9600, timeout=1.0):
        """
        Initialize the Maestro controller.
        
        Args:
            port (str): Serial port to connect to (e.g., 'COM3' on Windows)
            device_number (int): Device number for Pololu protocol (default 0x0C)
            baudrate (int): Baud rate for serial communication (default 9600)
            timeout (float): Serial timeout in seconds (default 1.0)
        """
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.device_number = device_number
        self.channels = {}  # Track state of each channel
    
    def __del__(self):
        """Clean up by closing the serial port when object is destroyed."""
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()
    
    def close(self):
        """Explicitly close the serial port."""
        if self.serial.is_open:
            self.serial.close()
    
    def _send_command(self, command, *data_bytes):
        """
        Send a command using the compact protocol.
        
        Args:
            command (int): Command byte
            *data_bytes: Additional data bytes
        """
        cmd_bytes = bytearray([command] + list(data_bytes))
        self.serial.write(cmd_bytes)
    
    def _send_pololu_command(self, command, *data_bytes):
        """
        Send a command using the Pololu protocol.
        
        Args:
            command (int): Command byte
            *data_bytes: Additional data bytes
        """
        cmd_bytes = bytearray([0xAA, self.device_number, command] + list(data_bytes))
        self.serial.write(cmd_bytes)
    
    def set_target(self, channel, target):
        """
        Set the target position of a servo.
        
        Args:
            channel (int): Channel number (0-23)
            target (int): Target position in quarter-microseconds (0-8000)
                          For servos, typical range is 4000-8000 (1000-2000 µs)
        """
        target = max(0, min(target, 8000))  # Clamp target to valid range
        
        # Track target for this channel
        self.channels[channel].update({
            'target': target,
            'last_update': time.time()
        })
        
        # Compact protocol: Command 0x84, channel number, target low bits, target high bits
        low_bits = target & 0x7F  # 7 bits for least significant byte
        high_bits = (target >> 7) & 0x7F  # 7 bits for most significant byte
        self._send_command(0x84, channel, low_bits, high_bits)
    
    def set_multiple_targets(self, start_channel, targets):
        """
        Set multiple targets at once (Mini Maestro 12, 18, and 24 only).
        
        Args:
            start_channel (int): First channel number (0-23)
            targets (list): List of target positions in quarter-microseconds
        """
        if not targets:
            return
            
        data_bytes = [0x9F, len(targets), start_channel]
        
        for i, target in enumerate(targets):
            target = max(0, min(target, 8000))  # Clamp target to valid range
            
            # Track target for this channel
            channel = start_channel + i
            self.channels[channel] = {
                'target': target,
                'last_update': time.time()
            }
            
            # Add target bytes (low bits, high bits)
            low_bits = target & 0x7F
            high_bits = (target >> 7) & 0x7F
            data_bytes.extend([low_bits, high_bits])
        
        self.serial.write(bytearray(data_bytes))
    
    def set_speed(self, channel, speed):
        """
        Set the speed limit for a servo.
        
        Args:
            channel (int): Channel number (0-23)
            speed (int): Speed limit in units of (0.25 µs)/(10 ms)
                         0 = unlimited, 1 = slowest, 255 = fastest
        """
        speed = max(0, min(speed, 255))  # Clamp speed to valid range
        
        # Compact protocol: Command 0x87, channel number, speed low bits, speed high bits
        low_bits = speed & 0x7F  # 7 bits for least significant byte
        high_bits = (speed >> 7) & 0x7F  # 7 bits for most significant byte
        self._send_command(0x87, channel, low_bits, high_bits)
    
    def set_acceleration(self, channel, acceleration):
        """
        Set the acceleration limit for a servo.
        
        Args:
            channel (int): Channel number (0-23)
            acceleration (int): Acceleration limit (0-255)
                               0 = unlimited, 1 = slowest, 255 = fastest
        """
        acceleration = max(0, min(acceleration, 255))  # Clamp acceleration to valid range
        
        # Compact protocol: Command 0x89, channel number, acceleration low bits, acceleration high bits
        low_bits = acceleration & 0x7F  # 7 bits for least significant byte
        high_bits = (acceleration >> 7) & 0x7F  # 7 bits for most significant byte
        self._send_command(0x89, channel, low_bits, high_bits)
    
    def get_position(self, channel):
        """
        Get the current position of a servo.
        
        Args:
            channel (int): Channel number (0-23)
        
        Returns:
            int: Position in quarter-microseconds
        """
        # Compact protocol: Command 0x90, channel number
        self._send_command(0x90, channel)
        
        # Response: position low 8 bits, position high 8 bits
        low_bits = ord(self.serial.read(1))
        high_bits = ord(self.serial.read(1))
        
        # Combine high and low bits to get position
        position = (high_bits << 8) | low_bits
        
        return position
    
    def get_moving_state(self):
        """
        Check if any servos are moving.
        
        Note: This command only works reliably on Mini Maestros, not on Micro Maestro.
        
        Returns:
            bool: True if any servos are moving, False otherwise
        """
        # Compact protocol: Command 0x93
        self._send_command(0x93)
        
        # Response: 0x00 if no servos are moving, 0x01 if servos are moving
        response = ord(self.serial.read(1))
        
        return bool(response)
    
    def get_errors(self):
        """
        Get the error register and clear it.
        
        Returns:
            int: Error register value
        """
        # Compact protocol: Command 0xA1
        self._send_command(0xA1)
        
        # Response: error low 8 bits, error high 8 bits
        low_bits = ord(self.serial.read(1))
        high_bits = ord(self.serial.read(1))
        
        # Combine high and low bits to get error register
        errors = (high_bits << 8) | low_bits
        
        return errors
    
    def go_home(self):
        """Send all servos to their home positions."""
        # Compact protocol: Command 0xA2
        self._send_command(0xA2)
    
    def convert_to_quarter_us(self, microseconds):
        """
        Convert microseconds to quarter-microseconds for Maestro commands.
        
        Args:
            microseconds (float): Value in microseconds (e.g., 1500)
            
        Returns:
            int: Value in quarter-microseconds (e.g., 6000)
        """
        return int(microseconds * 4)
    
    def convert_to_microseconds(self, quarter_us):
        """
        Convert quarter-microseconds to microseconds.
        
        Args:
            quarter_us (int): Value in quarter-microseconds (e.g., 6000)
            
        Returns:
            float: Value in microseconds (e.g., 1500.0)
        """
        return quarter_us / 4.0
    
    def set_servo_range(self, channel, min_us=1000, max_us=2000, neutral_us=1500):
        """
        Configure the range for a servo in microseconds.
        This is a convenience method to remember channel configuration.
        
        Args:
            channel (int): Channel number (0-23)
            min_us (float): Minimum pulse width in microseconds (default 1000)
            max_us (float): Maximum pulse width in microseconds (default 2000)
            neutral_us (float): Neutral position in microseconds (default 1500)
        """
        if channel not in self.channels:
            self.channels[channel] = {}
            
        self.channels[channel].update({
            'min_us': min_us,
            'max_us': max_us,
            'neutral_us': neutral_us,
            'min_qus': self.convert_to_quarter_us(min_us),
            'max_qus': self.convert_to_quarter_us(max_us),
            'neutral_qus': self.convert_to_quarter_us(neutral_us)
        })
    
    def set_servo_normalized(self, channel, position):
        """
        Set servo position using a normalized value between -1.0 and 1.0.
        
        Args:
            channel (int): Channel number (0-23)
            position (float): Normalized position from -1.0 (min) to 1.0 (max),
                              with 0.0 being the neutral position
        """
            
        cfg = self.channels[channel]
        position = max(-1.0, min(position, 1.0))  # Clamp position
        
        if position < 0:
            # Map from -1.0...0.0 to min...neutral
            target = cfg['min_qus'] + ((cfg['neutral_qus'] - cfg['min_qus']) * (position + 1.0))
        else:
            # Map from 0.0...1.0 to neutral...max
            target = cfg['neutral_qus'] + ((cfg['max_qus'] - cfg['neutral_qus']) * position)
        #print(f"Setting channel {channel} to target: {target}") 
        self.set_target(channel, int(target))
    
    def generate_motor_ranges_yaml(self, channels_to_check=None, filename="motor_ranges.yaml"):
        """
        Generate a YAML file with motor range information by testing channel limits.
        
        This function:
        1. Sends all servos to home position
        2. For each specified channel:
           - Records neutral position
           - Sets position to minimum (0)
           - Records actual position
           - Sets position to maximum (8000)
           - Records actual position
        3. Saves all data to a YAML file
        
        Args:
            channels_to_check (list): List of channel numbers to check. If None, checks channels 0-5.
            filename (str): Name of the YAML file to save
            
        Returns:
            dict: The motor ranges data that was saved to the file
        """
        if channels_to_check is None:
            channels_to_check = range(6)  # Default to first 6 channels
            
        # First send all servos to home position
        self.go_home()
        time.sleep(1.0)  # Wait for servos to reach home position
        
        motor_ranges = {}
        
        for channel in channels_to_check:
            print(f"Testing channel {channel}...")
            
            # Get neutral position
            neutral_position = self.get_position(channel)
            neutral_us = self.convert_to_microseconds(neutral_position)
            
            # if position is 0, motor is disabled, skip...
            if neutral_position == 0:               
                print(f"  Skipping channel {channel} as the motor is disabled.")
                continue
            
            # Test minimum position
            print(f"  Setting channel {channel} to minimum...")
            self.set_target(channel, 1000)
            time.sleep(1.0)  # Wait for servo to reach position
            min_position = self.get_position(channel)
            min_us = self.convert_to_microseconds(min_position)
            
            # Test maximum position
            print(f"  Setting channel {channel} to maximum...")
            self.set_target(channel, 8000)
            time.sleep(1.0)  # Wait for servo to reach position
            max_position = self.get_position(channel)
            max_us = self.convert_to_microseconds(max_position)
            
            # Return to neutral position
            print(f"  Returning channel {channel} to neutral...")
            self.set_target(channel, neutral_position)
            
            # Store the data
            motor_ranges[f"channel_{channel}"] = {
                "neutral_position": neutral_position,
                "neutral_us": neutral_us,
                "min_position": min_position,
                "min_us": min_us,
                "max_position": max_position,
                "max_us": max_us
            }
            
            # Update channel configuration with these values
            self.set_servo_range(channel, min_us=min_us, max_us=max_us, neutral_us=neutral_us)
            
            # Wait before moving to next channel
            time.sleep(0.5)
        
        # Return all servos to home position
        self.go_home()
        
        # Save to YAML file
        with open(filename, 'w') as yaml_file:
            yaml.dump(motor_ranges, yaml_file, default_flow_style=False)
            
        print(f"Motor ranges saved to {filename}")
        return motor_ranges
    
    def load_motor_ranges_from_yaml(self, yaml_file):
        """
        Load motor ranges from a YAML file and configure servos accordingly.
        
        Args:
            yaml_file (str): Path to the YAML file with motor calibration data
            
        Returns:
            list: List of channel numbers that were configured
        """
        try:
            with open(yaml_file, 'r') as file:
                motor_ranges = yaml.safe_load(file)
                
            configured_channels = []
            
            # Configure each channel found in the YAML file
            for channel_key, ranges in motor_ranges.items():
                # Extract channel number from key (e.g., "channel_3" -> 3)
                try:
                    channel = int(channel_key.split('_')[1])
                    
                    # Configure this channel with the ranges from YAML
                    self.set_servo_range(
                        channel, 
                        min_us=ranges['min_us'],
                        max_us=ranges['max_us'],
                        neutral_us=ranges['neutral_us']
                    )
                    
                    configured_channels.append(channel)
                    print(f"Configured {channel_key}: min={ranges['min_us']}, max={ranges['max_us']}, neutral={ranges['neutral_us']}")
                except (ValueError, KeyError) as e:
                    print(f"Error configuring {channel_key}: {e}")
            
            return configured_channels
            
        except Exception as e:
            print(f"Error loading motor ranges from {yaml_file}: {e}")
            return []