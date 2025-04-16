"""
Motor position optimization algorithm
This module contains the algorithm for optimizing motor positions
to match human facial expressions
"""
import asyncio
import numpy as np
import torch
import cv2
import mediapipe as mp
from ..utils.pid_controller import PIDController

async def optimize_motor_positions(human_blendshapes, model, motor_controller, face_landmarker, robot_cap, device, 
                                  max_iterations=10, improvement_threshold=0.0001, patience=3):
    """
    Iteratively adjust motor positions to minimize the difference between human and robot blendshapes.
    Uses PID controllers and gradient descent to efficiently optimize motor positions.
    
    Args:
        human_blendshapes: Facial blendshapes from the human
        model: Neural network model to predict motor positions
        motor_controller: Controller to adjust robot motors
        face_landmarker: MediaPipe face landmarker for robot
        robot_cap: OpenCV video capture for robot camera
        device: Torch device (CPU/CUDA)
        max_iterations: Maximum number of optimization iterations
        improvement_threshold: Minimum improvement to continue optimization
        patience: Number of iterations to continue without improvement before stopping
        
    Returns:
        human_blendshapes: The human blendshapes used as input
        best_motor_positions: Motor positions that achieved the best result
    """
    # Initial blendshape difference and motor positions
    best_mse = float('inf')
    best_motor_positions = motor_controller.get_motor_positions().copy()
    patience_counter = 0
    
    # Start with model prediction as initial guess
    try:
        human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
        predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
        motor_controller.adjust_motors(predicted_motor_positions)
        await asyncio.sleep(0.1)  # Allow motors to move
    except Exception as e:
        print(f"Error making initial prediction: {e}")
    
    # Keep track of MSE history to detect oscillations
    mse_history = []
    
    # Initialize PID controllers for each motor with motor-specific settings
    num_motors = len(motor_controller.get_motor_positions())
    pid_controllers = []
    
    # Setup PID controllers with motor-specific parameters from YAML if available
    try:
        # Try to get motor calibration info from the controller
        motor_info = {}
        if hasattr(motor_controller, 'maestro') and motor_controller.connected:
            for i, channel in enumerate(motor_controller.get_active_channels()):
                cfg = motor_controller.maestro.channels[channel]
                # Calculate motor range as a ratio compared to "standard" range
                # Standard range would be 1000 quarter-microseconds from min to neutral
                # and 1000 from neutral to max
                min_range = cfg['neutral_qus'] - cfg['min_qus']
                max_range = cfg['max_qus'] - cfg['neutral_qus']
                avg_range = (min_range + max_range) / 2
                
                # Calculate responsiveness factor (how much physical movement per command)
                responsiveness = 1000 / avg_range if avg_range > 0 else 1.0
                
                # Motors with smaller ranges need bigger commands (higher gain)
                # Motors with larger ranges need smaller commands (lower gain)
                motor_info[i] = {
                    'channel': channel,
                    'responsiveness': responsiveness,
                    'min_range': min_range,
                    'max_range': max_range
                }
                
                # Set initial gain based on motor range
                gain = 2.0 / responsiveness  # Inverse relationship: smaller range = higher gain
                
                # Set maximum adjustment size based on responsiveness
                max_adjustment = 0.15 * responsiveness  # More responsive motors get smaller adjustments
                
                # Create PID controller with motor-specific parameters
                pid = PIDController(kp=0.3, ki=0.05, kd=0.1, motor_gain=gain)
                pid.set_max_adjustment(min(0.2, max_adjustment))  # Cap at 0.2 to prevent extreme movements
                pid_controllers.append(pid)
                
                print(f"Motor {i} (channel {channel}): responsiveness={responsiveness:.2f}, gain={gain:.2f}, max_adj={max_adjustment:.2f}")
        
    except Exception as e:
        print(f"Error setting up motor-specific PID controllers: {e}")
        # Fallback to default PID controllers if motor info not available
        pid_controllers = [PIDController(kp=0.3, ki=0.05, kd=0.1) for _ in range(num_motors)]
    
    # If we couldn't get motor info, create default controllers
    if len(pid_controllers) != num_motors:
        pid_controllers = [PIDController(kp=0.3, ki=0.05, kd=0.1) for _ in range(num_motors)]
    
    # Gradient descent parameters
    learning_rate = 0.05
    momentum = 0.8
    previous_gradients = np.zeros(num_motors)
    
    # Keep track of previous blendshape differences for calculating gradients
    previous_blendshape_diff = None
    previous_motor_positions = None
    
    # Keep track of each motor's effect on blendshapes
    motor_effectiveness = np.ones(num_motors)  # Start with equal effectiveness
    
    # Track motor movement response over iterations
    motor_response_tracking = [[] for _ in range(num_motors)]
    
    for i in range(max_iterations):
        # Capture current robot frame
        ret_robot, robot_frame = robot_cap.read()
        if not ret_robot:
            print("Error: Could not read frame from robot camera.")
            return human_blendshapes, best_motor_positions
            
        # Process robot frame to get blendshapes
        robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=robot_rgb_frame)
        robot_results = face_landmarker.detect(mp_image)
        robot_blendshapes = extract_blendshapes(robot_results)
        
        if robot_blendshapes is None:
            print("Warning: No robot face detected during optimization.")
            await asyncio.sleep(0.1)  # Brief pause before retry
            continue
            
        # Check blendshape dimensions
        if len(human_blendshapes) != len(robot_blendshapes):
            print(f"Warning: Blendshape dimension mismatch. Human: {len(human_blendshapes)}, Robot: {len(robot_blendshapes)}")
            continue
            
        # Calculate current error between human and robot blendshapes
        current_mse = calculate_blendshape_mse(human_blendshapes, robot_blendshapes)
        mse_history.append(current_mse)
        
        print(f"Optimization iteration {i+1}, MSE: {current_mse:.6f}")
        
        # If this is a better result, save it
        if current_mse < best_mse:
            improvement = best_mse - current_mse
            best_mse = current_mse
            best_motor_positions = motor_controller.get_motor_positions().copy()
            patience_counter = 0  # Reset patience counter when we see improvement
            
            # If improvement is below threshold, count as low improvement
            if i > 0 and improvement < improvement_threshold:
                patience_counter += 1
                print(f"Small improvement (below threshold) in iteration {i+1}")
                
                # Only break if we've had multiple iterations with small improvements
                if patience_counter >= patience:
                    print(f"Optimization converged after {i+1} iterations (small improvements)")
                    break
        else:
            # If no improvement in this iteration
            patience_counter += 1
            print(f"No improvement in iteration {i+1}, patience counter: {patience_counter}/{patience}")
            
            # Only stop if we've had multiple iterations without improvement
            if patience_counter >= patience:
                print(f"Optimization stopped after {i+1} iterations (no improvement for {patience} iterations)")
                # Restore best motor positions
                motor_controller.adjust_motors(best_motor_positions)
                break
        
        # Calculate the blendshape differences (this is our error signal)
        blendshape_diff = np.array(human_blendshapes) - np.array(robot_blendshapes)
        
        # Get current motor positions
        current_motor_positions = motor_controller.get_motor_positions().copy()
        
        # Update motor effectiveness if we have previous data
        if previous_blendshape_diff is not None and previous_motor_positions is not None:
            delta_motors = current_motor_positions - previous_motor_positions
            delta_error = np.sum(np.square(blendshape_diff)) - np.sum(np.square(previous_blendshape_diff))
            
            # For each motor that moved significantly
            for j in range(num_motors):
                if abs(delta_motors[j]) > 0.05:  # Only consider significant movements
                    # Record the motor movement and resulting error change
                    motor_response_tracking[j].append((delta_motors[j], delta_error))
                    
                    # Keep only the last 3 responses
                    if len(motor_response_tracking[j]) > 3:
                        motor_response_tracking[j].pop(0)
                    
                    # If we have enough data, calculate effectiveness
                    if len(motor_response_tracking[j]) >= 2:
                        # Calculate average error change per unit movement
                        effectiveness = 0
                        count = 0
                        for move, err_change in motor_response_tracking[j]:
                            if abs(move) > 0.001:  # Avoid division by very small values
                                effectiveness += abs(err_change / move)
                                count += 1
                        
                        if count > 0:
                            avg_effectiveness = effectiveness / count
                            # Update motor effectiveness (with smoothing)
                            motor_effectiveness[j] = 0.7 * motor_effectiveness[j] + 0.3 * avg_effectiveness
                            
                            # Adjust PID gain based on effectiveness
                            if motor_effectiveness[j] > 0:
                                # More effective motors need lower gain
                                response_factor = motor_effectiveness[j] / np.mean(motor_effectiveness)
                                pid_controllers[j].adjust_gain(response_factor)
        
        # Adaptive learning rate based on iteration and oscillation detection
        if len(mse_history) >= 3:
            if (mse_history[-3] < mse_history[-2] and mse_history[-2] > mse_history[-1]) or \
               (mse_history[-3] > mse_history[-2] and mse_history[-2] < mse_history[-1]):
                # If oscillating, reduce learning rate
                learning_rate *= 0.7
        
        # Initialize motor adjustments
        motor_adjustments = np.zeros(num_motors)
        
        # Method 1: PID Control - use blendshape differences as error input
        # Weight more heavily the blendshapes with larger differences
        weighted_error = np.zeros(num_motors)
        
        # Calculate weights based on blendshape importance
        # Prioritize the most significant differences
        blendshape_importance = np.abs(blendshape_diff)
        top_indices = np.argsort(blendshape_importance)[-min(num_motors, len(blendshape_importance)):]
        
        # Apply PID control using weighted error signals
        for motor_idx in range(num_motors):
            # Use the top blendshape differences as error signals for each motor
            if motor_idx < len(top_indices):
                bs_idx = top_indices[motor_idx]
                # Use PID controller to calculate adjustment for this motor
                pid_adjustment = pid_controllers[motor_idx].update(blendshape_diff[bs_idx])
                motor_adjustments[motor_idx] += pid_adjustment
                
                # Analyze PID controller error trend and adapt parameters if needed
                trend = pid_controllers[motor_idx].get_error_trend()
                if trend == "oscillating":
                    # Reduce gain to stabilize
                    pid_controllers[motor_idx].adjust_gain(1.2)  # Increase divisor = reduce gain
                elif trend == "stuck":
                    # Increase gain to overcome sticking
                    pid_controllers[motor_idx].adjust_gain(0.8)  # Decrease divisor = increase gain
        
        # Method 2: Gradient Descent
        if previous_blendshape_diff is not None and previous_motor_positions is not None:
            # Calculate approximate gradients based on how MSE changed with motor movements
            delta_motors = current_motor_positions - previous_motor_positions
            delta_error = np.sum(np.square(blendshape_diff)) - np.sum(np.square(previous_blendshape_diff))
            
            # Only update if we have meaningful motor movement
            if np.any(np.abs(delta_motors) > 0.001):
                # Approximate the gradient: change in error / change in motor position
                gradients = np.zeros(num_motors)
                for j in range(num_motors):
                    if abs(delta_motors[j]) > 1e-6:  # Avoid division by very small values
                        gradients[j] = delta_error / delta_motors[j]
                    else:
                        gradients[j] = 0
                
                # Apply momentum to smooth out gradient updates
                gradients = momentum * previous_gradients + (1 - momentum) * gradients
                previous_gradients = gradients.copy()
                
                # Update motors using gradients (negative because we want to minimize error)
                # Scale by motor_effectiveness to give less responsive motors bigger adjustments
                for j in range(num_motors):
                    # Scale learning rate by inverse of effectiveness
                    motor_specific_lr = learning_rate
                    if motor_effectiveness[j] > 0:
                        rel_effectiveness = motor_effectiveness[j] / np.mean(motor_effectiveness)
                        if rel_effectiveness < 1.0:
                            # Less effective motors get higher learning rate
                            motor_specific_lr = learning_rate * (1.5 - 0.5 * rel_effectiveness)
                    
                    motor_adjustments[j] -= motor_specific_lr * gradients[j]
        
        # Store current state for next iteration
        previous_blendshape_diff = blendshape_diff.copy()
        previous_motor_positions = current_motor_positions.copy()
        
        # Method 3: Use model guidance with prediction
        try:
            # Get direction from the model as additional guidance
            human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
            predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
            
            # Blend between current position, PID/gradient adjustments, and model prediction
            # Weight between current position (stability), adjustments (improvement), and prediction (guidance)
            new_motor_positions = (
                0.4 * current_motor_positions +                      # Stability component
                0.3 * (current_motor_positions + motor_adjustments) + # PID and gradient adjustments
                0.3 * predicted_motor_positions                      # Model prediction component
            )
        except Exception as e:
            print(f"Error predicting motor positions: {e}")
            # Fallback to just using adjustments if model prediction fails
            new_motor_positions = current_motor_positions + motor_adjustments
        
        # Add small exploration term that decreases over time
        # Scale exploration by motor effectiveness - less effective motors get more exploration
        exploration_scale = 0.03 * (1.0 - min(0.9, i / max_iterations))
        exploration = np.random.normal(0, exploration_scale, size=len(new_motor_positions))
        
        # Scale exploration by inverse of motor effectiveness
        for j in range(num_motors):
            if motor_effectiveness[j] > 0:
                rel_effectiveness = motor_effectiveness[j] / np.mean(motor_effectiveness)
                if rel_effectiveness < 1.0:
                    # Less effective motors get more exploration
                    exploration[j] *= (1.5 - 0.5 * rel_effectiveness)
        
        new_motor_positions += exploration
        
        # Ensure values are within valid range
        new_motor_positions = np.clip(new_motor_positions, -1.0, 1.0)
        
        # Print motor effectiveness and adjustments for debugging
        if i % 3 == 0:  # Print every 3rd iteration to avoid clutter
            print("Motor effectiveness:", ", ".join([f"{e:.2f}" for e in motor_effectiveness]))
            print("Motor adjustments:", ", ".join([f"{a:.2f}" for a in motor_adjustments]))
        
        # Apply the adjusted motor positions
        motor_controller.adjust_motors(new_motor_positions)
            
        # Small delay to allow motors to move
        await asyncio.sleep(0.1)
    
    print(f"Optimization complete. Best MSE: {best_mse:.6f}, Total iterations: {i+1}")
    print("Final motor effectiveness:", ", ".join([f"{e:.2f}" for e in motor_effectiveness]))
    return human_blendshapes, best_motor_positions


# Utility functions for blendshapes
def extract_blendshapes(result):
    """
    Extracts blendshapes from a FaceLandmarkerResult
    
    Args:
        result: MediaPipe FaceLandmarkerResult
        
    Returns:
        List of blendshape values or None if no face detected
    """
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        # Convert face_blendshapes to a flat list of values
        blendshapes = []
        for blendshape in result.face_blendshapes[0]:
            blendshapes.append(blendshape.score)
        return blendshapes
    return None


def calculate_blendshape_mse(blendshapes1, blendshapes2):
    """
    Calculate mean squared error between two sets of blendshapes
    
    Args:
        blendshapes1: First set of blendshape values
        blendshapes2: Second set of blendshape values
        
    Returns:
        Mean squared error between the blendshapes
    """
    if blendshapes1 is None or blendshapes2 is None:
        return float('inf')
    return np.mean(np.square(np.array(blendshapes1) - np.array(blendshapes2)))


async def generate_initial_dataset(motor_controller, face_landmarker, robot_cap, device, samples_per_position=3):
    """
    Generate initial data points for the dataset with neutral position and min/max for each motor.
    
    Args:
        motor_controller: Controller to adjust robot motors
        face_landmarker: MediaPipe face landmarker for robot
        robot_cap: OpenCV video capture for robot camera
        device: Torch device (CPU/CUDA)
        samples_per_position: Number of samples to collect for each position to average out noise
        
    Returns:
        A list of tuples containing (blendshapes, motor_positions)
    """
    initial_data = []
    
    # Get the number of motors
    num_motors = motor_controller.get_num_motors()
    print(f"Generating initial dataset for {num_motors} motors...")
    
    # First, capture the neutral position (all motors at 0)
    print("Setting all motors to neutral position...")
    motor_controller.center_all_motors()
    await asyncio.sleep(0.5)  # Give motors time to reach position
    
    # Capture neutral blendshapes
    neutral_blendshapes = await capture_average_blendshapes(face_landmarker, robot_cap, samples_per_position)
    if neutral_blendshapes is not None:
        neutral_motors = motor_controller.get_motor_positions().copy()
        initial_data.append((neutral_blendshapes, neutral_motors))
        print(f"Added neutral position to dataset with {len(neutral_blendshapes)} blendshapes")
    else:
        print("Failed to capture neutral position blendshapes")
    
    # For each motor, capture min and max positions
    for motor_idx in range(num_motors):
        # Reset all motors to neutral
        motor_controller.center_all_motors()
        await asyncio.sleep(0.2)
        
        # Set this motor to max position (1.0)
        print(f"Setting motor {motor_idx} to maximum position...")
        max_motors = np.zeros(num_motors)
        max_motors[motor_idx] = 1.0
        motor_controller.adjust_motors(max_motors)
        await asyncio.sleep(0.5)  # Give motors time to reach position
        
        # Capture max blendshapes
        max_blendshapes = await capture_average_blendshapes(face_landmarker, robot_cap, samples_per_position)
        if max_blendshapes is not None:
            initial_data.append((max_blendshapes, max_motors.copy()))
            print(f"Added max position for motor {motor_idx} to dataset")
        else:
            print(f"Failed to capture max position blendshapes for motor {motor_idx}")
        
        # Set this motor to min position (-1.0)
        print(f"Setting motor {motor_idx} to minimum position...")
        min_motors = np.zeros(num_motors)
        min_motors[motor_idx] = -1.0
        motor_controller.adjust_motors(min_motors)
        await asyncio.sleep(0.5)  # Give motors time to reach position
        
        # Capture min blendshapes
        min_blendshapes = await capture_average_blendshapes(face_landmarker, robot_cap, samples_per_position)
        if min_blendshapes is not None:
            initial_data.append((min_blendshapes, min_motors.copy()))
            print(f"Added min position for motor {motor_idx} to dataset")
        else:
            print(f"Failed to capture min position blendshapes for motor {motor_idx}")
    
    # Return all motors to neutral position
    motor_controller.center_all_motors()
    
    print(f"Generated {len(initial_data)} initial data points")
    return initial_data


async def capture_average_blendshapes(face_landmarker, cap, num_samples=3):
    """
    Capture multiple frames and average the blendshapes to reduce noise.
    
    Args:
        face_landmarker: MediaPipe face landmarker
        cap: OpenCV video capture
        num_samples: Number of samples to average
        
    Returns:
        Average blendshapes or None if face detection fails
    """
    all_blendshapes = []
    
    for _ in range(num_samples):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
            
        # Process frame to get blendshapes
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_landmarker.detect(mp_image)
        blendshapes = extract_blendshapes(results)
        
        if blendshapes is not None:
            all_blendshapes.append(blendshapes)
        
        # Brief pause between captures
        await asyncio.sleep(0.1)
    
    # If we have at least one valid sample, compute the average
    if all_blendshapes:
        avg_blendshapes = np.mean(all_blendshapes, axis=0)
        return avg_blendshapes
    
    return None