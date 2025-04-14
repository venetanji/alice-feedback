import cv2
import mediapipe as mp
import numpy as np
import asyncio
import queue
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from maestro import MaestroController

# Mediapipe Modules with updated imports for face landmarker
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Import mediapipe vision module for face landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Thread-safe queue for sharing data
data_queue = queue.Queue()

# PyTorch Dataset for real-time data
class RealTimeDataset(Dataset):
    def __init__(self):
        self.human_blendshapes = []  # Human facial blendshapes (input)
        self.motor_positions = []    # Optimized motor positions (output)

    def add_data(self, human_blendshapes, motor_positions):
        self.human_blendshapes.append(human_blendshapes)
        self.motor_positions.append(motor_positions)

    def __len__(self):
        return len(self.human_blendshapes)

    def __getitem__(self, idx):
        return torch.tensor(self.human_blendshapes[idx], dtype=torch.float32), torch.tensor(self.motor_positions[idx], dtype=torch.float32)

# Neural network model - updated with dynamic input size
class LandmarkToMotorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LandmarkToMotorModel, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Check if input shape matches expected shape
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        return self.fc(x)

# Optimize motor positions using PID controllers and gradient descent
async def optimize_motor_positions(human_blendshapes, model, motor_controller, face_landmarker, robot_cap, device, max_iterations=10, improvement_threshold=0.0001, patience=3):
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
    
    # Initialize PID controllers for each motor
    num_motors = len(motor_controller.get_motor_positions())
    pid_controllers = [PIDController(kp=0.3, ki=0.05, kd=0.1) for _ in range(num_motors)]
    
    # Gradient descent parameters
    learning_rate = 0.05
    momentum = 0.8
    previous_gradients = np.zeros(num_motors)
    
    # Keep track of previous blendshape differences for calculating gradients
    previous_blendshape_diff = None
    previous_motor_positions = None
    
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
                for j in range(num_motors):
                    motor_adjustments[j] -= learning_rate * gradients[j]
        
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
                0.4 * current_motor_positions +                    # Stability component
                0.3 * (current_motor_positions + motor_adjustments) + # PID and gradient adjustments
                0.3 * predicted_motor_positions                    # Model prediction component
            )
        except Exception as e:
            print(f"Error predicting motor positions: {e}")
            # Fallback to just using adjustments if model prediction fails
            new_motor_positions = current_motor_positions + motor_adjustments
        
        # Add small exploration term that decreases over time
        exploration_scale = 0.03 * (1.0 - min(0.9, i / max_iterations))
        exploration = np.random.normal(0, exploration_scale, size=len(new_motor_positions))
        new_motor_positions += exploration
        
        # Ensure values are within valid range
        new_motor_positions = np.clip(new_motor_positions, -1.0, 1.0)
        
        # Apply the adjusted motor positions
        motor_controller.adjust_motors(new_motor_positions)
            
        # Small delay to allow motors to move
        await asyncio.sleep(0.1)
    
    print(f"Optimization complete. Best MSE: {best_mse:.6f}, Total iterations: {i+1}")
    return human_blendshapes, best_motor_positions

# Simple PID Controller implementation
class PIDController:
    def __init__(self, kp=0.2, ki=0.05, kd=0.1, setpoint=0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        
    def update(self, measurement, dt=0.1):
        # Calculate error
        error = self.setpoint - measurement
        
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
        
        # Limit output to valid range
        output = np.clip(output, -0.1, 0.1)  # Limit size of adjustments for stability
        
        return output

# Mimicry loop - converted to async and updated to use blendshapes
async def mimicry_loop(human_cam_id, robot_cam_id, motor_controller, model, dataset, training_control):
    # Open cameras first to get dimensions
    human_cap = cv2.VideoCapture(human_cam_id)
    robot_cap = cv2.VideoCapture(robot_cam_id)

    if not human_cap.isOpened() or not robot_cap.isOpened():
        print("Error: Could not open one or both cameras.")
        return
    
    # Get image dimensions
    ret_human, test_human_frame = human_cap.read()
    ret_robot, test_robot_frame = robot_cap.read()
    
    if not ret_human or not ret_robot:
        print("Error: Could not read test frames from cameras.")
        return
    
    # Download the MediaPipe Face Landmarker model if not available
    try:
        # Attempt to download the model if not present
        model_path = "face_landmarker.task"
        import urllib.request
        import os
        
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the Face Landmarker model manually from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    
    # Create MediaPipe Face Landmarker with blendshapes enabled
    try:
        face_landmarker = create_face_landmarker()
        print("MediaPipe Face Landmarker initialized with blendshapes")
    except Exception as e:
        print(f"Error creating Face Landmarker: {e}")
        return
    
    # Determine the device being used
    device = next(model.parameters()).device
    
    # Store first received blendshape dimension to check consistency
    blendshape_dim = None

    # Status indicator for display
    status_text = "Training: OFF (Press 'T' to toggle)"
    collect_data = False
    # Flag to toggle face mesh rendering
    show_face_mesh = True

    # For drawing face mesh on the display (still useful for visualization)
    human_face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5,
        refine_landmarks=True
    )

    while True:
        ret_human, human_frame = human_cap.read()
        if not ret_human:
            print("Error: Could not read frame from human camera.")
            break

        # Process human frame with MediaPipe Face Landmarker for blendshapes
        human_rgb_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=human_rgb_frame)
        human_results = face_landmarker.detect(mp_image)
        
        # Extract blendshapes from results
        human_blendshapes = extract_blendshapes(human_results)
        
        # Also process with regular face mesh for visualization
        human_mesh_results = human_face_mesh.process(human_rgb_frame)
        
        # Draw facial landmarks on human frame for visualization if enabled
        if show_face_mesh and human_mesh_results.multi_face_landmarks:
            for face_landmarks in human_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=human_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # Check if we have human blendshapes to work with
        if human_blendshapes is not None:
            # Store blendshape dimension if this is the first time
            if blendshape_dim is None:
                blendshape_dim = len(human_blendshapes)
                print(f"Detected blendshape dimension: {blendshape_dim}")
                
                # If the model input dimension doesn't match, recreate it
                if model.input_dim != blendshape_dim:
                    print(f"Recreating model with input dim {blendshape_dim} instead of {model.input_dim}")
                    model = LandmarkToMotorModel(blendshape_dim, model.fc[-1].out_features).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Use model to directly predict motor positions for inference
            human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
            predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
            motor_controller.adjust_motors(predicted_motor_positions)
            
            # If data collection is enabled, optimize and collect data points
            if collect_data:
                # Use the optimization function to iteratively adjust motors for this frame
                print("Starting motor position optimization for current frame...")
                optimized_blendshapes, best_motor_positions = await optimize_motor_positions(
                    human_blendshapes=human_blendshapes,
                    model=model,
                    motor_controller=motor_controller,
                    face_landmarker=face_landmarker,
                    robot_cap=robot_cap,
                    device=device,
                    max_iterations=50,
                    patience=10,
                    improvement_threshold=0.0001
                )
                
                # If optimization was successful, add the best data point to the dataset
                if optimized_blendshapes is not None:
                    print("Optimization complete, saving optimized data point")
                    dataset.add_data(optimized_blendshapes, best_motor_positions)
                    
                    # Set motors to best position found 
                    motor_controller.adjust_motors(best_motor_positions)

        # Get a final robot frame to display with the optimized positions
        ret_robot, robot_frame = robot_cap.read()
        if ret_robot:
            robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
            
            # Process robot frame with MediaPipe Face Landmarker
            mp_robot_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=robot_rgb_frame)
            robot_results = face_landmarker.detect(mp_robot_image)
            
            # Process with regular face mesh for visualization
            robot_mesh_results = human_face_mesh.process(robot_rgb_frame)
            
            # Draw facial landmarks on robot frame if enabled
            if show_face_mesh and robot_mesh_results.multi_face_landmarks:
                for face_landmarks in robot_mesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=robot_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            # Display blendshape info if available
            robot_blendshapes = extract_blendshapes(robot_results)
            if robot_blendshapes is not None and human_blendshapes is not None:
                # Calculate MSE between human and robot blendshapes
                current_mse = calculate_blendshape_mse(human_blendshapes, robot_blendshapes)
                cv2.putText(robot_frame, f"Blendshape MSE: {current_mse:.4f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the robot frame with status text
            cv2.putText(robot_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Display face mesh status
            mesh_status = "Face Mesh: ON (Press 'M' to toggle)" if show_face_mesh else "Face Mesh: OFF (Press 'M' to toggle)"
            cv2.putText(robot_frame, mesh_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Robot Camera", robot_frame)

        # Display the human frame
        cv2.putText(human_frame, f"Dataset Size: {len(dataset)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Display face mesh status
        mesh_status = "Face Mesh: ON (Press 'M' to toggle)" if show_face_mesh else "Face Mesh: OFF (Press 'M' to toggle)"
        cv2.putText(human_frame, mesh_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display blendshape info if available
        if human_blendshapes is not None:
            # Show top 3 active blendshapes
            top_indices = np.argsort(human_blendshapes)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                cv2.putText(human_frame, f"Blend[{idx}]: {human_blendshapes[idx]:.2f}", 
                           (10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Human Camera", human_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
            
        # 't' to toggle training on/off
        elif key == ord('t'):
            training_control['enabled'] = not training_control['enabled']
            if training_control['enabled']:
                status_text = "Training: ON (Press 'T' to toggle)"
                print("Training enabled")
            else:
                status_text = "Training: OFF (Press 'T' to toggle)"
                print("Training disabled")
                
        # 'd' to toggle data collection on/off
        elif key == ord('d'):
            collect_data = not collect_data
            if collect_data:
                print("Data collection enabled")
            else:
                print("Data collection disabled")
        
        # 'm' to toggle face mesh rendering on/off
        elif key == ord('m'):
            show_face_mesh = not show_face_mesh
            if show_face_mesh:
                print("Face mesh rendering enabled")
            else:
                print("Face mesh rendering disabled")
                
        # 's' to save the model
        elif key == ord('s'):
            model_path = f"facial_mimicry_model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.input_dim,
                'output_dim': model.fc[-1].out_features
            }, model_path)
            print(f"Model saved to {model_path}")
            
        # Add a small delay to allow other coroutines to run
        await asyncio.sleep(0.01)

    # Clean up resources
    human_face_mesh.close()
    human_cap.release()
    robot_cap.release()
    cv2.destroyAllWindows()

# Training loop - converted to async
async def training_loop(model, dataset, optimizer, criterion, training_control):
    """
    Asynchronous training loop for the facial mimicry model.
    
    Trains the neural network model on collected data points when training is enabled.
    Uses the human facial blendshapes to predict motor positions.
    
    Args:
        model: The neural network model to train
        dataset: The RealTimeDataset containing data points
        optimizer: The optimizer for training
        criterion: The loss function
        training_control: Dictionary controlling training state
    """
    device = next(model.parameters()).device
    
    while True:
        if training_control['enabled'] and len(dataset) > 0:
            try:
                # Use DataLoader to fetch batches
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                for human_blendshapes, motor_positions in dataloader:
                    # Move tensors to the correct device
                    human_blendshapes = human_blendshapes.to(device)
                    motor_positions = motor_positions.to(device)
                    
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(human_blendshapes)

                    # Compute loss
                    loss = criterion(predictions, motor_positions)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    print(f"Training Loss: {loss.item()}")
            except Exception as e:
                print(f"Error during training: {e}")

        # Sleep for a short time to avoid overloading the CPU
        await asyncio.sleep(0.1)

# Motor Controller using Maestro servo controller
class MotorController:
    def __init__(self, port="COM6", baudrate=9600, yaml_file="motor_ranges.yaml"):
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

# Setup the Face Landmarker with blendshapes support
def create_face_landmarker(model_path="face_landmarker.task"):
    """
    Creates a MediaPipe Face Landmarker with blendshapes enabled
    
    Args:
        model_path: Path to the face landmarker task file
        
    Returns:
        A configured FaceLandmarker instance
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,  # Enable blendshapes
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# Extract blendshapes from the FaceLandmarkerResult
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

# Calculate the mean squared error between two sets of blendshapes
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

# Main function
async def main():
    # Define YAML file path for motor calibration
    yaml_file = "motor_ranges.yaml"
    
    # Initialize motor controller with motor ranges from YAML
    motor_controller = MotorController(
        port="COM6",  # Update with your actual port
        yaml_file=yaml_file
    )
    
    # Get number of motors from the controller (automatically determined from YAML)
    output_dim = motor_controller.get_num_motors()
    print(f"Using {output_dim} motors from channels: {motor_controller.get_active_channels()}")
    
    # Initialize with a default blendshape dimension
    # MediaPipe FaceLandmarker typically provides 52 blendshapes
    # The exact dimension will be adjusted dynamically once we detect the first face
    input_dim = 52  # Default blendshape dimension from MediaPipe

    # Create dataset and model
    dataset = RealTimeDataset()
    model = LandmarkToMotorModel(input_dim, output_dim)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training control dictionary
    training_control = {'enabled': False}

    # Start both loops as tasks
    mimicry_task = asyncio.create_task(mimicry_loop(0, 5, motor_controller, model, dataset, training_control))
    training_task = asyncio.create_task(training_loop(model, dataset, optimizer, criterion, training_control))
    
    # Wait for both tasks to complete
    await asyncio.gather(mimicry_task, training_task)

if __name__ == "__main__":
    asyncio.run(main())