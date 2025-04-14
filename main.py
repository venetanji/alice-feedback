import cv2
import mediapipe as mp
import numpy as np
import asyncio
import queue
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Mediapipe Modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thread-safe queue for sharing data
data_queue = queue.Queue()

# PyTorch Dataset for real-time data
class RealTimeDataset(Dataset):
    def __init__(self):
        self.human_landmarks = []  # Human facial landmarks (input)
        self.motor_positions = []  # Optimized motor positions (output)

    def add_data(self, human_landmarks, motor_positions):
        self.human_landmarks.append(human_landmarks)
        self.motor_positions.append(motor_positions)

    def __len__(self):
        return len(self.human_landmarks)

    def __getitem__(self, idx):
        return torch.tensor(self.human_landmarks[idx], dtype=torch.float32), torch.tensor(self.motor_positions[idx], dtype=torch.float32)

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

# Extract landmarks as a flattened list
def extract_landmarks(results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks
    return None

# Calculate the mean squared error between two sets of landmarks
def calculate_landmark_mse(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return float('inf')
    return np.mean(np.square(np.array(landmarks1) - np.array(landmarks2)))

# Optimize motor positions to minimize landmark difference
async def optimize_motor_positions(human_landmarks, model, motor_controller, robot_face_mesh, robot_cap, device, max_iterations=10, improvement_threshold=0.0001, patience=3):
    """
    Iteratively adjust motor positions to minimize the difference between human and robot landmarks.
    The goal is to find the best motor positions that make the robot's face match the human's expression.
    
    Args:
        human_landmarks: Facial landmarks from the human (input for the model)
        model: Neural network model to predict motor positions
        motor_controller: Controller to adjust robot motors
        robot_face_mesh: MediaPipe face mesh for robot
        robot_cap: OpenCV video capture for robot camera
        device: Torch device (CPU/CUDA)
        max_iterations: Maximum number of optimization iterations
        improvement_threshold: Minimum improvement to continue optimization
        patience: Number of iterations to continue without improvement before stopping
        
    Returns:
        human_landmarks: The human landmarks used as input
        best_motor_positions: Motor positions that achieved the best result
    """
    # Initial landmark difference and motor positions
    best_mse = float('inf')
    best_motor_positions = motor_controller.get_motor_positions().copy()
    patience_counter = 0
    
    # Start with model prediction as initial guess
    try:
        human_landmarks_tensor = torch.tensor(human_landmarks, dtype=torch.float32).to(device).unsqueeze(0)
        predicted_motor_positions = model(human_landmarks_tensor).detach().cpu().numpy()[0]
        motor_controller.adjust_motors(predicted_motor_positions)
        await asyncio.sleep(0.1)  # Allow motors to move
    except Exception as e:
        print(f"Error making initial prediction: {e}")
    
    # Keep track of MSE history to detect oscillations
    mse_history = []
    
    for i in range(max_iterations):
        # Capture current robot frame
        ret_robot, robot_frame = robot_cap.read()
        if not ret_robot:
            print("Error: Could not read frame from robot camera.")
            return human_landmarks, best_motor_positions
            
        # Process robot frame to get landmarks
        robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
        robot_results = robot_face_mesh.process(robot_rgb_frame)
        robot_landmarks = extract_landmarks(robot_results)
        
        if robot_landmarks is None:
            print("Warning: No robot face detected during optimization.")
            await asyncio.sleep(0.1)  # Brief pause before retry
            continue
            
        # Check landmark dimensions
        if len(human_landmarks) != len(robot_landmarks):
            print(f"Warning: Landmark dimension mismatch during optimization.")
            continue
            
        # Calculate current error between human and robot landmarks
        current_mse = calculate_landmark_mse(human_landmarks, robot_landmarks)
        mse_history.append(current_mse)
        
        print(f"Optimization iteration {i+1}, MSE: {current_mse:.6f}")
        
        # If this is a better result, save it
        if current_mse < best_mse:
            improvement = best_mse - current_mse
            best_mse = current_mse
            best_motor_positions = motor_controller.get_motor_positions().copy()
            patience_counter = 0  # Reset patience counter when we see improvement
            
            # If improvement is below threshold, still continue but count as low improvement
            if i > 0 and improvement < improvement_threshold:
                patience_counter += 1
                print(f"Small improvement (below threshold) in iteration {i+1}")
                
                # Only break if we've had multiple iterations with small improvements
                if patience_counter >= patience:
                    print(f"Optimization converged after {i+1} iterations (small improvements)")
                    break
            
            # Calculate the gradient for motor adjustment
            landmarks_diff = np.array(human_landmarks) - np.array(robot_landmarks)
            
            # Adjust motor positions based on landmark differences
            try:
                # Get current model prediction
                current_prediction = model(human_landmarks_tensor).detach().cpu().numpy()[0]
                
                # Gradually decrease adjustment rate as optimization progresses
                base_adjustment_rate = 0.1
                adjustment_rate = base_adjustment_rate * (1.0 - min(0.9, i / max_iterations))
                
                # Calculate motor adjustment based on error
                motor_adjustment = adjustment_rate * np.sum(np.abs(landmarks_diff)) / len(landmarks_diff)
                
                # Apply random exploration with decreasing magnitude over iterations
                exploration_magnitude = 0.1 * (1.0 - min(0.9, i / max_iterations))
                random_exploration = np.random.uniform(-exploration_magnitude, exploration_magnitude, size=len(current_prediction))
                
                # Combine current prediction with adjustments
                new_motor_positions = current_prediction + random_exploration * motor_adjustment
                motor_controller.adjust_motors(new_motor_positions)
            except Exception as e:
                print(f"Error adjusting motor positions: {e}")
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
                
            # Try a more aggressive random exploration if we're not improving
            try:
                # Get current motor positions
                current_positions = motor_controller.get_motor_positions().copy()
                
                # More aggressive random exploration to escape local minimum
                exploration_magnitude = 0.15 * (1.0 + 0.5 * patience_counter / patience)
                random_exploration = np.random.uniform(-exploration_magnitude, exploration_magnitude, size=len(current_positions))
                
                # Apply exploration to best positions found so far, not current positions
                new_motor_positions = best_motor_positions + random_exploration
                motor_controller.adjust_motors(new_motor_positions)
            except Exception as e:
                print(f"Error during aggressive exploration: {e}")
            
        # Small delay to allow motors to move
        await asyncio.sleep(0.1)
    
    print(f"Optimization complete. Best MSE: {best_mse:.6f}, Total iterations: {i+1}")
    return human_landmarks, best_motor_positions

# Mimicry loop - converted to async
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
        
    human_height, human_width = test_human_frame.shape[:2]
    robot_height, robot_width = test_robot_frame.shape[:2]
    
    # Create separate face mesh instances for each camera
    human_face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5,
        refine_landmarks=True
    )
    
    robot_face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5,
        refine_landmarks=True
    )
    
    # Determine the device being used
    device = next(model.parameters()).device
    
    # Store first received landmark dimension to check consistency
    landmark_dim = None

    # Status indicator for display
    status_text = "Training: OFF (Press 'T' to toggle)"
    collect_data = False

    while True:
        ret_human, human_frame = human_cap.read()
        if not ret_human:
            print("Error: Could not read frame from human camera.")
            break

        # Process human frame
        human_rgb_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
        human_results = human_face_mesh.process(human_rgb_frame)
        human_landmarks = extract_landmarks(human_results)
        
        # Draw facial landmarks on human frame
        if human_results.multi_face_landmarks:
            for face_landmarks in human_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=human_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # Check if we have human landmarks to work with
        if human_landmarks is not None:
            # Store landmark dimension if this is the first time
            if landmark_dim is None:
                landmark_dim = len(human_landmarks)
                print(f"Detected landmark dimension: {landmark_dim}")
                
                # If the model input dimension doesn't match, recreate it
                if model.input_dim != landmark_dim:
                    print(f"Recreating model with input dim {landmark_dim} instead of {model.input_dim}")
                    model = LandmarkToMotorModel(landmark_dim, model.fc[-1].out_features).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Use model to directly predict motor positions for inference
            human_landmarks_tensor = torch.tensor(human_landmarks, dtype=torch.float32).to(device).unsqueeze(0)
            predicted_motor_positions = model(human_landmarks_tensor).detach().cpu().numpy()[0]
            motor_controller.adjust_motors(predicted_motor_positions)
            
            # If data collection is enabled, optimize and collect data points
            if collect_data:
                # Use the optimization function to iteratively adjust motors for this frame
                print("Starting motor position optimization for current frame...")
                optimized_landmarks, best_motor_positions = await optimize_motor_positions(
                    human_landmarks=human_landmarks,
                    model=model,
                    motor_controller=motor_controller,
                    robot_face_mesh=robot_face_mesh,
                    robot_cap=robot_cap,
                    device=device,
                    max_iterations=10,
                    improvement_threshold=0.001
                )
                
                # If optimization was successful, add the best data point to the dataset
                if optimized_landmarks is not None:
                    print("Optimization complete, saving optimized data point")
                    dataset.add_data(optimized_landmarks, best_motor_positions)
                    
                    # Set motors to best position found 
                    motor_controller.adjust_motors(best_motor_positions)

        # Get a final robot frame to display with the optimized positions
        ret_robot, robot_frame = robot_cap.read()
        if ret_robot:
            robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
            robot_results = robot_face_mesh.process(robot_rgb_frame)
            
            # Draw facial landmarks on robot frame
            if robot_results.multi_face_landmarks:
                for face_landmarks in robot_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=robot_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            # Display the robot frame with status text
            cv2.putText(robot_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Robot Camera", robot_frame)

        # Display the human frame
        cv2.putText(human_frame, f"Dataset Size: {len(dataset)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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

    # Clean up both face mesh instances
    human_face_mesh.close()
    robot_face_mesh.close()
    human_cap.release()
    robot_cap.release()
    cv2.destroyAllWindows()

# Training loop - converted to async
async def training_loop(model, dataset, optimizer, criterion, training_control):
    device = next(model.parameters()).device
    
    while True:
        if training_control['enabled'] and len(dataset) > 0:
            try:
                # Use DataLoader to fetch batches
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                for human_landmarks, motor_positions in dataloader:
                    # Move tensors to the correct device
                    human_landmarks = human_landmarks.to(device)
                    motor_positions = motor_positions.to(device)
                    
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(human_landmarks)

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

# Dummy Motor Controller (Replace with actual motor control logic)
class MotorController:
    def __init__(self, num_motors):
        self.motor_positions = np.zeros(num_motors)

    def adjust_motors(self, predicted_positions):
        self.motor_positions = np.clip(predicted_positions, -1.0, 1.0)
        #print(f"Updated Motor Positions: {self.motor_positions}")

    def get_motor_positions(self):
        return self.motor_positions

# Main function
async def main():
    # Get initial input_dim for the model
    # We'll use 1434 since that's what the error indicated, but the model will adapt dynamically
    input_dim = 1434  # Updated from 1404 based on the error message
    output_dim = 10   # Number of motors

    # Create dataset and model
    dataset = RealTimeDataset()
    model = LandmarkToMotorModel(input_dim, output_dim)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialize motor controller
    motor_controller = MotorController(num_motors=output_dim)

    # Training control dictionary
    training_control = {'enabled': False}

    # Start both loops as tasks
    mimicry_task = asyncio.create_task(mimicry_loop(0, 5, motor_controller, model, dataset, training_control))
    training_task = asyncio.create_task(training_loop(model, dataset, optimizer, criterion, training_control))
    
    # Wait for both tasks to complete
    await asyncio.gather(mimicry_task, training_task)

if __name__ == "__main__":
    asyncio.run(main())