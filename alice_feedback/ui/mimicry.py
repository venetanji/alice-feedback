"""
Camera and UI module for the facial mimicry system
This module handles camera input and user interface for the system
"""
import asyncio
import cv2
import numpy as np
import torch
import time
import mediapipe as mp
from torch.utils.data import DataLoader
import torch.optim as optim

from ..utils.facial_landmarks import create_face_landmarker, create_face_mesh, draw_face_mesh
from ..models.optimizer import (
    extract_blendshapes, 
    calculate_blendshape_mse, 
    JacobianOptimizer, 
)
from ..models.neural_network import LandmarkToMotorModel

async def mimicry_loop(human_cam_id, robot_cam_id, motor_controller, model, dataset, training_control):
    """
    Main mimicry loop that captures human facial expressions and controls robot motors
    
    Args:
        human_cam_id (int): Camera ID for the human-facing camera
        robot_cam_id (int): Camera ID for the robot-facing camera
        motor_controller: Controller for the robot motors
        model: Neural network model for predicting motor positions
        dataset: RealTimeDataset for collecting training data
        training_control (dict): Dictionary to control training settings
    """
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

    # Status indicators for display
    training_status = "Training: OFF (Press 'T' to toggle)"
    record_status = "Record: OFF (Press 'R' to toggle)"
    mimicry_status = "Mimicry: ON (Press 'C' to toggle)"
    
    # Control flags
    collect_data = False  # 'R' toggles this
    mimicry_enabled = True  # 'C' toggles this
    
    # Flag to toggle face mesh rendering
    show_face_mesh = True

    # For drawing face mesh on the display (still useful for visualization)
    human_face_mesh = create_face_mesh()
    robot_face_mesh = create_face_mesh()
    
    jacobian_optimizer = None

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
                draw_face_mesh(human_frame, face_landmarks)

        # Check if we have human blendshapes to work with
        if human_blendshapes is not None:
            # Store blendshape dimension if this is the first time
            if blendshape_dim is None:
                blendshape_dim = len(human_blendshapes)
                print(f"Detected blendshape dimension: {blendshape_dim}")
                
                # If the model input dimension doesn't match, recreate it
                if model.input_dim != blendshape_dim:
                    print(f"Recreating model with input dim {blendshape_dim} instead of {model.input_dim}")
                    output_dim = model.fc[-1].out_features
                    model = LandmarkToMotorModel(blendshape_dim, output_dim).to(device)
                
                # Also update the Jacobian optimizer's blendshape dimension if needed
                print(f"Initializing Jacobian with dimensions: {blendshape_dim}")
                
                # Create a new optimizer with the correct dimension
                jacobian_optimizer = JacobianOptimizer(
                    motor_controller.get_num_motors(), 
                    blendshape_dim
                )
                
                # Recalibrate with correct dimension using the direct method
                success = await jacobian_optimizer.calibrate_with_robot(
                    motor_controller=motor_controller,
                    face_landmarker=face_landmarker,
                    robot_cap=robot_cap,
                    device=device,
                    dataset=dataset,
                    add_to_dataset=True
                )
                
                if not success:
                    print("[FATAL] Jacobian calibration failed. Exiting.")
                    break
            
            
            # If mimicry is enabled, adjust motors based on human blendshapes
            if mimicry_enabled:
                # Capture current robot frame to get current blendshapes
                ret_robot, robot_frame = robot_cap.read()
                if ret_robot:
                    robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
                    mp_robot_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=robot_rgb_frame)
                    robot_results = face_landmarker.detect(mp_robot_image)
                    robot_blendshapes = extract_blendshapes(robot_results)
                    
                    # If we have both human and robot blendshapes, use iterative optimization
                    if robot_blendshapes is not None and jacobian_optimizer and jacobian_optimizer.is_calibrated:
                        _, new_motor_positions = await jacobian_optimizer.optimize_iteratively(
                            human_blendshapes=human_blendshapes,
                            model=model,
                            motor_controller=motor_controller,
                            face_landmarker=face_landmarker,
                            robot_cap=robot_cap,
                            device=device,
                            max_iterations=10,  # Keep iterations low for real-time response
                            improvement_threshold=0.001,
                            patience=2
                        )
                        # Apply the motor positions if optimization was successful
                        if new_motor_positions is not None:
                            motor_controller.adjust_motors(new_motor_positions)
                            await asyncio.sleep(0.1)
                    else:
                        # Fallback to model if we can't use Jacobian
                        human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
                        predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
                        motor_controller.adjust_motors(predicted_motor_positions)
                else:
                    print("Warning: Could not read robot camera frame for blendshape comparison")
                
                if collect_data:
                    current_motor_positions = motor_controller.get_motor_positions().copy()
                    print(f"Adding data point: blendshape dim {len(human_blendshapes)}, motor dim {len(current_motor_positions)}")
                    dataset.add_data(human_blendshapes, current_motor_positions)
            else:
                # Mimicry is disabled, use model to make predictions
                human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
                predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
                motor_controller.adjust_motors(predicted_motor_positions)
            
            # If data collection is enabled, add the current blendshape and motor position to the dataset


        # Get a final robot frame to display with the current positions
        ret_robot, robot_frame = robot_cap.read()
        if ret_robot:
            robot_rgb_frame = cv2.cvtColor(robot_frame, cv2.COLOR_BGR2RGB)
            
            # Process robot frame with MediaPipe Face Landmarker
            mp_robot_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=robot_rgb_frame)
            robot_results = face_landmarker.detect(mp_robot_image)
            
            # Process with regular face mesh for visualization
            robot_mesh_results = robot_face_mesh.process(robot_rgb_frame)
            
            # Draw facial landmarks on robot frame if enabled
            if show_face_mesh and robot_mesh_results.multi_face_landmarks:
                for face_landmarks in robot_mesh_results.multi_face_landmarks:
                    draw_face_mesh(robot_frame, face_landmarks)
            
            # Display blendshape info if available
            robot_blendshapes = extract_blendshapes(robot_results)
            if robot_blendshapes is not None and human_blendshapes is not None:
                # Calculate MSE between human and robot blendshapes
                current_mse = calculate_blendshape_mse(human_blendshapes, robot_blendshapes)
                cv2.putText(robot_frame, f"Blendshape MSE: {current_mse:.4f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the robot frame with status text
            cv2.putText(robot_frame, training_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Display record status
            cv2.putText(robot_frame, record_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Display mimicry status
            cv2.putText(robot_frame, mimicry_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Display face mesh status
            mesh_status = "Face Mesh: ON (Press 'M' to toggle)" if show_face_mesh else "Face Mesh: OFF (Press 'M' to toggle)"
            cv2.putText(robot_frame, mesh_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Robot Camera", robot_frame)

        # Display the human frame
        cv2.putText(human_frame, f"Dataset Size: {len(dataset)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Display face mesh status
        mesh_status = "Face Mesh: ON (Press 'M' to toggle)" if show_face_mesh else "Face Mesh: OFF (Press 'M' to toggle)"
        cv2.putText(human_frame, mesh_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display record status
        cv2.putText(human_frame, record_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display mimicry status
        cv2.putText(human_frame, mimicry_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display blendshape info if available
        if human_blendshapes is not None:
            # Show top 3 active blendshapes
            top_indices = np.argsort(human_blendshapes)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                cv2.putText(human_frame, f"Blend[{idx}]: {human_blendshapes[idx]:.2f}", 
                           (10, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
                training_status = "Training: ON (Press 'T' to toggle)"
                print("Training enabled")
            else:
                training_status = "Training: OFF (Press 'T' to toggle)"
                print("Training disabled")
                
        # 'r' to toggle data recording on/off
        elif key == ord('r'):
            collect_data = not collect_data
            if collect_data:
                record_status = "Record: ON (Press 'R' to toggle)"
                print("Data recording enabled")
            else:
                record_status = "Record: OFF (Press 'R' to toggle)"
                print("Data recording disabled")
                
        # 'c' to toggle mimicry on/off
        elif key == ord('c'):
            mimicry_enabled = not mimicry_enabled
            if mimicry_enabled:
                mimicry_status = "Mimicry: ON (Press 'C' to toggle)"
                print("Mimicry enabled")
            else:
                mimicry_status = "Mimicry: OFF (Press 'C' to toggle)"
                print("Mimicry disabled")
                # Center all motors when turning off mimicry
                motor_controller.center_all_motors()
        
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
            model.save_to_file(model_path)
            print(f"Model saved to {model_path}")
                
        # Add a small delay to allow other coroutines to run
        await asyncio.sleep(0.01)

    # Clean up resources
    human_face_mesh.close()
    robot_face_mesh.close()
    human_cap.release()
    robot_cap.release()
    cv2.destroyAllWindows()


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