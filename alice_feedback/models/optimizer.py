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
from ..utils.blendshapes import extract_blendshapes, calculate_blendshape_mse, capture_average_blendshapes

class JacobianOptimizer:
    """
    Jacobian-based motor position optimizer.
    Uses a Jacobian matrix to map changes in motor positions to changes in blendshapes.
    """
    def __init__(self, num_motors, num_blendshapes):
        self.num_motors = num_motors
        self.num_blendshapes = num_blendshapes
        # Initialize Jacobian matrix with zeros
        self.jacobian = np.zeros((num_blendshapes, num_motors))
        self.is_calibrated = False
        # Initialize motor effectiveness (how much each motor affects blendshapes)
        self.motor_effectiveness = np.ones(num_motors)
        # Learning rate for optimization
        self.learning_rate = 0.7
        # Safety clip limit for motor adjustments
        self.max_adjustment = 0.5
        # Stored blendshape samples from calibration
        self.blendshape_samples = {}
        self.motor_positions_samples = {}
    
    def calibrate(self, blendshape_changes, motor_changes):
        """
        Update the Jacobian matrix based on observed changes in blendshapes and motor positions.
        
        Args:
            blendshape_changes: Array of shape (num_samples, num_blendshapes) containing blendshape changes
            motor_changes: Array of shape (num_samples, num_motors) containing motor position changes
        """
        # For each motor, compute the average effect on each blendshape
        for motor_idx in range(self.num_motors):
            valid_samples = []
            
            # Find samples where this motor moved significantly
            for i in range(len(motor_changes)):
                if abs(motor_changes[i, motor_idx]) > 0.001:  # Only consider significant movements
                    # Compute the normalized effect: blendshape_change / motor_change
                    effect = blendshape_changes[i] / motor_changes[i, motor_idx]
                    valid_samples.append(effect)
            
            # If we have valid samples, update this column of the Jacobian
            if valid_samples:
                motor_effect = np.mean(valid_samples, axis=0)
                self.jacobian[:, motor_idx] = motor_effect
                
                # Update motor effectiveness (sum of absolute blendshape changes)
                self.motor_effectiveness[motor_idx] = np.sum(np.abs(motor_effect))
                
        self.is_calibrated = True
        print("Jacobian calibrated.")
        
    def build_jacobian_from_samples(self, blendshape_samples, motor_positions):
        """
        Build Jacobian from discrete motor position samples.
        
        Args:
            blendshape_samples: Dictionary mapping motor position tuples to blendshape arrays
            motor_positions: Dictionary mapping motor position descriptions to position arrays
        """
        # Store samples for potential future use
        self.blendshape_samples = blendshape_samples
        self.motor_positions_samples = motor_positions
        
        # Neutral blendshapes (all motors at 0)
        neutral_blendshapes = blendshape_samples.get('neutral')
        if neutral_blendshapes is None:
            print("Error: No neutral position in samples. Jacobian calibration failed.")
            return
            
        # For each motor, compute effect of moving from min to max
        for motor_idx in range(self.num_motors):
            # Get blendshapes at min and max for this motor
            min_key = f"motor_{motor_idx}_min"
            max_key = f"motor_{motor_idx}_max"
            
            min_blendshapes = blendshape_samples.get(min_key)
            max_blendshapes = blendshape_samples.get(max_key)
            
            if min_blendshapes is None or max_blendshapes is None:
                print(f"Warning: Missing samples for motor {motor_idx}. Using zeros for Jacobian column.")
                continue
            
            # Compute effect of moving from min to max
            blendshape_change = max_blendshapes - min_blendshapes
            motor_change = motor_positions[max_key][motor_idx] - motor_positions[min_key][motor_idx]
            
            if abs(motor_change) > 0.01:  # Avoid division by small values
                # Effect per unit motor change
                motor_effect = blendshape_change / motor_change
                self.jacobian[:, motor_idx] = motor_effect
                
                # Update motor effectiveness
                self.motor_effectiveness[motor_idx] = np.sum(np.abs(motor_effect))
        
        self.is_calibrated = True
        print(self.get_jacobian())
        print("Jacobian built from discrete samples.")
        print("Motor effectiveness:", ", ".join([f"{e:.2f}" for e in self.motor_effectiveness]))
    
    def optimize_motor_positions(self, current_motor_positions, target_blendshapes, current_blendshapes):
        """
        Compute optimal motor position adjustments to move from current_blendshapes toward target_blendshapes
        
        Args:
            current_motor_positions: Current motor positions array
            target_blendshapes: Target blendshape values to achieve
            current_blendshapes: Current blendshape values
            
        Returns:
            New motor positions array
        """
        if not self.is_calibrated:
            print("Warning: Jacobian not calibrated. Cannot optimize motor positions.")
            return current_motor_positions
        
        # Compute blendshape error
        blendshape_error = np.array(target_blendshapes) - np.array(current_blendshapes)
        
        # Compute motor adjustments using pseudoinverse of Jacobian
        # (approximate solution to J * motor_adjustments = blendshape_error)
        jacobian_pinv = np.linalg.pinv(self.jacobian)
        motor_adjustments = jacobian_pinv @ blendshape_error
        
        # Scale adjustments based on learning rate
        motor_adjustments *= self.learning_rate
        
        # Clip motor adjustments to avoid large movements
        motor_adjustments = np.clip(motor_adjustments, -self.max_adjustment, self.max_adjustment)
        
        # Calculate new motor positions
        new_motor_positions = current_motor_positions + motor_adjustments
        
        # Ensure values are within valid range
        new_motor_positions = np.clip(new_motor_positions, -1.0, 1.0)
        
        return new_motor_positions

    async def optimize_iteratively(self, human_blendshapes, model, motor_controller, face_landmarker, robot_cap, device, 
                                max_iterations=10, improvement_threshold=0.0001, patience=3):
        """
        Iteratively adjust motor positions to minimize the difference between human and robot blendshapes.
        
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
        if not self.is_calibrated:
            print("Warning: Jacobian not calibrated. Cannot optimize motor positions.")
            return human_blendshapes, None

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
                await asyncio.sleep(0.1)
                continue
                
            # Calculate current error between human and robot blendshapes
            current_mse = calculate_blendshape_mse(human_blendshapes, robot_blendshapes)
            print(f"Optimization iteration {i+1}, MSE: {current_mse:.6f}")
            
            print(current_mse < best_mse)
            
            # If this is a better result, save it
            if current_mse < best_mse:
                improvement = best_mse - current_mse
                best_mse = current_mse
                best_motor_positions = motor_controller.get_motor_positions().copy()
                patience_counter = 0
                
                if i > 0 and improvement < improvement_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Optimization converged after {i+1} iterations (small improvements)")
                        break
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Optimization stopped after {i+1} iterations (no improvement)")
                    motor_controller.adjust_motors(best_motor_positions)
                    break
            
            # Get optimized motor positions using Jacobian
            current_motor_positions = motor_controller.get_motor_positions().copy()
            new_motor_positions = self.optimize_motor_positions(
                current_motor_positions, 
                human_blendshapes, 
                robot_blendshapes
            )
            
            # Blend with model prediction for stability
            try:
                human_blendshapes_tensor = torch.tensor(human_blendshapes, dtype=torch.float32).to(device).unsqueeze(0)
                predicted_motor_positions = model(human_blendshapes_tensor).detach().cpu().numpy()[0]
                
                # Blend between Jacobian optimization and model prediction
                model_weight = min(0.5, len(model.dataset) / 200) if hasattr(model, "dataset") else 0.3
                new_motor_positions = (
                    (1.0 - model_weight) * new_motor_positions + 
                    model_weight * predicted_motor_positions
                )
            except Exception as e:
                print(f"Error predicting motor positions: {e}")
            
            # Ensure values are within valid range and apply
            new_motor_positions = np.clip(new_motor_positions, -1.0, 1.0)
            motor_controller.adjust_motors(new_motor_positions)
            await asyncio.sleep(0.1)  # Allow motors to move
        
        print(f"Optimization complete. Best MSE: {best_mse:.6f}")
        return human_blendshapes, best_motor_positions

    async def calibrate_with_robot(self, motor_controller, face_landmarker, robot_cap, device, 
                                dataset=None, add_to_dataset=True, samples_per_position=3):
        """
        Calibrate the robot by moving each motor from min to max, measuring blendshapes, building the Jacobian,
        and optionally adding the calibration data to the dataset.
        
        Args:
            motor_controller: Controller to adjust robot motors
            face_landmarker: MediaPipe face landmarker for robot
            robot_cap: OpenCV video capture for robot camera
            device: Torch device (CPU/CUDA)
            dataset: Optional dataset to add calibration data to
            add_to_dataset: Whether to add calibration data to the dataset
            samples_per_position: Number of samples to capture for each position
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        print("Starting calibration (Jacobian + dataset generation)...")
        print(f"Calibrating for {self.num_motors} motors and {self.num_blendshapes} blendshapes...")
        
        blendshape_samples = {}
        motor_positions = {}
        initial_data = []

        # Helper for robust blendshape capture
        async def safe_capture_blendshapes(label):
            for attempt in range(5):
                blendshapes = await capture_average_blendshapes(face_landmarker, robot_cap, samples_per_position)
                if blendshapes is not None:
                    return blendshapes
                print(f"[WARN] Failed to capture blendshapes for {label} (attempt {attempt+1}/5)")
                await asyncio.sleep(0.2)
            print(f"[ERROR] Could not capture blendshapes for {label} after 5 attempts.")
            return None

        # Neutral position
        print("Setting all motors to neutral position...")
        motor_controller.center_all_motors()
        await asyncio.sleep(0.5)
        neutral_blendshapes = await safe_capture_blendshapes('neutral')
        if neutral_blendshapes is not None:
            blendshape_samples['neutral'] = neutral_blendshapes
            motor_positions['neutral'] = motor_controller.get_motor_positions().copy()
            if add_to_dataset and dataset is not None:
                initial_data.append((neutral_blendshapes, motor_positions['neutral'].copy()))
            print(f"Captured neutral position with {len(neutral_blendshapes)} blendshapes")
        else:
            print("[FATAL] Failed to capture neutral position blendshapes. Calibration aborted.")
            return False

        # For each motor, min and max
        for motor_idx in range(self.num_motors):
            # Max
            motor_controller.center_all_motors()
            await asyncio.sleep(0.2)
            max_motors = motor_controller.get_motor_positions().copy()
            max_motors[motor_idx] = 0.8
            motor_controller.adjust_motors(max_motors)
            await asyncio.sleep(0.5)
            max_key = f"motor_{motor_idx}_max"
            max_blendshapes = await safe_capture_blendshapes(max_key)
            if max_blendshapes is not None:
                blendshape_samples[max_key] = max_blendshapes
                motor_positions[max_key] = motor_controller.get_motor_positions().copy()
                if add_to_dataset and dataset is not None:
                    initial_data.append((max_blendshapes, motor_positions[max_key].copy()))
                print(f"Captured max position for motor {motor_idx}")
            else:
                print(f"[WARN] Failed to capture max position blendshapes for motor {motor_idx}")
            
            # Min
            min_motors = motor_controller.get_motor_positions().copy()
            min_motors[motor_idx] = -0.8
            motor_controller.adjust_motors(min_motors)
            await asyncio.sleep(0.5)
            min_key = f"motor_{motor_idx}_min"
            min_blendshapes = await safe_capture_blendshapes(min_key)
            await asyncio.sleep(0.5)
            if min_blendshapes is not None:
                blendshape_samples[min_key] = min_blendshapes
                motor_positions[min_key] = motor_controller.get_motor_positions().copy()
                if add_to_dataset and dataset is not None:
                    initial_data.append((min_blendshapes, motor_positions[min_key].copy()))
                print(f"Captured min position for motor {motor_idx}")
            else:
                print(f"[WARN] Failed to capture min position blendshapes for motor {motor_idx}")

        motor_controller.center_all_motors()

        # Build Jacobian
        if 'neutral' in blendshape_samples and len(blendshape_samples) > 1:
            self.build_jacobian_from_samples(blendshape_samples, motor_positions)
            print("Jacobian calibration complete.")
        else:
            print("[FATAL] Not enough samples to build Jacobian. Calibration failed.")
            return False

        # Optionally add to dataset
        if add_to_dataset and dataset is not None:
            for blendshapes, motors in initial_data:
                dataset.add_data(blendshapes, motors)
            print(f"Added {len(initial_data)} calibration data points to dataset.")

        return True

    def set_learning_rate(self, rate):
        """Set the learning rate for the optimizer"""
        self.learning_rate = rate
        
    def set_max_adjustment(self, max_adj):
        """Set the maximum allowed adjustment per iteration"""
        self.max_adjustment = max_adj
        
    def get_jacobian(self):
        """Return the current Jacobian matrix"""
        return self.jacobian.copy()




