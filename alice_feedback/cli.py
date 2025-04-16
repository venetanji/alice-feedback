"""
Command Line Interface module for Alice Feedback
This module provides command-line argument parsing and the main entry point
"""
import os
import asyncio
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import mediapipe as mp
from torch.utils.data import DataLoader

from alice_feedback.models.neural_network import RealTimeDataset, LandmarkToMotorModel
from alice_feedback.controllers.motor_controller import MotorController
from alice_feedback.ui.mimicry import mimicry_loop, training_loop
from alice_feedback.utils.facial_landmarks import create_face_landmarker, create_face_mesh, draw_face_mesh
from alice_feedback.models.optimizer import generate_initial_dataset

async def process_camera(camera_id, window_name):
    """
    Process video from a camera with facial landmark detection
    
    Args:
        camera_id (int): Camera device ID
        window_name (str): Name for the display window
    """
    # Create a face mesh instance for each camera
    face_mesh = create_face_mesh()
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from camera {camera_id}")
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Draw facial landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_mesh(frame, face_landmarks)

        # Display the processed frame
        cv2.imshow(window_name, frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Add a small delay to allow other coroutines to run
        await asyncio.sleep(0.01)

    # Clean up
    face_mesh.close()
    cap.release()
    cv2.destroyWindow(window_name)

async def test_cameras_async(args):
    """
    Test multiple cameras simultaneously with facial landmark detection
    
    Args:
        args: Command-line arguments containing camera IDs
    """
    print("Starting real-time monitoring with cameras...")

    # Create tasks for cameras
    tasks = []
    for i, camera_id in enumerate([args.human_camera, args.robot_camera]):
        name = "Human Camera" if i == 0 else "Robot Camera"
        tasks.append(asyncio.create_task(process_camera(camera_id, name)))

    # Wait for all tasks to complete (will run until user presses 'q')
    for task in tasks:
        await task

    print("Camera monitoring ended.")

def list_cameras():
    """
    Find all available camera devices
    
    Returns:
        list: List of available camera IDs
    """
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras

def show_cameras():
    """
    Open a window for each detected camera to visually identify them
    """
    index = 0
    cameras = []
    caps = []
    
    # Open all available cameras
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        cameras.append(index)
        caps.append(cap)
        index += 1
    
    if not cameras:
        print("No cameras found.")
        return
    
    try:
        print(f"Displaying {len(cameras)} camera(s). Press 'q' to exit.")
        while True:
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Camera {i}", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Properly release all resources
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

def list_cameras_cmd(args):
    """
    Command handler for listing available cameras
    """
    cameras = list_cameras()
    if cameras:
        print("Available cameras:")
        for cam_id in cameras:
            print(f"Camera {cam_id}")
        
        if args.show:
            print("\nOpening camera windows...")
            show_cameras()
    else:
        print("No cameras found.")

async def main_async(args):
    """
    Main asynchronous function to run the facial mimicry system
    
    Args:
        args: Parsed command-line arguments
    """
    # Define YAML file path for motor calibration
    yaml_file = args.motor_ranges_file
    
    # Initialize motor controller with motor ranges from YAML
    motor_controller = MotorController(
        port=args.port,
        yaml_file=yaml_file
    )
    
    # Get number of motors from the controller (automatically determined from YAML)
    output_dim = motor_controller.get_num_motors()
    print(f"Using {output_dim} motors from channels: {motor_controller.get_active_channels()}")
    
    # Initialize with a default blendshape dimension
    # MediaPipe FaceLandmarker typically provides 52 blendshapes
    # The exact dimension will be adjusted dynamically once we detect the first face
    input_dim = 52  # Default blendshape dimension from MediaPipe

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Create dataset and model
    dataset = RealTimeDataset()
    
    # Load model if specified, otherwise create a new one
    if args.model_file and os.path.exists(args.model_file):
        print(f"Loading model from {args.model_file}")
        model = LandmarkToMotorModel.load_from_file(args.model_file, device)
        print(f"Loaded model with input dimension {model.input_dim}, output dimension {model.fc[-1].out_features}")
    else:
        print(f"Creating new model with input dimension {input_dim}, output dimension {output_dim}")
        model = LandmarkToMotorModel(input_dim, output_dim)
        model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generate initial dataset if requested
    if args.generate_initial_dataset:
        # Set up robot capture for initial data collection
        robot_cap = cv2.VideoCapture(args.robot_camera)
        if not robot_cap.isOpened():
            print("Error: Could not open robot camera for initial data collection.")
            return
        
        try:
            # Create MediaPipe Face Landmarker with blendshapes enabled
            face_landmarker = create_face_landmarker()
            print("MediaPipe Face Landmarker initialized with blendshapes")
            
            # Generate initial dataset with neutral, min, and max positions for each motor
            print("Generating initial data points for the dataset...")
            initial_data = await generate_initial_dataset(
                motor_controller=motor_controller,
                face_landmarker=face_landmarker,
                robot_cap=robot_cap,
                device=device,
                samples_per_position=3
            )
            
            # Add initial data points to the dataset
            for blendshapes, motor_positions in initial_data:
                dataset.add_data(blendshapes, motor_positions)
            
            print(f"Added {len(initial_data)} initial data points to the dataset")
            
            # Perform initial training to establish a baseline model
            if len(dataset) > 0:
                print("Performing initial training with baseline data...")
                dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
                
                # Do a few epochs of initial training
                for epoch in range(10):
                    total_loss = 0
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
                        
                        total_loss += loss.item()
                    
                    # Print progress
                    avg_loss = total_loss / len(dataloader)
                    print(f"Initial training - Epoch {epoch+1}/10, Loss: {avg_loss:.6f}")
        
        except Exception as e:
            print(f"Error during initial data generation: {e}")
        finally:
            # Release the camera when done
            robot_cap.release()

    # Training control dictionary
    training_control = {'enabled': args.training_enabled}

    # Start both loops as tasks
    mimicry_task = asyncio.create_task(
        mimicry_loop(
            args.human_camera, 
            args.robot_camera, 
            motor_controller, 
            model, 
            dataset, 
            training_control
        )
    )
    training_task = asyncio.create_task(
        training_loop(
            model, 
            dataset, 
            optimizer, 
            criterion, 
            training_control
        )
    )
    
    # Wait for both tasks to complete
    await asyncio.gather(mimicry_task, training_task)

def main():
    """Main entry point for the application"""
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Alice Feedback - Facial Mimicry System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments used by multiple commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--human-camera", type=int, default=0,
                        help="Camera ID for human-facing camera (default: 0)")
    common_parser.add_argument("--robot-camera", type=int, default=1,
                        help="Camera ID for robot-facing camera (default: 1)")
    
    # Parser for the main mimicry command (default)
    mimicry_parser = subparsers.add_parser("mimicry", parents=[common_parser],
                                        help="Run the facial mimicry system")
    # Motor controller settings
    mimicry_parser.add_argument("--port", type=str, default="COM6",
                        help="Serial port for the Maestro controller (default: COM6)")
    mimicry_parser.add_argument("--motor-ranges-file", type=str, default="motor_ranges.yaml",
                        help="Path to YAML file with motor range calibration (default: motor_ranges.yaml)")
    # Model settings
    mimicry_parser.add_argument("--model-file", type=str, default=None,
                        help="Path to saved model file to load (default: None, creates new model)")
    mimicry_parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if CUDA is available")
    # Training settings
    mimicry_parser.add_argument("--training-enabled", action="store_true",
                        help="Enable training mode at startup")
    mimicry_parser.add_argument("--generate-initial-dataset", action="store_true",
                        help="Generate initial dataset with neutral and extremes for each motor")
    
    # Parser for the test_cameras command
    test_parser = subparsers.add_parser("test_cameras", parents=[common_parser],
                                    help="Test cameras with facial landmark detection")
    
    # Parser for the list_cameras command
    list_parser = subparsers.add_parser("list_cameras",
                                    help="List all available cameras")
    list_parser.add_argument("--show", action="store_true",
                        help="Show preview windows for all cameras")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command was specified, default to mimicry
    if not args.command:
        args.command = "mimicry"
    
    # Route to the appropriate command handler
    if args.command == "mimicry":
        # Add default arguments needed for mimicry if they weren't added by the subparser
        if not hasattr(args, 'port'):
            args.port = "COM6"
        if not hasattr(args, 'motor_ranges_file'):
            args.motor_ranges_file = "motor_ranges.yaml"
        if not hasattr(args, 'model_file'):
            args.model_file = None
        if not hasattr(args, 'cpu'):
            args.cpu = False
        if not hasattr(args, 'training_enabled'):
            args.training_enabled = False
        if not hasattr(args, 'generate_initial_dataset'):
            args.generate_initial_dataset = False
        
        # Run the main mimicry system
        asyncio.run(main_async(args))
    elif args.command == "test_cameras":
        # Run the camera test
        asyncio.run(test_cameras_async(args))
    elif args.command == "list_cameras":
        # List available cameras
        list_cameras_cmd(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()