"""
Facial landmarks detection module using MediaPipe.
This module provides functions for detecting and working with facial landmarks.
"""
import mediapipe as mp
import os
import urllib.request

def create_face_landmarker(model_path="face_landmarker.task"):
    """
    Creates a MediaPipe Face Landmarker with blendshapes enabled
    
    Args:
        model_path: Path to the face landmarker task file
        
    Returns:
        A configured FaceLandmarker instance
    """
    # Ensure the model is available
    try:
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the Face Landmarker model manually from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    
    # Initialize the face landmarker
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,  # Enable blendshapes
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def create_face_mesh():
    """
    Creates a MediaPipe FaceMesh for visualization
    
    Returns:
        A configured FaceMesh instance
    """
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5,
        refine_landmarks=True
    )

def draw_face_mesh(image, face_landmarks):
    """
    Draw facial landmarks on an image
    
    Args:
        image: The image to draw on
        face_landmarks: MediaPipe face landmarks
        
    Returns:
        Image with landmarks drawn
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
    )
    
    return image