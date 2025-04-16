"""
Utility functions for working with facial blendshapes
"""
import asyncio
import cv2
import mediapipe as mp
import numpy as np

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
    await asyncio.sleep(0.1)
    
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
        await asyncio.sleep(0.3)
        
    print(f"Captured {len(all_blendshapes)} valid samples")
    # If we have at least one valid sample, compute the average
    if all_blendshapes:
        avg_blendshapes = np.mean(all_blendshapes, axis=0)
        return avg_blendshapes
    
    return None