import cv2
import mediapipe as mp
import time
import asyncio

# Mediapipe Modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

async def process_camera(camera_id, window_name):
    # Create a face mesh instance for each camera
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5,
        refine_landmarks=True
    )
    
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
            #print(f"Face detected on {window_name}")
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

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

async def main():
    print("Starting real-time monitoring with two cameras...")

    # Create tasks for two cameras
    human_task = asyncio.create_task(process_camera(0, "Human Camera"))
    robot_task = asyncio.create_task(process_camera(5, "Robot Camera"))

    # Wait for both tasks to complete
    await human_task
    await robot_task

    print("Monitoring ended.")

if __name__ == "__main__":
    asyncio.run(main())