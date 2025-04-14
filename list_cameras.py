import cv2

def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        cameras.append(f"Camera {index}")
        cap.release()
        index += 1
    return cameras

def show_cameras():
    """Open a window for each detected camera."""
    index = 0
    cameras = []
    caps = []
    
    # Open all available cameras
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        cameras.append(f"Camera {index}")
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

if __name__ == "__main__":
    cameras = list_cameras()
    if cameras:
        print("Available cameras:")
        for cam in cameras:
            print(cam)
        print("\nOpening camera windows...")
        show_cameras()
    else:
        print("No cameras found.")