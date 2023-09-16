import cv2
import mediapipe as mp

# Initialize Mediapipe Hand Landmark model
import mediapipe as mp

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures != []:
        print(result.gestures[0][0].category_name)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

def main():

    timestamp = 0

    # Initialize OpenCV webcamq
    cap = cv2.VideoCapture(0)

    with GestureRecognizer.create_from_options(options) as recognizer:

        while cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert the frame to RGB (Mediapipe uses RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe Hand Landmarks
            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = recognizer.recognize_async(mp_image, timestamp)
            
            if result:
                print("J")
                print(type(result))

            # Display the frame with landmarks
            cv2.imshow('Hand Landmarks', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()