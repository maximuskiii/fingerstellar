import cv2

# Initialize Mediapipe Hand Landmark model
import mediapipe as mp
import mediapipe as mp
import mediapipe as mp
import mediapipe as mp

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
hand_model_path = '/Users/mahathimanda/fingerstellar/hand_landmarker.task'
base_options = BaseOptions(model_asset_path=model_path)

gestureDict = {
    "None": "",
    "Closed_Fist": "Stop",
    "Open_Palm": "Go",
    "Pointing_Up": "Speed Up",
    "Thumb_Down": "Not Ok",
    "Thumb_Up": "Ok",
    "Victory": "Mission Complete",
    "ILoveYou": "I love you!",
}

handDict = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_finger_mcp': 5,
    'index_finger_pip': 6,
    'index_finger_dip': 7,
    'index_finger_tip': 8,
    'middle_finger_mcp': 9,
    'middle_finger_pip': 10,
    'middle_finger_dip': 11,
    'middle_finger_tip': 12,
    'ring_finger_mcp': 13,
    'ring_finger_pip': 14,
    'ring_finger_dip': 15,
    'ring_finger_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}

# Create a gesture recognizer instance with the live stream mode:
def gest_print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # pass
    if result.gestures != []:
        gesture = result.gestures[0][0].category_name
        print(gestureDict[gesture])

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gest_print_result)

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    result_callback=print_result)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def main():
    timestamp = 0

    # Initialize OpenCV webcamq
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(hand_options) as handlandmarker:
        with GestureRecognizer.create_from_options(gesture_options) as recognizer:
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
                handlandmarker.detect_async(mp_image, timestamp)
                recognizer.recognize_async(mp_image, timestamp)
                hand_result = hand_detector.detect(mp_image)

                # Display the frame with landmarks
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_result)
                cv2.imshow('Gesture Recognizer', annotated_image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")
                    break

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()