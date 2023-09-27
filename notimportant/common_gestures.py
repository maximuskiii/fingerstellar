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

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
pose_model_path = 'pose_landmarker_lite.task'
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

poseDict = {
  'nose': 0,
  'left_eye_inner': 1,
  'left_eye': 2,
  'left_eye_outer': 3,
  'right_eye_inner': 4,
  'right_eye': 5,
  'right_eye_outer': 6,
  'left_ear': 7,
  'right_ear': 8,
  'left_mouth': 9,
  'right_mouth': 10,
  'left_shoulder': 11,
  'right_shoulder': 12,
  'left_elbow': 13,
  'right_elbow': 14,
  'left_wrist': 15,
  'right_wrist': 16,
  'left_pinky': 17,
  'right_pinky': 18,
  'left_index': 19,
  'right_index': 20,
  'left_thumb': 21,
  'right_thumb': 22,
  'left_hip': 23,
  'right_hip': 24,
  'left_knee': 25,
  'right_knee': 26,
  'left_ankle': 27,
  'right_ankle': 28,
  'left_heel': 29,
  'right_heel': 30,
  'left_foot_index': 31,
  'right_foot_index': 32,
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

# helper functions
def is_left(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    
def is_inside_triangle(p, a, b, c):
    d1, d2, d3 = is_left(p, a, b), is_left(p, b, c), is_left(p, c, a)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)

# wrist, pinky, pointer, thumb form a quadrilateral so check if eye point is inside that 
def is_overlapping(A, B, C, D, point): 
    return is_inside_triangle(point, A, B, C) or is_inside_triangle(point, A, C, D) or is_inside_triangle(point, B, C, D)

# Create a pose landmarker instance with the live stream mode:
def pose_print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.pose_landmarks:
        pose = result.pose_landmarks[0]

        right_eye = (pose[poseDict['right_eye']].x, pose[poseDict['right_eye']].y)
        right_eye_inner = (pose[poseDict['right_eye_inner']].x, pose[poseDict['right_eye_inner']].y)

        left_eye = (pose[poseDict['left_eye']].x, pose[poseDict['left_eye']].y)
        left_eye_inner = (pose[poseDict['left_eye_inner']].x, pose[poseDict['left_eye_inner']].y)

        right_mouth = (pose[poseDict['right_mouth']].x, pose[poseDict['right_mouth']].y)
        left_mouth = (pose[poseDict['left_mouth']].x, pose[poseDict['left_mouth']].y)

        right_wrist = (pose[poseDict['right_wrist']].x, pose[poseDict['right_wrist']].y)
        right_pinky = (pose[poseDict['right_pinky']].x, pose[poseDict['right_pinky']].y)
        right_index = (pose[poseDict['right_index']].x, pose[poseDict['right_index']].y)
        right_thumb = (pose[poseDict['right_thumb']].x, pose[poseDict['right_thumb']].y)

        left_wrist = (pose[poseDict['left_wrist']].x, pose[poseDict['left_wrist']].y)
        left_pinky = (pose[poseDict['left_pinky']].x, pose[poseDict['left_pinky']].y)
        left_index = (pose[poseDict['left_index']].x, pose[poseDict['left_index']].y)
        left_thumb = (pose[poseDict['left_thumb']].x, pose[poseDict['left_thumb']].y)

        isRightEyeBlind = (is_overlapping(right_wrist, right_pinky, right_index, right_thumb, right_eye) or
                           is_overlapping(left_wrist, left_pinky, left_index, left_thumb, right_eye) or 
                           is_overlapping(right_wrist, right_pinky, right_index, right_thumb, right_eye_inner) or
                           is_overlapping(left_wrist, left_pinky, left_index, left_thumb, right_eye_inner))
        
        isLeftEyeBlind =  (is_overlapping(right_wrist, right_pinky, right_index, right_thumb, left_eye) or
                           is_overlapping(left_wrist, left_pinky, left_index, left_thumb, left_eye) or
                           is_overlapping(right_wrist, right_pinky, right_index, right_thumb, left_eye_inner) or
                           is_overlapping(left_wrist, left_pinky, left_index, left_thumb, left_eye_inner))
        isBlind = isRightEyeBlind and isLeftEyeBlind
        isRightMouthCovered = (is_overlapping(right_wrist, right_pinky, right_index, right_thumb, right_mouth) or 
                               is_overlapping(left_wrist, left_pinky, left_index, left_thumb, right_mouth))
        isLeftMouthCovered = (is_overlapping(right_wrist, right_pinky, right_index, right_thumb, left_mouth) or 
                              is_overlapping(left_wrist, left_pinky, left_index, left_thumb, left_mouth))
        isMouthCovered = isRightMouthCovered and isLeftMouthCovered
        
        # if eyes are not present / hands on eyes: cannot see
        # if isBlind: print("Fully Blind")
        # elif isRightEyeBlind: print("Right Eye Blind")
        # elif isLeftEyeBlind: print("Left Eye Blind")
        # print(pose[poseDict['right_eye_inner']].presence)

        # if hands on ears : cannot hear

        # if mouth is not present / hands on mouth: cannot breathe
        # if isMouthCovered: print("Cannot Breathe")

        # triangle shape with hands = emergency
        # for num in range(len(result.pose_landmarks[0])):
        #     print(poseList[num], "visibility: ", result.pose_landmarks[0][num].visibility)
    pass

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pose_print_result)

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

def main():
    timestamp = 0

    # Initialize OpenCV webcamq
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(hand_options) as landmarker:
        with GestureRecognizer.create_from_options(gesture_options) as recognizer:
            with PoseLandmarker.create_from_options(pose_options) as landmarker:

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
                    landmarker.detect_async(mp_image, timestamp)
                    recognizer.recognize_async(mp_image, timestamp)

                    # Display the frame with landmarks
                    cv2.imshow('Gesture Recognizer', frame)
                    
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