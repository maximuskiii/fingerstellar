import cv2

# Initialize Mediapipe Hand Landmark model
import mediapipe as mp
import mediapipe as mp
import mediapipe as mp
import mediapipe as mp
from gtts import gTTS
import os

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import numpy as np


model_path = 'gesture_recognizer.task'
hand_model_path = 'hand_landmarker.task'
language = 'en'


hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options,
                                       num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

gest_base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
gest_options = vision.GestureRecognizerOptions(base_options=gest_base_options)
gest_recognizer = vision.GestureRecognizer.create_from_options(gest_options)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

gestureDict = {
    "None": "",
    "Closed_Fist": "Stop",
    "Open_Palm": "Go",
    "Pointing_Up": "Speed Up",
    "Thumb_Down": "Not Ok",
    "Thumb_Up": "Ok",
    "Victory": "Mission Complete",
    "ILoveYou": "Fatal Injury! Need Help",
}

outputs = ["None"]

def playSound(mytext):
   myobj = gTTS(text=mytext, lang=language, slow=False)
   myobj.save("sound.mp3")
   os.system("afplay sound.mp3") 

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures != []:
        gesture = result.gestures[0][0].category_name
        # print(gestureDict[gesture]) 
        outputs.append(gestureDict[gesture])
        # print(outputs)
        # if (len(outputs)>10):
        #    playSound(outputs[0])
        #    outputs = []


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


# logic stuff - new poses

# helper functions
def is_left(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    
def is_inside_triangle(p, a, b, c):
    d1, d2, d3 = is_left(p, a, b), is_left(p, b, c), is_left(p, c, a)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)

# wrist, pinky, pointer, thumb form a quadrilateral so check if eye point is inside that 
def is_overlapping(A, B, C, D, point): 
    return is_inside_triangle(point, A, B, C) or is_inside_triangle(point, A, C, D) or is_inside_triangle(point, B, C, D)

def handOverEyes(handLandmarks, faceLandmarks):
   # if hand over eyes return Cannot See
   wrist = (handLandmarks[handDict["wrist"]].x, handLandmarks[handDict["wrist"]].y)
   thumb_tip = (handLandmarks[handDict["thumb_tip"]].x, handLandmarks[handDict["thumb_tip"]].y)
   middle_finger_tip = (handLandmarks[handDict["middle_finger_tip"]].x, handLandmarks[handDict["middle_finger_tip"]].y)
   pinky_tip = (handLandmarks[handDict["pinky_tip"]].x, handLandmarks[handDict["pinky_tip"]].y)

   count = 0
   for rightEyePoint in faceDict["rightEyeIris"]:
      if is_overlapping(wrist, thumb_tip, middle_finger_tip, pinky_tip, (faceLandmarks[rightEyePoint].x, faceLandmarks[rightEyePoint].y)):
         count += 1
   for leftEyePoint in faceDict["leftEyeIris"]:
      if is_overlapping(wrist, thumb_tip, middle_finger_tip, pinky_tip, (faceLandmarks[leftEyePoint].x, faceLandmarks[leftEyePoint].y)):
         count += 1
   if count >= (len(faceDict["rightEyeIris"])+len(faceDict["leftEyeIris"]))/2: return "Cannot See"
  

def handOverCheeks():
   # if hand over cheeks return Cannot Hear
   pass

def handOverMouth(handLandmarks, faceLandmarks):
   # if hand over mouth return Cannot Breathe
   wrist = (handLandmarks[handDict["wrist"]].x, handLandmarks[handDict["wrist"]].y)
   thumb_tip = (handLandmarks[handDict["thumb_tip"]].x, handLandmarks[handDict["thumb_tip"]].y)
   middle_finger_tip = (handLandmarks[handDict["middle_finger_tip"]].x, handLandmarks[handDict["middle_finger_tip"]].y)
   pinky_tip = (handLandmarks[handDict["pinky_tip"]].x, handLandmarks[handDict["pinky_tip"]].y)

   count = 0
   for mouthPoint in faceDict["lipsLowerInner"]:
      if is_overlapping(wrist, thumb_tip, middle_finger_tip, pinky_tip, (faceLandmarks[mouthPoint].x, faceLandmarks[mouthPoint].y)):
         count += 1
   for mouthPoint in faceDict["lipsUpperInner"]:
      if is_overlapping(wrist, thumb_tip, middle_finger_tip, pinky_tip, (faceLandmarks[mouthPoint].x, faceLandmarks[mouthPoint].y)):
         count += 1
   if count >= len(faceDict["lipsLowerInner"])+len(faceDict["lipsUpperInner"])/2: return "Cannot Breathe"


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
faceDict = {
  "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
  "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
  "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
  "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
  "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
  "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
  "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

  "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
  "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

  "rightEyeIris": [473, 474, 475, 476, 477],

  "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
  "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
  "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
  "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
  "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
  "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
  "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

  "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
  "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

  "leftEyeIris": [468, 469, 470, 471, 472],

  "midwayBetweenEyes": [168],

  "noseTip": [1],
  "noseBottom": [2],
  "noseRightCorner": [98],
  "noseLeftCorner": [327],

  "rightCheek": [205],
  "leftCheek": [425]
}


# drawing things
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_hand_landmarks_on_image(rgb_image, detection_result):
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
def draw_face_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image
def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

def main():
    timestamp = 0
    timer = 0


    # Initialize OpenCV webcamq
    cap = cv2.VideoCapture(0)
    with GestureRecognizer.create_from_options(options) as recognizer:
      while cap.isOpened():
          # Read a frame from the webcam
          ret, frame = cap.read()
          if not ret:
              continue
          
          # Process the frame with Mediapipe Hand Landmarks
          timestamp += 1
          timer += 1
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
          hand_detection_result = hand_detector.detect(mp_image)
          face_detection_result = face_detector.detect(mp_image)
          recognizer.recognize_async(mp_image, timestamp)

          if hand_detection_result.hand_landmarks and face_detection_result.face_landmarks:
             handLandmarks = hand_detection_result.hand_landmarks[0]
             faceLandmarks = face_detection_result.face_landmarks[0]
             outputs.append(handOverEyes(handLandmarks, faceLandmarks))
             outputs.append(handOverMouth(handLandmarks, faceLandmarks))
          
          print(outputs[-1])
          if len(outputs) > 5 and timer >= 40 and outputs[-1] != "None" and outputs[-1] != "" and outputs[-1]!= None:
             playSound(outputs[-1])
             timer = 0
          # Display the frame with landmarks
          hand_annotated_image = draw_hand_landmarks_on_image(mp_image.numpy_view(), hand_detection_result)
          annotated_image = draw_face_landmarks_on_image(hand_annotated_image, face_detection_result)
          
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