import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def landmark_detection_from_img (BGR_img):
    base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 1)

    detector = vision.HandLandmarker.create_from_options(options)
    RGB_img = cv.cvtColor(BGR_img, cv.COLOR_BGR2RGB)
    RGB_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_img)  # create an mp.Img object to detect
    detection_res = detector.detect(RGB_img)
    return detection_res


def get_landmarks_handedness (detection_res):
    hand_landmarks_list = detection_res.hand_landmarks
    handedness_list = detection_res.handedness

    if (len (hand_landmarks_list) != 0):
        return hand_landmarks_list[0], handedness_list[0]   # assert len () == 1
    else:
        return None, None
        

def get_bb_normalized (hand_landmarks):
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    bb = {}
    bb["top_left"] = (x_min, y_min)
    bb["top_right"] = (x_max, y_min)
    bb["bottom_left"] = (x_min, y_max)
    bb["bottom_right"] = (x_max, y_max)
    return bb

OFFSET = 25

def extract_bb (BGR_img, detection_res):
    hand_landmarks, _ = get_landmarks_handedness (detection_res)
    bb = get_bb_normalized (hand_landmarks)
    height, width, _ = BGR_img.shape
    x_min, y_min = bb["top_left"]
    x_max, y_max = bb["bottom_right"]
    BGR_img_cropped = BGR_img [int(y_min*height) - OFFSET : int(y_max*height) + OFFSET, int(x_min*width) - OFFSET : int(x_max*width) + OFFSET]
    return BGR_img_cropped
    

MARGIN = 15  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def hand_detected (detection_res):
    hand_landmarks, _ = get_landmarks_handedness (detection_res)
    if (hand_landmarks == None): return False
    else: return True

def draw_landmarks_on_image(BGR_img, detection_res):
    if not (hand_detected(detection_res)): return False, BGR_img, {}

    hand_landmarks, handedness = get_landmarks_handedness (detection_res)

    bb = get_bb_normalized (hand_landmarks)
    annotated_img = np.copy(BGR_img)

    # draw hand landmarks
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
    annotated_img,
    hand_landmarks_proto,
    solutions.hands.HAND_CONNECTIONS,
    solutions.drawing_styles.get_default_hand_landmarks_style(),
    solutions.drawing_styles.get_default_hand_connections_style())

    # put handedness on
    height, width, _ = annotated_img.shape
    text_x = int(bb["top_left"][0]*width)
    text_y = int(bb["top_left"][1]*height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_img, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
    return True, annotated_img, bb