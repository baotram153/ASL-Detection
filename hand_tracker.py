import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class HandLandmarkDetector:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task'),
            num_hands = 1,
            running_mode = vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def get_detection (self, BGR_img):
        '''
        input: BGR image
        output: mediapipe object (Handedness, Landmarks and WorldLandmarks)
        '''
        RGB_img = cv.cvtColor(BGR_img, cv.COLOR_BGR2RGB)
        RGB_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_img)  # create an mp.Img object to detect
        detection_res = self.detector.detect(RGB_img)     # return a mediapipe HandLandmarkerResult object, including: Handedness, Landmarks and WorldLandmarks
        print(f"Detected result from Mediapipe HandLandmarker: {detection_res}")
        return detection_res


    def get_landmarks_handedness (self, detection_res):
        '''
        input: mediapipe object (Handedness, Landmarks and WorldLandmarks)
        output: list of hand_landmarks, list of handedness (left or right)
        '''
        hand_landmarks_list = detection_res.hand_landmarks
        handedness_list = detection_res.handedness

        if (len (hand_landmarks_list) != 0):
            return hand_landmarks_list[0], handedness_list[0]   # assert len () == 1
        else:
            return None, None
        
    def hand_detected (self, detection_res):
        '''
        Return True if hand is detected, False otherwise
        input: mediapipe object (Handedness, Landmarks and WorldLandmarks)
        output: bool
        '''
        hand_landmarks, _ = self.get_landmarks_handedness (detection_res)
        if (hand_landmarks == None): return False
        else: return True
        
class HandLandmarkDrawer:
    def __init__(self,
        img_shape = (512, 512),
        margin = 15,
        font_style = cv.FONT_HERSHEY_DUPLEX,
        font_size = 1,
        font_thickness = 1,
        handedness_text_color = (88, 205, 54), # vibrant green
        offset = 20
    ):
        self.MARGIN = margin  # pixels
        self.FONT_STYLE = font_style
        self.FONT_SIZE = font_size
        self.FONT_THICKNESS = font_thickness
        self.HANDEDNESS_TEXT_COLOR = handedness_text_color # vibrant green
        self.OFFSET = offset
        
        # variables for image shape
        self.IMG_HEIGHT = img_shape[0]
        self.IMG_WIDTH = img_shape[1]
        
        self.detector = HandLandmarkDetector()

    def get_bb_normalized (self, hand_landmarks):
        '''
        input: list of hand_landmarks
        output: bounding box coordinates (normalized to [0, 1] according to the image size)
        '''
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

    def extract_bb (self, BGR_img, hand_landmarks):
        '''
        input: BGR image, list of hand_landmarks
        output: BGR image cropped to the bounding box of the hand
        '''
        bb = self.get_bb_normalized (hand_landmarks)
        x_min, y_min = bb["top_left"]
        x_max, y_max = bb["bottom_right"]
        # origin of the image is at the top left corner
        BGR_img_cropped = BGR_img [int(y_min*self.IMG_HEIGHT) - self.OFFSET : int(y_max*self.IMG_HEIGHT) + self.OFFSET, int(x_min*self.IMG_WIDTH) - self.OFFSET : int(x_max*self.IMG_WIDTH) + self.OFFSET]
        return BGR_img_cropped     

    def draw_landmarks(self, BGR_img, hand_landmarks, handedness):
        bb = self.get_bb_normalized (hand_landmarks)
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
        text_x = int(bb["top_left"][0]*self.IMG_WIDTH)
        text_y = int(bb["top_left"][1]*self.IMG_HEIGHT) - self.MARGIN

        # draw handedness (left or right hand) on the image.
        cv.putText(annotated_img, f"{handedness[0].category_name}",
                    (text_x, text_y), self.FONT_STYLE,
                    self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv.LINE_AA)
        return True, annotated_img, bb