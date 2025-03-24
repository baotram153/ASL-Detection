import cv2 as cv
import mediapipe as mp
import numpy as np

from hand_tracker import HandLandmarkDetector, HandLandmarkDrawer

frame = cv.VideoCapture(0)

class DataPreprocessor:
    def __init__(
        self, 
        img_size = 300
    ):
        self.hand_landmark_detector = HandLandmarkDetector()
        self.hand_landark_drawer = HandLandmarkDrawer()
        self.IMG_SIZE = img_size
        self.folder_path = "test/"
        
    def pad_cropped_img (self, cropped_img):
        '''
        Pad the cropped image to make it square
        input: BGR image
        output: BGR image of size (300, 300, 3)
        '''
        height, width, _ = cropped_img.shape
        ratio = height / width  # avoid divided by zero
        if (ratio > 1):
            new_height = self.IMG_SIZE
            new_width = int(self.IMG_SIZE / ratio)
        else:
            new_height = int(self.IMG_SIZE * ratio)
            new_width = self.IMG_SIZE
        
        resized_img = cv.resize(cropped_img, (new_width, new_height))
        white_background = np.ones ((self.IMG_SIZE, self.IMG_SIZE, 3), np.uint8) * 255

        if (ratio > 1):
            margin = int((self.IMG_SIZE - new_width) / 2)
            white_background[:, margin : new_width + margin, :] = resized_img
        else:
            margin = int((self.IMG_SIZE - new_height) / 2)
            white_background[margin : new_height + margin, :, :] = resized_img
        return white_background

    def get_landmarked_img (self, BGR_img):
        '''
        input: BGR image
        return:
            detected: bool
            detection_res: mediapipe object
            annotated_img: BGR image (no annotation if no hand detected)
            bb: bounding box
        '''
        detection_res = self.hand_landmark_detector.get_detection(BGR_img)
        if not (self.hand_landmark_detector.hand_detected(detection_res)): 
            return False, None, BGR_img, {}
        hand_landmarks, handedness = self.hand_landmark_detector.get_landmarks_handedness(detection_res)
        detected, annotated_img, bb = self.hand_landark_drawer.draw_landmarks(BGR_img, hand_landmarks, handedness)
        return detected, detection_res, annotated_img, bb

    def get_cropped_img (self, detection_res, annotated_img):
        '''
        Get hand-focus cropped image from the annotated image, padding with predefined size (300) if hand is detected, else return None
        input: mediapipe object (Handedness, Landmarks and WorldLandmarks), BGR image
        output: BGR image of size (300, 300, 3) or None
        '''
        if (self.hand_landmark_detector.hand_detected(detection_res)):
            hand_landmarks, _ = self.hand_landmark_detector.get_landmarks_handedness(detection_res)
            cropped_img = self.hand_landark_drawer.extract_bb (annotated_img, hand_landmarks)
            _, width, _ = cropped_img.shape
            if (width == 0): 
                return None     # avoid divided by 0 when passing into pad_cropped_image
            cropped_img_padded = self.pad_cropped_img (cropped_img)
            return cropped_img_padded
        return None