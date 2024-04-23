import cv2 as cv
import mediapipe as mp
import numpy as np
import hand_track
from hand_track import landmark_detection_from_img

IMG_SIZE = 300
folder_path = "test/"
counter = 5

frame = cv.VideoCapture(0)

def resize_cropped_img (cropped_img):
    height, width, _ = cropped_img.shape
    ratio = height / width  # avoid divided by zero
    if (ratio > 1):
        new_height = IMG_SIZE
        new_width = int(IMG_SIZE / ratio)
    else:
        new_height = int(IMG_SIZE * ratio)
        new_width = IMG_SIZE
    # print(new_width, new_height)
    resized_img = cv.resize(cropped_img, (new_width, new_height))
    white_background = np.ones ((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    if (ratio > 1):
        margin = int((IMG_SIZE - new_width) / 2)
        white_background[:, margin : new_width + margin, :] = resized_img
    else:
        margin = int((IMG_SIZE - new_height) / 2)
        white_background[margin : new_height + margin, :, :] = resized_img
    return white_background

def get_landmarked_img (BGR_img):
    detection_res = landmark_detection_from_img(BGR_img)
    detected, annotated_img, bb = hand_track.draw_landmarks_on_image(BGR_img, detection_res)
    return detected, detection_res, annotated_img, bb

def get_cropped_img (detection_res, annotated_img):
    if (hand_track.hand_detected(detection_res)):
        cropped_img = hand_track.extract_bb (annotated_img, detection_res)
        # cv.imshow("Cropped image", cropped_img)
        _, width, _ = cropped_img.shape
        if (width == 0): return np.zeros(1)
        cropped_img_resized = resize_cropped_img (cropped_img)
        return cropped_img_resized
    return np.zeros(1)

if (__name__ == "__main__"):
    while True:
        success, BGR_img = frame.read()   
        detection_res = landmark_detection_from_img(BGR_img)
        detected, annotated_img, bb = hand_track.draw_landmarks_on_image(BGR_img, detection_res)
        cv.imshow("Image from video", annotated_img)

        if (hand_track.hand_detected(detection_res)):
            cropped_img = hand_track.extract_bb (annotated_img, detection_res)
            # cv.imshow("Cropped image", cropped_img)
            _, width, _ = cropped_img.shape
            if (width == 0): continue
            cropped_img_resized = resize_cropped_img (cropped_img)
            cv.imshow("Cropped image - resized", cropped_img_resized)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # key = cv.waitKey(1)
        # if (key == ord("s")):
        #     cv.imwrite(f"{folder_path}/img_{counter}.png", cropped_img_resized)
        #     print(counter)
        #     counter += 1