import cv2 as cv
import mediapipe as mp
import numpy as np
import handTracking
from handTracking import landmark_detection_from_img

IMG_SIZE = 300
folder_path = "data/A"
counter = 0

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
    print(new_width, new_height)
    resized_img = cv.resize(cropped_img, (new_width, new_height))
    white_background = np.ones ((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    if (ratio > 1):
        margin = int((IMG_SIZE - new_width) / 2)
        white_background[:, margin : new_width + margin, :] = resized_img
    else:
        margin = int((IMG_SIZE - new_height) / 2)
        white_background[margin : new_height + margin, :, :] = resized_img
    return white_background

while True:
    success, BGR_img = frame.read()   
    detection_res = landmark_detection_from_img(BGR_img)
    annotated_img = handTracking.draw_landmarks_on_image(BGR_img, detection_res)
    cv.imshow("Image from video", annotated_img)

    if (handTracking.hand_detected(detection_res)):
        cropped_img = handTracking.extract_bb (annotated_img, detection_res)
        # cv.imshow("Cropped image", cropped_img)
        _, width, _ = cropped_img.shape
        if (width == 0): continue
        cropped_img_resized = resize_cropped_img (cropped_img)
        cv.imshow("Cropped image - resized", cropped_img_resized)

    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
    key = cv.waitKey(1)
    if (key == ord("s")):
        cv.imwrite(f"{folder_path}/img_{counter}.png", cropped_img_resized)
        print(counter)
        counter += 1