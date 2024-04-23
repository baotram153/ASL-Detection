from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import cv2 as cv
from data_preprocess import get_cropped_img, get_landmarked_img


def label (class_name, confidence_score, annotated_img, detected, bb):
    # print(bb["top_left"])
    # print(bb["bottom_right"])
    labeled_img = annotated_img
    if (confidence_score > 0.5):
        # print(bb["top_left"])
        # print(bb["bottom_right"])
        # labeled_img = cv.rectangle(labeled_img, (int(bb["top_left"][0]), int(bb["top_left"][1])), (int(bb["bottom_right"][0]), int(bb["bottom_right"][1])), (0, 0, 255), 5)
        # cv.putText(labeled_img, class_name, (int(bb["bottom_left"][0]), int(bb["bottom_left"][1])), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv.putText(labeled_img, class_name, (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return labeled_img

# specify path to model
model_dir = "model"
model_file = "keras_model.h5"
label_file = "labels.txt"

model_path = os.path.join(model_dir, model_file)
label_path = os.path.join(model_dir, label_file)

# Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
class_names = open(label_path, "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# # Replace this with the path to your image
# image = Image.open("<IMAGE_PATH>").convert("RGB")

frame = cv.VideoCapture(0)

while True:
    success, BGR_img = frame.read()
    detected, detection_res, annotated_img, bb = get_landmarked_img (BGR_img)
    labeled_img = annotated_img
    cropped_img = get_cropped_img (detection_res, annotated_img)
    if (cropped_img.any() != 0):
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        # image = ImageOps.fit(cropped_img, size, Image.Resampling.LANCZOS)
        image = cv.resize(cropped_img, size)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][:-1]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        # print("Class:", class_name[2:], end="")
        # print("Confidence Score:", confidence_score)
    
        labeled_img = label(class_name[2:], confidence_score, annotated_img, detected, bb)
    cv.imshow("Labeled image", labeled_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
