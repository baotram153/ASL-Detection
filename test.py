from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import cv2 as cv
from data_preprocess import DataPreprocessor
from output_visualizer import OutputVisualizer

# specify path to model
model_dir = "model"
model_file = "keras_model.h5"
label_file = "labels.txt"

model_path = os.path.join(model_dir, model_file)
label_path = os.path.join(model_dir, label_file)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
class_names = open(label_path, "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

data_preprocessor = DataPreprocessor()
visualizer = OutputVisualizer()

frame = cv.VideoCapture(0)


if (__name__ == "__main__"):
    while True:
        success, BGR_img = frame.read()
        detected, detection_res, annotated_img, bb = data_preprocessor.get_landmarked_img(BGR_img)
        labeled_img = annotated_img
        if detected:
            cropped_img = data_preprocessor.get_cropped_img (detection_res, annotated_img)
            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            # image = ImageOps.fit(cropped_img, size, Image.Resampling.LANCZOS)
            image = cv.resize(cropped_img, size)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # load the image into the array
            data[0] = normalized_image_array

            # get prediction from model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index][:-1]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            # print("Class:", class_name[2:], end="")
            # print("Confidence Score:", confidence_score)
        
            labeled_img = visualizer.label(class_name[2:], confidence_score, annotated_img, detected, bb)
        cv.imshow("Labeled image", labeled_img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
