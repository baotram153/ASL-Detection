from data_preprocess import DataPreprocessor
import cv2 as cv
import numpy as np

class Trainer:
    def __init__(self, preprocessor: DataPreprocessor, model, class_names, output_visualizer):
        self.preprocessor = preprocessor
        self.model = model
        self.class_names = class_names
        self.output_visualizer = output_visualizer
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
    def preprocess_and_predict(self, BGR_img):
        detected, detection_res, annotated_img, bb = self.preprocessor.get_landmarked_img(BGR_img)
        class_name = None
        confidence_score = None
        if detected:
            cropped_img = self.preprocessor.get_cropped_img (detection_res, annotated_img)
            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            # image = ImageOps.fit(cropped_img, size, Image.Resampling.LANCZOS)
            image = cv.resize(cropped_img, size)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # load the image into the array
            self.data[0] = normalized_image_array

            # get prediction from model
            prediction = self.model.predict(self.data)
            index = np.argmax(prediction)
            class_name = self.class_names[index][:-1]
            confidence_score = prediction[0][index]
        return detected, class_name, confidence_score
    