import cv2 as cv

class OutputVisualizer:
    def __init__(self):
        pass

    def label (self, class_name, confidence_score, annotated_img, detected, bb):
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