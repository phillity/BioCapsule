import os
import cv2
import numpy as np
from face_models.face_model import ArcFaceModel, FaceNetModel


class FaceNetUtils:
    def __init__(self, gpu="-1"):
        self.__model = FaceNetModel(gpu)

    def preprocess(self, image):
        return self.__model.get_input(image)

    def extract(self, image, align=True):
        if align:
            image = self.preprocess(image)
        if image.shape != (160, 160, 3):
            image = cv2.resize(image, (160, 160))
        return self.__model.get_feature(image)


class ArcFaceUtils:
    def __init__(self, gpu="-1"):
        self.__model = ArcFaceModel(gpu)

    def preprocess(self, image):
        return self.__model.get_input(image)

    def extract(self, image, align=True):
        if align:
            image = self.preprocess(image)
        if image.shape != (3, 112, 112):
            image = cv2.resize(image, (112, 112))
            image = np.rollaxis(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 2, 0)
        return self.__model.get_feature(image)


if __name__ == "__main__":
    facenet_utils = FaceNetUtils()
    path = os.path.dirname(os.path.realpath(__file__))

    image = cv2.imread(os.path.join(path, "face_models", "tom1.jpg"))
    f1 = facenet_utils.extract(image)
    image = cv2.imread(os.path.join(path, "face_models", "adrien.jpg"))
    f2 = facenet_utils.extract(image)
    dist = np.sum(np.square(f1 - f2))
    print(dist)
    sim = np.dot(f1, f2.T)
    print(sim)

    cv2.imshow("before", image)
    image = facenet_utils.preprocess(image)
    cv2.imshow("after", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arcface_utils = ArcFaceUtils()
    path = os.path.dirname(os.path.realpath(__file__))

    image = cv2.imread(os.path.join(path, "face_models", "tom1.jpg"))
    f1 = arcface_utils.extract(image)
    image = cv2.imread(os.path.join(path, "face_models", "adrien.jpg"))
    f2 = arcface_utils.extract(image)
    dist = np.sum(np.square(f1 - f2))
    print(dist)
    sim = np.dot(f1, f2.T)
    print(sim)

    cv2.imshow("before", image)
    image = arcface_utils.preprocess(image)
    cv2.imshow("after", cv2.cvtColor(
        np.rollaxis(image, 0, 3), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
