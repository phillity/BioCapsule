import os
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import walk, progress_bar
from face_models.face_model import ArcFaceModel, FaceNetModel


np.random.seed(42)


class FaceNet:
    def __init__(self, gpu=-1):
        self.__model = FaceNetModel(gpu)

    def preprocess(self, image):
        return self.__model.get_input(image)

    def extract(self, image, align=True):
        if align:
            image = self.preprocess(image)

        if image.shape != (160, 160, 3):
            image = cv2.resize(image, (160, 160))

        return self.__model.get_feature(image)


class ArcFace:
    def __init__(self, gpu=-1):
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


def extract_dataset(dataset, extractor="arcface", gpu=-1):
    if extractor == "arcface":
        face = ArcFace(gpu)
    else:
        face = FaceNet(gpu)

    dataset_path = os.path.join(os.path.abspath(""), "images", dataset)

    file_cnt = len(walk(dataset_path))
    features = np.zeros((file_cnt, 513))
    features_flip = np.zeros((file_cnt, 513))

    image_cnt = 0
    subjects = os.listdir(dataset_path)
    subjects = [x for _, x in sorted(
        zip([subject.lower() for subject in subjects], subjects))]
    for subject_id, subject in enumerate(subjects):
        progress_bar(dataset + " " + extractor,
                     float(image_cnt + 1) / file_cnt)

        for image in os.listdir(os.path.join(dataset_path, subject)):
            image = cv2.imread(os.path.join(dataset_path, subject, image))

            feature = face.extract(image)
            features[image_cnt, :] = np.append(feature, subject_id + 1)

            feature_flip = face.extract(cv2.flip(image, 1))
            features_flip[image_cnt, :] = np.append(
                feature_flip, subject_id + 1)

            image_cnt += 1

    return features, features_flip


if __name__ == "__main__":
    '''
    facenet = FaceNet()

    img_1 = cv2.imread(os.path.join(
        os.path.abspath(""), "src", "face_models", "tom1.jpg"))
    img_2 = cv2.imread(os.path.join(
        os.path.abspath(""), "src", "face_models", "adrien.jpg"))

    feat_1 = facenet.extract(img_1)
    feat_2 = facenet.extract(img_2)
    print(np.sum(np.square(feat_1 - feat_2)))
    print(np.dot(feat_1, feat_2.T))

    cv2.imshow("before", img_1)
    img = facenet.preprocess(img_1)
    cv2.imshow("after", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arcface = ArcFace()

    feat_1 = arcface.extract(img_1)
    feat_2 = arcface.extract(img_2)
    print(np.sum(np.square(feat_1 - feat_2)))
    print(np.dot(feat_1, feat_2.T))

    cv2.imshow("before", img_2)
    img = arcface.preprocess(img_2)
    cv2.imshow("after", cv2.cvtColor(
        np.rollaxis(img, 0, 3), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in feature extraction")
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1,
                        help="gpu to use in feature extraction")
    args = vars(parser.parse_args())

    features, features_flip = extract_dataset(
        args["dataset"], args["method"], args["gpu"])

    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "data", "{}_{}_feat.npz".format(args["dataset"], args["method"])), features)
    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "data", "{}_{}_feat_flip.npz".format(args["dataset"], args["method"])), features_flip)
