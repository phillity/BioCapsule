import os
import cv2
import numpy as np
from argparse import ArgumentParser
from face_utils import ArcFaceUtils, FaceNetUtils


def drawProgressBar(text, percent, barLen=20):
    print(text + " -- [{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100), end="\r")


def walk(path):
    files = []
    dirs = []
    contents = os.listdir(path)
    for content in contents:
        if os.path.isfile(os.path.join(path, content)):
            files += [content]
        else:
            dirs += [content]
    for d in dirs:
        files += walk(os.path.join(path, d))
    return files


def get_features(dataset, extractor="arcface", gpu="-1", save_features=True):
    if extractor == "arcface":
        face_utils = ArcFaceUtils(gpu)
    else:
        face_utils = FaceNetUtils(gpu)
    path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.abspath(os.path.join(path, "..", "images", dataset))
    features = np.zeros((len(walk(dataset_path)), 513))
    features_flip = np.zeros((len(walk(dataset_path)), 513))
    image_cnt = 0
    for subject_cnt, subject in enumerate(os.listdir(dataset_path)):
        drawProgressBar(dataset + " " + extractor, float(subject_cnt + 1) / len(os.listdir(dataset_path)))
        for image in os.listdir(os.path.join(dataset_path, subject)):
            image = cv2.imread(os.path.join(dataset_path, subject, image))
            feature = face_utils.extract(image)
            features[image_cnt, :] = np.append(feature, subject_cnt + 1)
            feature_flip = face_utils.extract(cv2.flip(image, 1))
            features_flip[image_cnt, :] = np.append(
                feature_flip, subject_cnt + 1)
            image_cnt += 1
    print("")

    if save_features:
        np.savez_compressed(os.path.abspath(os.path.join(
            path, "..", extractor + "_data", dataset + "_feat.npz")), features)
        np.savez_compressed(os.path.abspath(os.path.join(
            path, "..", extractor + "_data", dataset + "_feat_flip.npz")), features_flip)

    return features, features_flip


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in feature extraction")
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1,
                        help="gpu to use in feature extraction")
    args = vars(parser.parse_args())
    features, features_flip = get_features(args["dataset"], args["method"], args["gpu"])
