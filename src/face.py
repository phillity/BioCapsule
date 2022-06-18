import os
from argparse import ArgumentParser

import cv2
import numpy as np

from face_models import ArcFaceModel, FaceNetModel
from utils import progress_bar, walk

np.random.seed(42)


class FaceNet:
    """FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1503.03832

    Uses https://github.com/davidsandberg/facenet implementation

    """

    def __init__(self, gpu: int = -1):
        """Initialize FaceNet model object. Provide an int
        corresponding to a GPU id to use for models. If -1 is given
        CPU is used rather than GPU.

        """
        self.__model = FaceNetModel(gpu)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for FaceNet feature extraction
        using a MTCNN preprocessing model. MTCNN detects facial bounding
        boxes and facial landmarks to get face region-of-interest and perform
        facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 160x160x3 with RGB colorspace

        """
        return self.__model.get_input(face_img)

    def extract(self, face_img: np.ndarray, align: bool = True) -> np.ndarray:
        """Perform FaceNet feature extraction on input image. Optionally
        apply preprocessing before extract.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR or RCG colorspace channels
        align: bool = True
            Flag if preprocessing should be applied prior to feature extraction

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        if align:
            face_img = self.preprocess(face_img)

        if face_img.shape != (160, 160, 3):
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (160, 160))

        return self.__model.get_feature(face_img)


class ArcFace:
    def __init__(self, gpu: int = -1):
        self.__model = ArcFaceModel(gpu)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for ArcFace feature extraction
        using a MTCNN preprocessing model. MTCNN detects facial bounding
        boxes and facial landmarks to get face region-of-interest and perform
        facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 3x112x112 with RGB colorspace

        """
        return self.__model.get_input(face_img)

    def extract(self, face_img: np.ndarray, align: bool = True) -> np.ndarray:
        """Perform ArcFace feature extraction on input preprocessed image.

        Parameters
        ----------
        aligned_img: np.ndarray
            Aligned facial image of shape 3x112x112 with RGB colorspace
        align: bool = True
            Flag if preprocessing should be applied prior to feature extraction

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        if align:
            face_img = self.preprocess(face_img)

        if face_img.shape != (3, 112, 112):
            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = np.rollaxis(face_img, 2, 0)

        return self.__model.get_feature(face_img)


def extract_dataset(
    dataset: str, extractor: str = "arcface", gpu: int = -1
) -> np.ndarray:
    """Extract feature vectors of each image within a dataset.
    Return array conatining all extracted features.

    Parameters
    ----------
    dataset: str
        Dataset to extract features from. Examples would be gtdb or lfw
    extractor: str = "arcface"
        Model to use for feature extraction. Currently supported options are
        arcface/facenet
    gpu: int = -1
        GPU id to use for feature extraction and preprocessing models. If -1
        is given, CPU is used rather than GPU

    Returns
    -------
    np.ndarray
        Array of features corresponding each image from a dataset. Subject ids
        are appended to end of feature vectors. Resulting output will be of
        shape (number of dataset images)x513

    """
    if extractor == "arcface":
        face = ArcFace(gpu)
    else:
        face = FaceNet(gpu)

    dataset_path = f"images/{dataset}"

    file_cnt = len(walk(dataset_path))
    features = np.zeros((file_cnt, 513))

    subjects = sorted(
        os.listdir(dataset_path), key=lambda subject: subject.lower()
    )

    image_cnt = 0
    for subject_id, subject in enumerate(subjects):
        progress_bar(f"{dataset} {extractor}", (image_cnt + 1) / file_cnt)

        for image in os.listdir(f"{dataset_path}/{subject}"):
            image = cv2.imread(f"{dataset_path}/{subject}/{image}")

            feature = face.extract(image)
            features[image_cnt, :] = np.append(feature, subject_id + 1)

            image_cnt += 1

    return features


if __name__ == "__main__":
    """
    facenet = FaceNet()

    img_1 = cv2.imread("src/face_models/examples/tom1.jpg"))
    img_2 = cv2.imread("src/face_models/examples/adrien.jpg"))

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
    cv2.imshow(
        "after",
        cv2.cvtColor(np.rollaxis(img, 0, 3), cv2.COLOR_RGB2BGR)
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="dataset to use in feature extraction",
    )
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["arcface", "facenet"],
        help="method to use in feature extraction",
    )
    parser.add_argument(
        "-gpu",
        "--gpu",
        required=False,
        type=int,
        default=-1,
        help="gpu to use in feature extraction",
    )
    args = vars(parser.parse_args())

    features = extract_dataset(args["dataset"], args["method"], args["gpu"])

    np.savez_compressed(
        os.path.join(
            "data/{}_{}_feat.npz".format(args["dataset"], args["method"])
        ),
        features,
    )
