from __future__ import absolute_import, division, print_function

import os

import cv2
import mxnet as mx
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

from face_models import facenet
from face_models.face_preprocess import preprocess
from face_models.mtcnn_detector import MtcnnDetector

__all__ = ["ArcFaceModel", "FaceNetModel"]


class FaceNetModel:
    """FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1503.03832

    Uses https://github.com/davidsandberg/facenet implementation

    """

    def __init__(self, gpu: int):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.model_path = "src/face_models/model-ir-v1/20180402-114759.pb"

        self.__load_model()

        self.__detector = MtcnnDetector(
            model_dir="src/face_models/model-mtcnn",
            ctx=ctx,
            accurate_landmark=True,
        )

    def __load_model(self):
        """Load pretrained FaceNet model from:
        src/face_models/model-ir-v1/20180402-114759.pb

        Model originally downlaoded from:
        https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view

        FaceNet model with Inception-ResNet-v1 backbone, pretrained on
        VGGFace2, achieves 0.9965 on LFW.

        """
        self.__sess = tf.compat.v1.Session()

        facenet.load_model(self.model_path)

        self.__images_placeholder = (
            tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        )
        self.__embeddings = (
            tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        )
        self.__phase_train_placeholder = (
            tf.compat.v1.get_default_graph().get_tensor_by_name(
                "phase_train:0"
            )
        )

    def get_input(self, face_img: np.ndarray) -> np.ndarray:
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
        # Get face bounding box and landmark detections
        # from MTCNN model
        ret = self.__detector.detect_face(face_img, det_type=0)

        # If no detections were made, return input image
        if ret is None:
            return face_img

        # Otherwise, the MTCNN model detected facial bounding boxes
        # and landmarks
        bbox, points = ret

        # Make sure detections are not empty lists
        if bbox.shape[0] == 0:
            return face_img

        # If multiple faces were detected, we will use the centermost face
        if bbox.shape[0] > 1:
            img_center = np.array(
                [face_img.shape[0] / 2, face_img.shape[1] / 2]
            )
            bbox, points = _get_center_face(bbox, points, img_center)

        # Format bounding box and landmarks arrays
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T

        # Perform facial alignment using MTCNN bounding box and facial
        # landmarks to get 112x112x3 facial image
        aligned_img = preprocess(face_img, bbox, points, image_size="112,112")
        # Convert facial image from BGR to RGB colorspace
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        # Resize facial image to 160x160x3
        aligned_img = cv2.resize(aligned_img, (160, 160))

        return aligned_img

    def get_feature(self, aligned_img: np.ndarray) -> np.ndarray:
        """Perform FaceNet feature extraction on input preprocessed image.

        Parameters
        ----------
        aligned_img: np.ndarray
            Aligned facial image of shape 160x160x3 with RGB colorspace

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        aligned_img = _prewhiten(aligned_img)

        if len(aligned_img.shape) == 3:
            aligned_img = np.expand_dims(aligned_img, axis=0)

        feed_dict = {
            self.__images_placeholder: aligned_img,
            self.__phase_train_placeholder: False,
        }

        embedding = self.__sess.run(self.__embeddings, feed_dict=feed_dict)

        return embedding[0]


class ArcFaceModel:
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698

    Uses https://github.com/deepinsight/insightface implementation

    """

    def __init__(self, gpu: int):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.model_path = "src/face_models/model-r100-ii/model"

        self.__model = self.__load_model(ctx)
        self.__detector = MtcnnDetector(
            model_dir=os.path.join(os.path.dirname(__file__), "model-mtcnn"),
            ctx=ctx,
            accurate_landmark=True,
        )

    def __load_model(self, ctx: int):
        """Load pretrained ArcFace model from:
        src/face_models/model-r100-ii/model-symbol.json
        src/face_models/model-r100-ii/model-0000.params

        Model originally downlaoded from:
        https://drive.google.com/file/d/1Hc5zUfBATaXUgcU2haUNa7dcaZSw95h2/view

        ArcFace model with ResNet100 backbonem pretrained on MS1MV2,
        achieves 0.9977 on LFW.

        """
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            self.model_path, 0
        )
        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]

        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, 112, 112))])
        model.set_params(arg_params, aux_params)

        return model

    def get_input(self, face_img: np.ndarray) -> np.ndarray:
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
        # Get face bounding box and landmark detections
        # from MTCNN model
        ret = self.__detector.detect_face(face_img, det_type=0)

        # If no detections were made, return input image
        if ret is None:
            return face_img

        # Otherwise, the MTCNN model detected facial bounding boxes
        # and landmarks
        bbox, points = ret

        # Make sure detections are not empty lists
        if bbox.shape[0] == 0:
            return face_img

        # If multiple faces were detected, we will use the centermost face
        if bbox.shape[0] > 1:
            img_center = np.array(
                [face_img.shape[0] / 2, face_img.shape[1] / 2]
            )
            bbox, points = _get_center_face(bbox, points, img_center)

        # Format bounding box and landmarks arrays
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T

        # Perform facial alignment using MTCNN bounding box and facial
        # landmarks to get 112x112x3 facial image
        aligned_img = preprocess(face_img, bbox, points, image_size="112,112")
        # Convert facial image from BGR to RGB colorspace
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        # Format image array to have RGB color channels along first dimension
        aligned_img = np.transpose(aligned_img, (2, 0, 1))

        return aligned_img

    def get_feature(self, aligned_img: np.ndarray) -> np.ndarray:
        """Perform ArcFace feature extraction on input preprocessed image.

        Parameters
        ----------
        aligned_img: np.ndarray
            Aligned facial image of shape 3x112x112 with RGB colorspace

        Returns
        -------
        np.ndarray
            Extracted FaceNet feature vector of shape 512x1

        """
        if len(aligned_img.shape) == 3:
            aligned_img = np.expand_dims(aligned_img, axis=0)

        data = mx.nd.array(aligned_img)
        db = mx.io.DataBatch(data=(data,))

        self.__model.forward(db, is_train=False)

        embedding = self.__model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()

        return embedding


def _prewhiten(x: np.ndarray):
    """Normalization step for image prior to feature extraction.
    Used in FaceNet pipeline.

    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
    return y


def _get_center_face(
    bbox: np.ndarray, points: np.ndarray, img_center: np.ndarray
):
    """Using face bounding boxes, facial landmark points and
    image center point, find the centermost detected face
    and return its corresponding bounding box and facial
    landmarks.

    """
    dists = []
    for i in range(bbox.shape[0]):
        face_rect = np.array(
            [[bbox[i, 0], bbox[i, 1]], [bbox[i, 2], bbox[i, 3]]]
        )
        dists.append(_rect_point_dist(face_rect, img_center))

    idx = dists.index(min(dists))

    bbox = bbox[idx, :]
    points = points[idx, :]

    return bbox, points


def _rect_point_dist(bbox: np.ndarray, point: np.ndarray) -> float:
    """Get distance between a bounding box and a point.

    Parameters
    ----------
    bbox: np.ndarray
        Bounding box array containing top-left and bottom-right
        coordinates [(x1, y1), (x2, x2)]
    point: np.ndarray
        Point coordinates [(x1, y1)]

    Returns
    -------
    float:
        Distance between bbox and point

    """
    bbox_center = (bbox[0] + bbox[1]) / 2

    bbox_height = bbox[1, 0] - bbox[0, 0]
    bbox_width = bbox[1, 1] - bbox[0, 1]

    dx = max(np.abs(point[1] - bbox_center[1]) - bbox_width / 2, 0)
    dy = max(np.abs(point[0] - bbox_center[0]) - bbox_height / 2, 0)
    return dx * dx + dy * dy
