from __future__ import absolute_import, division, print_function

import cv2
import mxnet as mx
import numpy as np
import tensorflow as tf
from insightface.model_zoo.face_detection import FaceDetector
from sklearn.preprocessing import normalize

from face_models.facenet import load_facenet_model
from face_models.mtcnn import MtcnnDetector
from face_models.utils import align, get_center_face

__all__ = ["ArcFaceModel", "FaceNetModel", "MtcnnModel", "RetinaFaceModel"]


class RetinaFaceModel:
    """RetinaFace: Single-stage Dense Face Localisation in the Wild
    https://arxiv.org/abs/1905.00641

    Uses https://github.com/deepinsight/insightface implementation

    """

    def __init__(self, gpu: int):
        self.model_path = "src/face_models/models/retinaface/R50-0000.params"
        self.__model = self.__load_model(gpu)

    def __load_model(self, gpu: int):
        """Load pretrained RetinaFace model from:
        src/face_models/models/retinaface/R50-0000.params

        Model originally downloaded from:
        https://drive.google.com/file/d/1wm-6K688HQEx_H90UdAIuKv-NAsKBu85/view

        """
        model = FaceDetector(self.model_path, rac="net3")
        model.prepare(gpu)
        return model

    def get_input(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for feature extraction using RetinaFace model.
        RetinaFace detects facial bounding boxes and facial landmarks to get
        face region-of-interest and perform facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 112x112x3 with RGB colorspace

        """
        # Get face bounding box and landmark detections
        # from MTCNN model
        ret = self.__model.detect(face_img)

        # If no detections were made, return input image
        if ret is None:
            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            return face_img

        # Otherwise, the MTCNN model detected facial bounding boxes
        # and landmarks
        bbox, points = ret

        # Make sure detections are not empty lists
        if bbox.shape[0] == 0:
            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            return face_img

        # If multiple faces were detected, we will use the centermost face
        if bbox.shape[0] > 1:
            img_center = np.array(
                [face_img.shape[0] / 2, face_img.shape[1] / 2]
            )
            bbox, points = get_center_face(bbox, points, img_center)

        else:
            bbox = bbox[0]
            points = points[0]

        # Format bounding box and landmarks arrays
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T

        # Perform facial alignment using MTCNN bounding box and facial
        # landmarks to get 112x112x3 facial image
        aligned_img = align(face_img, bbox=bbox)
        # Convert facial image from BGR to RGB colorspace
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)

        return aligned_img


class MtcnnModel:
    """Joint Face Detection and Alignment using Multi-task
    Cascaded Convolutional Networks
    https://arxiv.org/abs/1604.02878

    Uses https://github.com/davidsandberg/facenet implementation

    """

    def __init__(self, gpu: int):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.model_path = "src/face_models/models/mtcnn"
        self.__model = self.__load_model(ctx)

    def __load_model(self, ctx: int):
        """Load pretrained MTCNN model from:
        src/face_models/models/det[1-4]*

        """
        model = MtcnnDetector(
            model_dir=self.model_path, ctx=ctx, accurate_landmark=True,
        )
        return model

    def get_input(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess facial image for feature extraction using MTCNN model.
        MTCNN detects facial bounding boxes and facial landmarks to get face
        region-of-interest and perform facial alignment.

        Parameters
        ----------
        face_img: np.ndarray
            Face image of any width/height with BGR colorspace channels

        Returns
        -------
        np.ndarray
            Aligned facial image of shape 112x112x3 with RGB colorspace

        """
        # Get face bounding box and landmark detections
        # from MTCNN model
        ret = self.__model.detect_face(face_img)

        # If no detections were made, return input image
        if ret is None:
            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            return face_img

        # Otherwise, the MTCNN model detected facial bounding boxes
        # and landmarks
        bbox, points = ret

        # Make sure detections are not empty lists
        if bbox.shape[0] == 0:
            face_img = cv2.resize(face_img, (112, 112))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            return face_img

        # If multiple faces were detected, we will use the centermost face
        if bbox.shape[0] > 1:
            img_center = np.array(
                [face_img.shape[0] / 2, face_img.shape[1] / 2]
            )
            bbox, points = get_center_face(bbox, points, img_center)

        else:
            bbox = bbox[0]
            points = points[0]

        # Format bounding box and landmarks arrays
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T

        # Perform facial alignment using MTCNN bounding box and facial
        # landmarks to get 112x112x3 facial image
        aligned_img = align(face_img, points=points)
        # Convert facial image from BGR to RGB colorspace
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)

        return aligned_img


class FaceNetModel:
    """FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1503.03832

    Uses https://github.com/davidsandberg/facenet implementation

    """

    def __init__(self, gpu: int, detector: str):
        self.model_path = "src/face_models/models/facenet/20180402-114759.pb"
        self.__load_model()

        if detector == "mtcnn":
            self.__detector = MtcnnModel(gpu)
        else:
            self.__detector = RetinaFaceModel(gpu)

    def __load_model(self):
        """Load pretrained FaceNet model from:
        src/face_models/models/facenet/20180402-114759.pb

        Model originally downloaded from:
        https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view

        FaceNet model with Inception-ResNet-v1 backbone, pretrained on
        VGGFace2, achieves 0.9965 on LFW.

        """
        self.__sess = tf.compat.v1.Session()

        load_facenet_model(self.model_path)

        self.__images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "input:0"
        )
        self.__embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "embeddings:0"
        )
        self.__phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "phase_train:0"
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
        aligned_img = self.__detector.get_input(face_img)

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
        aligned_img = self.__prewhiten(aligned_img)

        if len(aligned_img.shape) == 3:
            aligned_img = np.expand_dims(aligned_img, axis=0)

        feed_dict = {
            self.__images_placeholder: aligned_img,
            self.__phase_train_placeholder: False,
        }

        embedding = self.__sess.run(self.__embeddings, feed_dict=feed_dict)

        return embedding[0]

    @staticmethod
    def __prewhiten(x: np.ndarray):
        """Normalization step for image prior to feature extraction.
        Used in FaceNet pipeline.

        """
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
        return y


class ArcFaceModel:
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698

    Uses https://github.com/deepinsight/insightface implementation

    """

    def __init__(self, gpu: int, detector: str):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.model_path = "src/face_models/models/arcface/model"
        self.__model = self.__load_model(ctx)

        if detector == "mtcnn":
            self.__detector = MtcnnModel(gpu)
        else:
            self.__detector = RetinaFaceModel(gpu)

    def __load_model(self, ctx: int):
        """Load pretrained ArcFace model from:
        src/face_models/models/arcface/model-symbol.json
        src/face_models/models/arcface/model-0000.params

        Model originally downloaded from:
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
        aligned_img = self.__detector.get_input(face_img)

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
