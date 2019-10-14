from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import numpy as np
import mxnet as mx
import tensorflow as tf
from sklearn.preprocessing import normalize
from face_models import facenet
from face_models.face_preprocess import preprocess
from face_models.mtcnn_detector import MtcnnDetector


__all__ = ["ArcFaceModel", "FaceNetModel"]


def rect_point_dist(rect, center):
        rect_center = (rect[0] + rect[1]) / 2
        y = rect_center[0]
        x = rect_center[1]
        height = rect[1, 0] - rect[0, 0]
        width = rect[1, 1] - rect[0, 1]
        py = center[0]
        px = center[1]
        dx = max(np.abs(px - x) - width / 2, 0)
        dy = max(np.abs(py - y) - height / 2, 0)
        return dx * dx + dy * dy


def get_center_face(bbox, points, img_center):
    dists = []
    for i in range(bbox.shape[0]):
        face_rect = np.array([[bbox[i, 0], bbox[i, 1]], [
                              bbox[i, 2], bbox[i, 3]]])
        dists.append(rect_point_dist(face_rect, img_center))

    idx = dists.index(min(dists))
    bbox = bbox[idx, :]
    points = points[idx, :]
    return bbox, points


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
    return y


class FaceNetModel:
    def __init__(self, gpu):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)
        self.__sess = None
        self.__images_placeholder = None
        self.__embeddings = None
        self.__phase_train_placeholder = None
        self.__get_model()
        self.__detector = MtcnnDetector(model_folder=os.path.join(
            os.path.dirname(__file__), "mtcnn-model"), ctx=ctx, accurate_landmark=True)

    def __get_model(self):
        self.__sess = tf.Session()
        model_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "model-ir-v1", "20180402-114759.pb")
        print("loading", model_path)
        facenet.load_model(model_path)
        self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.__phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")

    def get_input(self, face_img):
        ret = self.__detector.detect_face(face_img, det_type=0)
        if ret is None:
            return face_img
        img_center = np.array([face_img.shape[0] / 2, face_img.shape[1] / 2])
        bbox, points = ret
        if bbox.shape[0] == 0:
            return face_img
        if bbox.shape[0] > 1:
            bbox, points = get_center_face(bbox, points, img_center)
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T
        nimg = preprocess(face_img, bbox, points, image_size="112,112")
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = cv2.resize(nimg, (160, 160))
        return aligned

    def get_feature(self, aligned):
        image = prewhiten(aligned)
        input_blob = np.zeros((1, 160, 160, 3))
        input_blob[0, :, :, :] = image
        feed_dict = {self.__images_placeholder: input_blob,
                     self.__phase_train_placeholder: False}
        embedding = self.__sess.run(
            self.__embeddings, feed_dict=feed_dict)
        return embedding[0]


class ArcFaceModel:
    def __init__(self, gpu):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)
        self.__model = self.__get_model(ctx)
        self.__detector = MtcnnDetector(model_folder=os.path.join(
            os.path.dirname(__file__), "mtcnn-model"), ctx=ctx, accurate_landmark=True)

    def __get_model(self, ctx):
        model_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "model-r100-ii" + "\\")
        print("loading", model_path, 0)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        return model

    def get_input(self, face_img):
        ret = self.__detector.detect_face(face_img, det_type=0)
        if ret is None:
            return face_img
        img_center = np.array([face_img.shape[0] / 2, face_img.shape[1] / 2])
        bbox, points = ret
        if bbox.shape[0] == 0:
            return face_img
        if bbox.shape[0] > 1:
            bbox, points = get_center_face(bbox, points, img_center)
        bbox = bbox[0:4]
        points = points.reshape((2, 5)).T
        nimg = preprocess(face_img, bbox, points, image_size="112,112")
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.__model.forward(db, is_train=False)
        embedding = self.__model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding
