from typing import Tuple

import cv2
import numpy as np
from skimage import transform


def get_center_face(
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


def align(
    face_img: np.ndarray,
    bbox: np.ndarray = None,
    points: np.ndarray = None,
    image_size: Tuple[int, int] = (112, 112),
    margin: int = 44,
):
    assert len(image_size) == 2
    assert image_size[0] == 112
    assert image_size[0] == 112 or image_size[1] == 96

    if points is not None:
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        if image_size[1] == 112:
            src[:, 0] += 8.0

        dst = points.astype(np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)

        warped = cv2.warpAffine(
            face_img,
            tform.params[0:2, :],
            (image_size[1], image_size[0]),
            borderValue=0.0,
        )

        return warped

    else:
        bb = np.zeros(4, dtype=np.int32)

        bb[0] = np.maximum(bbox[0] - margin / 2, 0)
        bb[1] = np.maximum(bbox[1] - margin / 2, 0)
        bb[2] = np.minimum(bbox[2] + margin / 2, face_img.shape[1])
        bb[3] = np.minimum(bbox[3] + margin / 2, face_img.shape[0])

        ret = face_img[bb[1] : bb[3], bb[0] : bb[2], :]

        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
