# Preprocessing script
import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
from face_models.MTCNN import MtcnnService


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


def align(image):
    # Convert image to BGR if grayscale
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Get RGB version of BGR OpenCV image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face bounding box and landmarks using MTCNN method
    # Face bounding box points [top_left, bottom_right]
    # Face landmark points [eye_right, eye_left, nose, mouth_right, mouth_left]
    face_bb, face_pts = MtcnnService().detect(image_rgb)

    # Get number of faces detected
    nrof_faces = face_bb.shape[0]

    # No faces are detected
    if nrof_faces == 0:
        # raise ValueError('No faces detected in user image! (align)')
        print('No faces detected in user image! (align)')
        return image

    # Multiple faces are detected
    if nrof_faces > 1:
        print('Multiple faces detected in user image! (align)')

        # Choose face closest to center of image
        img_center = np.array([image.shape[0] / 2, image.shape[1] / 2])
        dists = []
        for i in range(face_bb.shape[0]):
            face_rect = np.array([[face_bb[i, 0], face_bb[i, 1]], [face_bb[i, 2], face_bb[i, 3]]])
            dists.append(rect_point_dist(face_rect, img_center))

        idx = dists.index(min(dists))
        face_pts = face_pts[:, idx]

    # One face is detected (or center-most of multiple face detections)
    # Get left and right eye points
    eye_left = (face_pts[1], face_pts[6])
    eye_right = (face_pts[0], face_pts[5])

    eye_center = np.zeros((3,))
    eye_center[0] = (eye_left[0] + eye_right[0]) / 2
    eye_center[1] = (eye_left[1] + eye_right[1]) / 2
    eye_center[2] = 1.

    # Compute angle between eyes
    dY = eye_right[1] - eye_left[1]
    dX = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Get size and center point of image
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Get size of image after rotation
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform rotation
    image = cv2.warpAffine(image, M, (nW, nH))
    eye_center = M @ eye_center

    return crop(image, aligned=True, rot_center=eye_center)


def crop(image, out_size=160, margin=44, aligned=False, rot_center=None):
    # Convert image to BGR if grayscale
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Get RGB version of BGR OpenCV image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face bounding box and landmarks using MTCNN method
    # Face bounding box points [top_left, bottom_right]
    bb, pts = MtcnnService().detect(image_rgb)

    # Get number of faces detected
    nrof_faces = bb.shape[0]

    # No faces are detected
    if nrof_faces == 0:
        # raise ValueError('No faces detected in user image! (crop)')
        print('No faces detected in user image! (crop)')
        return image
    # Multiple faces are detected
    if nrof_faces > 1:
        print('Multiple faces detected in user image! (crop)')

        # Choose face closest to center of image
        img_center = np.array([image.shape[0] / 2, image.shape[1] / 2])
        if not aligned:
            dists = []
            for i in range(bb.shape[0]):
                face_rect = np.array([[bb[i, 0], bb[i, 1]], [bb[i, 2], bb[i, 3]]])
                dists.append(rect_point_dist(face_rect, img_center))

            idx = dists.index(min(dists))
            bb[0] = bb[idx]

        # Choose face we used for alignment
        else:
            dists = []
            for i in range(pts.shape[1]):
                eye_left = (pts[1][i], pts[6][i])
                eye_right = (pts[0][i], pts[5][i])

                eye_center = np.zeros((2,))
                eye_center[0] = (eye_left[0] + eye_right[0]) / 2
                eye_center[1] = (eye_left[1] + eye_right[1]) / 2

                dist = np.linalg.norm(eye_center - rot_center)

                dists.append(dist)

            idx = dists.index(min(dists))
            bb[0] = bb[idx]

    # One face is detected (or center-most of multiple face detections)
    # Format face bounding box
    bb = np.around(bb[0]).astype(int)
    bb[0] = np.maximum(bb[0] - margin / 2, 0)
    bb[1] = np.maximum(bb[1] - margin / 2, 0)
    bb[2] = np.maximum(bb[2] + margin / 2, 0)
    bb[3] = np.maximum(bb[3] + margin / 2, 0)

    face_bb = []
    for i in range(0, 3, 2):
        face_bb.append((bb[i], bb[i + 1]))

    # Get face detection
    face = image[face_bb[0][1]:face_bb[1][1], face_bb[0][0]:face_bb[1][0]]

    # Resize face image
    face = cv2.resize(face, (out_size, out_size))

    # Return detected face
    return face


def preprocess(database, mode):
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    database_path = os.path.join(cur_path, 'images', database)
    for subject in os.listdir(database_path):
        subject_path = os.path.join(database_path, subject)
        print(subject_path)
        for image in os.listdir(subject_path):
            image_path = os.path.join(subject_path, image)
            print(image_path)

            img = cv2.imread(image_path)
            if mode == 'crop':
                img = crop(img)
            else:
                img = align(img)

            out_path = os.path.join(cur_path, 'images_align', database, subject, image)
            cv2.imwrite(out_path, img)


# Initialize MTCNN model
MtcnnService()

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to perform preprocessing upon')
parser.add_argument('-m', '--mode', choices=['crop', 'align'], default='align',
                    help='preprocess using crop or align')
args = vars(parser.parse_args())

# Perform preprocessing
preprocess(args['database'], args['mode'])
