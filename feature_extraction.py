# Feature extraction script
import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
from face_models.FNET import FnetService


def extract(database):
    cur_path = os.path.dirname(__file__)
    database_path = os.path.join(cur_path, 'images_align', database)

    extracted_features = np.empty((0, 513))
    extracted_features_flip = np.empty((0, 513))

    subject_idx = 0
    for subject in os.listdir(database_path):
        subject_path = os.path.join(database_path, subject)
        print(subject_path)
        subject_idx += 1
        for image in os.listdir(subject_path):
            image_path = os.path.join(subject_path, image)
            print(image_path)

            # Read image
            img = cv2.imread(image_path)
            img_flip = cv2.flip(img, 1)

            # Change channels from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_flip = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract features
            feature = FnetService().feature(img)
            feature_flip = FnetService().feature(img_flip)

            # Add class label to extracted feature
            feature = np.reshape(np.append(feature, subject_idx), (1, 513))
            feature_flip = np.reshape(np.append(feature_flip, subject_idx), (1, 513))

            # Add extracted feature to extracted features
            extracted_features = np.vstack([extracted_features, feature])
            extracted_features_flip = np.vstack([extracted_features_flip, feature_flip])

    # Write out features to npz file
    data_path = os.path.join(cur_path, 'data', database + '.npz')
    data_flip_path = os.path.join(cur_path, 'data', database + '_flip.npz')
    np.savez_compressed(data_path, extracted_features)
    np.savez_compressed(data_flip_path, extracted_features_flip)


# Initialize FaceNet model
FnetService()

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to perform feature extraction on')
args = vars(parser.parse_args())

# Perform feature extraction
extract(args['database'])
