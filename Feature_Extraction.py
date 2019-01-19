# Feature Extraction

import cv2
import numpy as np
import sys
import os
import time
import pickle

from face_models.FNET import FnetService

def extract(database_name):
    database = 'images_align//' + database_name

    extracted_features = np.empty((0,513))
    extracted_features_flip = np.empty((0,513))

    s = 0
    for dir in os.listdir(database):
        print(dir)
        s = s+1
        for image_filename in os.listdir(database + '//' + dir):
            print('        ' + image_filename)
            # Read in each image
            image_path = database + '//' + dir + '//' + image_filename
            image = cv2.imread(image_path)
            image_flip = cv2.flip(image,1)

            # Change channels from BGR to RGB
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image_flip = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            # Extract features
            feature = FnetService().feature(image)
            feature_flip = FnetService().feature(image_flip)

            # Add class label to extracted feature
            feature = np.reshape(np.append(feature,s),(1,513))
            feature_flip = np.reshape(np.append(feature_flip,s),(1,513))

            # Add extracted feature to extracted features
            extracted_features = np.vstack([extracted_features,feature])
            extracted_features_flip = np.vstack([extracted_features_flip,feature_flip])

    # Write out features to npy file
    output_file = 'data//' + database_name + '.npy' 
    np.save(output_file,extracted_features)
    output_file_flip = 'data//' + database_name + '_flip.npy'
    np.save(output_file_flip,extracted_features_flip)

FnetService()
database_name = 'lfw'
extract(database_name)


