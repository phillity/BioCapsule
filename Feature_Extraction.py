# Feature Extraction

# References:
# [1] Y. Sui, X. Zou, E. Y. Du and F. Li, "Secure and privacy-preserving biometrics based active authentication," 2012 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Seoul, 2012, pp. 1291-1296.
# https://ieeexplore.ieee.org/document/6377911/
# [2] K. Zhang, Z. Zhang, Z. Li and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," in IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499-1503, Oct. 2016.
# https://ieeexplore.ieee.org/document/7553523/
# [3] F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, pp. 815-823.
# http://ieeexplore.ieee.org/document/7298682/
# [4] O. M. Parkhi, A. Vedaldi, and A. Zisserman, "Deep face recognition," in Proc. Brit. Mach. Vis. Conf., 2015, vol. 1. no. 3, p. 6.
# https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
# [5] B. Amos, B. Ludwiczuk, M. Satyanarayanan, "Openface: A general-purpose face recognition library with mobile applications," CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.
# http://cmusatyalab.github.io/openface/

import cv2
import numpy as np
import sys
import os

sys.path.append("models")
from models import FNET

# FaceNet database feature extraction function [3][4][5]
#   database_name - image database to extact all features
#   model_name - name of model to use for feature extraction
#   mode - specify if input images were cropped, aligned or neither
#   openface - if set to true, use OpenFace FaceNet model 
def extract(database_name,model_name=None,mode="crop",openface=False):
    if model_name:
        database = "images_" + mode + "//" + database_name
    else:
        database = "images//" + database_name

    if openface == False:
        facenet = FNET.FNET(model_name)
        extracted_features = np.empty((0,513))
        extracted_features_flip = np.empty((0,513))
    else:
        facenet = cv2.dnn.readNetFromTorch("models//openface.nn4.small2.v1//openface.nn4.small2.v1.t7")
        extracted_features = np.empty((0,129))
        extracted_features_flip = np.empty((0,129))
    
    s = 0
    for dir in os.listdir(database):
        print(dir)
        s = s+1

        for image_filename in os.listdir(database + "//" + dir):
            print("        " + image_filename)

            # Read in each image
            image_path = database + "//" + dir + "//" + image_filename
            image = cv2.imread(image_path)
            image_flip = cv2.flip(image,1)

            # Extract feature using FaceNet embedding method [3][4]
            if openface == False:
                feature = embedding(image,facenet)
                feature_flip = embedding(image_flip,facenet)
            # Extract feature using FaceNet embedding method [5]
            else:
                feature = embedding_cv(image,facenet)
                feature_flip = embedding_cv(image_flip,facenet)

            # Add class label to extracted feature
            if openface == False:
                feature = np.reshape(np.append(feature,s),(1,513))
                feature_flip = np.reshape(np.append(feature_flip,s),(1,513))
            else:
                feature = np.reshape(np.append(feature,s),(1,129))
                feature_flip = np.reshape(np.append(feature_flip,s),(1,129))

            # Add extracted feature to extracted features
            extracted_features = np.vstack([extracted_features,feature])
            extracted_features_flip = np.vstack([extracted_features_flip,feature_flip])

    # Write out features to text file
    if mode:
        if openface == False:
            output_file = database_name + "_" + mode + "_" + model_name + ".txt" 
            np.savetxt(output_file,extracted_features)
            output_file_flip = database_name + "_" + mode + "_" + model_name + "_flip.txt" 
            np.savetxt(output_file_flip,extracted_features_flip)
        else:
            output_file = database_name + "_" + mode + "_openface.txt" 
            np.savetxt(output_file,extracted_features)
            output_file_flip = database_name + "_" + mode + "_openface_flip.txt" 
            np.savetxt(output_file_flip,extracted_features_flip)
    else:
        if openface == False:
            output_file = database_name + "_" + model_name + ".txt" 
            np.savetxt(output_file,extracted_features)
            output_file_flip = database_name + "_" + model_name + "_flip.txt" 
            np.savetxt(output_file_flip,extracted_features_flip)
        else:
            output_file = database_name + "_openface.txt" 
            np.savetxt(output_file,extracted_features)
            output_file_flip = database_name + "_openface_flip.txt" 
            np.savetxt(output_file_flip,extracted_features_flip)


# FaceNet feature computation helper function [3][4] 
#   image - image to retrieve FaceNet embedding
#   model - FaceNet model to use for feature embedding
def embedding(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    feature = model.feature(image)
    return feature

# FaceNet feature computation helper function [5] 
#   image - image to retrieve FaceNet embedding
#   model - FaceNet model to use for feature embedding
def embedding_cv(image,model):
    blob = cv2.dnn.blobFromImage(image,1./255,(96,96),(0,0),True,False)
    model.setInput(blob)
    feature = model.forward()
    return feature



# 20180402-114759 0.9905 CASIA-WebFace Inception ResNet v1
# 20180408-102900 0.9965 VGGFace2 Inception ResNet v1
database_name = "caltech"
model_name = "20180402-114759"
#model_name = "20180408-102900"
#mode = "crop"
mode = "align"

extract(database_name,mode=mode,model_name=model_name)

