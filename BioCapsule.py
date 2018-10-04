# BioCapsule Computation

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
from models import MTCNN
from models import FNET

# BioCapsule computation function
#   image - user image filename
#   RS - RS image filename
def BioCapsule(image,RS):



    return image

# FaceNet database feature extraction function [3][4]
#   database - image database to extact all features
#   openface - if set to true, use OpenFace FaceNet model 
def extract(database,openface=False):
    if openface == False:
        facenet = FNET.FNET()
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
            image_flip = cv2.flip(image, 1)

            # Extract feature using FaceNet embedding method [3][4]
            if openface == False:
                feature = embedding(image,facenet)
                feature_flip = embedding(image_flip,facenet)
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
    np.savetxt("features.txt",extracted_features)
    np.savetxt("features_flip.txt",extracted_features_flip)


# FaceNet feature computation helper function [3][4] 
#   image - image to retrieve FaceNet embedding
#   model - FaceNet model to use for feature embedding
def embedding(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    feature = model.feature(image)
    return feature

def embedding_cv(image,model):
    blob = cv2.dnn.blobFromImage(image,1./255,(96,96),(0,0),True,False)
    model.setInput(blob)
    feature = model.forward()
    return feature

# MTCNN face detection/alignment function [2]
#   image - image filename to perform facial detection/alignment
#   mtcnn - MTCNN model to use for facial detection/alignment
#   out_size - output size of image
#   ratio - scaling factor to determine how "zoomed in" on face
def align(image,mtcnn,out_size=160,ratio=0.25):
    # Get RGB version of BGR OpenCV image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face bounding box and landmarks using MTCNN method [2]
    # Face bounding box points [top_left,bottom_right]
    # Face landmark points [eye_right,eye_left,nose,mouth_right,mouth_left]
    face_bb, face_pts = mtcnn.detect(image_rgb)

    # Get number of faces detected
    nrof_faces = face_bb.shape[0]

    # No faces are detected
    if nrof_faces == 0:
        raise ValueError("No faces detected in user image!")
    
    # Multiple faces are detected
    if nrof_faces > 1:
        print("Multiple faces detected in user image!")

        sizes = []
        for i in range(face_bb.shape[0]):
            l = np.maximum(face_bb[i][2],0) - np.maximum(face_bb[i][0],0)
            w = np.maximum(face_bb[i][3],0) - np.maximum(face_bb[i][1],0)
            sizes.append(l*w)

        idx = sizes.index(max(sizes))
        face_pts = face_pts[:,idx] 
    
    # One face is detected (or largest of multiple face detections)
    # Get left and right eye points
    eye_left = (face_pts[1],face_pts[6])
    eye_right = (face_pts[0],face_pts[5])

    # Compute angle between eyes
    dY = eye_right[1] - eye_left[1]
    dX = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Compute median point between eyes
    eye_center = (((eye_right[0]+eye_left[0])/2,(eye_right[1]+eye_left[1])/2))

    # Get desired left and right eye coordinates based on ratio parameter
    eye_left_desired = ratio
    eye_right_desired = 1.0 - ratio

    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (eye_right_desired - eye_left_desired)
    desired_dist *= out_size
    scale = desired_dist / dist

    # Get affine transformation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # Update the translation component of the matrix
    tX = out_size * 0.5
    tY = out_size * eye_left_desired
    M[0,2] += (tX - eye_center[0])
    M[1,2] += (tY - eye_center[1])

    (w, h) = (out_size, out_size)
    output = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC)

    return output

# MTCNN face detection (without alignment) function [2]
#   image - image filename to perform facial detection/alignment
#   mtcnn - MTCNN model to use for facial detection/alignment
#   out_size - output size of image
#   margin - pixel area around detected face to preserve
def crop(image,mtcnn,out_size=160,margin=44):
    # Get RGB version of BGR OpenCV image
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Get face bounding box and landmarks using MTCNN method [2]
    # Face bounding box points [top_left,bottom_right]
    bb, _ = mtcnn.detect(image_rgb)

    # Get number of faces detected
    nrof_faces = bb.shape[0]

    # No faces are detected
    if nrof_faces == 0:
        raise ValueError("No faces detected in user image!")
    # Multiple faces are detected
    if nrof_faces > 1:
        print("Multiple faces detected in user image!")

        sizes = []
        for i in range(bb.shape[0]):
            l = np.maximum(bb[i][2]+margin/2,0) - np.maximum(bb[i][0]-margin/2,0)
            w = np.maximum(bb[i][3]+margin/2,0) - np.maximum(bb[i][1]-margin/2,0)
            sizes.append(l*w)

        idx = sizes.index(max(sizes))
        bb[0] = bb[idx] 

    # One face is detected (or largest of multiple face detections)
    # Format face boudning box
    bb = np.around(bb[0]).astype(int)
    bb[0] = np.maximum(bb[0]-margin/2,0)
    bb[1] = np.maximum(bb[1]-margin/2,0)
    bb[2] = np.maximum(bb[2]+margin/2,0)
    bb[3] = np.maximum(bb[3]+margin/2,0)

    face_bb = []
    for i in range(0,3,2):
        face_bb.append((bb[i],bb[i+1]))

    # Get face detection
    face = image[face_bb[0][1]:face_bb[1][1],face_bb[0][0]:face_bb[1][0]]
        
    # Resize face image
    face = cv2.resize(face,(out_size,out_size))

    # Return detected face
    return face




#database = "images_align//caltech"
#extract(database,openface=False)

# Load MCTNN model [2]
mtcnn = MTCNN.MTCNN()

database = "images//lfw"
for dir in os.listdir(database):
    print(dir)
    for file_name in os.listdir(database + "//" + dir):
        print("       " + file_name)
        image_path = database + "//" + dir + "//" + file_name
        out_path = "images_crop//lfw" + "//" + dir + "//" + file_name

        image = cv2.imread(image_path)
        image = crop(image,mtcnn)

        cv2.imwrite(out_path,image)





