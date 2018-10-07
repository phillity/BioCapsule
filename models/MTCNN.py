import facenet
import detect_face
import tensorflow as tf
import numpy as np

class MTCNN():
    def __init__(self):
        print('Creating networks and loading parameters')
    
        sess = tf.Session()
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

        self.minsize = 100 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

    def detect(self,image):
        bounding_boxes, facial_pts = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        return bounding_boxes, facial_pts