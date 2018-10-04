import facenet
import detect_face
import tensorflow as tf
import numpy as np
import cv2

class FNET():
    def __init__(self):
        print('Loading feature extraction model')
    
        self.sess = tf.Session()
        
        # 20180402-114759//20180402-114759.pb 0.9905 CASIA-WebFace Inception ResNet v1
        #facenet.load_model("models//20180402-114759//20180402-114759.pb")
        # 20180408-102900//20180408-102900.pb 0.9965 VGGFace2 Inception ResNet v1
        facenet.load_model("models//20180408-102900//20180408-102900.pb")

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  

    def feature(self,image):
        # Ensure image is color image
        if len(image.shape) != 3:
            raise ValueError("Color image is required for FaceNet feature embedding!")
        
        # Ensure image is 160x160x3
        if image.shape != (160,160,3):
            image = cv2.resize(image,(160,160))
        
        # Normalize image
        image = FNET.prewhiten(image)

        # Put image into input array
        input = np.zeros((1, 160, 160, 3))
        input[0,:,:,:] = image

        # Perform FaceNet feature embedding
        feed_dict = { self.images_placeholder:input, self.phase_train_placeholder:False }
        output = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return output