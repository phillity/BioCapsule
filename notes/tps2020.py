#this module is alone and not used by others.
#Uses two datasets: lfw and gtdb
import datetime
import os
import shutil
#from sklearn.neighbors import KNeighborsClassifier
from argparse import ArgumentParser

import cv2
#from queue import Queue #FIFO
#from threading import Thread
import numpy as np
#from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

#from utils import progress_bar
from biocapsule import BioCapsuleGenerator
from face import ArcFace, extract_dataset

np.random.seed(42)

def filter_lfw(features): #only used in this module; second input features_flip removed by Kai
    y = np.unique(features[:, -1])
    mask = np.ones(features[:, -1].shape, dtype=bool)
    for y_i in y:
        if features[features[:, -1] == y_i].shape[0] < 5:
            idxes = np.where(features[:, -1] == y_i)
            mask[idxes] = False
    features = features[mask]
    #features_flip = features_flip[mask]

    y_map = {}
    y = np.unique(features[:, -1])
    for i, y_i in enumerate(y):
        y_map[y_i] = i + 1

    for i in range(features[:, -1].shape[0]):
        features[i, -1] = y_map[features[i, -1]]
        #features_flip[i, -1] = y_map[features_flip[i, -1]]

    #return features, features_flip #second return value features_flip removed by Kai
    return features

#returns a 2D array of 6 by 512
def get_rs_features(): #only used in this module
    arcface = ArcFace() #an object of ArcFace class; this can be customized with a different feature extraction method

    # if os.path.isdir(os.path.join(os.path.abspath(""), "images", "rs_aligned")):
        # shutil.rmtree(os.path.join(os.path.abspath(""), "images", "rs_aligned"))

    rs_features = np.zeros((6, 512))
    #os.mkdir(os.path.join(os.path.abspath(""), "images", "rs_aligned"))
    for s_id, subject in enumerate(os.listdir(os.path.join(os.path.abspath(""), "images", "rs"))[4:]): #here listdir should return a list of 10 directory names rs_00 to rs_09
        for image in os.listdir(os.path.join(os.path.abspath(""), "images", "rs", subject)): #image will be sth like rs_04.jpg ... rs_09.jpg; subject is rs_04 ... rs_09
            img = cv2.imread(os.path.join(os.path.abspath(""), "images", "rs", subject, image)) #img is of class 'numpy.ndarray'
            img_aligned = arcface.preprocess(img) #get an aligned image with just facial region (five facial landmarks)
            feature = arcface.extract(img_aligned, align=False) #the return value of extract here should be a row vector of 512 elements
            rs_features[s_id] = feature

            if img_aligned.shape != (3, 112, 112): #this is unnecessary since extract function has already done this?
                img_aligned = cv2.resize(img_aligned, (112, 112))
                img_aligned = np.rollaxis(cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR), 2, 0)

            #cv2.imwrite(os.path.join(os.path.abspath(""), "images", "rs_aligned", image), cv2.cvtColor(np.rollaxis(img_aligned, 0, 3), cv2.COLOR_RGB2BGR))

    return rs_features

#yLen is the number of subjects (LFW: 423; GTDB: 50)
#return a vector of random values (0~5) of length yLen
def rs_rbac(yLen, dist): #only used in this module; REVISED BY KAI
    if dist == "unbal":
        rs_map = np.random.choice(6, yLen, p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
    else:
        rs_map = np.random.choice(6, yLen)
    return rs_map

#return biocapsules
#input features is a 2D array: number of rows is the image count; number of columns is 513
#input rs_features is of shape 6 by 512
#original input rs_map is also removed by Kai
def get_bcs(features, rs_features): #only used in this module; #second input features_flip and second return value bcs_flip removed by Kai
    bcs = np.zeros((rs_features.shape[0], features.shape[0], 513)) # 3D array of 6 by image_count by 513
    #bcs_flip = np.zeros((rs_features.shape[0], features_flip.shape[0], 513))
#features[:, :-1] is of shape image_count by 512
    bc_gen = BioCapsuleGenerator()
    for i in range(rs_features.shape[0]): #i: 0~5
        bcs[i, :, :] = np.hstack([bc_gen.biocapsule_batch(features[:, :-1], rs_features[i]), features[:, -1][:, np.newaxis]]) #note: the features 2D array will be updated here (but its last column remains the same)!
        #bcs_flip[i, :, :] = np.hstack([bc_gen.biocapsule_batch(features_flip[:, :-1], rs_features[i]), features_flip[:, -1][:, np.newaxis]])
#last column features[:, -1][:, np.newaxis] is subject_id
    #return bcs, bcs_flip #second return value bcs_flip removed by Kai
    return bcs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, choices=["lfw", "gtdb"], help="dataset to use in experiment")
    parser.add_argument("-m", "--mode", required=True, choices=["under", "bc"], help="feature mode to use in experiment")
    parser.add_argument("-r", "--role_dist", required=False, choices=["bal", "unbal"], default="unbal", help="role distribution to use in experiment")
    parser.add_argument("-t", "--thread_cnt", required=False, type=int, default=1, help="thread count to use in classifier training")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1, help="gpu to use in feature extraction")
    args = vars(parser.parse_args())

    if args["mode"] == "under":
        fi = open(os.path.join(os.path.abspath(""), "results", "tps2020_{}_under.txt".format(args["dataset"])), "w")
    else:
        fi = open(os.path.join(os.path.abspath(""), "results", "tps2020_{}_bc_{}.txt".format(args["dataset"], args["role_dist"])), "w")
    print("computing features:",datetime.datetime.now())
    # extract features for experiment: extract_dataset is in face.py
    if args["dataset"]=="lfw":
        features = np.load(os.path.join(os.path.abspath(""), "data", "lfw_arcface_feat.npz"))["arr_0"]
    elif args["dataset"]=="gtdb":
        features = np.load(os.path.join(os.path.abspath(""), "data", "gtdb_arcface_feat.npz"))["arr_0"]
    else:
        features = extract_dataset(args["dataset"], "arcface", args["gpu"]) #second return value features_flip removed by Kai
# features is a 2D array: number of rows is the image count; number of columns is 513 (last column is 1-based subject_id). Each row is a feature vector plus subject_id.
    print("done computing features",datetime.datetime.now())
    # remove all subjects with less than 5 images from LFW dataset
    if args["dataset"] == "lfw": #filter_lfw is in this module
        print("filtering lfw features.")
        features = filter_lfw(features) #second input and return value features_flip removed by Kai

    # if biocapsules are used, we can perform authn-authz operation using reference subjects
    if args["mode"] == "bc":
        # get reference subjects for roles; get_rs_features is in this module
        print("computing bcs:",datetime.datetime.now())
        rs_features = get_rs_features() #a 2D array of 6 by 512

        # assign subjects their reference subjects/roles; rs_rbac is in this module
        rs_map = rs_rbac(len(np.unique(features[:, -1])), args["role_dist"]) #each element (0~5) of the vector rs_map (of length number_of_subjects) represents a reference_subject/role for a subject
        cnts = np.unique(rs_map, return_counts=True)[1]
        for i, cnt in enumerate(cnts): #histogram: how many subjects for each role 0~5
            fi.write("Role {} -- {} Subjects\n".format(i + 1, cnt))

        # create all possible biocapsules: get_bcs is in this module; note: features will get updated by the get_bcs call
        bcs = get_bcs(features, rs_features) #second input features_flip and fourth input rs_map and second return value bcs_flip removed by Kai

    # tn, fp, fn, tp
    #conf_mat = np.zeros((4,))
    ctp=0
    ctn=0
    cfp=0
    cfn=0
    cfp1=0
    print("begin skf...",datetime.datetime.now())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(skf.split(features[:, :-1], features[:, -1])): #k will be 0 to 4; test_index is a vector of length image_count/5; train_index is a vector of length image_count*4/5
        print("fold",k)
        if args["mode"] == "under":
            X_train = features[:, :-1][train_index] #2D array of shape train_image_count by 512
            y_train = features[:, -1][train_index] #a vector of subject_id's of length train_image_count
            X_test = features[:, :-1][test_index] #2D array of shape test_image_count by 512
            y_test = features[:, -1][test_index] #a vector of subject_id's of length test_image_count
            # labels = np.unique(y_train) #a vector of  unique subject_id's
            # labels_test=np.unique(y_test)
            # assert labels.size==labels_test.size
            # knn = KNeighborsClassifier() #typically no better than LR?
            # print("fold",k,"KNN score:", knn.fit(X_train, y_train).score(X_test, y_test))
            clf = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train, y_train)
            #print("fold",k,"LR score:", clf.score(X_test, y_test))
            y_pred=clf.predict(X_test)
            for j in range(len(test_index)):
                if y_pred[j]==y_test[j]:
                    ctp=ctp+1
                else:
                    cfn=cfn+1
                    cfp=cfp+1
        else: #args["mode"] == "bc"
            for i in range(len(rs_features)): #i: 0~5
                X_train = bcs[i, :, :-1][train_index]
                y_train = bcs[i, :, -1][train_index] #based on bcs construction, equivalent to features[:, -1][train_index]
                X_test = bcs[i, :, :-1][test_index]
                y_test = bcs[i, :, -1][test_index] #based on bcs construction, equivalent to features[:, -1][test_index]
                # knn = KNeighborsClassifier() #typically no better than LR?
                # print("fold",k,"rs",i,"KNN score:", knn.fit(X_train, y_train).score(X_test, y_test))
                clf = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train, y_train)
                #print("fold",k,"rs",i," LR score:", clf.score(X_test, y_test))
                y_pred=clf.predict(X_test)
                #indices = [idx+1 for idx, el in enumerate(rs_map) if el == i] #subject ids who are assigned rs role i
                for j in range(len(test_index)):
                    if rs_map[int(y_test[j]-1)]==i: #subject y_test[j] is known to be in role i
                        if y_pred[j]==y_test[j]:
                            ctp=ctp+1
                        else:
                            cfn=cfn+1
                    else: #subject y_test[j] is known to be not in role i
                        if y_pred[j]!=y_test[j]:
                            cfp1=cfp1+1
                            if rs_map[int(y_pred[j]-1)]==i:
                                cfp=cfp+1
            #labels = np.unique(y_train[0]) #y_train[0] is the first row of y_train: unique 1-based subject ids; equuivalently, we could use y_train[1~5]

    print("ctp =",ctp)
    print("ctn =",ctn)
    print("cfp =",cfp)
    print("cfn =",cfn)
    print("cfp1 =",cfp1)
    # (tn + tp) / (tn + fp + fn + tp)
    # acc = (conf_mat[0] + conf_mat[3]) / np.sum(conf_mat)
    # # fp / (tn + fp)
    # far = conf_mat[1] / (conf_mat[0] + conf_mat[1])
    # # fn / (fn + tp)
    # frr = conf_mat[2] / (conf_mat[2] + conf_mat[3])

    fi.write("Dataset -- {}\n".format(args["dataset"]))
    fi.write("BC  -- {}\n".format(args["mode"]))
    fi.write("RS  -- {}\n".format(args["role_dist"]))
    # fi.write("TN -- {:.6f}\n".format(conf_mat[0]))
    # fi.write("TP -- {:.6f}\n".format(conf_mat[3]))
    # fi.write("FP -- {:.6f}\n".format(conf_mat[1]))
    # fi.write("FN -- {:.6f}\n".format(conf_mat[2]))
    # fi.write("ACC -- {:.6f}\n".format(acc))
    # fi.write("FAR -- {:.6f}\n".format(far))
    # fi.write("FRR -- {:.6f}\n".format(frr))
    fi.close()

#on lfw, I got 5,5(under);27,29(bc bal);31,50(bc unbal) for fp,fn 
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# confusion_matrix is a function that computes confusion matrix to evaluate the accuracy of a classification.
# By definition a confusion matrix C is such that C_i,j is equal to the number of observations known to be in group i and predicted to be in group j.
# Thus in binary classification, the count of true negatives is C_0,0, false negatives is C_1,0, true positives is C_1,1 and false positives is C_0,1.
