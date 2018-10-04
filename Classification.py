import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

#def svm_recognition(data,data_flip,labels):
#    # Initialize result metrics
#    FAR = []
#    FRR = []
#    Accuracy = []

#    # Get k-fold split of dataset (k=5)
#    #cv = ShuffleSplit(n_splits=5, test_size=0.2)
#    cv = StratifiedKFold(n_splits=5,shuffle=True)
#    cv.get_n_splits(data,labels)

#    for k,(train_index,test_index) in enumerate(cv.split(data,labels)):
#        # Get training and testing sets
#        train = np.vstack([data[train_index,:],data_flip[train_index,:]])
#        train_labels = np.append(labels[train_index],labels[train_index])
#        test = data[test_index,:]
#        test_labels = labels[test_index]

#        # Perform training
#        svm = SVC(kernel='linear', probability=True)
#        svm.fit(train,train_labels)

#        # Perform testing
#        prediction = svm.predict(test)
#        prediction_prob = svm.predict_proba(test)

#        # Get training classes
#        classes = np.unique(train_labels)

#        confusion_mat = metrics.confusion_matrix(test_labels,prediction)
#        TP = np.diagonal(confusion_mat)
#        FP = np.sum(confusion_mat,axis=0) - TP
#        FN = np.sum(confusion_mat,axis=1) - TP

#        TP = np.sum(TP)
#        FP = np.sum(FP)
#        FN = np.sum(FN)
#        TN = np.sum(confusion_mat) - TP - FP - FN
    
#        # Get FAR, FRR and Accuracy
#        FAR.append(FP / (FP + TN))
#        FRR.append(FN / (TP + FN))
#        Accuracy.append((TP + TN) / (TP + TN + FP + FN))

#    # Print results
#    print("Authentication Results:")
#    print("FAR: " + str(np.mean(FAR)) + " (+/- " + str(np.std(FAR)) + ")")
#    print("FRR: " + str(np.mean(FRR)) + " (+/- " + str(np.std(FRR)) + ")")
#    print("Accuracy: " + str(np.mean(Accuracy)) + " (+/- " + str(np.std(Accuracy)) + ")")


def auth_confusion_matrix(y_true,f_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and f_pred[i] == 0:
            TP = TP + 1
        elif y_true[i] == 1 and f_pred[i] == 1:
            TN = TN + 1
        elif y_true[i] == 1 and f_pred[i] == 0:
            FP = FP + 1
        else:
            FN = FN + 1

    return TP, TN, FP, FN

def svm_authentication(data,data_flip,labels):
    # Initialize output information
    thresholds = np.arange(0,1.1,0.1)
    FAR = np.zeros((5,thresholds.shape[0]))
    FRR = np.zeros((5,thresholds.shape[0]))
    ACC = np.zeros((5,thresholds.shape[0]))

    # Get k-fold split of dataset (k=5)
    cv = StratifiedKFold(n_splits=5,shuffle=True)
    cv.get_n_splits(data,labels)

    for k,(train_index,test_index) in enumerate(cv.split(data,labels)):
        # Initialize prediction information
        TP = np.zeros((thresholds.shape[0],1))
        TN = np.zeros((thresholds.shape[0],1))
        FP = np.zeros((thresholds.shape[0],1))
        FN = np.zeros((thresholds.shape[0],1))

        # Get training and testing sets
        train = np.vstack([data[train_index,:],data_flip[train_index,:]])
        train_labels = np.append(labels[train_index],labels[train_index])
        test = data[test_index,:]
        test_labels = labels[test_index]

        # Get training classes
        classes = np.unique(train_labels)

        # Get binary svm for each training class
        class_svms = []
        for i,c in enumerate(classes):
            binary_labels = np.zeros(train_labels.shape)
            binary_labels[train_labels != c] = 1

            svm = SVC(kernel='linear', probability=True)
            svm.fit(train,binary_labels)
            class_svms.append(svm)

        for i,c in enumerate(classes):
            binary_labels = np.zeros(test_labels.shape)
            binary_labels[test_labels != c] = 1

            # Get class c svm
            svm = class_svms[i]

            # Use binary class svm to accept/reject test set
            prediction_prob = svm.predict_proba(test)

            for t,thresh in enumerate(thresholds):
                thresh_labels = np.copy(binary_labels)
                for p,pred in enumerate(prediction_prob):
                    if pred[0] > thresh:
                        thresh_labels[p] = 0
                    else:
                        thresh_labels[p] = 1

                TP_c_t, TN_c_t, FP_c_t, FN_c_t = auth_confusion_matrix(binary_labels,thresh_labels)
                TP[t] = TP[t] + TP_c_t
                TN[t] = TN[t] + TN_c_t
                FP[t] = FP[t] + FP_c_t
                FN[t] = FN[t] + FN_c_t
        
        # FAR = FPR = FP/(FP+TN)
        FAR[k,:] = np.reshape(FP / (FP + TN), (thresholds.shape[0],))
        # FRR = FNR = FN / (TP + FN)
        FRR[k,:] = np.reshape(FN / (TP + FN), (thresholds.shape[0],))
        # ACC = (TP + TN) / (TP + TN + FP + FN)
        ACC[k,:] = np.reshape((TP + TN) / (TP + TN + FP + FN), (thresholds.shape[0],))
        

    # Print results
    print("--------------------------------------------------------------------------------------")
    print("Authentication Results:")
    for t,thresh in enumerate(thresholds):
        print("Threshold: " + str(thresh) + "   " +
        "FAR: " + str(np.mean(FAR,axis=0)[t]) + " (+/- " + str(np.std(FAR,axis=0)[t]) + ")   " +
        "FRR: " + str(np.mean(FRR,axis=0)[t]) + " (+/- " + str(np.std(FRR,axis=0)[t]) + ")   " +
        "Accuracy: " + str(np.mean(ACC,axis=0)[t]) + " (+/- " + str(np.std(ACC,axis=0)[t]) + ")")

data_path = "caltech_crop_20180408-102900.txt"
print(data_path)

data = np.loadtxt(data_path)
data_flip = np.loadtxt(data_path[:-4] + "_flip.txt")

labels =  data[:,-1]
data = data[:,:-1]
data_flip = data_flip[:,:-1]

#svm_recognition(data,data_flip,labels)
svm_authentication(data,data_flip,labels)

