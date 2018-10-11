import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from threading import Thread
from Queue import Queue
import sys

def split_list(l,n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

### Authentication training thread function
def svm_authentication_train(c,train,train_labels,que):
    #print("          Training - Class " + str(c))
    binary_labels = np.zeros(train_labels.shape)
    binary_labels[train_labels != c] = 1

    svm = SVC(kernel='linear', probability=True)
    svm.fit(train,binary_labels)

    que.put((c,svm))

### Authentication testing thread function
def svm_authentication_test(c,svm,test,test_labels,que):
    #print("          Testing - Class " + str(c))
    
    binary_labels = np.zeros(test_labels.shape)
    binary_labels[test_labels != c] = 1

    # Use binary class svm to predict accept/reject test set
    prediction_prob = svm.predict_proba(test)

    result = []
    result.append(prediction_prob[:,0])
    result.append(binary_labels)

    que.put(result)

### AUTHENTICATION
def svm_authentication(data_path,data,data_flip,labels):
    #print("Authentication")
    ### Initialize output information
    FAR = np.array([])
    FRR = np.array([])
    ACC = np.array([])
    EER = np.array([])
    EER_thresh = np.array([])

    # Get k-fold split of dataset (k=5)
    cv = StratifiedKFold(n_splits=5,shuffle=True)
    cv.get_n_splits(data,labels)

    ### Perform k-fold cross validation
    for k,(train_index,test_index) in enumerate(cv.split(data,labels)):
        #print("     Fold - " + str(k))

        # Get training and testing sets
        train = np.vstack([data[train_index,:],data_flip[train_index,:]])
        train_labels = np.append(labels[train_index],labels[train_index])
        test = data[test_index,:]
        test_labels = labels[test_index]

        # Get training classes
        classes = np.unique(train_labels)
        # Split classes into groups of 50 so we will only have 50 threads at once!!!
        classes_split = list(split_list(classes.tolist(),50))

        ### TRAINING
        # Binary SVM for each class
        class_svms = []
        c_idxes = []
        threads = []
        que = Queue()

        # Thread to train each class binary SVM
        for li in classes_split:
            for i,c in enumerate(li):
                threads.append(Thread(target=svm_authentication_train,args=(c,train,train_labels,que)))
                threads[-1].start()
            
            # Collect training thread results
            _ = [ t.join() for t in threads ]
            while not que.empty():
                (c_idx,svm) = que.get()
                c_idxes.append(c_idx)
                class_svms.append(svm)

        ### TESTING
        y_prob = np.array([])
        y_true = np.array([])
        threads = []
        que = Queue()
        for li in classes_split:
            for i,c in enumerate(li):
                c_idx = c_idxes.index(c)
                threads.append(Thread(target=svm_authentication_test,args=(c,class_svms[c_idx],test,test_labels,que)))
                threads[-1].start()

            # Collect testing thread results
            _ = [ t.join() for t in threads ]
            while not que.empty():
                result = que.get()
                y_prob = np.append(y_prob,result[0])
                y_true = np.append(y_true,result[1])
           
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=0)
        fnr = 1 - tpr
        tnr = 1 - fpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        idx = [i for i,v in enumerate(fpr) if v <= 0.001 and i > 0]
        min_idx = np.argmin(fnr[idx])
        idx = idx[min_idx]

        threshold = thresholds[idx]
        y_pred = np.zeros(y_true.shape)
        y_pred[y_prob < threshold] = 1
        acc = metrics.accuracy_score(y_true, y_pred)

        if not idx:
            FAR = np.append(FAR,fpr[1])
            FRR = np.append(FRR,fnr[1])
        else:
            FAR = np.append(FAR,fpr[idx])
            FRR = np.append(FRR,fnr[idx])
        ACC = np.append(ACC,acc)
        EER = np.append(EER,eer)
        EER_thresh = np.append(EER_thresh,eer_threshold)

        #print("FAR: " + str(FAR))
        #print("FAR: " + str(FAR))
        #print("ACC: " + str(ACC))
        #print("EER: " + str(EER))
        

    # Print results
    print(data_path)
    print("--------------------------------------------------------------------------------------")
    print("Authentication Results:")
    print("Threshold: " + str(np.mean(EER_thresh)) + "   " +
    "FAR: " + str(np.mean(FAR)) + " (+/- " + str(np.std(FAR)) + ")   " +
    "FRR: " + str(np.mean(FRR)) + " (+/- " + str(np.std(FRR)) + ")   " +
    "ACC: " + str(np.mean(ACC)) + " (+/- " + str(np.std(ACC)) + ")   " +
    "EER: " + str(np.mean(EER)) + " (+/- " + str(np.std(EER)) + ")")


# Get the arguments list 
argv = str(sys.argv)
print(str(argv))

argc = len(sys.argv)
if argc < 2:
    sys.exit("Datafile argument is required!")

data_path = sys.argv[1]
print(data_path)

print("Loading data")
data = np.loadtxt(data_path)
print("Loading data_flip")
data_flip = np.loadtxt(data_path[:-4] + "_flip.txt")
print("\n\n")


labels =  data[:,-1]
data = data[:,:-1]
data_flip = data_flip[:,:-1]

svm_authentication(data_path,data,data_flip,labels)


