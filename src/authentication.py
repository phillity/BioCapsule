import os
import numpy as np
from queue import Queue
from threading import Thread
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from biocapsule import BioCapsuleGenerator


tf.compat.v1.set_random_seed(42)
np.random.seed(42)


def drawProgressBar(text, percent, barLen=20):
    print(text + " -- [{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100), end="\r")


def get_mlp(input_shape):
    inputs = Input(shape=(input_shape,))
    # (1) FC - BN - ReLU - BN
    fc1 = Dense(256)(inputs)
    bn1 = BatchNormalization()(fc1)
    ru1 = Activation("relu")(bn1)
    do1 = Dropout(0.2)(ru1)
    # (2) FC - BN - ReLU - BN
    fc2 = Dense(128)(do1)
    bn2 = BatchNormalization()(fc2)
    ru2 = Activation("relu")(bn2)
    do2 = Dropout(0.2)(ru2)
    # (3) FC - BN - ReLU - BN
    fc3 = Dense(64)(do2)
    bn3 = BatchNormalization()(fc3)
    ru3 = Activation("relu")(bn3)
    do3 = Dropout(0.2)(ru3)
    # (4) FC - BN - Sigmoid
    fc4 = Dense(1)(do3)
    bn4 = BatchNormalization()(fc4)
    predictions = Activation("sigmoid")(bn4)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["acc"])
    return model


def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train_test_mlp(c, X_train, y_train, X_test, y_test, queue):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.keras.backend.set_session(sess)

        mlp = get_mlp(X_train.shape[1])
        mlp._make_predict_function()
        class_weights = compute_class_weight(
            "balanced", np.unique(y_train), y_train)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)

        y_train_bin = np.zeros(y_train.shape)
        y_train_bin[y_train == c] = 1
        y_val_bin = np.zeros(y_val.shape)
        y_val_bin[y_val == c] = 1
        y_test_bin = np.zeros(y_test.shape)
        y_test_bin[y_test == c] = 1

        es = EarlyStopping(monitor="val_acc", mode="max",
                           verbose=0, patience=100, restore_best_weights=True)
        mlp.fit(x=X_train, y=y_train_bin, epochs=100000, batch_size=X_train.shape[0],
                verbose=0, callbacks=[es], validation_data=[X_val, y_val_bin],
                class_weight=class_weights)

        y_prob = mlp.predict(X_test)[:, 0]
        queue.put((c, y_test_bin, y_prob))


def get_biocapsules(data, data_flip, rs_data):
    user_y = data[:, -1]
    user_feat = data[:, :-1]
    user_feat_flip = data[:, :-1]

    rs_y = rs_data[:, -1]
    rs_feat = rs_data[:, :-1]

    bc = np.zeros((user_feat.shape[0], 513))
    bc_flip = np.zeros((user_feat_flip.shape[0], 513))
    bc_gen = BioCapsuleGenerator()
    for i, y in enumerate(user_y):
        bc[i] = np.append(bc_gen.biocapsule(user_feat[i, :], rs_feat[0]), y)
        bc_flip[i] = np.append(bc_gen.biocapsule(
            user_feat_flip[i, :], rs_feat[0]), y)
    return bc, bc_flip


def authentication(dataset, feat_mode, bc_mode, thread_cnt):
    path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.abspath(os.path.join(
        path, "..", feat_mode + "_data", dataset))
    data = np.load(dataset_path + "_feat.npz")["arr_0"]
    data_flip = np.load(dataset_path + "_feat_flip.npz")["arr_0"]

    if bc_mode != "underlying":
        rs_path = os.path.abspath(os.path.join(
            path, "..", feat_mode + "_data", "lfw"))
        rs_data = np.load(rs_path + "_feat.npz")["arr_0"]
        data, data_flip = get_biocapsules(data, data_flip, rs_data)

    X = data[:, :-1]
    X_flip = data_flip[:, :-1]
    y = data[:, -1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ACC, FAR, FRR, PRE, REC, F1M, EER = [], [], [], [], [], [], []
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(dataset + " " + feat_mode + " " + bc_mode + " -- fold " + str(k))
        X_train, X_train_flip, X_test = X[train_index], X_flip[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = np.vstack([X_train, X_train_flip])
        y_train = np.hstack([y_train, y_train])

        classes = np.unique(y_train)
        classes_split = list(split_list(classes.tolist(), thread_cnt))
        y_prob = np.zeros((X_test.shape[0] * classes.shape[0],))
        y_true = np.zeros((X_test.shape[0] * classes.shape[0],))
        for li in classes_split:
            threads = []
            queue = Queue()
            for c in li:
                threads.append(Thread(target=train_test_mlp, args=(
                    c, X_train, y_train, X_test, y_test, queue)))
                threads[-1].start()
            _ = [t.join() for t in threads]
            while not queue.empty():
                drawProgressBar("fold " + str(k), c / len(classes))
                (c_idx, true, prob) = queue.get()
                c_idx = int(c_idx - 1) * X_test.shape[0]
                y_true[c_idx: c_idx + X_test.shape[0]] = true
                y_prob[c_idx: c_idx + X_test.shape[0]] = prob
        print("")

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        EER.append(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))

        TN, FP, FN, TP = confusion_matrix(y_true, np.around(y_prob)).ravel()
        ACC.append((TP + TN) / (TP + TN + FP + FN))
        FAR.append(FP / (FP + TN))
        FRR.append(FN / (FN + TP))
        PRE.append(TP / (TP + FP))
        REC.append(TP / (TP + FN))
        F1M.append(2 * TP / (2 * TP + FP + FN))

    out_path = os.path.abspath(os.path.join(
        path, "..", "results", "authentication_" + dataset + "_" + feat_mode + "_" + bc_mode + ".txt"))
    with open(out_path, "w") as out_file:
        out_file.write(dataset + " " + feat_mode + " " + bc_mode + "\n")
        out_file.write("ACC -- " + str(np.average(ACC) * 100) + "\n")
        out_file.write("FAR -- " + str(np.average(FAR) * 100) + "\n")
        out_file.write("FRR -- " + str(np.average(FRR) * 100) + "\n")
        out_file.write("PRE -- " + str(np.average(PRE) * 100) + "\n")
        out_file.write("REC -- " + str(np.average(REC) * 100) + "\n")
        out_file.write("F1M -- " + str(np.average(F1M) * 100) + "\n")
        out_file.write("EER -- " + str(np.average(EER) * 100) + "\n")
        out_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in authentication")
    parser.add_argument("-f", "--feat_mode", required=True, choices=["arcface", "facenet"],
                        help="feat_mode to use in authentication")
    parser.add_argument("-b", "--bc_mode", required=True, choices=["underlying", "same"],
                        help="bc_mode to use in authentication")
    parser.add_argument("-t", "--thread", required=False, type=int, default=1,
                        help="thread count to use in authentication")
    args = vars(parser.parse_args())

    authentication(args["dataset"], args["feat_mode"], args["bc_mode"], args["thread"])
