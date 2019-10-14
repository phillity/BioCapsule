import os
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from biocapsule import BioCapsuleGenerator


tf.compat.v1.set_random_seed(42)
np.random.seed(42)


def get_mlp(input_shape, output_shape):
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
    # (4) FC - BN - SoftMax
    fc4 = Dense(output_shape)(do3)
    bn4 = BatchNormalization()(fc4)
    predictions = Activation("softmax")(bn4)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    return model


def train_mlp(X_train, y_train, mlp):
    class_weights = compute_class_weight(
        "balanced", np.unique(np.argmax(y_train, axis=1)), np.argmax(y_train, axis=1))
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    es = EarlyStopping(monitor="val_loss", mode="min",
                       verbose=0, patience=100, restore_best_weights=True)
    mlp.fit(x=X_train, y=y_train, epochs=100000, batch_size=X_train.shape[0],
            verbose=0, callbacks=[es], validation_data=[X_val, y_val],
            class_weight=class_weights)
    return mlp


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
        bc_flip[i] = np.append(bc_gen.biocapsule(user_feat_flip[i, :], rs_feat[0]), y)
    return bc, bc_flip


def identification(dataset, feat_mode, bc_mode):
    path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.abspath(os.path.join(path, "..", feat_mode + "_data", dataset))
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
    ACC, FAR, FRR, PRE, REC, F1M = [], [], [], [], [], []
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(dataset + " " + feat_mode + " " + bc_mode + " -- fold " + str(k))
        X_train, X_train_flip, X_test = X[train_index], X_flip[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = np.vstack([X_train, X_train_flip])
        y_train = np.hstack([y_train, y_train])

        n_values = np.max(y_train.astype(int) - 1) + 1
        y_train = np.eye(n_values)[y_train.astype(int) - 1]
        y_test = np.eye(n_values)[y_test.astype(int) - 1]

        mlp = get_mlp(X_train.shape[1], y_train.shape[1])
        mlp = train_mlp(X_train, y_train, mlp)

        y_pred = mlp.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        cnf_matrix = confusion_matrix(y_test, y_pred)

        FP = np.sum(cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
        FN = np.sum(cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
        TP = np.sum(np.diag(cnf_matrix)).astype(float)
        TN = 0.
        for i in range(cnf_matrix.shape[0]):
            temp = np.delete(cnf_matrix, i, 0)
            temp = np.delete(temp, i, 1)
            TN += float(np.sum(np.sum(temp)))

        ACC.append((TP + TN) / (TP + TN + FP + FN))
        FAR.append(FP / (FP + TN))
        FRR.append(FN / (FN + TP))
        PRE.append(TP / (TP + FP))
        REC.append(TP / (TP + FN))
        F1M.append(2 * TP / (2 * TP + FP + FN))

    out_path = os.path.abspath(os.path.join(
        path, "..", "results", "identification_" + dataset + "_" + feat_mode + "_" + bc_mode + ".txt"))
    with open(out_path, "w") as out_file:
        out_file.write(dataset + " " + feat_mode + " " + bc_mode + "\n")
        out_file.write("ACC -- " + str(np.average(ACC) * 100) + "\n")
        out_file.write("FAR -- " + str(np.average(FAR) * 100) + "\n")
        out_file.write("FRR -- " + str(np.average(FRR) * 100) + "\n")
        out_file.write("PRE -- " + str(np.average(PRE) * 100) + "\n")
        out_file.write("REC -- " + str(np.average(REC) * 100) + "\n")
        out_file.write("F1M -- " + str(np.average(F1M) * 100) + "\n")
        out_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in identification")
    parser.add_argument("-f", "--feat_mode", required=True, choices=["arcface", "facenet"],
                        help="feat_mode to use in identification")
    parser.add_argument("-b", "--bc_mode", required=True, choices=["underlying", "same"],
                        help="bc_mode to use in identification")
    args = vars(parser.parse_args())

    identification(args["dataset"], args["feat_mode"], args["bc_mode"])
