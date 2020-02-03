import os
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import sensitivity_score, specificity_score
from argparse import ArgumentParser
from model import get_mlp, train_mlp, predict_mlp


np.random.seed(42)


def identification(method, rs_cnt, batch_size):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "{}_{}_vggface2.txt".format(method, rs_cnt)), "w")
    data_file = os.path.join(os.path.abspath(""), "data", "vggface2_{}_dataset.hdf5".format(method))
    train, val, test = [h5py.File(data_file, "r")[part] for part in ["train", "val", "test"]]
    rs = np.load(os.path.join(os.path.abspath(""), "data",
                              "rs_{}_feat.npz".format(method)))["arr_0"]

    mlp = get_mlp(512, np.unique(train[:, -1].astype(int) - 1).shape[0])

    mlp = train_mlp(mlp, train, val, rs, rs_cnt, method, batch_size)

    y_prob = predict_mlp(mlp, test, rs, rs_cnt)
    y_pred = np.argmax(y_prob, axis=-1)
    y_true = test[:, -1].astype(int) - 1

    out_file.write("ACC -- {:.6f}\n".format(accuracy_score(y_true, y_pred)))
    out_file.write("FPR -- {:.6f}\n".format(
        1 - sensitivity_score(y_true, y_pred, average="micro")))
    out_file.write("FRR -- {:.6f}\n".format(
        1 - specificity_score(y_true, y_pred, average="micro")))
    out_file.write(
        "PRE -- {:.6f}\n".format(precision_score(y_true, y_pred, average="micro")))
    out_file.write(
        "REC -- {:.6f}\n".format(recall_score(y_true, y_pred, average="micro")))
    out_file.write(
        "F1 -- {:.6f}\n".format(f1_score(y_true, y_pred, average="micro")))
    out_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-rs", "--rs_cnt", default=0, type=int, choices=range(0, 11),
                        help="number of rs in biocapsule generation")
    parser.add_argument("-bs", "--batch_size", default=512, type=int,
                        help="batch size to use in training")
    args = vars(parser.parse_args())

    identification(args["method"], args["rs_cnt"], args["batch_size"])
