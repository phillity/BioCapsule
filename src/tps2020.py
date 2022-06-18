import datetime
from inspect import stack
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from biocapsule import BioCapsuleGenerator
from face import ArcFace, extract_dataset

# Set random seed for reproducibility
np.random.seed(42)


def filter_lfw(features: np.ndarray) -> np.ndarray:
    """Filter out subjects from LFW dataset with less than
    5 corresponding images.

    """
    y = np.unique(features[:, -1])
    mask = np.ones(features[:, -1].shape, dtype=bool)

    # Remove subjects with less than 5 images
    for y_i in y:
        if features[features[:, -1] == y_i].shape[0] < 5:
            idxes = np.where(features[:, -1] == y_i)
            mask[idxes] = False

    features = features[mask]

    # Update subject ids
    y_map = {}
    y = np.unique(features[:, -1])
    for i, y_i in enumerate(y):
        y_map[y_i] = i + 1

    # Update subject ids in feature vectors
    for i in range(features[:, -1].shape[0]):
        features[i, -1] = y_map[features[i, -1]]

    return features


def get_rs_features():
    """Return ArcFace features for 6 predetermined Reference Subjects
    (RSs) used in this experiment for reproducibility.

    """
    arcface = ArcFace()

    rs_subjects = sorted(os.listdir("images/rs"))
    rs_subjects = rs_subjects[4:]

    rs_features = np.zeros((6, 512))
    for s_id, subject in enumerate(rs_subjects):
        for image in os.listdir(f"images/rs/{subject}"):
            img = cv2.imread(f"images/rs/{subject}/{image}")
            feature = arcface.extract(img)
            rs_features[s_id] = feature

    return rs_features


def rs_rbac(subject_cnt: int, distribution: str) -> np.ndarray:
    """Generate RS-RBAC by assigning each subject a corresponding
    RS according to specified input distribution.

    Parameters
    ----------
    subject_cnt: int
        Number of subjects to assign to RSs
    distribution: str
        RS assignment distribution, either bal/unbal
        for balanced/unbalanced distribution

    Returns
    -------
    np.ndarray
        Array containing each subject's RS assignment

    """
    if distribution == "unbal":
        rs_map = np.random.choice(
            6, subject_cnt, p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
        )
    else:
        rs_map = np.random.choice(6, subject_cnt)

    return rs_map


def get_bcs(features: np.ndarray, rs_features: np.ndarray) -> np.ndarray:
    """Compute BioCapsules for each given feature and Reference Subject.

    Parameters
    ----------
    features: np.ndarray
        Array of ArcFace feature vectors for dataset images. Last column
        will contain a subject id, resulting in shape of (number of images)x513
    rs_features: np.ndarray
        Array of ArcFace feature vectors for RS images. Will be of shape
        (number of RSs)x512

    Returns
    -------
    np.ndarray
        BioCapsules generated from combining each feature vector with
        each RS. Will be of shape (number of RSs * number of images)x514

    """
    bc_gen = BioCapsuleGenerator()

    bcs = []

    for i in range(rs_features.shape[0]):
        bcs.append(
            np.hstack(
                [
                    bc_gen.biocapsule_batch(features[:, :-1], rs_features[i]),
                    features[:, -1][:, np.newaxis],
                    np.ones(features[:, -1][:, np.newaxis].shape) * (i + 1),
                ]
            )
        )

    bcs = np.vstack(bcs)

    return bcs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["lfw", "gtdb"],
        help="dataset to use in experiment",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["under", "bc"],
        help="feature mode to use in experiment",
    )
    parser.add_argument(
        "-r",
        "--role_dist",
        required=False,
        choices=["bal", "unbal"],
        default="unbal",
        help="role distribution to use in experiment",
    )
    parser.add_argument(
        "-gpu",
        "--gpu",
        required=False,
        type=int,
        default=-1,
        help="gpu to use in feature extraction",
    )
    args = vars(parser.parse_args())

    if args["mode"] == "under":
        fi = open("results/tps2020_{}_under.txt".format(args["dataset"]), "w")
    else:
        fi = open(
            "results/tps2020_{}_bc_{}.txt".format(
                args["dataset"], args["role_dist"]
            ),
            "w",
        )

    print(f"Computing Features: {datetime.datetime.now()}")

    # If already extracted features, use the precomputed features
    if args["dataset"] == "lfw" and os.path.exists(
        "data/lfw_arcface_feat.npz"
    ):
        features = np.load("data/lfw_arcface_feat.npz")["arr_0"]
    elif args["dataset"] == "gtdb" and os.path.exists(
        "data/gtdb_arcface_feat.npz"
    ):
        features = np.load("data/gtdb_arcface_feat.npz")["arr_0"]
    else:
        features = extract_dataset(args["dataset"], "arcface", args["gpu"])

    print(f"Done Computing Features {datetime.datetime.now()}")

    # Remove all subjects with less than 5 images from LFW dataset
    if args["dataset"] == "lfw":
        features = filter_lfw(features)

    # Comput BioCapsules for features using Reference Subjects
    if args["mode"] == "bc":
        print(f"Computing BCs: {datetime.datetime.now()}")

        rs_features = get_rs_features()
        rs_map = rs_rbac(len(np.unique(features[:, -1])), args["role_dist"])
        cnts = np.unique(rs_map, return_counts=True)[1]
        for i, cnt in enumerate(cnts):
            fi.write(f"Role {i + 1} -- {cnt} Subjects\n")

        bcs = get_bcs(features, rs_features)

        print(f"Done Computing BCs: {datetime.datetime.now()}")

    # TP, FP, FN, FP
    conf_mat = np.zeros((4,))

    print(f"Starting KFold Experiment: {datetime.datetime.now()}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(
        skf.split(features[:, :-1], features[:, -1])
    ):
        print(f"Fold {k} : {datetime.datetime.now()}")

        if args["mode"] == "under":
            X_train, y_train = (
                features[:, :-1][train_index],
                features[:, -1][train_index],
            )
            X_test, y_test = (
                features[:, :-1][test_index],
                features[:, -1][test_index],
            )

            clf = LogisticRegression(
                class_weight="balanced", random_state=42
            ).fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            # Aggregate prediction results with respect to each subject
            for subject_id in y_test:
                y_test_subject = (y_test == subject_id).astype(int)
                y_pred_subject = (y_pred == subject_id).astype(int)
                conf_mat += confusion_matrix(
                    y_test_subject, y_pred_subject
                ).ravel()

        else:
            train_mask = np.zeros(
                (train_index.shape[0] + test_index.shape[0]),
                dtype=bool,
            )
            train_mask[train_index] = True
            train_mask = np.concatenate(
                [train_mask for _ in range(np.unique(rs_map).shape[0])]
            )

            test_mask = np.zeros(
                (train_index.shape[0] + test_index.shape[0]),
                dtype=bool,
            )
            test_mask[test_index] = True
            test_mask = np.concatenate(
                [test_mask for _ in range(np.unique(rs_map).shape[0])]
            )

            X_train, y_train_subject, y_train_rs = (
                bcs[:, :-2][train_mask],
                bcs[:, -2][train_mask],
                bcs[:, -1][train_mask],
            )
            X_test, y_test_subject, y_test_rs = (
                bcs[:, :-2][test_mask],
                bcs[:, -2][test_mask],
                bcs[:, -1][test_mask],
            )

            for subject_id in np.unique(bcs[:, -2]):
                rs_id = float(rs_map[int(subject_id) - 1] + 1)

                y_train = np.zeros(X_train.shape[0])
                y_train[
                    np.logical_and(
                        y_train_subject == subject_id,
                        y_train_rs == rs_id,
                    )
                ] = 1

                y_test = np.zeros(X_test.shape[0])
                y_test[
                    np.logical_and(
                        y_test_subject == subject_id,
                        y_test_rs == rs_id,
                    )
                ] = 1

                clf = LogisticRegression(
                    class_weight="balanced", random_state=42
                ).fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                conf_mat += confusion_matrix(y_test, y_pred).ravel()

    print(f"Finished KFold Experiment: {datetime.datetime.now()}")

    # (tn + tp) / (tn + fp + fn + tp)
    acc = (conf_mat[0] + conf_mat[3]) / np.sum(conf_mat)
    # fp / (tn + fp)
    far = conf_mat[1] / (conf_mat[0] + conf_mat[1])
    # fn / (fn + tp)
    frr = conf_mat[2] / (conf_mat[2] + conf_mat[3])

    fi.write("Dataset -- {}\n".format(args["dataset"]))
    fi.write("BC  -- {}\n".format(args["mode"]))
    fi.write("RS  -- {}\n".format(args["role_dist"]))
    fi.write("TN -- {}\n".format(conf_mat[0]))
    fi.write("TP -- {}\n".format(conf_mat[3]))
    fi.write("FP -- {}\n".format(conf_mat[1]))
    fi.write("FN -- {}\n".format(conf_mat[2]))
    fi.write("ACC -- {:.6f}\n".format(acc))
    fi.write("FAR -- {:.6f}\n".format(far))
    fi.write("FRR -- {:.6f}\n".format(frr))
    fi.close()
