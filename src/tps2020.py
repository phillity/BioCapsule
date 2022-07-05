import datetime
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


def experiment(
    dataset: str, mode: str, role_dist: str, flipped: bool, gpu: int
):
    if mode == "under":
        fi = open(f"results/tps2020_{dataset}_under.txt", "w")
    else:
        fi = open(f"results/tps2020_{dataset}_bc_{role_dist}.txt", "w")

    print(f"Computing Features: {datetime.datetime.now()}")

    # If already extracted features, use the precomputed features
    if not flipped and os.path.exists(
        f"data/{dataset}_arcface_mtcnn_feat.npz"
    ):
        features = np.load(f"data/{dataset}_arcface_mtcnn_feat.npz")["arr_0"]

    elif (
        flipped
        and os.path.exists(f"data/{dataset}_arcface_mtcnn_feat.npz")
        and os.path.exists(f"data/{dataset}_arcface_mtcnn_flip_feat.npz")
    ):
        features = np.load(f"data/{dataset}_arcface_mtcnn_feat.npz")["arr_0"]
        flipped_features = np.load(
            f"data/{dataset}_arcface_mtcnn_flip_feat.npz"
        )["arr_0"]

    else:
        extract_dataset(dataset, "arcface", "mtcnn", flipped, gpu)

        features = np.load(f"data/{dataset}_arcface_mtcnn_feat.npz")["arr_0"]

        if flipped:
            flipped_features = np.load(
                f"data/{dataset}_arcface_mtcnn_flip_feat.npz"
            )["arr_0"]

    print(f"Done Computing Features {datetime.datetime.now()}")

    # Remove all subjects with less than 5 images from LFW dataset
    if dataset == "lfw":
        features = filter_lfw(features)

        if flipped:
            flipped_features = filter_lfw(flipped_features)

    # Compute BioCapsules for features using Reference Subjects
    if mode == "bc":
        print(f"Computing BCs: {datetime.datetime.now()}")

        rs_features = get_rs_features()
        rs_map = rs_rbac(len(np.unique(features[:, -1])), role_dist)
        cnts = np.unique(rs_map, return_counts=True)[1]
        for i, cnt in enumerate(cnts):
            fi.write(f"Role {i + 1} -- {cnt} Subjects\n")

        bcs = get_bcs(features, rs_features)

        if flipped:
            flipped_bcs = get_bcs(flipped_features, rs_features)

        print(f"Done Computing BCs: {datetime.datetime.now()}")

    # TP, FP, FN, FP
    conf_mat = np.zeros((4,))

    print(f"Starting KFold Experiment: {datetime.datetime.now()}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(
        skf.split(features[:, :-1], features[:, -1])
    ):
        print(f"Fold {k} : {datetime.datetime.now()}")

        if mode == "under":
            X_train, y_train = (
                features[:, :-1][train_index],
                features[:, -1][train_index],
            )
            X_test, y_test = (
                features[:, :-1][test_index],
                features[:, -1][test_index],
            )

            if flipped:
                X_train_flip, y_train_flip = (
                    flipped_features[:, :-1][train_index],
                    flipped_features[:, -1][train_index],
                )

                X_train = np.vstack([X_train, X_train_flip])
                y_train = np.hstack([y_train, y_train_flip])

            clf = LogisticRegression(
                class_weight="balanced", random_state=42
            ).fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            # Aggregate prediction results with respect to each subject
            for subject_id in np.unique(y_test):
                y_test_subject = (y_test == subject_id).astype(int)
                y_pred_subject = (y_pred == subject_id).astype(int)
                conf_mat += confusion_matrix(
                    y_test_subject, y_pred_subject
                ).ravel()

        else:
            train_mask = np.zeros(
                (train_index.shape[0] + test_index.shape[0]), dtype=bool,
            )
            train_mask[train_index] = True
            train_mask = np.concatenate(
                [train_mask for _ in range(np.unique(rs_map).shape[0])]
            )

            test_mask = np.zeros(
                (train_index.shape[0] + test_index.shape[0]), dtype=bool,
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

            if flipped:
                X_train_flip, y_train_subject_flip, y_train_rs_flip = (
                    flipped_bcs[:, :-2][train_mask],
                    flipped_bcs[:, -2][train_mask],
                    flipped_bcs[:, -1][train_mask],
                )

                X_train = np.vstack([X_train, X_train_flip])
                y_train_subject = np.hstack(
                    [y_train_subject, y_train_subject_flip]
                )
                y_train_rs = np.hstack([y_train_rs, y_train_rs_flip])

            for subject_id in np.unique(y_test_subject):
                rs_id = float(rs_map[int(subject_id) - 1] + 1)

                y_train_binary = np.zeros(X_train.shape[0])
                y_train_binary[
                    np.logical_and(
                        y_train_subject == subject_id, y_train_rs == rs_id,
                    )
                ] = 1

                y_test_binary = np.zeros(X_test.shape[0])
                y_test_binary[
                    np.logical_and(
                        y_test_subject == subject_id, y_test_rs == rs_id,
                    )
                ] = 1

                # We assume the RS is automatically specified when
                # a username is given during an authentication attempt
                X_train_bianry = X_train[y_train_rs == rs_id]
                y_train_binary = y_train_binary[y_train_rs == rs_id]
                X_test_bianry = X_test[y_test_rs == rs_id]
                y_test_binary = y_test_binary[y_test_rs == rs_id]

                clf = LogisticRegression(
                    class_weight="balanced", random_state=42
                ).fit(X_train_bianry, y_train_binary)

                y_pred = clf.predict(X_test_bianry)

                conf_mat += confusion_matrix(y_test_binary, y_pred).ravel()

    print(f"Finished KFold Experiment: {datetime.datetime.now()}")

    # (tn + tp) / (tn + fp + fn + tp)
    acc = (conf_mat[0] + conf_mat[3]) / np.sum(conf_mat)
    # fp / (tn + fp)
    far = conf_mat[1] / (conf_mat[0] + conf_mat[1])
    # fn / (fn + tp)
    frr = conf_mat[2] / (conf_mat[2] + conf_mat[3])

    fi.write(f"Dataset -- {dataset}\n")
    fi.write(f"BC  -- {mode}\n")
    fi.write(f"RS  -- {role_dist}\n")
    fi.write(f"TN -- {conf_mat[0]}\n")
    fi.write(f"TP -- {conf_mat[3]}\n")
    fi.write(f"FP -- {conf_mat[1]}\n")
    fi.write(f"FN -- {conf_mat[2]}\n")
    fi.write(f"ACC -- {acc:.6f}\n")
    fi.write(f"FAR -- {far:.6f}\n")
    fi.write(f"FRR -- {frr:.6f}\n")
    fi.close()


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
        "-f",
        "--flipped",
        required=False,
        action="store_true",
        default=False,
        help="use flipped features in experiment",
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

    experiment(
        args["dataset"],
        args["mode"],
        args["role_dist"],
        args["flipped"],
        args["gpu"],
    )
