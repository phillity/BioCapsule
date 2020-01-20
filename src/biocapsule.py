import os
import numpy as np
from scipy.signal import convolve2d
from argparse import ArgumentParser
from utils import draw_progress


class BioCapsuleGenerator:
    def __signature_extraction(self, feature):
        lvl1 = convolve2d(feature.reshape(32, 16), np.ones(
            (5, 5)) / 25., mode="same", boundary="wrap")
        lvl2 = feature.reshape(32, 16) - lvl1
        signature = np.around(np.average(lvl2, axis=1) * 100.).astype(int) % 9
        return signature

    def __key_generation(self, signature):
        key = np.empty((0,))
        for sig in signature:
            np.random.seed(sig)
            key = np.append(key, np.random.choice(2, 16))
        key = (key * 2) - 1
        return key

    def biocapsule(self, user_feature, rs_feature):
        user_signature = self.__signature_extraction(user_feature)
        user_key = self.__key_generation(user_signature)
        rs_signature = self.__signature_extraction(rs_feature)
        rs_key = self.__key_generation(rs_signature)
        return np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)


def biocapsule_dataset(user_data, user_data_flip, rs_data):
    bc_gen = BioCapsuleGenerator()

    user_y = user_data[:, -1]
    user_feat = user_data[:, :-1]
    user_feat_flip = user_data_flip[:, :-1]

    rs_y = rs_data[:, -1]
    rs_feat = rs_data[:, :-1]

    bc = np.zeros((user_feat.shape[0], 513))
    bc_flip = np.zeros((user_feat_flip.shape[0], 513))

    for i, y in enumerate(user_y):
        draw_progress("BC Generation", float(image_cnt + 1) / len(file_cnt))

        bc[i] = np.append(bc_gen.biocapsule(user_feat[i, :], rs_feat[0]), y)
        bc_flip[i] = np.append(bc_gen.biocapsule(
            user_feat_flip[i, :], rs_feat[0]), y)

    return bc, bc_flip


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--user_dataset", required=True,
                        help="user feature dataset to use in biocapsule generation")
    parser.add_argument("-r", "--rs_dataset", required=True,
                        help="rs feature dataset to use in biocapsule generation")
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    args = vars(parser.parse_args())

    user_data = np.load(os.path.join(os.path.abspath(
        ""), "data", args["user_dataset"] + "_" + args["method"] + "_feat.npz"))["arr_0"]
    user_data_flip = np.load(os.path.join(os.path.abspath(
        ""), "data", args["user_dataset"] + "_" + args["method"] + "_feat_flip.npz"))["arr_0"]
    rs_data = np.load(os.path.join(os.path.abspath(
        ""), "data", args["rs_dataset"] + "_" + args["method"] + "_feat.npz"))["arr_0"]

    features, features_flip = biocapsule_dataset(
        user_data, user_data_flip, rs_data)

    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "data", args["dataset"] + "_" + args["method"] + "_bc.npz"), features)
    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "data", args["dataset"] + "_" + args["method"] + "_bc_flip.npz"), features_flip)
