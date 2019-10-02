import os
import numpy as np


__all__ = ["biocapsule"]


def signature_extraction(feature):
    feature_grid = np.reshape(feature, (32, 16))
    signature = np.zeros(feature_grid.shape)

    inc = 3
    B_size = np.floor(np.array([3, 3]) / 2).astype(int)
    T_size = np.floor(np.array([5, 5]) / 2).astype(int)

    for r in range(0, feature_grid.shape[0], inc):
        for c in range(0, feature_grid.shape[1], inc):
            # Get T window
            T_r_s = max([0, r - T_size[0]])
            T_r_e = min([feature_grid.shape[0], r + T_size[0]])
            T_c_s = max([0, c - T_size[1]])
            T_c_e = min([feature_grid.shape[1], c + T_size[0]])
            T = feature_grid[T_r_s:T_r_e + 1, T_c_s:T_c_e + 1]

            # Get T window mean
            T_mean = np.mean(T)

            # Get B window
            B_r_s = max([0, r - B_size[0]])
            B_r_e = min([feature_grid.shape[0], r + B_size[0]])
            B_c_s = max([0, c - B_size[1]])
            B_c_e = min([feature_grid.shape[1], c + B_size[0]])
            B = feature_grid[B_r_s:B_r_e + 1, B_c_s:B_c_e + 1]

            signature[B_r_s:B_r_e + 1, B_c_s:B_c_e + 1] = np.abs(B - T_mean)
    signature = np.around(np.average(signature, axis=1) * 100).astype(int)
    return signature


def key_generation(signature):
    key = np.empty((0,))
    for sig in signature:
        np.random.seed(sig)
        key = np.append(key, np.random.choice(2, 16))
    key = (key * 2) - 1
    return key


def biocapsule(user_feature, rs_feature):
    user_signature = signature_extraction(user_feature)
    user_key = key_generation(user_signature)
    rs_signature = signature_extraction(rs_feature)
    rs_key = key_generation(rs_signature)
    return np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)
