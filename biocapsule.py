# BioCapsule Generation script
import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np


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


def biocapsule_unique_rs(database):
    cur_path = os.path.dirname(__file__)
    user_path = os.path.join(cur_path, 'data', database + '.npz')
    user_flip_path = os.path.join(cur_path, 'data', database + '_flip.npz')
    rs_path = os.path.join(cur_path, 'data', 'rs.npz')

    user_data = np.load(user_path)['arr_0']
    user_flip_data = np.load(user_flip_path)['arr_0']
    rs_data = np.load(rs_path)['arr_0']

    user_y = user_data[:, -1]
    user_data = user_data[:, :-1]
    user_flip_data = user_flip_data[:, :-1]

    rs_y = rs_data[:, -1]
    rs_data = rs_data[:, :-1]

    generated_bcs = np.empty((0, 513))
    generated_flip_bcs = np.empty((0, 513))

    for i, y in enumerate(user_y):
        print('Unique: ' + str(i + 1) + '/' + str(user_y.shape[0]))
        user_feature = user_data[i, :]
        user_flip_feature = user_flip_data[i, :]

        rs_idx = int(y - 1)
        rs_feature = rs_data[rs_idx, :]

        user_signature = signature_extraction(user_feature)
        rs_signature = signature_extraction(rs_feature)
        user_key = key_generation(user_signature)
        rs_key = key_generation(rs_signature)
        bc = np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)

        user_flip_signature = signature_extraction(user_flip_feature)
        user_flip_key = key_generation(user_flip_signature)
        bc_flip = np.multiply(user_flip_feature, rs_key) + np.multiply(rs_feature, user_flip_key)

        bc = np.reshape(np.append(bc, y), (1, 513))
        bc_flip = np.reshape(np.append(bc_flip, y), (1, 513))

        generated_bcs = np.vstack([generated_bcs, bc])
        generated_flip_bcs = np.vstack([generated_flip_bcs, bc_flip])

    bcs_path = os.path.join(cur_path, 'data', database + '_bc_uni.npz')
    bcs_flip_path = os.path.join(cur_path, 'data', database + '_flip_bc_uni.npz')

    np.savez_compressed(bcs_path, generated_bcs)
    np.savez_compressed(bcs_flip_path, generated_flip_bcs)


def biocapsule_same_rs(database):
    cur_path = os.path.dirname(__file__)
    user_path = os.path.join(cur_path, 'data', database + '.npz')
    user_flip_path = os.path.join(cur_path, 'data', database + '_flip.npz')
    rs_path = os.path.join(cur_path, 'data', 'rs.npz')

    user_data = np.load(user_path)['arr_0']
    user_flip_data = np.load(user_flip_path)['arr_0']
    rs_data = np.load(rs_path)['arr_0']

    user_y = user_data[:, -1]
    user_data = user_data[:, :-1]
    user_flip_data = user_flip_data[:, :-1]

    rs_y = rs_data[:, -1]
    rs_data = rs_data[:, :-1]

    rs_idx = 0
    rs_feature = rs_data[rs_idx, :]
    rs_signature = signature_extraction(rs_feature)
    rs_key = key_generation(rs_signature)

    generated_bcs = np.empty((0, 513))
    generated_flip_bcs = np.empty((0, 513))

    for i, y in enumerate(user_y):
        print('Same: ' + str(i + 1) + '/' + str(user_y.shape[0]))
        user_feature = user_data[i, :]
        user_flip_feature = user_flip_data[i, :]

        user_signature = signature_extraction(user_feature)
        user_flip_signature = signature_extraction(user_flip_feature)

        user_key = key_generation(user_signature)
        user_flip_key = key_generation(user_flip_signature)

        bc = np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)
        bc_flip = np.multiply(user_flip_feature, rs_key) + np.multiply(rs_feature, user_flip_key)

        bc = np.reshape(np.append(bc, y), (1, 513))
        bc_flip = np.reshape(np.append(bc_flip, y), (1, 513))

        generated_bcs = np.vstack([generated_bcs, bc])
        generated_flip_bcs = np.vstack([generated_flip_bcs, bc_flip])

    bcs_path = os.path.join(cur_path, 'data', database + '_bc_same.npz')
    bcs_flip_path = os.path.join(cur_path, 'data', database + '_flip_bc_same.npz')

    np.savez_compressed(bcs_path, generated_bcs)
    np.savez_compressed(bcs_flip_path, generated_flip_bcs)

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to use in biocapsule generation')
args = vars(parser.parse_args())

# Perform feature extraction
biocapsule_unique_rs(args['database'])
biocapsule_same_rs(args['database'])
