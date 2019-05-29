# Signature/Key Experiment script
import os
import sys
from argparse import ArgumentParser
import numpy as np
import matplotlib
from matplotlib import pyplot
from latex_format import latexify


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


def signature_key_experiment(database):
    cur_path = os.path.dirname(__file__)
    user_path = os.path.join(cur_path, 'data', database + '.npz')

    user_data = np.load(user_path)['arr_0']
    user_y = user_data[:, -1]
    user_data = user_data[:, :-1]

    generated_keys = np.empty((0, 513))
    generated_signatures = np.empty((0, 33))

    for i, y in enumerate(user_y):
        print('Sig/Key: ' + str(i + 1) + '/' + str(user_y.shape[0]))
        user_feature = user_data[i, :]
        user_signature = signature_extraction(user_feature)
        user_key = key_generation(user_signature)

        user_signature = np.reshape(np.append(user_signature, y), (1, 33))
        user_key = np.reshape(np.append(user_key, y), (1, 513))

        generated_signatures = np.vstack([generated_signatures, user_signature])
        generated_keys = np.vstack([generated_keys, user_key])

    sig_same_cmp = []
    key_same_cmp = []
    sig_diff_cmp = []
    key_diff_cmp = []
    for i in range(generated_signatures.shape[0]):
        print('Comp: ' + str(i + 1) + '/' + str(user_y.shape[0]))
        for j in range(generated_signatures.shape[0]):
            if j > i:
                sig_diff = np.where(generated_signatures[i, :] != generated_signatures[j, :])[0].shape[0]
                key_diff = np.where(generated_keys[i, :] != generated_keys[j, :])[0].shape[0]
                if user_y[i] == user_y[j]:
                    sig_same_cmp.append(sig_diff)
                    key_same_cmp.append(key_diff)
                else:
                    sig_diff_cmp.append(sig_diff)
                    key_diff_cmp.append(key_diff)
    sig_same_cmp = np.asarray(sig_same_cmp)
    key_same_cmp = np.asarray(key_same_cmp)
    sig_diff_cmp = np.asarray(sig_diff_cmp)
    key_diff_cmp = np.asarray(key_diff_cmp)

    np.random.seed(42)
    idx = np.random.choice(sig_diff_cmp.shape[0], sig_same_cmp.shape[0])
    sig_diff_cmp = sig_diff_cmp[idx]
    key_diff_cmp = key_diff_cmp[idx]

    latexify()
    if database == 'caltech':
        pyplot.title('Caltech Faces 1999 Signature Stability/Distinguishability')
    elif database == 'gt':
        pyplot.title('GeoTech Face Database Signature Stability/Distinguishability')
    else:
        pyplot.title('LFW Signature Stability/Distinguishability')
    pyplot.ylabel('Occurrences')
    pyplot.xlabel('Number of Differences in Compared Signatures')
    pyplot.hist([sig_same_cmp, sig_diff_cmp], np.arange(33), alpha=0.5, color=['blue', 'red'])
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(database + '_sig.pdf')
    pyplot.close()

    if database == 'caltech':
        pyplot.title('Caltech Faces 1999 Key Stability/Distinguishability')
    elif database == 'gt':
        pyplot.title('GeoTech Face Database Key Stability/Distinguishability')
    else:
        pyplot.title('LFW Key Stability/Distinguishability')
    pyplot.ylabel('Occurrences')
    pyplot.xlabel('Number of Differences in Compared Keys')
    pyplot.hist([key_same_cmp, key_diff_cmp], np.arange(0, 520, 10), alpha=0.5, label=['Intraclass Comparisons', 'Interclass Comparisons'], color=['blue', 'red'])
    pyplot.legend(loc='upper right')
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(database + '_key.pdf')


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to use in signature/key experiment')
args = vars(parser.parse_args())

# Perform signature/key experiment
signature_key_experiment(args['database'])
