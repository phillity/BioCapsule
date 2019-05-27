# BioCapsule Generation script
import cv2
import numpy as np
import os
import time


def signature_extraction(feature):
    feature_grid = np.reshape(feature, (32, 16))
    LTP = np.zeros(feature_grid.shape)

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

            LTP[B_r_s:B_r_e + 1, B_c_s:B_c_e + 1] = np.abs(B - T_mean)
    LTP = np.around(np.average(LTP, axis=1) * 100).astype(int)
    return LTP


def key_generation(signature):
    '''
    signature = np.abs(signature - np.mean(signature))
    signature = np.around(signature).astype(int)
    '''
    key = np.zeros((512,))
    idx = 0
    for i, s in enumerate(signature):
        np.random.seed(s)
        for j in range(16):
            key[idx] = np.random.rand()
            idx = idx + 1
    key = np.around(key)
    key = (key * 2) - 1
    return key


def biocapsule_unique_rs(user_database, user_database_flip, rs_database, 
                         database_name, mode, model_name):
    print('BC Unique RS Generation')
    generated_bcs = np.empty((0, 513))
    generated_bcs_flip = np.empty((0, 513))

    user_y = user_database[:, -1]
    user_database = user_database[:, :-1]
    user_database_flip = user_database_flip[:, :-1]

    rs_y = rs_database[:, -1]
    rs_database = rs_database[:, :-1]

    for i, y in enumerate(user_y):
        print('     Class ' + str(y))
        user_feature = user_database[i, :]
        user_feature_flip = user_database_flip[i, :]

        rs_idx = int(y - 1)
        rs_feature = rs_database[rs_idx, :]

        #start = time.time()
        user_signature = signature_extraction(user_feature)
        rs_signature = signature_extraction(rs_feature)
        user_key = key_generation(user_signature)
        rs_key = key_generation(rs_signature)
        bc = np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)
        #end = time.time()
        #print(end - start)

        user_signature_flip = signature_extraction(user_feature_flip)
        user_key_flip = key_generation(user_signature_flip)
        bc_flip = np.multiply(user_feature_flip, rs_key) + np.multiply(rs_feature, user_key_flip)

        bc = np.reshape(np.append(bc, y), (1, 513))
        bc_flip = np.reshape(np.append(bc_flip, y), (1, 513))

        generated_bcs = np.vstack([generated_bcs, bc])
        generated_bcs_flip = np.vstack([generated_bcs_flip, bc_flip])

    if mode == '':
        out_file = database_name + '_' + model_name + '_bc_uni.txt'
        out_file_flip = database_name + '_' + model_name + '_bc_uni_flip.txt'
    else:
        out_file = database_name + '_' + mode + '_' + model_name + '_bc_uni.txt'
        out_file_flip = database_name + '_' + mode + '_' + model_name + '_bc_uni_flip.txt'

    np.savetxt(out_file, generated_bcs)
    np.savetxt(out_file_flip, generated_bcs_flip)    


def biocapsule_same_rs(user_database, user_database_flip, rs_database, database_name, mode, model_name):
    print('BC Same RS Generation')
    generated_bcs = np.empty((0, 513))
    generated_bcs_flip = np.empty((0, 513))

    user_y = user_database[:, -1]
    user_database = user_database[:, :-1]
    user_database_flip = user_database_flip[:, :-1]

    rs_y = rs_database[:, -1]
    rs_database = rs_database[:, :-1]

    rs_idx = 0
    rs_feature = rs_database[rs_idx, :]
    rs_signature = signature_extraction(rs_feature)
    rs_key = key_generation(rs_signature)

    for i, y in enumerate(user_y):
        print('     Class ' + str(y))
        user_feature = user_database[i, :]
        user_feature_flip = user_database_flip[i, :]

        user_signature = signature_extraction(user_feature)
        user_signature_flip = signature_extraction(user_feature_flip)
       
        user_key = key_generation(user_signature)
        user_key_flip = key_generation(user_signature_flip)
        
        bc = np.multiply(user_feature, rs_key) + np.multiply(rs_feature, user_key)
        bc_flip = np.multiply(user_feature_flip, rs_key) + np.multiply(rs_feature, user_key_flip)

        bc = np.reshape(np.append(bc, y), (1, 513))
        bc_flip = np.reshape(np.append(bc_flip, y), (1, 513))

        generated_bcs = np.vstack([generated_bcs, bc])
        generated_bcs_flip = np.vstack([generated_bcs_flip, bc_flip])

    if mode == '':
        out_file = database_name + '_' + model_name + '_bc_same.txt'
        out_file_flip = database_name + '_' + model_name + '_bc_same_flip.txt'
    else:
        out_file = database_name + '_' + mode + '_' + model_name + '_bc_same.txt'
        out_file_flip = database_name + '_' + mode + '_' + model_name + '_bc_same_flip.txt'

    np.savetxt(out_file, generated_bcs)
    np.savetxt(out_file_flip, generated_bcs_flip)

database_name = 'cmu_pie_scifi'
mode = 'align'
'''
mode ='crop'
mode = ''
model_name = '20180402-114759'
'''
model_name = '20180408-102900'

if mode == '':
    print(database_name + '_' + model_name + '.txt')
    user_database = np.loadtxt('features//' + database_name + '//' + database_name + '_' + model_name + '.txt')
    user_database_flip = np.loadtxt('features//' + database_name + '//' + database_name + '_' + model_name + '_flip.txt')
else:
    print(database_name + '_' + mode + '_' + model_name + '.txt')
    user_database = np.loadtxt('features//' + database_name + '//' + database_name + '_' + mode + '_' + model_name + '.txt')
    user_database_flip = np.loadtxt('features//' + database_name + '//' + database_name + '_' + mode + '_' + model_name + '_flip.txt')


rs_database = np.loadtxt('features//rs//rs_' + model_name + '.txt')

biocapsule_unique_rs(user_database, user_database_flip, rs_database, database_name, mode, model_name)
biocapsule_same_rs(user_database, user_database_flip, rs_database, database_name, mode, model_name)
