import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation

np.random.seed(42)
tf.compat.v1.set_random_seed(42)


def draw_progress(text, percent, barLen=20):
    print(text + " -- [{:<{}}] {:.0f}%".format("=" *
                                               int(barLen * percent), barLen, percent * 100), end="\r")
    if percent == 1:
        print("\n")


def walk(path):
    files = []
    dirs = []
    contents = os.listdir(path)
    for content in contents:
        if os.path.isfile(os.path.join(path, content)):
            files += [content]
        else:
            dirs += [content]
    for d in dirs:
        files += walk(os.path.join(path, d))
    return files


def get_mlp(input_shape, output_shape):
    inputs = Input(shape=(input_shape,))
    # (1) FC - ReLU - BN - Dropout
    fc1 = Dense(256)(inputs)
    ru1 = Activation("relu")(fc1)
    bn1 = BatchNormalization()(ru1)
    do1 = Dropout(0.2)(bn1)
    # (2) FC - ReLU - BN - Dropout
    fc2 = Dense(128)(do1)
    ru2 = Activation("relu")(fc2)
    bn2 = BatchNormalization()(ru2)
    do2 = Dropout(0.2)(bn2)
    # (3) FC - ReLU - BN - Dropout
    fc3 = Dense(64)(do2)
    ru3 = Activation("relu")(do2)
    bn3 = BatchNormalization()(ru3)
    do3 = Dropout(0.2)(bn3)
    # (4) FC - Sigmoid/SoftMax
    fc4 = Dense(output_shape)(do3)
    if output_shape == 1:
        predictions = Activation("sigmoid")(fc4)
        loss = "binary_crossentropy"
    else:
        predictions = Activation("softmax")(fc4)
        loss = "categorical_crossentropy"
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer="adam",
                  loss=loss,
                  metrics=["acc"])
    return model


def get_lfw(mode):
    people = []
    with open(os.path.join(os.path.abspath(""), "images", "people.txt"), "r") as people_file:
        people_list = list(csv.reader(people_file, delimiter="\t"))
        assert(len(people_list[2:603]) == 601)
        people.append(people_list[2:603])
        assert(len(people_list[604:1159]) == 555)
        people.append(people_list[604:1159])
        assert(len(people_list[1160:1712]) == 552)
        people.append(people_list[1160:1712])
        assert(len(people_list[1713:2273]) == 560)
        people.append(people_list[1713:2273])
        assert(len(people_list[2274:2841]) == 567)
        people.append(people_list[2274:2841])
        assert(len(people_list[2842:3369]) == 527)
        people.append(people_list[2842:3369])
        assert(len(people_list[3370:3967]) == 597)
        people.append(people_list[3370:3967])
        assert(len(people_list[3968:4569]) == 601)
        people.append(people_list[3968:4569])
        assert(len(people_list[4570:5150]) == 580)
        people.append(people_list[4570:5150])
        assert(len(people_list[5151:]) == 609)
        people.append(people_list[5151:])

    pairs = []
    with open(os.path.join(os.path.abspath(""), "images", "pairs.txt"), "r") as pairs_file:
        pairs_list = list(csv.reader(pairs_file, delimiter="\t"))
        for i in range(10):
            idx = i * 600 + 1
            pairs.append(pairs_list[idx: idx + 600])
            assert (len(pairs[i]) == 600)

    features = np.load(os.path.join(os.path.abspath(
        ""), "data", "lfw_" + mode + "_feat.npz"))["arr_0"]

    subject = {}
    for s_id, s in enumerate(os.listdir(os.path.join(os.path.abspath(""), "images", "lfw"))):
        subject[s] = s_id + 1

    lfw = {}
    for i in range(10):
        train = people[i]
        train_cnt = np.sum([int(s[-1]) for s in train])
        test = pairs[i]

        lfw["train_" + str(i)] = np.zeros((train_cnt, 513))
        lfw["test_" + str(i)] = np.zeros((600, 2, 513))

        train_idx = 0
        for s in train:
            s_id = subject[s[0]]
            s_features = features[features[:, -1] == s_id]
            assert (s_features.shape[0] == int(s[1]))

            for j in range(s_features.shape[0]):
                lfw["train_" + str(i)][train_idx] = s_features[j]
                train_idx += 1

        assert (train_idx == train_cnt)

        for test_idx, s in enumerate(test):
            if len(s) == 3:
                s_id = subject[s[0]]
                s_features = features[features[:, -1] == s_id]
                lfw["test_" + str(i)][test_idx, 0] = s_features[int(s[1]) - 1]
                lfw["test_" + str(i)][test_idx, 1] = s_features[int(s[2]) - 1]
            else:
                s_id_1 = subject[s[0]]
                s_features = features[features[:, -1] == s_id_1]
                lfw["test_" + str(i)][test_idx, 0] = s_features[int(s[1]) - 1]
                s_id_2 = subject[s[2]]
                s_features = features[features[:, -1] == s_id_2]
                lfw["test_" + str(i)][test_idx, 1] = s_features[int(s[3]) - 1]

        assert (test_idx == 599)

    return lfw
