import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from biocapsule import BioCapsuleGenerator


np.random.seed(42)
tf.compat.v1.set_random_seed(42)


def get_mlp(in_size, out_size):
    inputs = Input(shape=(in_size,))
    x = Dense(256)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(out_size)(x)
    predictions = Activation("softmax")(x)
    mlp = Model(inputs=inputs, outputs=predictions)
    mlp.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy", metrics=["acc"])
    return mlp


def train_mlp(mlp, train, val, rs, rs_cnt, method, batch_size):
    X_train, y_train = train[:, :-1], train[:, -1].astype(int) - 1
    X_val, y_val = val[:, :-1], val[:, -1].astype(int) - 1

    bc_gen = BioCapsuleGenerator()
    for i in range(rs_cnt):
        X_train = bc_gen.biocapsule_batch(X_train, rs[i, :-1])
        X_val = bc_gen.biocapsule_batch(X_val, rs[i, :-1])

    class_weights = compute_class_weight("balanced",
                                         np.unique(y_train),
                                         y_train)

    mc = ModelCheckpoint(os.path.join(os.path.abspath(""), "results",
                                      "{}_{}_vggface2.h5".format(method, rs_cnt)),
                         monitor="val_loss", verbose=1, save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=3,
                       verbose=1, restore_best_weights=True)
    log = CSVLogger(os.path.join(os.path.abspath(""), "results",
                                 "{}_{}_vggface2.log".format(method, rs_cnt)))

    mlp.fit(x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            epochs=1000, batch_size=batch_size,
            callbacks=[mc, es, log],
            class_weight=class_weights)
    return mlp


def predict_mlp(mlp, test, rs, rs_cnt):
    X_test, y_true = test[:, :-1], test[:, -1].astype(int) - 1

    bc_gen = BioCapsuleGenerator()
    for i in range(rs_cnt):
        X_test = bc_gen.biocapsule_batch(X_test, rs[i, :-1])

    y_prob = mlp.predict(X_test)
    return y_prob
