import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


np.random.seed(42)
tf.compat.v1.set_random_seed(42)


def get_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class FeatureGenerator:
    def __init__(self, dataset, batch_size=512):
        super().__init__()
        self.dataset = dataset
        self.idx = np.arange(dataset.shape[0])
        self.batch_size = batch_size
        np.random.shuffle(self.idx)

    def __len__(self):
        return len(list(get_batches(self.idx, self.batch_size)))

    def flow(self):
        batch_idxs = list(get_batches(self.idx, self.batch_size))
        for idx in batch_idxs:
            batch = self.dataset[np.sort(idx), :]
            X, y = batch[:, :-1], np.around(batch[:, -1]) - 1
            yield X, y


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
    predictions = Dense(out_size, activation="softmax")(x)
    mlp = Model(inputs=inputs, outputs=predictions)
    mlp.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy", metrics=["acc"])
    return mlp


def train_mlp(mlp, train_gen, val_gen, method, rs_cnt):
    y_train = train_gen.dataset[:, -1]
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

    mlp.fit_generator(train_gen.flow(), steps_per_epoch=len(train_gen),
                      epochs=10000, callbacks=[mc, es, log],
                      validation_data=val_gen.flow(), validation_steps=len(val_gen),
                      class_weight=class_weights)
    return mlp
