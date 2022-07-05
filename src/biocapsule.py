import numpy as np
from scipy.signal import convolve2d


class BioCapsuleGenerator:
    def __signature_extraction(self, feature):
        lvl1 = convolve2d(
            feature.reshape(32, 16),
            np.ones((5, 5)) / 25.0,
            mode="same",
            boundary="wrap",
        )

        lvl2 = feature.reshape(32, 16) - lvl1

        signature = np.around(np.average(lvl2, axis=1) * 100.0).astype(int) % 9

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

        bc = np.multiply(user_feature, rs_key) + np.multiply(
            rs_feature, user_key
        )

        return bc

    def biocapsule_batch(self, user_features, rs_feature):
        rs_signature = self.__signature_extraction(rs_feature)
        rs_key = self.__key_generation(rs_signature)

        for i in range(user_features.shape[0]):
            user_signature = self.__signature_extraction(user_features[i])
            user_key = self.__key_generation(user_signature)

            user_features[i] = np.multiply(
                user_features[i], rs_key
            ) + np.multiply(rs_feature, user_key)

        return user_features
