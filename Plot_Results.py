import matplotlib.pyplot as plt
import numpy as np

#intra = "intra_class_distance_20180402-114759.txt"
#inter = "inter_class_distance_20180402-114759.txt"

intra = "intra_class_distance_20180408-102900.txt"
inter = "inter_class_distance_20180408-102900.txt"

intra_mat = np.loadtxt(intra)
inter_mat = np.loadtxt(inter)

while intra_mat.shape[0] < inter_mat.shape[0]:
    idx = np.random.randint(inter_mat.shape[0])
    inter_mat = np.delete(inter_mat,idx)

plt.plot(intra_mat, np.zeros_like(intra_mat) + 1, 'gx')
plt.plot(inter_mat, np.zeros_like(inter_mat) - 1, 'rx')
plt.show()