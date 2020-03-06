import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.image import BboxImage
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from biocapsule import BioCapsuleGenerator


class HandlerLegendImage(HandlerBase):
    def __init__(self, path, space=5, offset=10, c="red"):
        super(HandlerLegendImage, self).__init__()
        self.space = space
        self.offset = offset
        self.color = c
        self.image_data = plt.imread(path)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        l = Line2D([xdescent+self.offset],
                   [ydescent+height/2.], c=self.color, ls="", marker="o", mfc=self.color, mec=self.color)
        l.set_clip_on(False)

        bb = Bbox.from_bounds(xdescent + (width+self.space)/3.+self.offset,
                              ydescent,
                              height *
                              self.image_data.shape[1] /
                              self.image_data.shape[0],
                              height)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        image.set_alpha(1.0)
        legend.set_alpha(1.0)

        self.update_prop(image, orig_handle, legend)
        return [l, image]


def feature_polt(method, rs_cnt, mode="pca"):
    user_data = np.load(os.path.join(os.path.abspath(
        ""), "data", "vggface2_visualize_{}_feat.npz".format(method)))["arr_0"]
    rs_data = np.load(os.path.join(os.path.abspath(
        ""), "data", "rs_{}_feat.npz".format(method)))["arr_0"]
    img_path = os.path.abspath(os.path.join(
        os.path.abspath(""), "images", "vggface2_visualize"))

    X = user_data[:, :-1]
    y = user_data[:, -1].astype(int)
    bc_gen = BioCapsuleGenerator()
    for i in range(rs_cnt):
        X = bc_gen.biocapsule_batch(X, rs_data[i, :-1])

    if mode == "pca":
        X_embedded = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

    plots = []
    colors = ["red", "green", "blue", "yellow", "black", "pink"]
    for i, label in enumerate(np.unique(y)):
        p = plt.scatter(X_embedded[y == label][:, 0],
                        X_embedded[y == label][:, 1], c=colors[i])
        plots.append(p)

    subjects = {}
    for subject in os.listdir(img_path):
        for i, label in enumerate(np.unique(y)):
            if len(os.listdir(os.path.join(img_path, subject))) == X_embedded[y == label].shape[0]:
                subjects[i] = subject

    paths = {}
    for i, label in enumerate(np.unique(y)):
        pic_file = os.listdir(os.path.join(img_path, subjects[i]))[0]
        paths[i] = os.path.join(img_path, subjects[i], pic_file)

    handler_map = {}
    for i, plot in enumerate(plots):
        handler_map[plot] = HandlerLegendImage(paths[i], c=colors[i])

    if method == "arcface":
        if rs_cnt == 0:
            plt.title("ArcFace Feature Visualization")
        else:
            plt.title(
                "ArcFace-BioCapsule Visualization (Fusion Count={})".format(rs_cnt))
    else:
        if rs_cnt == 0:
            plt.title("FaceNet Feature Visualization")
        else:
            plt.title(
                "FaceNet-BioCapsule Visualization (Fusion Count={})".format(rs_cnt))

    plt.grid(alpha=0.25, linestyle="--", color="grey")
    if mode == "pca":
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
    else:
        plt.xlabel("First t-SNE Dimension")
        plt.ylabel("Second t-SNE Dimension")
    lgd = plt.legend(plots, ["" for _ in plots],
                     handler_map=handler_map,
                     handlelength=2, labelspacing=0.0, fontsize=36, borderpad=0.15,
                     loc="center left", bbox_to_anchor=(1, 0.5),
                     handletextpad=0.2, borderaxespad=0.15)
    for p in plots:
        p.set_alpha(0.5)
    plt.savefig(os.path.join(os.path.abspath(""), "results",
                             "{}_{}_{}_plot.png").format(method, rs_cnt, mode),
                bbox_extra_artists=(lgd,), dpi=500, bbox_inches="tight")
    plt.close()


def accuracy_plot(acc_scores):

    plt.title("LFW Verification Accuracy")
    plt.xlabel("Number of Biometric-Capsule Fusions")
    plt.ylabel("Accuracy")

    plt.ylim(80, 100)

    plt.plot(acc_scores[0, :] * 100, label="ArcFace+BC")
    plt.plot(acc_scores[1, :] * 100, label="FaceNet+BC")
    plt.legend(loc="upper left")

    plt.grid(alpha=0.25, linestyle="--", color="grey")
    plt.savefig(os.path.join(os.path.abspath(""), "results", "lfw_acc_plot.png"), dpi=500, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    """
    for method in ["arcface", "facenet"]:
        for rs_cnt in range(6):
            feature_polt(method, rs_cnt, "pca")
            feature_polt(method, rs_cnt, "tsne")
    """

    acc_scores = np.zeros((2, 6))
    for method in ["arcface", "facenet"]:
        for rs_cnt in range(6):
            score = np.load(os.path.join(
                os.path.abspath(""), "results", "{}_{}_lfw_acc.npz").format(method, rs_cnt))["arr_0"]
            if method == "arcface":
                acc_scores[0, rs_cnt] = np.average(score)
            else:
                acc_scores[1, rs_cnt] = np.average(score)
    accuracy_plot(acc_scores)
