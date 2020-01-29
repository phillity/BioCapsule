import os
from math import sqrt
import numpy as np
from sklearn.metrics import roc_curve
from argparse import ArgumentParser
from matplotlib import rcParams
import matplotlib.pyplot as plt
from verification import verification


def latexify(fig_width=None, fig_height=None, columns=1):
    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0
        fig_height = fig_width*golden_mean

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {"backend": "ps",
              "text.latex.preamble": [r"\usepackage{gensymb}"],
              "axes.labelsize": 10,
              "axes.titlesize": 10,
              "font.size": 10,
              "legend.fontsize": 10,
              "xtick.labelsize": 10,
              "ytick.labelsize": 10,
              "text.usetex": True,
              "figure.figsize": [fig_width, fig_height],
              "font.family": "serif"
              }

    rcParams.update(params)


def format_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("gray")
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color="gray")

    return ax


def plot_roc(y_true, y_prob, title, out_file):
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_prob.flatten())

    latexify()

    plt.plot(fpr, tpr, label="Arcface Features")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.grid(True)
    ax = plt.axes()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()
    format_axes(ax)

    plt.savefig(out_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    args = vars(parser.parse_args())

    y_true, y_prob, acc = verification(args["method"])
    plot_roc(y_true, y_prob, "LFW Verification ROC Curve", "lfw.png")
        
