# BioCapsule Visualization script
import os
import sys
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot
from latex_format import latexify


def bc_visualization(database, mode):
    cur_path = os.path.dirname(__file__)
    if mode == 'under':
        data_path = os.path.join(cur_path, 'data', database + '.npz')
    elif mode == 'same':
        data_path = os.path.join(cur_path, 'data', database + '_bc_same.npz')
    else:
        data_path = os.path.join(cur_path, 'data', database + '_bc_uni.npz')

    data = np.load(data_path)['arr_0']
    data_y = data[:, -1]
    data = data[:, :-1]
    data = PCA(n_components=2).fit_transform(data)

    np.random.seed(42)
    latexify()
    if database == 'caltech':
        if mode == 'under':
            pyplot.title('Caltech Faces 1999 FaceNet Features')
        elif mode == 'same':
            pyplot.title('Caltech Faces 1999 Same RS BioCapsules')
        else:
            pyplot.title('Caltech Faces 1999 Unique RS BioCapsules')
    elif database == 'gt':
        if mode == 'under':
            pyplot.title('GeoTech Face Database FaceNet Features')
        elif mode == 'same':
            pyplot.title('GeoTech Face Database Same RS BioCapsules')
        else:
            pyplot.title('GeoTech Face Database Unique RS BioCapsules')
    else:
        if mode == 'under':
            pyplot.title('LFW FaceNet Features')
        elif mode == 'same':
            pyplot.title('LFW Same RS BioCapsules')
        else:
            pyplot.title('LFW Unique RS BioCapsules')
    pyplot.xlabel('First Principal Component')
    pyplot.ylabel('Second Principal Component')

    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])
    pyplot.xticks(())
    pyplot.yticks(())
    pyplot.xlim(min_x - .1 * max_x, max_x + .1 * max_x)
    pyplot.ylim(min_y - .1 * max_y, max_y + .1 * max_y)
    pyplot.scatter(data[:, 0], data[:, 1], s=5, c=data_y)
    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig(database + '_' + mode + '.pdf')
    pyplot.close()


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to use in bc visualization')
args = vars(parser.parse_args())

# Perform biocapsule visualization
bc_visualization(args['database'], 'under')
bc_visualization(args['database'], 'same')
bc_visualization(args['database'], 'uni')
