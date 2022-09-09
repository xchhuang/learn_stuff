import numpy as np
import torch
import matplotlib.pyplot as plt


def tsne():
    pass


def main():
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("../tsne_python_author/mnist2500_X.txt")
    labels = np.loadtxt("../tsne_python_author/mnist2500_labels.txt")
    # X = np.load('D:/xhuang/phd/repo/deepsampling/src/scripts/results_cluster/results_v4_latent_n1000/features_vgg_v19.npz')['features'].astype(np.float32)

    # Y = tsne(X, 2, 50, 20.0)
    # print('Y:', Y.shape)
    # # pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # pylab.scatter(Y[:, 0], Y[:, 1], 20)
    # pylab.show()


if __name__ == '__main__':
    main()
