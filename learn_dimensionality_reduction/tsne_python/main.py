from tsne import tsne
from tsne_pytorch import tsne_pytorch
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    print("===> Running t-SNE on your MNIST digits.")

    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    init_Y = np.random.randn(X.shape[0], 2)

    start_time = time.time()
    Y1 = tsne(X, init_Y)    # author's version
    end_time = time.time()
    print('Time of tsne: {:.4f}s'.format(end_time - start_time))

    start_time = time.time()
    Y2 = tsne_pytorch(X, init_Y)    # my pytorch version
    end_time = time.time()
    print('Time of tsne_pytorch: {:.4f}s'.format(end_time - start_time))

    plt.figure(1, figsize=(10, 10))
    plt.subplot(121)
    plt.title('author\'s version')
    plt.scatter(Y1[:, 0], Y1[:, 1], s=10, c=labels)
    plt.gca().set_aspect('equal')
    plt.subplot(122)
    plt.title('my pytorch version')
    plt.scatter(Y2[:, 0], Y2[:, 1], s=10, c=labels)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()
