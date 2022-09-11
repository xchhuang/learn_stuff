from tsne import tsne
from tsne_pytorch import tsne_pytorch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE


def main():
    print("===> Running t-SNE on your MNIST digits.")

    X = np.loadtxt("mnist2500_X.txt")
    # X = np.load('D:/xhuang/phd/repo/deepsampling/src/scripts/results_cluster/results_v4_latent_n1000/features_vgg_v19.npz')['features'].astype(np.float32)

    labels = np.loadtxt("mnist2500_labels.txt")
    init_Y = np.random.randn(X.shape[0], 2)

    start_time = time.time()
    # Y1 = tsne(X, init_Y)    # author's version

    t_sne = TSNE(
        n_components=2,
        perplexity=20,
        n_iter=1000,
        init=init_Y,
        random_state=42,
        early_exaggeration=4,
        n_jobs=4,
        learning_rate=200,
    )
    Y1 = t_sne.fit_transform(X)

    end_time = time.time()
    print('Time of tsne: {:.4f}s'.format(end_time - start_time))

    start_time = time.time()
    Y2 = tsne(X, init_Y)    # my pytorch version
    end_time = time.time()
    print('Time of tsne_pytorch: {:.4f}s'.format(end_time - start_time))

    plt.figure(1, figsize=(10, 10))
    plt.subplot(121)
    plt.title('sklearn\'s version')
    # plt.scatter(Y1[:, 0], Y1[:, 1], s=10, c=labels)
    plt.scatter(Y1[:, 0], Y1[:, 1], s=10)

    plt.gca().set_aspect('equal')
    plt.subplot(122)
    plt.title('my version')
    # plt.scatter(Y2[:, 0], Y2[:, 1], s=10, c=labels)
    plt.scatter(Y2[:, 0], Y2[:, 1], s=10)

    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()
