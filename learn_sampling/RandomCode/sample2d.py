import numpy as np
import matplotlib.pyplot as plt
from sampling_utils import ImportanceSamples


def gauss2d(x, mu, sigma):
    z = 1 / (np.sqrt(2 * np.pi) * sigma)
    return z * np.exp(-np.sum((x - mu) ** 2, -1) / (2 * sigma ** 2))


def main():
    res = 100
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=-1).reshape(res * res, -1)
    y = gauss2d(grid, 0, 1).reshape(res, res)
    print(y.shape, y.min(), y.max())
    
    den = (y - y.min()) / (y.max() - y.min())
    pts = ImportanceSamples(den, 1000)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(y)
    plt.subplot(122)
    plt.scatter(pts[:, 0], pts[:, 1], s=0.1)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()

