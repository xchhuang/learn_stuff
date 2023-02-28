import numpy as np
import matplotlib.pyplot as plt
import  sampling_utils


def gauss2d(x, mu, sigma):
    z = 1 / (np.sqrt(2 * np.pi) * sigma)
    return z * np.exp(-np.sum((x - mu) ** 2, -1) / (2 * sigma ** 2))


def main():
    res = 100
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=-1).reshape(res * res, -1)
    y = gauss2d(grid, 0, 1).reshape(res, res)
    print(y.shape, y.min(), y.max())
    
    den = (y - y.min()) / (y.max() - y.min())
    num_points = 1000
    pts_white = sampling_utils.ImportanceSamples(den, num_points)
    pts_blue = sampling_utils.ImportanceSamplesBlue(den, num_points)

    pts_white = (pts_white - 0.5) * 6
    pts_blue = (pts_blue - 0.5) * 6
    
    plt.figure(1)
    plt.subplot(131)
    plt.title('Density')
    plt.imshow(y)
    plt.axis('off')
    plt.subplot(132)
    plt.title('Random samples')
    plt.scatter(pts_white[:, 0], pts_white[:, 1], s=0.1)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.subplot(133)
    plt.title('Blue noise samples')
    plt.scatter(pts_blue[:, 0], pts_blue[:, 1], s=0.1)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    # plt.show()
    plt.savefig('results/sample2d.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()


if __name__ == '__main__':
    main()

