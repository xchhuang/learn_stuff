import numpy as np
import math
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1024
DIM = 2

poissonDiskSamples = np.zeros((NUM_SAMPLES, DIM))


def get_rmax(num_points):
    """
    minimal distance between points when packing them optimally in the unit square
    """
    return 2.0 * np.sqrt(1.0 / (2.0 * np.sqrt(3.0) * num_points))


def poissonDiskSampling():
    fail_count = 0
    sample_count = 0
    min_dist = get_rmax(NUM_SAMPLES)

    while sample_count < NUM_SAMPLES:
        # print('sample_count:', sample_count)
        rp = np.random.rand(2)
        is_fail = 0
        for i in range(0, sample_count):
            dist2 = (rp[0] - poissonDiskSamples[i][0]) ** 2 + (rp[1] - poissonDiskSamples[i][1]) ** 2
            if dist2 < min_dist ** 2:
                is_fail = 1
                break
        if is_fail:
            fail_count += 1
        else:
            fail_count = 0
            poissonDiskSamples[sample_count][0] = rp[0]
            poissonDiskSamples[sample_count][1] = rp[1]

            sample_count += 1
        if fail_count > 100:
            min_dist = 0.999 * min_dist
            fail_count = 0


def main():
    start = time.time()
    poissonDiskSampling()
    end = time.time()
    print('Time: {:.4f}'.format(end - start))

    randomSamples = np.random.rand(NUM_SAMPLES, 2)
    plt.figure(1)
    plt.subplot(121)
    plt.scatter(poissonDiskSamples[:, 0], poissonDiskSamples[:, 1], s=10)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal')
    plt.subplot(122)
    plt.scatter(randomSamples[:, 0], randomSamples[:, 1], s=10)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal')
    plt.show()

    pds = poissonDiskSamples * 2 - 1
    x_prime = pds[:, 0] * np.sqrt(1 - pds[:, 1] ** 2 / 2)
    y_prime = pds[:, 1] * np.sqrt(1 - pds[:, 0] ** 2 / 2)

    plt.figure(2)
    plt.subplot(121)
    plt.scatter(x_prime, y_prime, s=10)
    plt.gca().set_aspect('equal')
    plt.subplot(122)
    plt.scatter(pds[:, 0], pds[:, 1], s=10)
    plt.gca().set_aspect('equal')
    plt.show()
    

if __name__ == '__main__':
    main()
