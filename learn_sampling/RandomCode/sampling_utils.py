import numpy as np
import torch
import bisect
from tqdm import tqdm


def get_rmax(num_points):
    y = 2.0 * np.sqrt(1.0 / (2.0 * np.sqrt(3.0) * num_points))
    return y


def getCDF(importance):
    importance_size = importance.shape[0]
    cum = np.zeros_like(importance)
    cum[0] = importance[0]
    for i in range(1, importance_size):
        cum[i] = cum[i - 1] + importance[i]
    v_max = cum[importance_size - 1]
    if v_max != 0:
        cum = cum / v_max
    return cum


def randCDF(cum, importance_size):
    r = np.random.rand()
    pos = bisect.bisect_right(cum, r, lo=0, hi=cum.shape[0])
    return pos


def ImportanceSamples(den, num_points):
    pts = []
    height, width = den.shape[-2], den.shape[-1]
    importance = np.random.rand(height * width)

    for i in range(importance.shape[0]):
        importance[i] = den[int(i / width), i % width]

    cum = getCDF(importance)
    # print(cum.min(), cum.max(), den.min(), den.max())
    # plt.figure(1)
    # plt.imshow(importance.reshape(height, width))
    # plt.show()

    for i in range(num_points):
        pt = np.random.rand(2)
        num = randCDF(cum, importance.shape[0])
        pt[0] += float(num % width)
        pt[1] += float(num / width)
        pt[0] /= width
        pt[1] /= height
        pts.append(pt)
    pts = np.array(pts).astype(np.float32)
    pts[:, 1] = 1 - pts[:, 1]
    # pts = np.random.rand(num_points, 2)
    # plt.figure(1)
    # plt.title(str(pts.shape[0]))
    # plt.scatter(pts[:, 0], pts[:, 1], s=0.1)
    # plt.gca().set_aspect('equal')
    # plt.show()
    return pts





def ImportanceSamplesBlue(den, num_points):
    pts = np.random.rand(1, 2)

    # mindist = get_rmax(num_points)
    mindist = 0.015
    height, width = den.shape[-2], den.shape[-1]
    importance = np.random.rand(height * width)

    for i in range(importance.shape[0]):
        importance[i] = den[int(i / width), i % width]

    cum = getCDF(importance)

    for i in tqdm(range(num_points)):

        reject = True
        failcount = 0

        while reject:

            pt = np.random.rand(2)
            num = randCDF(cum, importance.shape[0])
            pt[0] += float(num % width)
            pt[1] += float(num / width)
            pt[0] /= width
            pt[1] /= height
            pt = pt[None, ...]

            if np.min(np.sum((pts - pt) ** 2, 1)) > mindist * mindist:
                pts = np.vstack((pts, pt))
                failcount = 0
                reject = False
            else:
                failcount += 1

            if failcount > 100:
                mindist = 0.999 * mindist
                failcount = 0
                reject = False

    pts = pts.astype(np.float32)
    pts[:, 1] = 1 - pts[:, 1]
    return pts



def blue(npts, mindist=0):
    """
    Generates well spaced points with an adaptative dart throw method
    Output is similar to blue noise

    Parameters
    ----------
    npts : number of points to generate
    mindist : initial minimal accepted distance between points
    """
    if mindist == 0:
        mindist = get_rmax(npts)
    failcount = 0
    pts = np.random.rand(1, 2)
    while np.size(pts, 0) < npts:
        pt = np.random.rand(1, 2)
        if np.min(np.sum((pts - pt) ** 2, 1)) > mindist * mindist:
            pts = np.vstack((pts, pt))
            failcount = 0
        else:
            failcount += 1

        if failcount > 100:
            mindist = 0.999 * mindist
            failcount = 0
    return pts
