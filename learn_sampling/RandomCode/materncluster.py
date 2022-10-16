# Simulate a Matern point process on a rectangle.
# Author: H. Paul Keeler, 2018.
# Website: hpaulkeeler.com
# Repository: github.com/hpaulkeeler/posts
# For more details, see the post:
# hpaulkeeler.com/simulating-a-matern-cluster-point-process/

from configparser import RawConfigParser
from os import replace
import numpy as np;  # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm


def run_matern_clustering(lambdaParent, lambdaDaughter, radiusCluster):
    # Simulation window parameters
    xMin = 0
    xMax = 1
    yMin = 0
    yMax = 1

    # Parameters for the parent and daughter point processes
    # lambdaParent = 20  # density of parent Poisson point process
    # lambdaDaughter = int(1024 / lambdaParent);  # mean number of points in each cluster
    # radiusCluster = 0.1  # radius of cluster disk (for daughter points)

    # Extended simulation windows parameters
    rExt = radiusCluster  # extension parameter -- use cluster radius
    xMinExt = xMin - rExt
    xMaxExt = xMax + rExt
    yMinExt = yMin - rExt
    yMaxExt = yMax + rExt
    # rectangle dimensions
    xDeltaExt = xMaxExt - xMinExt
    yDeltaExt = yMaxExt - yMinExt
    areaTotalExt = xDeltaExt * yDeltaExt;  # area of extended rectangle

    # Simulate Poisson point process for the parents
    numbPointsParent = np.random.poisson(areaTotalExt * lambdaParent);  # Poisson number of points

    # x and y coordinates of Poisson points for the parent
    xxParent = xMinExt + xDeltaExt * np.random.uniform(0, 1, numbPointsParent)
    yyParent = yMinExt + yDeltaExt * np.random.uniform(0, 1, numbPointsParent)

    # Simulate Poisson point process for the daughters (ie final poiint process)
    numbPointsDaughter = np.random.poisson(lambdaDaughter, numbPointsParent)

    numbPoints = sum(numbPointsDaughter);  # total number of points

    # Generate the (relative) locations in polar coordinates by
    # simulating independent variables.
    theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints);  # angular coordinates
    rho = radiusCluster * np.sqrt(np.random.uniform(0, 1, numbPoints));  # radial coordinates

    # Convert from polar to Cartesian coordinates
    xx0 = rho * np.cos(theta)
    yy0 = rho * np.sin(theta)

    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(xxParent, numbPointsDaughter)
    yy = np.repeat(yyParent, numbPointsDaughter)

    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + xx0
    yy = yy + yy0

    # thin points if outside the simulation window
    booleInside = ((xx >= xMin) & (xx <= xMax) & (yy >= yMin) & (yy <= yMax))
    # retain points inside simulation window
    xx = xx[booleInside]
    yy = yy[booleInside]

    if False:
        # Plotting
        plt.figure(1)
        plt.title(xx.shape[0])
        # plt.scatter(xx, yy, edgecolor='b', facecolor='none', alpha=0.5)
        plt.scatter(xx, yy, c='k', s=1)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.show()

    pts = np.stack([xx, yy], -1).astype(np.float32)
    return pts


def main():

    lambdaParent = 20  # density of parent Poisson point process
    lambdaDaughter = int(1024 / lambdaParent);  # mean number of points in each cluster
    radiusCluster = 0.1  # radius of cluster disk (for daughter points)

    num_realizations = 1000
    count = 0
    for i in tqdm(range(num_realizations)):
        lambdaParent = 30 + np.random.rand() * (120 - 30)
        # lambdaParent = 300
        lambdaDaughter = int(1024 / lambdaParent)
        radiusCluster = 0.04 + np.random.rand() * (0.08 - 0.04)
        # radiusCluster = 0.2
        pts = run_matern_clustering(lambdaParent, lambdaDaughter, radiusCluster)
        if pts.shape[0] > 1024 and pts.shape[0] < 1050:
            count += 1
            plt.figure(1)
            # plt.scatter(xx, yy, edgecolor='b', facecolor='none', alpha=0.5)
            pts = pts[np.random.choice(pts.shape[0], 1024, replace=False)]
            plt.scatter(pts[:, 0], pts[:, 1], c='k', s=1)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.gca().set_aspect('equal')
            plt.axis('off')
            plt.savefig('results/{:0>5}.png'.format(count), bbox_inches='tight', pad_inches=0, dpi=200)
            plt.clf()

            np.savetxt('results/{:0>5}.txt'.format(count), pts)
    print('count =', count)

if __name__ == '__main__':
    main()