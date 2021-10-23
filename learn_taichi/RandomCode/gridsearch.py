# my first taichi program

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from time import time

ti.init(arch=ti.cpu)

@ti.data_oriented
class GridSearchTest:
    def __init__(self):
        self.num_grids = 200
        self.grid_centers_np = self.generate_grid()
        self.N = self.grid_centers_np.shape[0]
        self.grid_centers_ti = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.M = 100
        self.random_vec_np = np.random.rand(self.M, 2)
        self.random_vec_ti = ti.Vector.field(2, dtype=ti.f32, shape=self.M)
        self.min_dist = ti.field(dtype=ti.f32, shape=())
        self.min_idx = ti.field(dtype=ti.i32, shape=())

        # materialization
        self.min_dist[None] = np.inf
        self.min_idx[None] = 0
        self.grid_centers_ti.from_numpy(self.grid_centers_np)
        self.random_vec_ti.from_numpy(self.random_vec_np)

    def generate_grid(self):
        """
        must be square
        """
        x = list(range(1, self.num_grids))
        y = list(range(1, self.num_grids))
        xv, yv = np.meshgrid(x, y)
        w, h = xv.shape[0], xv.shape[1]
        assert w == h
        # print(w, h, w * h)
        points = np.zeros((w * h, 2))
        points[:, 0] = xv.reshape(w * h)
        points[:, 1] = yv.reshape(w * h)
        points /= w
        # plt.figure(1)
        # plt.scatter(points[:, 0], points[:, 1], s=5)
        # plt.show()
        return points

    @staticmethod
    @ti.func
    def euclidean_distance(pi, pj):
        diff = pi - pj
        dist = ti.sqrt(diff.x ** 2 + diff.y ** 2)
        return dist

    @ti.kernel
    def grid_search_ti(self):
        # global min_dist
        for i in range(self.N):
            pi = self.grid_centers_ti[i]
            for j in range(self.M):
                pj = self.random_vec_ti[j]
                dist = self.euclidean_distance(pi, pj)
                if self.min_dist > dist:
                    self.min_dist = dist
                    self.min_idx = i
        # print(self.min_dist)

    def grid_search_np(self):
        start = time()
        min_idx = 0
        min_dist = np.inf
        for i in range(self.N):
            p = self.grid_centers_np[i]
            diff = p - self.random_vec_np
            dist = np.sqrt(np.sum(diff ** 2, 1))
            dist.sort()
            if min_dist > dist[0]:
                min_idx = i
                min_dist = dist[0]
        end = time()
        print('Time elapsed numpy: {:.4f}, min_dist: {:.4f}, min_idx: {:}'.format(end - start, min_dist, min_idx))


def main():
    grid_search_test = GridSearchTest()
    grid_search_test.grid_search_np()

    start = time()
    grid_search_test.grid_search_ti()
    end = time()
    print('Time elapsed taichi: {:.4f}, min_dist: {:.4f}, min_idx: {:}'.format(end - start, grid_search_test.min_dist[None], grid_search_test.min_idx[None]))


if __name__ == '__main__':
    main()
