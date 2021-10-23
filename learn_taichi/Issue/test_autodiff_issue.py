"""
optimize 2d points via matching pairwise distance histogram
"""
import taichi as ti
import random
import numpy as np

ti.init(arch=ti.cpu)

# parameters for computing pairwise distance histogram
num_hist = 1
num_bin = 50
num_points = 50
sigma = 0.2
r_max = 0.15
step_size = r_max / num_bin
bins_np = np.linspace(step_size, step_size * num_bin, num_bin)
lr = 0.1

x = ti.field(dtype=ti.f32, shape=(num_points, 2), needs_grad=True)
x_hist = ti.field(dtype=ti.f32, shape=(num_hist, num_bin), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(num_points, 2), needs_grad=False)
y_hist = ti.field(dtype=ti.f32, shape=(num_hist, num_bin), needs_grad=False)
bins = ti.field(dtype=ti.f32, shape=num_bin, needs_grad=False)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

# Initialize vectors
for i in range(num_points):
    x[i, 0] = random.random()
    x[i, 1] = random.random()
    y[i, 0] = random.random()
    y[i, 1] = random.random()
bins.from_numpy(bins_np)


@ti.kernel
def reduce():
    for p in range(num_hist):
        for i in range(num_bin):
            L[None] += 0.5 * (x_hist[p, i] - y_hist[p, i]) ** 2


@ti.kernel
def gradient_descent():
    for i in range(num_points):
        # print('grad:', x.grad[i, 0], x.grad[i, 1])    # weird, zero gradients
        for j in range(2):
            x[i, j] -= x.grad[i, j] * lr


@ti.func
def euclidean_distance(p_i, p_j):
    diff = p_i - p_j
    d = ti.sqrt(diff[0] ** 2 + diff[1] ** 2)
    return d


@ti.kernel
def compute_x_hist():
    for i in range(num_hist):
        for j in range(num_bin):
            x_hist[i, j] = 0.0

    # old: zero gradient
    # for i, j in ti.ndrange((0, num_points), (0, num_points)):
    #     if i != j:
    #         p_i = ti.Vector([x[i, 0], x[i, 1]])
    #         p_j = ti.Vector([x[j, 0], x[j, 1]])
    #         dist = euclidean_distance(p_i, p_j)
    #         for p in range(num_hist):
    #             for k in range(num_bin):
    #                 val = ti.exp(-((dist - bins[k]) ** 2) / (sigma ** 2))
    #                 x_hist[p, k] += val / (num_points * num_points)

    # new: Kernel Simplicity Rule
    for i, j in ti.ndrange((0, num_points), (0, num_points)):
        for p in range(num_hist):
            for k in range(num_bin):
                if i != j:
                    p_i = ti.Vector([x[i, 0], x[i, 1]])
                    p_j = ti.Vector([x[j, 0], x[j, 1]])
                    dist = euclidean_distance(p_i, p_j)
                    val = ti.exp(-((dist - bins[k]) ** 2) / (sigma ** 2))
                    x_hist[p, k] += val / (num_points * num_points)


@ti.kernel
def compute_y_hist():
    for i in range(num_hist):
        for j in range(num_bin):
            y_hist[i, j] = 0.0

    for i, j in ti.ndrange((0, num_points), (0, num_points)):
        if i != j:
            p_i = ti.Vector([y[i, 0], y[i, 1]])
            p_j = ti.Vector([y[j, 0], y[j, 1]])
            dist = euclidean_distance(p_i, p_j)
            for p in range(num_hist):
                for k in range(num_bin):
                    val = ti.exp(-((dist - bins[k]) ** 2) / (sigma ** 2))
                    y_hist[p, k] += val / (num_points * num_points)


if __name__ == '__main__':
    # compute_target
    compute_y_hist()

    # Optimize with 10 gradient descent iterations
    for k in range(10):
        with ti.Tape(loss=L):
            compute_x_hist()
            reduce()
        print('Loss =', L[None])  # weird, loss does not change
        gradient_descent()
