import taichi as ti
import random
import numpy as np

ti.init(arch=ti.cpu, debug=True)

num_hist = 1
num_bin = 50
num_points = 50
sigma = 0.2
rmax = 0.15  # 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * num_points))
step_size = rmax / num_bin
bins_np = np.linspace(step_size, step_size * num_bin, num_bin)
# print(rmax)

beta1 = 0.9
beta2 = 0.999
lr = 0.1
eps = 1e-8
use_tape = True

x = ti.field(dtype=ti.f32, shape=(num_points, 2), needs_grad=True)
x_hist = ti.field(dtype=ti.f32, shape=(num_hist, num_bin), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(num_points, 2), needs_grad=False)
y_hist = ti.field(dtype=ti.f32, shape=(num_hist, num_bin), needs_grad=False)
bins = ti.field(dtype=ti.f32, shape=num_bin, needs_grad=False)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

moment_1st = ti.field(dtype=ti.f32, shape=(num_points, 2))
moment_2nd = ti.field(dtype=ti.f32, shape=(num_points, 2))


@ti.kernel
def reduce():
    for p in range(num_hist):
        for i in range(num_bin):
            L[None] += 0.5 * (x_hist[p, i] - y_hist[p, i]) ** 2


# Initialize vectors
for i in range(num_points):
    x[i, 0] = random.random()
    x[i, 1] = random.random()
    y[i, 0] = random.random()
    y[i, 1] = random.random()
bins.from_numpy(bins_np)


@ti.kernel
def gradient_descent():
    for i in range(num_points):
        # print('grad:', x.grad[i, 0], x.grad[i, 1])
        for j in range(2):
            # print('grad:', x.grad[i, j])
            x[i, j] -= x.grad[i, j] * lr


@ti.kernel
def adam(itr: ti.i32):
    for i in range(num_points):
        # print('grad:', x.grad[i, 0], x.grad[i, 1])
        for j in range(2):
            g = x.grad[i, j]
            moment_1st[i, j] = beta1 * moment_1st[i, j] + (1 - beta1) * g
            moment_2nd[i, j] = beta2 * moment_2nd[i, j] + (1 - beta2) * (g ** 2)
            m_hat = moment_1st[i, j] / (1.0 - beta1 ** (itr + 1))
            v_hat = moment_2nd[i, j] / (1.0 - beta2 ** (itr + 1))
            x[i, j] -= lr * m_hat / (ti.sqrt(v_hat) + eps)


@ti.func
def euclideanDistance(p_i, p_j):
    diff = p_i - p_j
    d = ti.sqrt(diff[0] ** 2 + diff[1] ** 2)
    return d


@ti.kernel
def compute_x_hist():
    for i in range(num_hist):
        for j in range(num_bin):
            x_hist[i, j] = 0.0

    for i, j in ti.ndrange((0, num_points), (0, num_points)):
        if i != j:
            p_i = ti.Vector([x[i, 0], x[i, 1]])
            p_j = ti.Vector([x[j, 0], x[j, 1]])
            dist = euclideanDistance(p_i, p_j)
            # print('dist1:', dist)
            # dist = ti.sqrt((x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2)
            # print('dist2:', dist)
            # need to use ti.static
            for p in ti.static(range(num_hist)):
                for k in ti.static(range(num_bin)):
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
            dist = euclideanDistance(p_i, p_j)
            for p in range(num_hist):
                for k in range(num_bin):
                    val = ti.exp(-((dist - bins[k]) ** 2) / (sigma ** 2))
                    y_hist[p, k] += val / (num_points * num_points)


# compute_target
compute_y_hist()

# Optimize with 100 gradient descent iterations
for k in range(100):
    if use_tape:
        with ti.Tape(loss=L):
            compute_x_hist()
            reduce()
    if k % 10 == 0:
        print('Iter: {:}, Loss: {:.10f}'.format(k, L[None]))
    gradient_descent()
    # adam(k)

# for i in range(num_points):
#     # Now you should approximately have x[i] == y[i]
#     print(x[i, 0], y[i, 0], x[i, 1], y[i, 1])
