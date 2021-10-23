import taichi as ti
import random
import numpy as np

ti.init(arch=ti.cpu)

n = 8
beta1 = 0.9
beta2 = 0.999
lr = 0.1
eps = 1e-8
use_tape = True

x = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
x_cumsum = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
x_other = ti.field(dtype=ti.f32, shape=n)
y = ti.field(dtype=ti.f32, shape=n)
y_cumsum = ti.field(dtype=ti.f32, shape=n)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
ind = ti.field(dtype=ti.f32, shape=n)

moment_1st = ti.field(dtype=ti.f32, shape=n)
moment_2nd = ti.field(dtype=ti.f32, shape=n)

# init
for i in range(n):
    x_other[i] = 1
    ind[i] = i


@ti.kernel
def reduce():
    for i in range(n):
        L[None] += 0.5 * (x_cumsum[i] - y_cumsum[i]) ** 2
        # L[None] += 0.5 * (x[i] - y[i]) ** 2


# Initialize vectors
for i in range(n):
    x[i] = random.random()
    y[i] = random.random()


@ti.kernel
def gradient_descent():
    for i in x:
        # print('grad:', x.grad[i])
        x[i] -= x.grad[i] * lr


@ti.kernel
def adam(itr: ti.i32):
    for i in x:
        # print('grad:', x.grad[i])
        g = x.grad[i]
        moment_1st[i] = beta1 * moment_1st[i] + (1 - beta1) * g
        moment_2nd[i] = beta2 * moment_2nd[i] + (1 - beta2) * (g ** 2)
        m_hat = moment_1st[i] / (1.0 - beta1 ** (itr + 1))
        v_hat = moment_2nd[i] / (1.0 - beta2 ** (itr + 1))
        x[i] -= lr * m_hat / (ti.sqrt(v_hat) + eps)


@ti.kernel
def compute_x_cumsum():
    # TODO: nothing before the loop
    # a = 1
    # b = 1
    for i in range(n):
        x_cumsum[i] = 0.0
    for i in range(n):
        a = ti.Vector([x[ind[i]], x[ind[i]]])  # test intermediate vector
        x_cumsum[i] += a[0] * x_other[i]
        # x_cumsum[i] = x[i] * x_other[i]


@ti.kernel
def compute_y_cumsum():
    for i in range(n):
        y_cumsum[i] += y[i]


# compute_target
compute_y_cumsum()

# Optimize with 100 gradient descent iterations
for k in range(100):
    if use_tape:
        with ti.Tape(loss=L):
            compute_x_cumsum()
            reduce()
    # else:
    #     ti.clear_all_gradients()
    #     L[None] = 0
    #     L.grad[None] = 1
    #     compute_x_cumsum()
    #     reduce()
    #     reduce.grad()

    print('Loss =', L[None])
    # gradient_descent()
    adam(k)

# for i in range(n):
#     # Now you should approximately have x[i] == y[i]
#     print(x[i], y[i])
