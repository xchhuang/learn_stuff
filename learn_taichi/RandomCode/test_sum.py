import numpy as np
import taichi as ti
import time

np.random.seed(123)
real = ti.f32
ti.init(arch=ti.cpu)

n = 100000
a = ti.field(dtype=real, shape=n)
b = ti.field(dtype=real, shape=())

a.from_numpy(np.random.rand(n))


@ti.kernel
def test_while_kernel():
    b[None] = 0.0
    for i in range(n):
        b[None] += a[i]  # / n
    b[None] /= n


def test_while_python():
    tmp = 0
    for i in range(n):
        tmp += a[i]
    tmp /= n
    return tmp


for j in range(2):
    b[None] = 0
    test_while_kernel()
    print('kernel:', b[None])

for j in range(2):
    tmp = test_while_python()
    print('python:', tmp)
