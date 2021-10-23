import numpy as np
import taichi as ti

np.random.seed(123)
real = ti.f32
ti.init(ti.gpu)  # cpu works

n = 10000
a = ti.field(dtype=real, shape=n)
a.from_numpy(np.random.rand(n))


@ti.kernel
def find_minimum_serialized():
    min_error = np.inf
    min_index = 0
    for _ in range(1):
        for i in ti.ndrange((0, n)):
            if min_error > a[i]:
                min_error = a[i]
                min_index = i
    print('Minimum(serialized):', min_error, min_index)


@ti.kernel
def find_minimum_parallelized():
    min_error = np.inf
    min_index = 0
    for i in ti.ndrange((0, n)):
        ti.atomic_min(min_error, a[i])  # use like this
        # if min_error > a[i]:
        #     min_error = a[i]
        #     min_index = i
    print('Minimum(parallelized):', min_error, min_index)


find_minimum_serialized()       # correct
find_minimum_parallelized()     # wrong on gpu
