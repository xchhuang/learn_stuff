import taichi as ti
import numpy as np

a = ti.field(dtype=ti.f32, shape=2, needs_grad=True)
b = ti.field(dtype=ti.f32, shape=2)
c = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

a.from_numpy(np.random.rand(2))
b.from_numpy(np.random.rand(2))


@ti.kernel
def compute_c():
    c[None] = a[0] * b[0] + a[1] * b[1]


with ti.Tape(c):
    compute_c()

print(a.grad[0])

