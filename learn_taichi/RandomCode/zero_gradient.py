import numpy as np
import taichi as ti

np.random.seed(123)
real = ti.f32
# ti.init(ti.cpu, default_fp=real, print_ir=True)
ti.init(ti.cpu)

loss = ti.field(dtype=real, shape=(), needs_grad=True)
x = ti.field(dtype=real, shape=(), needs_grad=True)
acc = ti.field(dtype=real, shape=(), needs_grad=True)

@ti.kernel
def forces():
    acc[None] = x[None]

@ti.kernel
def vjp():
    loss[None] = acc[None]

x[None] = np.random.randn(1).astype(np.float32)[0]

with ti.Tape(loss):
    forces()
    vjp()

# These should be non-zero!
print(x.grad.to_numpy())