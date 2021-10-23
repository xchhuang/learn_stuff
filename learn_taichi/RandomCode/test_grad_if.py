import taichi as ti

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def bug_gen():
    if x[None] > x[None] - 1:  # Passed if delete this line
        if x[None] < x[None] - 1:  # useless
            m = x[None]
            loss[None] = m * m
        else:
            m = x[None]
            loss[None] = m * m


with ti.Tape(loss=loss):
    bug_gen()

print(x.grad[None])