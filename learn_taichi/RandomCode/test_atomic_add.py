import taichi as ti
from time import time

ti.init(arch=ti.cpu)

N = 10000
a = ti.field(dtype=ti.i32, shape=N)

@ti.kernel
def foo():
    for i in range(10000):
        x = 0
        # a[i] = a[i] + 1.0
        a[i] += 1.0


@ti.kernel
def bar():
    for i, j in ti.ndrange((0, 200), (0, 200)):
        for k in range(50):
            for l in range(2):
                if 1:
                    x = 0



# start = time()
# foo()
# end = time()
# print(end - start)

start = time()
i = 0
while i < N:
    bar()
    # a[i] += 1
    i += 1

end = time()
print(end - start)
