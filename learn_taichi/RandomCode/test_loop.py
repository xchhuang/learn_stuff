import taichi as ti
from time import time

ti.init(arch=ti.cpu)


@ti.kernel
def foo():
    for i in range(10000):
        x = 0


@ti.kernel
def bar():
    for i in ti.ndrange(100):
        for j in ti.ndrange(100):
            x = 0

foo()

start = time()
foo()
end = time()
print(end - start)

start = time()
bar()
end = time()
print(end - start)
