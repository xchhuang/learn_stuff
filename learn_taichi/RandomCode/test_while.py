import numpy as np
import taichi as ti
import time

np.random.seed(123)
real = ti.f32
ti.init(ti.gpu)

n = 10000000
a = ti.field(dtype=real, shape=n)
a.from_numpy(np.random.rand(n))


@ti.kernel
def test_while_kernel():
    # x = 0
    # for i in range(n):
    #     x += 1

    i = 0
    N = n
    while i < N:
        i += 1
        if i > 100:
            N = i
    # print(i)


def test_while_python():
    i = 0
    while i < n:
        i += 1


def foo():
    x = 0
    x += 1


# start = time.time()
# test_while_python()
# end = time.time()
# print(end - start)
#
# start = time.time()
# test_while_python()
# end = time.time()
# print(end - start)

start = time.time()
test_while_kernel()
end = time.time()
print(end - start)

start = time.time()
test_while_kernel()
end = time.time()
print(end - start)



start = time.time()
for j in range(n):
    foo()
end = time.time()
print(end - start)

start = time.time()
for j in range(n):
    # test_while_kernel()
    foo()
end = time.time()
print(end - start)
