import taichi as ti
from time import time
from tqdm import tqdm
from collections import defaultdict

ti.init(ti.cpu)


# expected, why
N = 10
a = defaultdict(list)
a[0].append(ti.field(dtype=ti.f32, shape=N))
a[1].append(ti.field(dtype=ti.f32, shape=N))
a[1].append(ti.field(dtype=ti.f32, shape=N))


@ti.kernel
def update_a():
    for i in range(N):
        a[0][0][i] = i * 0.3


@ti.kernel
def read_a():
    for i in range(N):
        print(a[0][0][i], sep=',', end='')
    print()


print('before:', read_a())
update_a()
print('after:', read_a())
