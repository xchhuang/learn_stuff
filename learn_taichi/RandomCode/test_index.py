import taichi as ti
from time import time
from tqdm import tqdm
from collections import defaultdict

ti.init(ti.cpu)

a = ti.field(dtype=ti.f32, shape=100)
b = []
b.append(ti.field(dtype=ti.f32, shape=100))


@ti.kernel
def foo(i: ti.i32):
    j = i + 1
    print(a[j])


# @ti.kernel
# def bar(i: ti.i32):
#     print(b[i])


foo(2)
# bar(2)
