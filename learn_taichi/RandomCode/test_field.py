import taichi as ti
from time import time
from tqdm import tqdm

ti.init(ti.cpu)

# small taichi field
start = time()
a = ti.field(dtype=ti.f32, shape=20*400*100)
end = time()
print('small: {:.4f}'.format(end - start))


# larger taichi field
start = time()
b = ti.field(dtype=ti.f32, shape=2*400*50)
end = time()
print('large: {:.4f}'.format(end - start))

c = ti.field(dtype=ti.f32, shape=())

# serialized, very slow -> almost 1 min
print('===> Python looping(might be slow): {:.4f}'.format(end - start))
start = time()
for i in tqdm(range(1000000)):
    c[None] += a[i] + b[i]
end = time()
print('python loop: {:.4f}'.format(end - start))


# parallelized
@ti.kernel
def add_ti():
    for i in range(1000000):
        c[None] += a[i] + b[i]

start = time()
add_ti()
end = time()
print('taichi loop: {:.4f}'.format(end - start))

