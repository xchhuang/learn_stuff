import taichi as ti

ti.init(arch=ti.cpu)

N = 0
array_py = []
array_ti = []


# expected behaviour
def test_append_py():
    if N > 0:
        array_py.append(1)


# weird behaviour
@ti.kernel
def test_append_ti():
    if N > 0:
        array_ti.append(ti.Vector([1, 1]))


# tests
print('array_py before:', array_py)
for i in range(10):
    test_append_py()
print('array_py after:', array_py)

print('array_ti before:', array_ti)
for i in range(10):
    test_append_ti()
print('array_ti after:', array_ti)
