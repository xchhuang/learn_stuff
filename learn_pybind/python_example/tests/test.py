import time
import python_example as m

# assert m.__version__ == '0.0.1'
# assert m.add(1, 2) == 3
# print('===> Passed all assert.')

start_time = time.time()
m.precompute_boundary(10)
end_time = time.time()
print('===> Time: {:.4f}'.format(end_time - start_time))



