import time
import numpy as np
import matplotlib.pyplot as plt

import python_example as m

# assert m.__version__ == '0.0.1'
# assert m.add(1, 2) == 3
# print('===> Passed all assert.')

def main():
    
    start_time = time.time()
    bonndary_term = m.precompute_boundary(1024, 256, 256, 20, 2)
    bonndary_term = np.array(bonndary_term).astype(np.float32).reshape((256, 256, 20))
    end_time = time.time()
    print('===> Time: {:.4f}'.format(end_time - start_time))
    # print('bonndary_term:', bonndary_term, len(bonndary_term))
    print('bonndary_term:', bonndary_term.shape)


if __name__ == '__main__':
    main()


