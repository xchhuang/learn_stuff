from multiprocessing import Pool
from time import time

def foo(x):
    return x * x


def main():
    a = list(range(10000))
    pool = Pool(processes=4)              # start 4 worker processes
    start = time()
    result_list = pool.map(foo, a)
    end = time()
    print(end-start)

    result_list = []
    start = time()
    for i in a:
        result_list.append(foo(i))
    end = time()
    print(end - start)


if __name__ == "__main__":
    main()

