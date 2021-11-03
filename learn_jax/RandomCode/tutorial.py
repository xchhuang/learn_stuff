# jax version: '0.2.24'

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def main():
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))
    size = 3000
    x = random.normal(key, (size, size), dtype=jnp.float32)
    %timeit jnp.dot(x, x.T).block_until_ready()  # runs on the GPU


if __name__ == '__main__':
    main()
