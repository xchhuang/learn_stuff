# jax version: '0.2.24'

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def func(W, b):
    out = 0
    for w in W:
        if w < b:
            out += w
        else:
            out += w * 0.5
    return out


key = random.PRNGKey(0)
key, W_key, b_key = random.split(key, 3)
W = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
b = jnp.array([0.5])

# W_grad, b_grad = grad(func, argnums=(0, 1))(W, b)

val = func(W, b)
# print(val)
df_dW = grad(func, argnums=0)
df_db = grad(func, argnums=1)

print(df_dW(W, b))
print(df_db(W, b))  # should not be zero

