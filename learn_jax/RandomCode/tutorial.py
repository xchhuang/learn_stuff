# jax version: '0.2.24'

from jax import grad
import jax.numpy as jnp


def tanh(x):  # Define a function
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


def main():
    grad_tanh = grad(tanh)  # Obtain its gradient function
    print(grad_tanh(1.0))  # Evaluate it at x = 1.0
    # prints 0.4199743


if __name__ == '__main__':
    main()
