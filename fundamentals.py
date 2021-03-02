import jax
import jax.numpy as jnp
import numpy as np

x = np.zeros(10)
y= jnp.zeros(10)

x

y

x = np.random.rand(1000,1000)
y = jnp.array(x)

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 1 -r 1 np.dot(x,x)

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 1 -r 1 jnp.dot(y,y).block_until_ready()

"""## Automatic differentiation with grad"""

from jax import grad

def f(x):
  return 3*x**2 + 2*x + 5

def f_prime(x):
  return 6*x +2

grad(f)(1.0)

f_prime(1.0)

"""## XLA and Jit"""

# Commented out IPython magic to ensure Python compatibility.
from jax import jit

x = np.random.rand(1000,1000)
y = jnp.array(x)

def f(x):
  for _ in range(10):
      x = 0.5*x + 0.1* jnp.sin(x)
  return x

g = jit(f)



# %timeit -n 5 -r 5 f(y).block_until_ready()

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 g(y).block_until_ready()

"""## pmap"""

from jax import pmap

def f(x):
  return jnp.sin(x) + x**2

f(np.arange(4))
# pmap(f)(np.arange(4))

## Note:colab doesn't allow to attach multiple GPUs to test this

from functools import partial
from jax.lax import psum

@partial(pmap, axis_name="i")
def normalize(x):
  return x/ psum(x,'i')

normalize(np.arange(8.))

## Note:colab doesn't allow to attach multiple GPUs to test this

"""## vmap"""

from jax import vmap

def f(x):
  return jnp.square(x)

f(jnp.arange(5))
vmap(f)(jnp.arange(5))

"""## Pseudo Random Number Generator"""

from jax import random
key = random.PRNGKey(5)
random.uniform(key)

"""## Profiler"""

import jax.profiler

def func1(x):
  return jnp.tile(x, 10) * 0.5

def func2(x):
  y = func1(x)
  return y, jnp.tile(x, 10) + 1

x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
y, z = func2(x)

z.block_until_ready()

jax.profiler.save_device_memory_profile("memory.prof")