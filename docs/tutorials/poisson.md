---
title: Two-dimensional Poisson Problem
parent: Tutorials
layout: default
nav_order: 2
---

### Poisson equation on a square

All routines in JCS are written with non-trivial mapping from logical to physical domain in mind, hence we need to define a mapping in any case.
```python
def F(x):
    return x
```

Next, we define some analytical solution for $\Delta u = f$.
```python
def f(x):
    return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
def u(x):
    return jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
```

...etc