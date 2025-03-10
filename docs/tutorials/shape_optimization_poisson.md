---
title: Two-dimensional Poisson Problem
parent: Tutorials
layout: default
nav_order: 2
---

### Optimizing a shape to fit the solution of a Poisson problem

In this script, we
* Solve a Poisson problem on a deformed disc-shaped domain. The domain is defined through the map $F: \hat x \mapsto x$, which depends on some parameter vector $\alpha$.
* Calculate the mismatch between the solution $u$ and a given desired function $\bar u$
* Optimize the deformation of the domain to find the optimal $\alpha^*$ such that
    $$
    \begin{align}
        \Delta u &= f \quad &x \in F(\alpha^*)(\hat \Omega)
        u &= 0 \quad &x \in F(\alpha^*)(\partial \hat \Omega)
    \end{align}
    $$

#### Define the mapping

The map $F$ is given by
$$
\begin{align}
    (r, \chi, \zeta) \mapsto ( \varsigma(\chi) r \cos(2 \pi \chi),  \varsigma(\chi) r \sin(2 \pi \chi), 2 \pi \zeta )
\end{align}
$$
The function $\varsigma$ is what depends on $\alpha$, namely it is a periodic function with $\alpha$ its Fourier components.
```python
nα = α.shape[0]
map_basis_fn = get_tensor_basis_fn((get_trig_fn(nα, 0, 1), ), (nα,))
α = α.at[0].set(1.0)
def ς(θ):
    return get_u_h(α, map_basis_fn)(θ * jnp.ones(1))
```
That defines the whole mapping $F$ as
```python
R0 = 0
Y0 = 0
def R(x):
    r, χ, z = x
    return ς(χ) * r * jnp.cos(2 * jnp.pi * χ)
def Y(x):
    r, χ, z = x
    return ς(χ) * r * jnp.sin(2 * jnp.pi * χ)
def F(x):
    r, χ, ζ = x
    return jnp.array([R_h(x), Y_h(x), 2 * jnp.pi * ζ])
```
