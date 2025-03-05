---
title: Discrete functions
parent: Bases
layout: default
nav_order: 2
---

### Scalar-valued functions

A degree of freedom vector `u_hat` and a (tensor) basis function `phi` can be passed to `get_u_h` to get a function $x \mapsto u_h(x) = \sum_i \hat u_i \phi_i(x)$.

This function will return a scalar (not an array with one entry).

### Vector-valued functions

To get these, `get_u_h_vec` will, with the same signature, return $x \mapsto u_h(x) = \sum_i \hat u_i \phi_i(x)$, where now the $\phi_i$ are vector-valued.