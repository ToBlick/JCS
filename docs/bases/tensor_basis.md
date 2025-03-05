---
title: Tensor Bases
parent: Bases
layout: default
nav_order: 2
---

### Tensor bases

The function `get_tensor_basis_fn(bases, shape)` builds from a tuple of basis functions (each with signature `x, i -> basis(x,i)`) and a tuple of integers (number of bases in every dimension) a tensor basis with linear indexing. 

The conversion from cartesian to linear indexing is done with `jax.numpy.unravel_index()`. 

This function works for any dimension.