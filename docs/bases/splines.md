---
title: Splines
parent: Bases
nav_order: 2
---

### 1d splines

Splines are defined recursively with the base case an indicator function on $[0,1]$.

The spline basis is represented as a function taking a number of arguments:
```python
def spline(x, i, T, p, n, type):
    ...
```
* `x` is the evaluation point.
* `i` is the number of the spline to be evaluated.
* `T` is the knot vector. For a spline of order zero, spline `i` is the indicator function of `[T[i], T[i+1])`.
* `p` is the order of the spline.
* `n` is the number of spline functions in the basis.
* `type` is either "clamped" or not (in which case we get a periodic spline)

The way spline bases are generated is through the function `get_spline(n, p, type='clamped')`, which constructs the knot vector and returns itself function `_spline(x, i)` that can be used to evaluate spline basis function number `i` at point `x`.
