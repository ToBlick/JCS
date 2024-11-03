from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

# def get_curl_form(basis_fn):
#     def _C(i,j,x):
#         # compute the inner product of the Laplacian of the ith basis function
#         # with the jth basis function
#         gradϕ_j = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, j)
#         gradϕ_i = vmap(jax.grad(basis_fn, argnums=0), (0, None))(x, i)
#         return -jnp.sum(gradϕ_i * gradϕ_j) / x.shape[0]

# # assemble
# K = vmap(vmap(_K, (0, None, None)), (None, 0, None))(n_s, n_s, x_q)