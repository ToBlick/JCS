import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax import vmap, grad, jit
from functools import partial
import orthax

from mhd_equilibria.bases import _lin_to_cart

# def get_vector_basis_fn(bases, shape):
#     n_max = jnp.max(jnp.array(shape))
#     d = len(shape)
#     def basis_fn(x, I):
#         j, i = _lin_to_cart(I, (d, n_max))
#         i = i % shape[j]
#         val = bases[j](x, i)
#         return jnp.zeros(d).at[j].set(val)
#     return jit(basis_fn, static_argnums=(1,))

#TODO: This feels very inelegant
def get_scalar_basis_fn(basis, shape):
    def basis_fn(x, i):
        return jnp.ones(1)*(basis(x, i))
    return basis_fn

def get_vector_basis_fn(bases, shape):
    def basis_fn_0(x, i):
        return jnp.zeros(3).at[0].set(bases[0](x, i))
    def basis_fn_1(x, i):
        i -= shape[0]
        return jnp.zeros(3).at[1].set(bases[1](x, i))
    def basis_fn_2(x, i):
        i -= shape[0] + shape[1]
        return jnp.zeros(3).at[2].set(bases[2](x, i))
    def basis_fn(x, I):
        return jax.lax.cond(I < shape[0], basis_fn_0, basis_fn_1_or_2, x, I)
    def basis_fn_1_or_2(x, I):
        return jax.lax.cond(I < shape[0] + shape[1], basis_fn_1, basis_fn_2, x, I)
    return basis_fn

# def get_u_h_vec(u_hat, basis_fns):
#     # u_hat: d-tuple with n_j elements
#     _d = jnp.arange(len(u_hat), dtype=jnp.int32)
#     def u_h(x):
#         return jnp.array([ jnp.sum(u_hat[i] * vmap(basis_fns[i], (None, 0))(x, jnp.arange(len(u_hat[i]), dtype=jnp.int32))) for i in _d ])
#     return u_h