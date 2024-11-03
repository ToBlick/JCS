import unittest
from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
import numpy.testing as npt
import numpy as np
from jax import numpy as jnp
from jax import vmap, jit, grad
import jax
from functools import partial

import matplotlib.pyplot as plt

class ProjectionTests(unittest.TestCase):
    
    # @partial(jax.jit, static_argnames=['self'])
    def test_inprod(self):
        def f(x):
            return jnp.dot(x,x)
        def g(x):
            return x[1]
    
        L = 1.0
        nx = 64
        h = 1/nx
        _x = jnp.linspace(h/2, 1-h/2, nx) #midpoints
        x = jnp.array(jnp.meshgrid(_x, _x, _x)) # shape 3, nx, nx, nx
        x = x.transpose(1, 2, 3, 0).reshape(nx**3, 3)
        w = h**3
        
        J = lambda x: 1.0        
        npt.assert_allclose(l2_product(f, g, lambda x: 1.0, x, w), 7/12, rtol=1e-4)
        
        def F(x):
            return x
        def G(x):
            return x
        npt.assert_allclose(l2_product(F, G, J, x, w), 1.0, rtol=1e-4)
        
        def f(x):
            return x[0]**3 * jnp.sin(2*jnp.pi*x[1]/L) + jnp.sin(4*jnp.pi*x[2]/L) + 4 * x[0]**2 * jnp.sin(8*jnp.pi*x[2]/L)
        
        n1, n2, n3 = 4, 6, 8
        _bases = (get_legendre_fn_x(n1, 0, 1), get_trig_fn_x(n2, 0, 1), get_trig_fn_x(n3, 0, 1))
        shape = (n1, n2, n3)
        basis_fn = get_basis_fn(_bases, shape) # basis_fn(x, k)
        
        l2_proj = get_l2_projection(basis_fn, J, x, w, n1*n2*n3)
        f_hat = l2_proj(f)
        # f_hat = vmap(lambda i: l2_product(f, lambda x: basis_fn(x, i), J, x, w), 0)(jnp.arange(n1*n2*n3))
        f_h = get_u_h(f_hat, basis_fn)
        
        # plt.plot(f_hat)
        # plt.show()
        
        def err(x):
            return (f(x) - f_h(x))**2
        
        f_slice = lambda x: f(jnp.array([x, x, x]))
        f_h_slice = lambda x: f_h(jnp.array([x, x, x]))   
        
        # plt.plot(_x, vmap(f_slice)(_x), label='f')
        # plt.plot(_x, vmap(f_h_slice)(_x), label='f_h')
        # plt.legend()
        # plt.show()
        
        npt.assert_allclose(jnp.sqrt(integral(err, J, x, w)), 0, atol=5e-3)