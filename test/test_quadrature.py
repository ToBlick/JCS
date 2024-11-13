import unittest
from mhd_equilibria.quadratures import *
import numpy.testing as npt
from jax import numpy as jnp
from jax import jit, vmap
import quadax as quad
import jax
jax.config.update("jax_enable_x64", True)

class QuadratureTests(unittest.TestCase):
    
    def test_quad(self):
        
        x_q, w_q = get_quadrature_periodic(32)(0, 1)
        f = lambda x: jnp.sin(x * 20 * jnp.pi)**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), 0.5, rtol=1e-15)
        
        x_q, w_q = get_quadrature(31)(0, 1)
        f = lambda x: jnp.exp(x)
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.exp(1) - 1, rtol=1e-15)
        
        x_q, w_q = quadrature_grid(get_quadrature(31)(0, 1),
                                   get_quadrature_periodic(16)(0, 2*jnp.pi),
                                   get_quadrature_periodic(16)(0, 2*jnp.pi))
        
        # x_q, w_q = get_quadrature_grid(jnp.array([31, 31, 31]))(jnp.array([0, 0, 0]), 
        #                                                         jnp.array([1, 2*jnp.pi, 2*jnp.pi]), 
        #                                                         jnp.array([False, True, True]))
        
        f = lambda x: x[0] * jnp.exp(x[0]) * jnp.sin(x[1])**2 * jnp.cos(x[2])**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.pi**2, rtol=1e-15)