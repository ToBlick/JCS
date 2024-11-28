import unittest
from mhd_equilibria.quadratures import *
from mhd_equilibria.bases import *
import numpy.testing as npt
from jax import numpy as jnp
from jax import jit, vmap
import quadax as quad
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)

class QuadratureTests(unittest.TestCase):
    
    def test_quad(self):
        
        x_q, w_q = get_quadrature_periodic(32)(0, 1)
        f = lambda x: jnp.sin(x * 20 * jnp.pi)**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), 0.5, rtol=1e-15)
        
        x_q, w_q = get_quadrature_spectral(31)(0, 1)
        f = lambda x: jnp.exp(x)
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.exp(1) - 1, rtol=1e-15)
        
        x_q, w_q = quadrature_grid(get_quadrature_spectral(31)(0, 1),
                                   get_quadrature_periodic(16)(0, 2*jnp.pi),
                                   get_quadrature_periodic(16)(0, 2*jnp.pi))
        
        # x_q, w_q = get_quadrature_grid(jnp.array([31, 31, 31]))(jnp.array([0, 0, 0]), 
        #                                                         jnp.array([1, 2*jnp.pi, 2*jnp.pi]), 
        #                                                         jnp.array([False, True, True]))
        
        f = lambda x: x[0] * jnp.exp(x[0]) * jnp.sin(x[1])**2 * jnp.cos(x[2])**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.pi**2, rtol=1e-15)
     
    def test_basis_orth(self):   
        # check orthogonality of tensor basis:
        n_r = 7
        n_θ = 5
        n_φ = 8
        
        Omega = ((0, 1), (0, 2*jnp.pi), (0, 2*jnp.pi))
        bases = (get_legendre_fn_x(n_r, *Omega[0]), get_trig_fn(n_θ, *Omega[1]), get_trig_fn(n_φ, *Omega[2]))
        shape = (n_r, n_θ, n_φ)
        basis_fn = jit(get_tensor_basis_fn(bases, shape))
        
        x_q, w_q = quadrature_grid(get_quadrature_spectral(31)(*Omega[0]),
                                   get_quadrature_periodic(32)(*Omega[1]),
                                   get_quadrature_periodic(32)(*Omega[2]))
        
        for _ in range(10):
            i = np.random.randint(n_r * n_θ * n_φ)
            j = np.random.randint(n_r * n_θ * n_φ)
            while i == j:
                j = np.random.randint(n_r * n_θ * n_φ)
            ψ0 = vmap(basis_fn, (0, None))(x_q, i)
            ψ1 = vmap(basis_fn, (0, None))(x_q, j)
        
            npt.assert_allclose(jnp.sum(w_q * ψ0**2), 1, rtol=1e-6)
            npt.assert_allclose(jnp.sum(w_q * ψ1**2), 1, rtol=1e-6)
            npt.assert_allclose(jnp.sum(w_q * ψ0 * ψ1), 0, atol=1e-12)
            
        bases = (get_zernike_fn_x(n_r*n_θ, *Omega[0], *Omega[1]), get_trig_fn(n_φ, *Omega[2]))
        shape = (n_r*n_θ, n_φ)
        basis_fn = jit(get_zernike_tensor_basis_fn(bases, shape))
        def J_analytic(x):
            r, _, _ = x
            return r
        J_at_x = vmap(J_analytic)(x_q)
        
        for _ in range(10):
            i = np.random.randint(n_r * n_θ * n_φ)
            j = np.random.randint(n_r * n_θ * n_φ)
            while i == j:
                j = np.random.randint(n_r * n_θ * n_φ)
            ψ0 = vmap(basis_fn, (0, None))(x_q, i)
            ψ1 = vmap(basis_fn, (0, None))(x_q, j)
        
            npt.assert_allclose(jnp.sum(w_q * J_at_x * ψ0**2), 1, rtol=1e-6)
            npt.assert_allclose(jnp.sum(w_q * J_at_x * ψ1**2), 1, rtol=1e-6)
            npt.assert_allclose(jnp.sum(w_q * J_at_x * ψ0 * ψ1), 0, atol=1e-12)