# %%
import unittest
from mhd_equilibria.splines import *
from mhd_equilibria.forms import *
from mhd_equilibria.bases import *
from mhd_equilibria.quadratures import *

import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import quadax as quad
import chex

import matplotlib.pyplot as plt

class SplineTests(unittest.TestCase):
# 1D projection
    def test_splines(self):
        
        n = 32
        p = 3
        for type in ["clamped", "periodic"]:

            sp = jit(get_spline(n, p, type))

            # if type == 'periodic':
            #     n = n - p

            def f(x):
                return jnp.cos(2*jnp.pi*x)
            x_q_1d, w_q_1d = get_quadrature(41)(0, 1)
            _M = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
            M = assemble(_M, jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32))
                    
            npt.assert_allclose(M - M.T, 0, atol=1e-16)
            proj = get_l2_projection(sp, x_q_1d, w_q_1d, n)
            f_hat = jnp.linalg.solve(M, proj(f))
            f_h = get_u_h(f_hat, sp)
            nx = 256
            _x = jnp.linspace(0, 1, nx)
            npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), rtol=1e-4)
            
# 1D projection
    def test_dx_splines(self):
        
        n = 32
        p = 3
        for type in ["clamped", "periodic"]:

            sp = jit(get_spline(n, p, type))

            def f(x):
                return jnp.cos(2*jnp.pi*x)
            x_q_1d, w_q_1d = get_quadrature(41)(0, 1)
            _M = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
            M = assemble(_M, jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32))
            proj = get_l2_projection(sp, x_q_1d, w_q_1d, n)
            f_hat = jnp.linalg.solve(M, proj(f))
            dx_f_h = grad(get_u_h(f_hat, sp))
            
            n_dx = n-1
            sp_dx = jit(get_spline(n_dx, p-1, type))
            
            # project dx_f_h onto sp_dx
            _M = get_mass_matrix_lazy(sp_dx, x_q_1d, w_q_1d, None)
            M = assemble(_M, 
                jnp.arange(n_dx, dtype=jnp.int32), 
                jnp.arange(n_dx, dtype=jnp.int32))
            proj = get_l2_projection(sp_dx, x_q_1d, w_q_1d, n_dx)
            f_dx_hat = jnp.linalg.solve(M, proj(dx_f_h))
            f_h_dx = get_u_h(f_dx_hat, sp_dx)
            
            nx = 256
            _x = jnp.linspace(0, 1, nx)
            npt.assert_allclose(vmap(f_h_dx)(_x), vmap(dx_f_h)(_x), atol=1e-15)

    def test_3d_splines(self):
        # Make a box-torus: [0, 1] x [0, 1] x [0, 1] periodic in 3rd direction
        n_r = 10
        p_r = 3
        n_θ = 10
        p_θ = 3
        n_ζ = 3
        N = n_r * n_θ * n_ζ
        Omega = ((0, 1), (0, 1), (0, 1))
                
        basis_r = jit(get_spline(n_r, p_r, 'clamped'))
        basis_θ = jit(get_spline(n_θ, p_θ, 'clamped'))
        basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))
        basis = get_tensor_basis_fn((basis_r, basis_θ, basis_ζ), (n_r, n_θ, n_ζ))

        @jit
        def f(x):
            x, y, ζ  = x
            r = jnp.sqrt(jnp.sum(( jnp.array([x,y]) - jnp.ones(2)/2)**2))
            return jnp.cos(2*jnp.pi*ζ) * jnp.exp(-10*r**2)

        x_q, w_q = quadrature_grid(get_quadrature(41)(*Omega[0]),
                                get_quadrature(41)(*Omega[1]),
                                get_quadrature_periodic(16)(*Omega[2]))

        _M = jit(get_mass_matrix_lazy(basis, x_q, w_q, None))
        M = assemble(_M, 
            jnp.arange(N, dtype=jnp.int32), 
            jnp.arange(N, dtype=jnp.int32))

        npt.assert_allclose(M - M.T, 0, atol=1e-16)
        proj = get_l2_projection(basis, x_q, w_q, N)
        f_hat = jnp.linalg.solve(M, proj(f))
        f_h = jit(get_u_h(f_hat, basis))

        nx = 32
        _r = jnp.linspace(0, 1, nx)
        _θ = jnp.linspace(0, 1, nx)
        _z = jnp.linspace(0, 1, nx)
        _x = jnp.array(jnp.meshgrid(_r, _θ, _z)) # shape 3, n_x, n_x, n_x
        _x = _x.transpose(1, 2, 3, 0).reshape((nx)**3, 3)

        npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), rtol=1e-2)
        
# # %%
# n_r = 10
# p_r = 3
# n_θ = 10
# p_θ = 3
# n_ζ = 3

# N = n_r * n_θ * n_ζ
# Omega = ((0, 1), (0, 1), (0, 1))
        
# basis_r = jit(get_spline(n_r, p_r, 'clamped'))
# basis_θ = jit(get_spline(n_θ, p_θ, 'clamped'))
# basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))
# zero_form_basis = get_tensor_basis_fn((basis_r, basis_θ, basis_ζ), (n_r, n_θ, n_ζ))
# @jit
# def f(x):
#     x, y, ζ  = x
#     r = jnp.sqrt(jnp.sum(( jnp.array([x,y]) - jnp.ones(2)/2)**2))
#     return jnp.cos(2*jnp.pi*ζ) * jnp.exp(-10*r**2)
# x_q, w_q = quadrature_grid(get_quadrature(41)(*Omega[0]),
#                         get_quadrature(41)(*Omega[1]),
#                         get_quadrature_periodic(16)(*Omega[2]))
# _M = jit(get_mass_matrix_lazy(zero_form_basis, x_q, w_q, None))
# M = assemble(_M, 
#     jnp.arange(N, dtype=jnp.int32), 
#     jnp.arange(N, dtype=jnp.int32))
# npt.assert_allclose(M - M.T, 0, atol=1e-16)
# proj = get_l2_projection(zero_form_basis, x_q, w_q, N)
# f_hat = jnp.linalg.solve(M, proj(f))
# f_h = jit(get_u_h(f_hat, zero_form_basis))