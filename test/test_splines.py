#%%
import unittest

import jax.experimental
import jax.experimental.sparse
from mhd_equilibria.splines import *
from mhd_equilibria.forms import *
from mhd_equilibria.bases import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.bases import *
from mhd_equilibria.vector_bases import *
from mhd_equilibria.operators import *
from mhd_equilibria.projections import *

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
            def f(x):
                return jnp.cos(2*jnp.pi*x)
            x_q_1d, w_q_1d = get_quadrature_composite(jnp.linspace(0, 1, n - p + 1), 15)
            _M0 = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
            M0 = assemble(_M0, jnp.arange(n), jnp.arange(n))
                    
            npt.assert_allclose(M0 - M0.T, 0, atol=1e-16)
            proj = get_l2_projection(sp, x_q_1d, w_q_1d, n)
            f_hat = jnp.linalg.solve(M0, proj(f))
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
            x_q_1d, w_q_1d = get_quadrature_composite(jnp.linspace(0, 1, n - p + 1), 15)
            _M0 = get_mass_matrix_lazy(sp, x_q_1d, w_q_1d, None)
            M0 = assemble(_M0, jnp.arange(n), jnp.arange(n))
            proj = get_l2_projection(sp, x_q_1d, w_q_1d, n)
            f_hat = jnp.linalg.solve(M0, proj(f))
            f_h = get_u_h(f_hat, sp)
            dx_f_h = grad(f_h)
            
            sp_dx = jit(get_spline(n-1, p-1, type))
            
            # project dx_f_h onto sp_dx
            _M0 = get_mass_matrix_lazy(sp_dx, x_q_1d, w_q_1d, None)
            M0 = assemble(_M0, jnp.arange(n-1), jnp.arange(n-1))
            proj = get_l2_projection(sp_dx, x_q_1d, w_q_1d, n-1)
            f_dx_hat = jnp.linalg.solve(M0, proj(dx_f_h))
            f_h_dx = get_u_h(f_dx_hat, sp_dx)
            
            nx = 256
            _x = jnp.linspace(0, 1, nx)
            npt.assert_allclose(vmap(f)(_x), vmap(f_h)(_x), rtol=1e-4)
            npt.assert_allclose(vmap(f_h_dx)(_x), vmap(dx_f_h)(_x), atol=1e-12)

    def test_3d_splines(self):
        # Make a box-torus: [0, 1] x [0, 1] x [0, 1] periodic in 3rd direction
        n_r = 8
        p_r = 3
        n_θ = 10
        p_θ = 3
        n_ζ = 3
        
        Omega = ((0, 1), (0, 1), (0, 1))
                
        basis_r = jit(get_spline(n_r, p_r, 'clamped'))
        basis_θ = jit(get_spline(n_θ, p_θ, 'periodic'))
        basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))
        basis = get_tensor_basis_fn((basis_r, basis_θ, basis_ζ), (n_r, n_θ, n_ζ))
        N = n_r * n_θ * n_ζ

        @jit
        def f(x):
            x, y, ζ  = x
            r = jnp.sqrt(jnp.sum(( jnp.array([x,y]) - jnp.ones(2)/2)**2))
            return jnp.cos(2*jnp.pi*ζ) * jnp.exp(-10*r**2)
        
        x_q, w_q = quadrature_grid(
                    get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
                    get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
                    get_quadrature_periodic(16)(*Omega[2]))

        _M0 = jit(get_mass_matrix_lazy(basis, x_q, w_q, None))
        M0 = assemble(_M0, jnp.arange(N), jnp.arange(N))

        npt.assert_allclose(M0 - M0.T, 0, atol=1e-16)
        proj = get_l2_projection(basis, x_q, w_q, N)
        f_hat = jnp.linalg.solve(M0, proj(f))
        f_h = jit(get_u_h(f_hat, basis))

        nx = 32
        _r = jnp.linspace(0, 1, nx)
        _θ = jnp.linspace(0, 1, nx)
        _z = jnp.linspace(0, 1, nx)
        _x = jnp.array(jnp.meshgrid(_r, _θ, _z)) # shape 3, n_x, n_x, n_x
        _x = _x.transpose(1, 2, 3, 0).reshape((nx)**3, 3)

        npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), rtol=1e-1)
        
    def test_3d_splines_grad(self):
        # %%
        n_r = 8
        p_r = 3
        n_θ = 8
        p_θ = 3
        n_ζ = 3
        Omega = ((0, 1), (0, 1), (0, 1))
        
        def F(x):
            return x
        
        basis_r = jit(get_spline(n_r, p_r, 'clamped'))
        basis_θ = jit(get_spline(n_θ, p_θ, 'periodic'))
        basis_ζ = jit(get_trig_fn(n_ζ, *Omega[2]))
        
        basis_dr = jit(get_spline(n_r-1, p_r-1, 'clamped'))
        basis_dθ = jit(get_spline(n_θ-1, p_θ-1, 'periodic')) 
        basis_dζ = basis_ζ
        
        basis0 = get_tensor_basis_fn(
                    (basis_r, basis_θ, basis_ζ), 
                    (n_r, n_θ, n_ζ))
        N0 = n_r * n_θ * n_ζ
        
        # basis0, N0 = get_slab_zero_form_basis((n_r, n_θ, n_ζ), (p_r, p_θ, 0))
        # basis1, N1 = get_slab_one_form_basis((n_r, n_θ, n_ζ), (p_r, p_θ, 0))
        # basis2, N2 = get_slab_two_form_basis((n_r, n_θ, n_ζ), (p_r, p_θ, 0))
        # basis3, N3 = get_slab_three_form_basis((n_r, n_θ, n_ζ), (p_r, p_θ, 0))
        
        @jit
        def f(x):
            x, y, ζ  = x
            r = jnp.sqrt(jnp.sum(( jnp.array([x,y]) - jnp.ones(2)/2)**2))
            return  jnp.exp(-10*r**2) * jnp.cos(2*jnp.pi*ζ) #
        x_q, w_q = quadrature_grid(
            # get_quadrature_spectral(41)(0, 1),
            # get_quadrature_spectral(41)(0, 1),
            get_quadrature_composite(jnp.linspace(0, 1, n_r - p_r + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, n_θ - p_θ + 1), 15),
            get_quadrature_periodic(8)(*Omega[2]))
        
        _M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
        M0 = get_sparse_operator(_M0, jnp.arange(N0), jnp.arange(N0))
        
        proj0 = get_l2_projection(basis0, x_q, w_q, N0)
        # f_hat = jnp.linalg.solve(M0, proj0(f))
        f_hat = jax.experimental.sparse.linalg.spsolve(
                    M0.data, M0.indices, M0.indptr, proj0(f))
        f_h = jit(get_u_h(f_hat, basis0))
        
        nx = 8
        _r = jnp.linspace(0, 1, nx)
        _θ = jnp.linspace(0, 1, nx)
        _z = jnp.linspace(0, 1, nx)
        _x = jnp.array(jnp.meshgrid(_r, _θ, _z)) # shape 3, n_x, n_x, n_x
        _x = _x.transpose(1, 2, 3, 0).reshape((nx)**3, 3)
        
        npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), rtol=1e-1)

        basis1_1 = get_tensor_basis_fn(
                    (basis_dr, basis_θ, basis_ζ), 
                    (n_r - 1, n_θ, n_ζ))
        N1_1 = (n_r - 1) * n_θ * n_ζ

        basis1_2 = get_tensor_basis_fn(
                    (basis_r, basis_dθ, basis_ζ), 
                    (n_r, n_θ - 1, n_ζ))
        N1_2 = n_r * (n_θ - 1) * n_ζ

        basis1_3 = get_tensor_basis_fn(
                    (basis_r, basis_θ, basis_dζ), 
                    (n_r, n_θ, n_ζ))
        N1_3 = n_r * n_θ * n_ζ

        basis1 = get_vector_basis_fn(
                    (basis1_1, basis1_2, basis1_3), 
                    (N1_1, N1_2, N1_3))
        N1 = N1_1 + N1_2 + N1_3

        _M1 = jit(get_mass_matrix_lazy_1(basis1, x_q, w_q, F))
        M1_1 = assemble(_M1, jnp.arange(N1_1), jnp.arange(N1_1))
        M1_2 = assemble(_M1, jnp.arange(N1_1, N1_1+N1_2), jnp.arange(N1_1, N1_1+N1_2))
        M1_3 = assemble(_M1, jnp.arange(N1_1+N1_2, N1), jnp.arange(N1_1+N1_2, N1))
        M1 = jax.experimental.sparse.bcsr_fromdense(jax.scipy.linalg.block_diag(M1_1, M1_2, M1_3))
        proj1 = get_l2_projection(basis1, x_q, w_q, N1)
        gradf_hat = proj1(grad(f_h))

        gradf_hat = jax.experimental.sparse.linalg.spsolve(
                        M1.data, M1.indices, M1.indptr, proj1(grad(f_h)))
        gradf_h = jit(get_u_h_vec(gradf_hat, basis1))

        npt.assert_allclose(vmap(grad(f_h))(_x), vmap(gradf_h)(_x), atol=1e-12)
        npt.assert_allclose(vmap(curl(gradf_h))(_x), 0, atol=1e-12)
        npt.assert_allclose(vmap(gradf_h)(_x), vmap(grad(f))(_x), atol=2e-1)

        def A(x):
            x, y, ζ  = x
            A1 = (x - 0.5)**2 * y * jnp.cos(jnp.pi*ζ)
            A2 = x * y**2 * jnp.sin(jnp.pi*ζ)
            A3 = x**2 * y * jnp.cos(jnp.pi*ζ)
            return jnp.array([A1, A2, A3])
        
        basis2_1 = get_tensor_basis_fn((basis_r, basis_dθ, basis_dζ), 
                                    (n_r, n_θ-1, n_ζ))
        N2_1 = (n_r) * (n_θ - 1) * n_ζ
        basis2_2 = get_tensor_basis_fn(
                    (basis_dr, basis_θ, basis_dζ), 
                    (n_r-1, n_θ, n_ζ))
        N2_2 = (n_r - 1) * (n_θ) * n_ζ
        basis2_3 = get_tensor_basis_fn(
                    (basis_dr, basis_dθ, basis_ζ), 
                    (n_r-1, n_θ-1, n_ζ))
        N2_3 = (n_r - 1) * (n_θ - 1) * n_ζ
        basis2 = get_vector_basis_fn(
                    (basis2_1, basis2_2, basis2_3), 
                    (N2_1, N2_2, N2_3))
        N2 = N2_1 + N2_2 + N2_3

        _M2 = jit(get_mass_matrix_lazy_2(basis2, x_q, w_q, F))
        M2_1 = assemble(_M2, jnp.arange(N2_1), jnp.arange(N2_1))
        M2_2 = assemble(_M2, jnp.arange(N2_1, N2_1 + N2_2), jnp.arange(N2_1, N2_1 + N2_2))
        M2_3 = assemble(_M2, jnp.arange(N2_1 + N2_2, N2), jnp.arange(N2_1 + N2_2, N2))
        M2 = jax.experimental.sparse.bcsr_fromdense(jax.scipy.linalg.block_diag(M2_1, M2_2, M2_3))
        proj2 = get_l2_projection(basis2, x_q, w_q, N2)
        # A_hat = jnp.linalg.solve(M1, proj1(A))
        A_hat = jax.experimental.sparse.linalg.spsolve(
                        M1.data, M1.indices, M1.indptr, proj1(A))
        A_h = jit(get_u_h_vec(A_hat, basis1))
        
        # curlA_hat = jnp.linalg.solve(M2, proj2(curl(A_h)))
        curlA_hat = jax.experimental.sparse.linalg.spsolve(
                        M2.data, M2.indices, M2.indptr, proj2(curl(A_h)))
        curlA_h = jit(get_u_h_vec(curlA_hat, basis2))
        
        npt.assert_allclose(vmap(curlA_h)(_x), vmap(curl(A_h))(_x), atol=1e-12)
        npt.assert_allclose(vmap(curl(gradf_h))(_x), 0, atol=1e-12)
        npt.assert_allclose(vmap(div(curlA_h))(_x), 0, atol=1e-12)
        npt.assert_allclose(vmap(curlA_h)(_x), vmap(curl(A))(_x), atol=5e0)
        
        basis3 = get_tensor_basis_fn(
                    (basis_dr, basis_dθ, basis_dζ), 
                    (n_r - 1, n_θ - 1, n_ζ))
        N3 = (n_r - 1) * (n_θ - 1) * n_ζ
        
        _M3 = jit(get_mass_matrix_lazy_3(basis3, x_q, w_q, F))
        M3 = get_sparse_operator(_M3, jnp.arange(N3), jnp.arange(N3))

        proj3 = get_l2_projection(basis3, x_q, w_q, N3)
        E_hat = jax.experimental.sparse.linalg.spsolve(
                    M2.data, M2.indices, M2.indptr, proj2(grad(f)))
        E_h = jit(get_u_h_vec(E_hat, basis2))
        
        div_E_hat = jax.experimental.sparse.linalg.spsolve(
                        M3.data, M3.indices, M3.indptr, proj3(div(E_h)))
        div_E_h = jit(get_u_h(div_E_hat, basis3))
        
        npt.assert_allclose(vmap(div_E_h)(_x), vmap(div(E_h))(_x), atol=1e-12)
