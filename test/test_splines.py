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
        
        def F(x):
            return x
        n = 8
        p = 3
        ns = (n, n, 1)
        ps = (p, p, 1)
        
        nx = 64
        _x1 = jnp.linspace(0, 1, nx)
        _x2 = jnp.linspace(0, 1, nx)
        _x3 = jnp.zeros(1)
        _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
        _x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
        
        def f(x):
            x, y, ζ  = x
            return x**2 * jnp.cos(2*jnp.pi*y)
        def A(x):
            x, y, ζ  = x
            A1 = x * jnp.cos(2*jnp.pi*y)
            A2 = x**2 * jnp.cos(2*jnp.pi*y)
            A3 = (1-x)**2 * jnp.sin(2*jnp.pi*y)
            return jnp.array([A1, A2, A3])
        B = A
        g = f
            
        types = ('clamped', 'periodic', 'fourier')
        boundary = ('free', 'free', 'periodic')
        basis0, shape0,  N0 = get_zero_form_basis( ns, ps, types, boundary)
        basis1, shapes1, N1 = get_one_form_basis(  ns, ps, types, boundary)
        basis2, shapes2, N2 = get_two_form_basis(  ns, ps, types, boundary)
        basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)
            
        x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
            get_quadrature_periodic(1)(0,1))
            
        ### 0-form projection
        proj0 = get_l2_projection(basis0, x_q, w_q, N0)
        M00 = assemble(get_mass_matrix_lazy_00(basis0, x_q, w_q, F), jnp.arange(N0), jnp.arange(N0))
        f_hat = jnp.linalg.solve(M00, proj0(f))
        f_h = get_u_h(f_hat, basis0)
        npt.assert_allclose(vmap(f_h)(_x), vmap(f)(_x), atol=1e-2)
        print("f - f_h: ", jnp.linalg.norm(vmap(f)(_x) - vmap(f_h)(_x))/nx)
        ### exact gradient
        M11 = assemble(get_mass_matrix_lazy_11(basis1, x_q, w_q, F), jnp.arange(N1), jnp.arange(N1))
        D = assemble(get_gradient_matrix_lazy_01(basis0, basis1, x_q, w_q, F), jnp.arange(N0), jnp.arange(N1)).T
        df_hat = jnp.linalg.solve(M11, D @ f_hat)
        df_h = get_u_h_vec(df_hat, basis1)
        npt.assert_allclose(vmap(df_h)(_x), vmap(grad(f_h))(_x), atol=1e-12)
        print("grad(f) - grad(f_h): ", jnp.linalg.norm(vmap(df_h)(_x) - vmap(grad(f_h))(_x))/nx)
        ### 1-form projection
        proj1 = get_l2_projection(basis1, x_q, w_q, N1)
        A_hat = jnp.linalg.solve(M11, proj1(A))
        A_h = get_u_h_vec(A_hat, basis1)
        npt.assert_allclose(vmap(A_h)(_x), vmap(A)(_x), atol=1e-2)
        print("A - A_h: ", jnp.linalg.norm(vmap(A)(_x) - vmap(A_h)(_x))/nx)
        ### exact curl
        C = assemble(get_curl_matrix_lazy_12(basis1, basis2, x_q, w_q, F), jnp.arange(N1), jnp.arange(N2)).T
        M22 = assemble(get_mass_matrix_lazy_22(basis2, x_q, w_q, F), jnp.arange(N2), jnp.arange(N2))
        dA_hat = jnp.linalg.solve(M22, C @ A_hat) 
        dA_h = get_u_h_vec(dA_hat, basis2)
        npt.assert_allclose(vmap(dA_h)(_x), vmap(curl(A_h))(_x), atol=1e-12)
        print("curl(A) - curl(A_h): ", jnp.linalg.norm(vmap(dA_h)(_x) - vmap(curl(A_h))(_x))/nx)
        ### 2-form projection
        proj2 = get_l2_projection(basis2, x_q, w_q, N2)
        B_hat = jnp.linalg.solve(M22, proj2(B))
        B_h = get_u_h_vec(B_hat, basis2)
        npt.assert_allclose(vmap(B_h)(_x), vmap(B)(_x), atol=1e-2)
        print("B - B_h: ", jnp.linalg.norm(vmap(B)(_x) - vmap(B_h)(_x))/nx)
        ### exact divergence
        D = assemble(get_divergence_matrix_lazy_23(basis2, basis3, x_q, w_q, F), jnp.arange(N2), jnp.arange(N3)).T
        M33 = assemble(get_mass_matrix_lazy_33(basis3, x_q, w_q, F), jnp.arange(N3), jnp.arange(N3))
        dB_hat = jnp.linalg.solve(M33, D @ B_hat)
        dB_h = get_u_h(dB_hat, basis3)
        npt.assert_allclose(vmap(dB_h)(_x), vmap(div(B_h))(_x), atol=1e-12)
        print("div(B) - div(B_h): ", jnp.linalg.norm(vmap(dB_h)(_x) - vmap(div(B_h))(_x))/nx)
        ### 3-form projection
        proj3 = get_l2_projection(basis3, x_q, w_q, N3)
        g_hat = jnp.linalg.solve(M33, proj3(g))
        g_h = get_u_h(g_hat, basis3)
        npt.assert_allclose(vmap(g_h)(_x), vmap(g)(_x), atol=1e-2)
        print("g - g_h: ", jnp.linalg.norm(vmap(g)(_x) - vmap(g_h)(_x))/nx)
            
            
