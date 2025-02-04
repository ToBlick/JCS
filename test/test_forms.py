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
from mhd_equilibria.pullbacks import *

import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import quadax as quad
import chex

import matplotlib.pyplot as plt
import time

class FormsTests(unittest.TestCase):
# 1D projection
    def test_assembly(self):

        alpha = jnp.pi/2
        # F maps the logical domain (unit cube) to the physical one by rotating it by 90 degrees
        def F(x):
            return jnp.array([ [ jnp.cos(alpha), jnp.sin(alpha), 0],
                               [-jnp.sin(alpha), jnp.cos(alpha), 0],
                               [0              , 0             , 1] ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
        def F_inv(x):
            return jnp.array([ [jnp.cos(alpha), -jnp.sin(alpha), 0],
                               [jnp.sin(alpha),  jnp.cos(alpha), 0],
                               [0             , 0              , 1]]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2

        n = 16
        p = 3
        ns = (n, n, 1)
        ps = (p, p, 1)
        types = ('clamped', 'clamped', 'fourier')
        boundary = ('free', 'free', 'periodic')
        basis0, shape0, N0  = get_zero_form_basis( ns, ps, types, boundary)
        basis1, shapes1, N1 = get_one_form_basis(  ns, ps, types, boundary)
        basis2, shapes2, N2 = get_two_form_basis(  ns, ps, types, boundary)
        basis3, shapes3, N3 = get_three_form_basis(ns, ps, types, boundary)

        x_q, w_q = quadrature_grid(
            get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
            get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
            get_quadrature_periodic(1)(0,1))

        nx = 32
        _x1 = jnp.linspace(0, 1, nx)
        _x2 = jnp.linspace(0, 1, nx)
        _x3 = jnp.zeros(1)
        _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
        _x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)

        D = assemble(get_divergence_matrix_lazy_23(basis2, basis3, x_q, w_q, F), jnp.arange(N2), jnp.arange(N3)).T
        M22 = assemble(get_mass_matrix_lazy_22(basis2, x_q, w_q, F), jnp.arange(N2), jnp.arange(N2))

        def f(x):
            return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
        def u(x):
            return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])
        proj = get_l2_projection(basis3, x_q, w_q, N3)

        ###
        # D σ = p(f) in L2
        # Mσ = D.T u in Hdiv
        # -> σ = M⁻¹ D.T u
        # -> Dσ = D M⁻¹ D.T u = p(f)
        # -> (D M⁻¹ D.T) u = p(f) 
        # or
        #
        # | M  -D.T | | σ | = |  0   |
        # | D   0   | | u | = | p(f) |
        #
        ###
        Q = jnp.block([[M22, -D.T], 
                       [D, jnp.zeros((N3, N3))]])
        b = jnp.block([jnp.zeros(N2), proj(f)])

        sigma_hat, u_hat = jnp.split(jnp.linalg.solve(Q, b), [N2])

        u_h = pullback_3form(get_u_h(u_hat, basis3), F_inv)
        sigma_h = get_u_h_vec(sigma_hat, basis2)

        def err(x):
            return jnp.sum((u_h(x) + u(x))**2)
        # + instead of minus because the rotation by 90 degrees flips the sign

        error = jnp.sqrt(integral(err, x_q, w_q))
        print(error)
        npt.assert_allclose(error, 0, atol = 1e-3)