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

import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import quadax as quad
import chex

import matplotlib.pyplot as plt
import time

class AssemblyTests(unittest.TestCase):
# 1D projection
    def test_assembly(self):

        # Map to physical domain: unit cube to [-3,3]^3
        def F(x):
            return 6*x - jnp.ones_like(x)*3

        def F_inv(x):    
            return (x + jnp.ones_like(x)*3) / 6

        # Bases
        ns = (6, 7, 3)
        ps = (3, 3, 0)
        types = ('clamped', 'periodic', 'fourier')
        # Alan: added this since get_zero_form_basis wants the BCs
        BCs = ('dirichlet', 'dirichlet', 'dirichlet')
        basis0, shape0, N0 = get_zero_form_basis( ns, ps, types, BCs)
        basis1, shapes1, N1 = get_one_form_basis( ns, ps, types, BCs)
        
        print(N0, N1)

        x_q, w_q = quadrature_grid(
                    get_quadrature_composite(jnp.linspace(0, 1, ns[0] - ps[0] + 1), 15),
                    get_quadrature_composite(jnp.linspace(0, 1, ns[1] - ps[1] + 1), 15),
                    get_quadrature_periodic(16)(0,1))

        # Mass matrices
        _M0 = jit(get_mass_matrix_lazy_0(basis0, x_q, w_q, F))
        assemble_M0 = jit(lambda : assemble(_M0, jnp.arange(N0), jnp.arange(N0)))
        # sparse_assemble_M0 = jit(lambda: sparse_assemble_3d(_M0, shape0, 3))
        vmap_assemble_M0 = jit(lambda: assemble_full_vmap(_M0, jnp.arange(N0), jnp.arange(N0)))

        start = time.time()
        M0_a = assemble_M0()
        M0_a = jax.experimental.sparse.bcsr_fromdense(M0_a)
        M0_a @ jnp.ones(N0)
        end = time.time()
        print("Assemble compilation: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M0 = assemble_M0()
        #     M0[0,0]
        # end = time.time()
        # print("Assemble: " , (end - start)/3)

        # start = time.time()
        # M0_s = sparse_assemble_M0()
        # M0_s @ jnp.ones(N0)
        # end = time.time()
        # print("Sparse assemble compilation: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M0 = sparse_assemble_M0()
        #     M0[0,0]
        # end = time.time()
        # print("Sparse assemble: " , (end - start)/3)

        start = time.time()
        M0_v = vmap_assemble_M0()
        M0_v = jax.experimental.sparse.bcsr_fromdense(M0_v)
        M0_v @ jnp.ones(N0)
        end = time.time()
        print("Vmap assemble compilation: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M0 = vmap_assemble_M0()
        #     M0[0,0]
        # end = time.time()
        # print("Vmap assemble: " , (end - start)/3)

        # npt.assert_allclose(M0_a - M0_s, 0, atol = 1e-15)
        v = jnp.ones(N0)
        npt.assert_allclose(M0_a @ v, M0_v @ v, atol = 1e-15)
        
        # Mass matrices
        _M1 = jit(get_mass_matrix_lazy_1(basis1, x_q, w_q, F))
        assemble_M1 = jit(lambda : assemble(_M1, jnp.arange(N1), jnp.arange(N1)))
        # sparse_assemble_M1 = jit(lambda: sparse_assemble_3d_vec(_M1, shapes1, 3))
        vmap_assemble_M1 = jit(lambda: assemble_full_vmap(_M1, jnp.arange(N1), jnp.arange(N1)))

        start = time.time()
        M1_a = assemble_M1()
        end = time.time()
        print("Assemble compilation: " , end - start)
        start = time.time()
        M1_a = jax.experimental.sparse.bcsr_fromdense(M1_a)
        end = time.time()
        print("Vmap assemble sparsification: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M1 = assemble_M1()
        #     M1[0,0]
        # end = time.time()
        # print("Assemble: " , (end - start)/3)

        # start = time.time()
        # M1_s = sparse_assemble_M1()
        # M1_s @ jnp.ones(N1)
        # end = time.time()
        # print("Sparse assemble compilation: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M1 = sparse_assemble_M0()
        #     M1[0,0]
        # end = time.time()
        # print("Sparse assemble: " , (end - start)/3)

        start = time.time()
        M1_v = vmap_assemble_M1()
        end = time.time()
        print("Vmap assemble compilation: " , end - start)
        start = time.time()
        M1_v = jax.experimental.sparse.bcsr_fromdense(M1_v)
        end = time.time()
        print("Vmap assemble sparsification: " , end - start)
        # start = time.time()
        # for _ in range(3):
        #     M1 = vmap_assemble_M0()
        #     M1[0,0]
        # end = time.time()
        # print("Vmap assemble: " , (end - start)/3)
        
        # npt.assert_allclose(M1_a - M1_s, 0, atol = 1e-15)
        v = jnp.ones(N1)
        npt.assert_allclose(M1_a @ v, M1_v @ v, atol = 1e-15)