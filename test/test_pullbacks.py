import unittest
from mhd_equilibria.operators import *
from mhd_equilibria.projections import *
from mhd_equilibria.coordinate_transforms import *
from mhd_equilibria.pullbacks import *

import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import quadax as quad
import chex

class PullbackTests(unittest.TestCase):
    
    def test_forms(self):
        key = jax.random.PRNGKey(0)
        n = 1000
        x_hat = jax.random.uniform(key, (n, 3))
        R0 = 2.0
        def f(x):
            return x[0]**2 + jnp.sin(x[1]) + x[1]**2 + x[2] + x[2]**2 * jnp.cos(x[0])
        def B(x):
            return jnp.array([x[0]**2 + x[2]*jnp.sin(x[1]), 
                              x[2]**2 + x[1], 
                              x[1]**2 * jnp.cos(x[0])])
        A = B
        p = f

        for (F, F_inv) in zip([cart_to_cyl, get_cart_to_tok(R0)], 
                              [cyl_to_cart, get_tok_to_cart(R0)]):
            x = jax.vmap(F)(x_hat)
            p_hat =    jit(pullback_0form(p, F))
            p_hathat = jit(pullback_0form(p_hat, F_inv))
            A_hat =    jit(pullback_1form(A, F))
            A_hathat = jit(pullback_1form(A_hat, F_inv))
            B_hat =    jit(pullback_2form(B, F))
            B_hathat = jit(pullback_2form(B_hat, F_inv))
            f_hat =    jit(pullback_3form(f, F))
            f_hathat = jit(pullback_3form(f_hat, F_inv))

            npt.assert_allclose(vmap(p)(x), vmap(p_hathat)(x), atol=1e-6)
            npt.assert_allclose(vmap(A)(x), vmap(A_hathat)(x), atol=1e-6)
            npt.assert_allclose(vmap(B)(x), vmap(B_hathat)(x), atol=1e-6)
            npt.assert_allclose(vmap(f)(x), vmap(f_hathat)(x), atol=1e-6)
            
    def test_logicalspace_int(self):
        
        n = 128
        hx, hz, hphi = 1 / n, 1 / n, 2 * jnp.pi / n
        _x = jnp.linspace(0 + hx/2, 1 - hx/2, n)
        _phi = jnp.linspace(0 + hphi/2, 2 * jnp.pi - hphi/2, n)
        _z = jnp.linspace(0 + hz/2, 1 - hz/2, n)
        x = jnp.array(jnp.meshgrid(_x, _phi, _z))
        x_hat = x.transpose(1, 2, 3, 0).reshape(n**3, 3)
        w_q = jnp.ones(n**3) * hx * hphi * hz
        
        # key = jax.random.PRNGKey(0)
        # x_hat = jax.random.uniform(key, (n, 3))
        R0 = 1.5
        
        def bump(x):
            return 1e2 * jnp.exp(-0.5 * jnp.sum((x - jnp.array([0.5, jnp.pi, 0.5]))**2 / 0.1**2))
        
        @jit 
        def f_hat(x):
            return (x[0] + x[1]) * jnp.cos(x[2])
        @jit 
        def B_hat(x):
            return jnp.array([(x[2])*jnp.sin(x[0] * jnp.pi), 
                              (x[2] + x[1]) * jnp.sin(x[1]), 
                              (x[1] + x[0]) * jnp.sin(x[2] * jnp.pi)])
        @jit 
        def A_hat(x):
            return jnp.array([(x[0] + x[1]) * jnp.sin(x[1]) * jnp.sin(x[2] * jnp.pi), 
                              x[2] * x[0] * jnp.sin(x[0] * jnp.pi) * jnp.sin(x[2] * jnp.pi), 
                              (x[2] + x[1]) * jnp.sin(x[0] * jnp.pi) * jnp.sin(x[1])])
        @jit 
        def p_hat(x):
            return jnp.sin(x[0] * jnp.pi) * jnp.sin(x[1]) * jnp.sin(x[2] * jnp.pi) * (x[0] + x[1] * x[2])
        
        for (F, F_inv) in zip([cart_to_cyl, get_cart_to_tok(R0)], 
                              [cyl_to_cart, get_tok_to_cart(R0)]):
            x = jax.vmap(F)(x_hat)
            p = jit(pullback_0form(p_hat, F_inv))
            A = jit(pullback_1form(A_hat, F_inv))
            B = jit(pullback_2form(B_hat, F_inv))
            f = jit(pullback_3form(f_hat, F_inv))
            
            def J(x_hat):
                return jnp.linalg.det(jax.jacfwd(F)(x_hat))
            def J_inv(x):
                return 1/jnp.linalg.det(jax.jacfwd(F_inv)(x))
                     
            integrand_pf =     vmap(p)(x) * vmap(f)(x) * vmap(J)(x_hat)
            # this is p_hat(x_hat) f_hat(x_hat) dx_hat
            integrand_pf_hat = vmap(p_hat)(x_hat) * vmap(f_hat)(x_hat) 
            # this is p(x) f(x) dx
            
            def inprod_hat(x):
                return A_hat(x) @ B_hat(x)
            
            def inprod(x):
                return A(x) @ B(x)
            
            integrand_AB = vmap(inprod)(x) * vmap(J)(x_hat)
            integrand_AB_hat = vmap(inprod_hat)(x_hat)
            
            npt.assert_allclose(integrand_pf, integrand_pf_hat, atol=1e-9)
            npt.assert_allclose(integrand_AB, integrand_AB_hat, atol=1e-9)
            
            # key = jax.random.PRNGKey(0)
            # x_mc =  jax.random.uniform(key, (n**3, 3), 
            #                            minval=jnp.array((-2.5, -2.5, -1)),
            #                            maxval=jnp.array((2.5, 2.5, 1)))
            # integral_AB_mc = jnp.mean(vmap(inprod)(x_mc))
             
            # Check that gradients are orthogonal to curls when using homogeneous boundary conditions
            def inprod_hat(x):
                return curl(A_hat)(x) @ grad(p_hat)(x)
            
            def inprod(x):
                return curl(A)(x) @ grad(p)(x)
            
            npt.assert_allclose(l2_product(inprod, J_inv, x, w_q), 0, atol=2e-3)
            npt.assert_allclose(l2_product(inprod_hat, lambda x: 1.0, x_hat, w_q), 0, atol=2e-3)
            
            npt.assert_allclose(jnp.mean(vmap(inprod)(x) * vmap(J)(x_hat)), 0, atol=1e-3)
            npt.assert_allclose(jnp.mean(vmap(inprod_hat)(x_hat)), 0, atol=1e-3)
            
            # print(jnp.mean(vmap(inprod)(x) * vmap(J)(x_hat)))
            # print(jnp.mean(vmap(inprod_hat)(x_hat)))
            
            # print(l2_product(inprod, J_inv, x, w_q))
            # print(l2_product(inprod_hat, lambda x: 1.0, x_hat, w_q))

    def test_realspace_int(self):
        
        # n = 128
        # hx, hz, hphi = 1 / n, 1 / n, 2 * jnp.pi / n
        # _x = jnp.linspace(0 + hx/2, 1 - hx/2, n)
        # _phi = jnp.linspace(0 + hphi/2, 2 * jnp.pi - hphi/2, n)
        # _z = jnp.linspace(0 + hz/2, 1 - hz/2, n)
        # x = jnp.array(jnp.meshgrid(_x, _phi, _z))
        # x_hat = x.transpose(1, 2, 3, 0).reshape(n**3, 3)
        # w_q = jnp.ones(n**3) * hx * hphi * hz
        
        nr = 31
        nz = nr
        nphi = 32

        q_r = quad.GaussKronrodRule(nr, 2)
        wq_r = q_r._wh * (1 - 0) / 2
        wq_z = q_r._wh * (1 - 0) / 2
        _r = (q_r._xh + 1) / 2 * (1 - 0) + 0
        _z = (q_r._xh + 1) / 2 * (1 - 0) + 0
        _phi = jnp.linspace(0, 2*jnp.pi * (1 - 1/nphi), nphi)
        wq_phi = 2 * jnp.pi / nphi * jnp.ones(nphi)

        x_hat = jnp.array(jnp.meshgrid(_r, _phi, _z)) # shape 3, nx, nx, nx
        x_hat = x_hat.transpose(1, 2, 3, 0).reshape(nr*nphi*nz, 3)
        w_q = jnp.array(jnp.meshgrid(wq_r, wq_phi, wq_z)).transpose(1, 2, 3, 0).reshape(nr*nphi*nz, 3)
        w_q = jnp.prod(w_q, 1)
        
        # in cylinder geometry, this is going to be the integral of r^2 sin(phi)**2 z, aka r^3/3 * pi * z^2/2
        def p(x):
            r, φ, z = cart_to_cyl(x)
            return jnp.sin(φ)**2
        def f(x):
            r, φ, z = cart_to_cyl(x)
            return r * z
        # this one will be the integral of (r^2 + z^2) r sin(φ)^2, i.e. 5pi/12
        def A(x):
            r, φ, z = cart_to_cyl(x)
            return x * jnp.sin(φ)
        def B(x):
            r, φ, z = cart_to_cyl(x)
            return x * jnp.sin(φ)

        F = cyl_to_cart
        
        p_hat = jit(pullback_0form(p, F))
        A_hat = jit(pullback_1form(A, F))
        B_hat = jit(pullback_2form(B, F))
        f_hat = jit(pullback_3form(f, F))
        
        
        
        npt.assert_allclose(l2_product(p_hat, f_hat, x_hat, w_q), 1/3 * 1/2 * jnp.pi, rtol=1e-15)
        npt.assert_allclose(l2_product(A_hat, B_hat, x_hat, w_q), 5 * jnp.pi / 12, rtol=1e-15)
        