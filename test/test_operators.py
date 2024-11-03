import unittest
from mhd_equilibria.coordinate_transforms import *
from mhd_equilibria.operators import *
import numpy.testing as npt
import numpy as np
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)

class OperatorTests(unittest.TestCase):
    
    def test_cartesian(self):

        def F(x):
            return jnp.array([x[0]**2 + x[1], x[1]**2 + x[2], x[2]**2])
        def f(x):
            return jnp.sin(x[0]) * jnp.cos(x[1]) * x[2]**2
        
        x = jnp.array([1.0, 2.0, 3.0])
        
        # divergence, curl, grad, laplacian, vector laplacian
        npt.assert_allclose(div(F)(x), 2*x[0] + 2*x[1] + 2*x[2], rtol=1e-9)
        npt.assert_allclose(curl(F)(x), jnp.array([-1, 0, -1]), rtol=1e-9)
        npt.assert_allclose(grad(f)(x), jnp.array([jnp.cos(x[0]) * jnp.cos(x[1]) * x[2]**2,
                                                   jnp.sin(x[0]) * (-1) * jnp.sin(x[1]) * x[2]**2,
                                                   jnp.sin(x[0]) * jnp.cos(x[1]) * 2*x[2]]), atol=1e-9)
        npt.assert_allclose(div(grad(f))(x), laplacian(f)(x), atol=1e-9)
        npt.assert_allclose(curl(curl(F))(x) - grad(div(F))(x), -2, atol=1e-9)
        
        # div(curl(F)) = 0 and curl(grad(f)) = 0
        npt.assert_allclose(div(curl(F))(x), 0, atol=1e-9)
        npt.assert_allclose(curl(grad(f))(x), 0, atol=1e-9)
        
    def test_cyl(self):
        def F(x):
            return jnp.array([jnp.sin(x[0]**2) + x[1]**2 * x[2], 
                              jnp.cos(x[0]**2 + x[1]**2), 
                              x[2]**2 * x[0] * x[1]])
        def f(x):
            return jnp.sin(x[0]) * jnp.sin(jnp.cos(x[1]) + x[2]**2)
        def F_c(r): 
            x = cyl_to_cart(r)
            return vector_cart_to_cyl(F(x), x)
        f_c = lambda r: f(cyl_to_cart(r))
        
        # check transformations
        x = jnp.array([1.0, 2.0, 3.0])
        r = cart_to_cyl(x)
        npt.assert_allclose(x, cyl_to_cart(r), atol=1e-9)
        npt.assert_allclose(r, cart_to_cyl(x), atol=1e-9)
        
        # divergence, curl, grad, laplacian checked against cartesian coords
        npt.assert_allclose(div(F)(x), cyl_div(F_c)(r), rtol=1e-6)
        npt.assert_allclose(curl(F)(x), vector_cyl_to_cart(cyl_curl(F_c)(r), r), rtol=1e-6)
        npt.assert_allclose(div(grad(f))(x), cyl_div(cyl_grad(f_c))(r), rtol=1e-6)
        
        # vector laplacian
        lapF = - curl(curl(F))(x) + grad(div(F))(x)
        lapF_c = - cyl_curl(cyl_curl(F_c))(r) + cyl_grad(cyl_div(F_c))(r)
        npt.assert_allclose(lapF, vector_cyl_to_cart(lapF_c, r), rtol=1e-6)
        npt.assert_allclose(vector_cart_to_cyl(lapF, x), lapF_c, rtol=1e-6)
        
        # div(curl(F)) = 0 and curl(grad(f)) = 0
        npt.assert_allclose(cyl_div(cyl_curl(F_c))(r), 0, atol=1e-9)
        npt.assert_allclose(cyl_curl(cyl_grad(f_c))(r), 0, atol=1e-9)
        
    def test_tok(self):
        R0 = 1.91
        tok_to_cart = jit(get_tok_to_cart(R0))
        cart_to_tok = jit(get_cart_to_tok(R0))
        vector_cart_to_tok = jit(get_vector_cart_to_tok(R0))
        vector_tok_to_cart = jit(get_vector_tok_to_cart(R0))
        
        # cyl_to_tok = get_cyl_to_tok(R0)
        # tok_to_cyl = get_tok_to_cyl(R0)
        # vector_cyl_to_tok = get_vector_cyl_to_tok(R0)
        # vector_tok_to_cyl = get_vector_tok_to_cyl(R0)
        
        tok_div =  get_tok_div(R0)
        tok_grad = get_tok_grad(R0)
        tok_curl = get_tok_curl(R0)
        
        def F(x):
            return jnp.array([jnp.sin(x[0]**2) + x[1]**2 * x[2], 
                              jnp.cos(x[0]**2 + x[1]**2), 
                              x[2]**2 * x[0] * x[1]])
        def f(x):
            return jnp.sin(x[0]) * jnp.sin(jnp.cos(x[1]) + x[2]**2)
        def F_tok(r): 
            x = tok_to_cart(r)
            return vector_cart_to_tok(F(x), x)
        f_tok = lambda r: f(tok_to_cart(r))
        
        # check transformations
        x = jnp.array([2.0, 3.0, 4.0])
        r = cart_to_tok(x)
        
        # divergence, curl, grad, laplacian checked against cartesian coords
        npt.assert_allclose(div(F)(x), tok_div(F_tok)(r), rtol=1e-6)
        npt.assert_allclose(curl(F)(x), vector_tok_to_cart(tok_curl(F_tok)(r), r), rtol=1e-6)
        npt.assert_allclose(div(grad(f))(x), tok_div(tok_grad(f_tok))(r), rtol=1e-6)
        
        # vector laplacian (check jitting)
        @jit
        def lapF(x):
            return - curl(curl(F))(x) + grad(div(F))(x)
        @jit
        def lap_F_tok(x):
            return - tok_curl(tok_curl(F_tok))(r) + tok_grad(tok_div(F_tok))(r)
        npt.assert_allclose(lapF(x), vector_tok_to_cart(lap_F_tok(x), r), rtol=1e-6)
        npt.assert_allclose(vector_cart_to_tok(lapF(x), x), lap_F_tok(x), rtol=1e-6)
        
        # div(curl(F)) = 0 and curl(grad(f)) = 0
        npt.assert_allclose(tok_div(tok_curl(F_tok))(r), 0, atol=1e-9)
        npt.assert_allclose(tok_curl(tok_grad(f_tok))(r), 0, atol=1e-9)