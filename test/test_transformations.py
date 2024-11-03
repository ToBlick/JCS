import unittest
from mhd_equilibria.operators import *
from mhd_equilibria.bases import _binom, _get_legendre_coeffs
from mhd_equilibria.coordinate_transforms import *
import numpy.testing as npt
import numpy as np
from jax import numpy as jnp
from jax import vmap, jit, grad, hessian, jacfwd, jacrev
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

class TransformsTests(unittest.TestCase):
    
    def test_cartesian_cyl(self): 
        x = jnp.array([1.7, -2.4, 3.1])
        r = cart_to_cyl(x)
        v = jnp.array([-4.1, 3.1, 2.7])
        v_c = vector_cart_to_cyl(v, x)
        
        npt.assert_allclose(x, cyl_to_cart(r), atol=1e-9)
        npt.assert_allclose(v, vector_cyl_to_cart(v_c, r), atol=1e-9)
        
    def test_cyl_tok(self):
        R0 = 1.5
        tok_to_cyl = get_tok_to_cyl(R0)
        cyl_to_tok = get_cyl_to_tok(R0)
        vector_tok_to_cyl = get_vector_tok_to_cyl(R0)
        vector_cyl_to_tok = get_vector_cyl_to_tok(R0)
        r = jnp.array([1.7, -2.4, 3.1])
        y = cyl_to_tok(r)
        v_c = jnp.array([-4.1, 3.1, 2.7])
        v_t = vector_cyl_to_tok(v_c, r)
        
        npt.assert_allclose(r, tok_to_cyl(y), atol=1e-9)
        npt.assert_allclose(vector_tok_to_cyl(v_t, y), v_c, atol=1e-9)
        
    def test_cartesian_tok(self):
        R0 = 1.5
        tok_to_cart = get_tok_to_cart(R0)
        cart_to_tok = get_cart_to_tok(R0)
        vector_tok_to_cart = get_vector_tok_to_cart(R0)
        vector_cart_to_tok = get_vector_cart_to_tok(R0)
        x = jnp.array([1.7, -2.4, 3.1])
        y = cart_to_tok(x)
        v = jnp.array([-4.1, 3.1, 2.7])
        v_t = vector_cart_to_tok(v, x)
        
        npt.assert_allclose(x, tok_to_cart(y), atol=1e-9)
        npt.assert_allclose(v, vector_tok_to_cart(v_t, y), atol=1e-9)