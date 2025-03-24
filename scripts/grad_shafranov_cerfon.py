# %%
# import time
from mhd_equilibria.vector_bases import get_vector_basis_fn
from mhd_equilibria.pullbacks import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import grad, jit, vmap, config, jacfwd
config.update("jax_enable_x64", True)
from functools import partial
import numpy.testing as npt
from jax.experimental.sparse.linalg import spsolve
import scipy as sp
from mhd_equilibria.bases import *
from mhd_equilibria.splines import *
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.operators import *

import matplotlib.pyplot as plt

# F maps from logical domain to physical domain

# Logical domain
Omega = ((0, 1), (0, 1), (0, 1))


# Define constants, based on ITER
κ = 1.5
q = 1.5
B0 = 1.0
R0 = 1.5
μ0 = 1.0
        
c0 = B0 / (R0**2 * κ * q)
c1 = B0 * ( κ**2 + 1) / ( R0**2 * κ * q )
c2 = 0.0

# Define F and F_inv

def F(x_hat):
    R_hat, Z_hat, phi_hat = x_hat
    R = R_hat * R0 * 1.2 + R0 * 0.2
    Z = Z_hat * 3 - 1.5
    phi = phi_hat * 2 * jnp.pi
    return jnp.array([R, Z, phi])

def F_inv(x):
    R, Z, phi = x
    R_hat = (R/R0 - 0.2) / 1.2
    Z_hat = (Z + 1.5) / 3
    phi_hat = phi / (2 * jnp.pi)
    return jnp.array([R_hat, Z_hat, phi_hat])
        

# The form of Antoine's analytic solution. The constants c_1,c_2,...are determined from boundary

def psi_analytic_cerfon(x,y,A,c_1, c_2, c_3, c_4, c_5,c_6,c_7):
    return (x**4)/8 + A*(0.5*(x**2)*jnp.log(x)-(x**4)/8) + c_1 +c_2*x**2 + c_3*(y**2-(x**2)*jnp.log(x)) 
    + c_4*(x**4-4*(x**2)*(y**2)) + c_5*(2*(y**4)-9*(y**2)*(x**2) + 3*(x**4)*jnp.log(x) -12*(x**2)*(y**2)*jnp.log(x))
    + c_6*(x**6 -12*(x**4)*(y**2) + 8*(x**2)*(y**4))+ c_7*(8*(y**6) -140*(y**4)*(x**2) 
    + 75*(y**2)*(x**4) - 15*(x**6)*jnp.log(x) + 180*(x**4)*(y**2)*jnp.log(x) - 120*(x**2)*(y**4)*jnp.log(x))
   
psi_analytic_cerfon_hat = pullback_0form(psi_analytic_cerfon, F)








# %%
