# %%
 
import quadax as quad
import matplotlib.pyplot as plt 
import jax.numpy as jnp

q = quad.GaussKronrodRule(51, 2)


# %%
wh_scaled = q._wh * (1 - 0) / 2
wl_scaled = q._wl * (1 - 0) / 2
xh_scaled = (q._xh + 1) / 2 * (1 - 0) + 0
# %%
f = lambda x: jnp.sin(x * 20 * jnp.pi)**2
# %%
jnp.sum( wh_scaled * f(xh_scaled) )
# %%
jnp.sum( wl_scaled * f(xh_scaled) )