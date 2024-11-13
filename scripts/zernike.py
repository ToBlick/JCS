# %%
from mhd_equilibria.bases import *
from mhd_equilibria.bases import _unravel_ansi_idx
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

nx = 512
_r = np.linspace(1/nx, 1, nx)
_θ = np.linspace(0, 2*np.pi, nx)
d = 2
J = 50
zernike_fn = get_zernike_fn_x(J, 0, 1, 0, 2*np.pi)

x = jnp.array(jnp.meshgrid(_r, _θ)).reshape(d, nx**2).T

# %%
j = 12
R, Phi = np.meshgrid(_r, _θ)
# Compute the function values on the grid
Z = jax.vmap(zernike_fn, (0, None))(x, j).reshape(nx, nx)

# Convert polar coordinates to Cartesian for plotting
X, Y = R * np.cos(Phi), R * np.sin(Phi)

# Plot the function on the unit disk
plt.figure(figsize=(6, 6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('{}th Zernike polynomial'.format(j))
plt.axis('equal')
plt.show()
# %%

for j in range(1, J):
    n, m = _unravel_ansi_idx(j)
    if m != 0:
        continue
    if m == 0:
        c = 2
    else:
        c = 1
    plt.plot(_r, (c * np.pi/(2*n + 2))**0.5 * vmap(zernike_fn, (0, None))( np.column_stack([_r, np.zeros_like(_r)]) , j))
plt.grid()
# %%
