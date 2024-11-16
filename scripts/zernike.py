# %%
from mhd_equilibria.bases import *
from mhd_equilibria.bases import _unravel_ansi_idx
from mhd_equilibria.quadratures import *
from mhd_equilibria.projections import *
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap, jit
jax.config.update("jax_enable_x64", True)

nx = 512
_r = np.linspace(1/nx, 1, nx)
_θ = np.linspace(0, 2*np.pi, nx)
d = 2
J = 50
zernike_fn = get_zernike_fn_x(J, 0, 1, 0, 2*np.pi)
zernike_fn_radial_derivative = get_zernike_fn_radial_derivative(J, 0, 1, 0, 2*np.pi)

x = jnp.array(jnp.meshgrid(_r, _θ)).reshape(d, nx**2).T

# %%
j = 12
R, Phi = np.meshgrid(_r, _θ)
# Compute the function values on the grid
Z = jax.vmap(zernike_fn_radial_derivative, (0, None))(x, j).reshape(nx, nx)

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
    plt.plot(_r, (c * np.pi/(2*n + 2))**0.5 * vmap(zernike_fn, (0, None))( np.column_stack([_r, np.zeros_like(_r)]) , j), label='n = {}'.format(n))
plt.grid()
plt.legend()
plt.show()
# %%

# project Gaussian onto Zernike polynomials
def f(x):
    r, θ = x 
    x1 = r * jnp.cos(θ)
    x2 = r * jnp.sin(θ)
    return jnp.exp(-(r)**2/(2 * 0.5**2)) * r * (1 + r**2 * jnp.cos(θ)**2)
    # return r**3 * jnp.sin(θ)
Z = jax.vmap(f)(x).reshape(nx, nx)
plt.figure(figsize=(6, 6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian')
plt.axis('equal')
plt.show()

def f_3d(x):
    r, θ, z = x
    return f(jnp.array([r, θ]))
# %%
# check radial derivative spaces
Omega = ((0, 1), (0, 2*jnp.pi), (0, 2*jnp.pi))
x_q, w_q = quadrature_grid(get_quadrature(31)(*Omega[0]),
                           get_quadrature_periodic(64)(*Omega[1]),
                           get_quadrature_periodic(1)(*Omega[2]))

def J_analytic(x):
    r, _, _ = x
    return r
J_at_x = vmap(J_analytic)(x_q)

n_r = 7
n_θ = 3
n_φ = 1
bases = (get_zernike_fn_radial_derivative(n_r*n_θ, *Omega[0], *Omega[1]), get_trig_fn_x(n_φ, *Omega[2]))
shape = (n_r*n_θ, n_φ)
basis_fn = jit(get_zernike_tensor_basis_fn(bases, shape))
N = n_r*n_θ*n_φ

# Assemble the mass matrix
M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

# %%
plt.imshow(jnp.log10(jnp.abs(M) + 1e-16))
plt.colorbar()

# %%
l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
f_hat = jnp.linalg.solve(M, f_hat)
plt.scatter(jnp.arange(N), f_hat)

# %%
### Do the same with a tensor basis
n_r = 4
n_θ = 4
n_φ = 1
bases = (get_legendre_fn_x(n_r, *Omega[0]), get_trig_fn_x(n_θ, *Omega[1]), get_trig_fn_x(n_φ, *Omega[2]))
shape = (n_r, n_θ, n_φ)
basis_fn = jit(get_tensor_basis_fn(bases, shape))
N = n_r*n_θ*n_φ

# Assemble the mass matrix
M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

# %%
plt.imshow(jnp.log10(jnp.abs(M) + 1e-16))
plt.colorbar()

# %%
l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
f_hat = jnp.linalg.solve(M, f_hat)
plt.scatter(jnp.arange(N), f_hat)

# %%

# Compute the projection of the Gaussian onto the Zernike basis

l2_errors_zernike = []
l2_errors_zernike_radial = []
l2_errors_legendre = []
l2_errors_cheb = []

for _n in range(2,9):
    n_r = _n
    n_θ = _n
    n_φ = 1
    N = n_r*n_θ*n_φ
    print(N)
    
    bases = (get_zernike_fn_radial_derivative(n_r*n_θ, *Omega[0], *Omega[1]), 
             get_trig_fn_x(n_φ, *Omega[2]))
    shape = (n_r*n_θ, n_φ)
    basis_fn = jit(get_zernike_tensor_basis_fn(bases, shape))
    
    M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
    M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

    l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
    f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
    f_hat = jnp.linalg.solve(M, f_hat)
    f_h = get_u_h_vec(f_hat, basis_fn)

    def err(x):
        return (f_h(x) - f_3d(x))**2 * x[0]

    def normalization(x):
        return (f_3d(x))**2 * x[0]
    
    l2_errors_zernike_radial.append( jnp.sqrt( integral(err, x_q, w_q)) 
                                    / jnp.sqrt( integral(normalization, x_q, w_q)) )
    
    bases = (get_zernike_fn_x(n_r*n_θ, *Omega[0], *Omega[1]), 
             get_trig_fn_x(n_φ, *Omega[2]))
    shape = (n_r*n_θ, n_φ)
    basis_fn = jit(get_zernike_tensor_basis_fn(bases, shape))
    
    M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
    M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

    l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
    f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
    f_hat = jnp.linalg.solve(M, f_hat)
    f_h = get_u_h_vec(f_hat, basis_fn)

    def err(x):
        return (f_h(x) - f_3d(x))**2 * x[0]

    def normalization(x):
        return (f_3d(x))**2 * x[0]
    
    l2_errors_zernike.append( jnp.sqrt( integral(err, x_q, w_q)) 
                                    / jnp.sqrt( integral(normalization, x_q, w_q)) )
    
    bases = (get_legendre_fn_x(n_r, *Omega[0]), 
             get_trig_fn_x(n_θ, *Omega[1]), 
             get_trig_fn_x(n_φ, *Omega[2]))
    shape = (n_r, n_θ, n_φ)
    basis_fn = jit(get_tensor_basis_fn(bases, shape))
    
    M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
    M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

    l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
    f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
    f_hat = jnp.linalg.solve(M, f_hat)
    f_h = get_u_h_vec(f_hat, basis_fn)

    def err(x):
        return (f_h(x) - f_3d(x))**2 * x[0]

    def normalization(x):
        return (f_3d(x))**2 * x[0]
    
    l2_errors_legendre.append( jnp.sqrt( integral(err, x_q, w_q)) 
                                    / jnp.sqrt( integral(normalization, x_q, w_q)) )
    
    # bases = (get_chebyshev_fn_x(n_r, *Omega[0]), 
    #          get_trig_fn_x(n_θ, *Omega[1]), 
    #          get_trig_fn_x(n_φ, *Omega[2]))
    # shape = (n_r, n_θ, n_φ)
    # basis_fn = jit(get_tensor_basis_fn(bases, shape))
    
    # M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
    # M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)

    # l2_proj = get_l2_projection(basis_fn, x_q, w_q, N)
    # f_hat = l2_proj(lambda x: f_3d(x) * x[0])[:, None]
    # f_hat = jnp.linalg.solve(M, f_hat)
    # f_h = get_u_h_vec(f_hat, basis_fn)

    # def err(x):
    #     return (f_h(x) - f_3d(x))**2 * x[0]

    # def normalization(x):
    #     return (f_3d(x))**2 * x[0]
    
    # l2_errors_cheb.append( jnp.sqrt( integral(err, x_q, w_q)) 
    #                                 / jnp.sqrt( integral(normalization, x_q, w_q)) )
    
    # Z = jax.vmap(f_h)(x).reshape(nx, nx)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('f_h')
    # plt.axis('equal')
    # plt.show()

    # Z = jax.vmap(f_h)(x).reshape(nx, nx) - jax.vmap(f_3d)(x).reshape(nx, nx)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('f_h - f')
    # plt.axis('equal')
    # plt.show()
    
    # if x.shape[1] != 3:
    #     x = jnp.column_stack([x, jnp.ones(x.shape[0])])
    # _j = N // 2
    # Z = jax.vmap(basis_fn, (0, None))(x, _j).reshape(nx, nx)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('{}th Zernike polynomial'.format(_j))
    # plt.axis('equal')
    # plt.show()

    # # Assemble the mass matrix
    # M_ij = get_mass_matrix_lazy(lambda x,j: basis_fn(x, j) * jnp.sqrt(J_analytic(x)), x_q, w_q, N)
    # M = jnp.array([ M_ij(i, j) for i in range(N) for j in range(N) ]).reshape(N, N)
    # plt.imshow(M)
    # plt.colorbar()

# %%
_Ns = jnp.array(range(2,9))**2
plt.scatter(_Ns, l2_errors_zernike, label='zernike')
plt.scatter(_Ns, l2_errors_zernike_radial, label='d/dr zernike')
plt.scatter(_Ns, l2_errors_legendre, label='legendre')
# plt.scatter(_Ns, l2_errors_cheb, label='chebyshev')
plt.plot(_Ns, 200 * np.exp(-( _Ns**0.5)), label='~ exp(- 2 sqrt N)', color='k', linestyle='--')
plt.yscale('log')
plt.xlabel('N')
plt.legend()
plt.grid()

# %%
