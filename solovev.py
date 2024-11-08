# %%
from jax import grad, jacrev, jacfwd, hessian, vmap, jit
import jax.numpy as jnp
from functools import partial

import numpy as np

from mhd_equilibria.bases import *
from mhd_equilibria.projections import *
from mhd_equilibria.forms import *
from mhd_equilibria.plotting import *
from mhd_equilibria.operators import *

import matplotlib.pyplot as plt

import time

# %%
### Constants
κ = 1.5
q = 1.5
B0 = 1.0
R0 = 2.0
μ0 = 1.0

π = jnp.pi

χ = 1.0 # friction coefficient
γ = 1.3 # adiabatic index
        
c0 = B0 / (R0**2 * κ * q)
c1 = B0 * ( κ**2 + 1) / ( R0**2 * κ * q )
c2 = 0.0
        
p0 = 1.8

Φ = jnp.pi/4
        
### Analytical solutions
def Psi(x):
    R, φ, Z = x
    return B0/(2 * R0**2 * κ * q) * ( R**2 * Z**2 + κ**2/4 * (R**2 - R0**2)**2 )

def get_B(Psi):
    def B(x):
        R, φ, Z = x
        gradPsi = grad(Psi)(x)
        return 1/R * jnp.cross(gradPsi, jnp.array([0, 1, 0]))
    return B

def get_p(Psi):
    def p(x):
        R, φ, Z = x
        return p0 - B0 * (κ**2 + 1) / (μ0 * R0**2 * κ * q) * Psi(x)
    return p

def Psi_restricted(x):
    T = Psi(jnp.array([R0 - 0.66, Φ, 0.0]))
    R, φ, Z = x
    return jax.lax.cond( Psi(x) < T, lambda x: Psi(x), lambda x: T, x)

def mask(f):
    def f_masked(x):
        T = Psi(jnp.array([R0 - 0.66, Φ, 0.0]))
        return jax.lax.cond( Psi(x) < T, lambda x: f(x), lambda x: f(x) * 0, x)
    return f_masked

B = get_B(Psi_restricted)
p = get_p(Psi_restricted)
# pressure-free
# p = lambda x: 0.0 

### Quadrature grid
Omega = ((1., 3.), (-1., 1.))
nx = 256
_R = jnp.linspace(*Omega[0], nx)
_Φ = jnp.array([Φ])
_Z = jnp.linspace(*Omega[1], nx)
hR = (_R[-1] - _R[0]) / nx
hZ = (_Z[-1] - _Z[0]) / nx
w_q = hR * hZ * 2 * jnp.pi
        
x = jnp.array(jnp.meshgrid(_R, _Φ, _Z)) # shape 3, n_x, n_x, 1
x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)

# ### Initial condition: perturbed Psi
def Psi_perturbed(x):
    R, φ, Z = x
    m = Omega[0][0] + 0.5 * (Omega[0][1] - Omega[0][0])
    bump = lambda x: jnp.exp(- 0.5 * jnp.sum((x - jnp.array([m, Φ, 0]))**2) / 0.3**2)
    perturb = lambda x: jnp.sin(4*jnp.pi * x[0]) * jnp.cos(4*jnp.pi * x[1])
    return Psi_restricted(x) + bump(x) * perturb(x) * 1e-2 * 0
p_perturbed = get_p(Psi_perturbed)

# %%
### Basis
n1, n2, n3 = 8, 1, 8
_bases = (get_trig_fn_x(n1, *Omega[0]), get_trig_fn_x(n2, 0, 1), get_trig_fn_x(n3, *Omega[1]))
#_bases = (get_legendre_fn_x(n1, *Omega[0]), get_legendre_fn_x(n2, 0, 2*jnp.pi), get_legendre_fn_x(n3, *Omega[1]))
shape = (n1, n2, n3)
basis_fn = get_tensor_basis_fn(_bases, shape) # basis_fn(x, k)
_ns = jnp.arange(n1*n2*n3, dtype=jnp.int32)

J = lambda x: x[0] # J(R, φ, Z) = R

l2_proj = get_l2_projection(basis_fn, J, x, w_q, n1*n2*n3)
M_ij = get_mass_matrix_lazy(basis_fn, J, x, w_q, n1*n2*n3)
M = vmap(vmap(M_ij, (0, None)), (None, 0))(_ns, _ns)

# %%
### Initial conditions

Psi_hat = jnp.linalg.solve(M, l2_proj(Psi_perturbed))
p_hat = jnp.linalg.solve(M, l2_proj(p))

@jit
def Psi_h(x):
    return get_u_h(Psi_hat, basis_fn)(x)

@jit
def p_h(x):
    return get_u_h(p_hat, basis_fn)(x)

# %%
### Plotting

# Plot Psi, perturbed Psi, discrete Psi:
plt.contour(_R, _Z, vmap(Psi_restricted)(x).reshape(nx, nx).T, 10, colors='black', alpha = 0.5)
plt.contour(_R, _Z, vmap(Psi_perturbed)(x).reshape(nx, nx).T, 10, colors='black')
plt.contour(_R, _Z, vmap(get_u_h(Psi_hat, basis_fn))(x).reshape(nx, nx).T, 10, colors='red')
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()
        
# %%
### Brute force!
def get_Psi_h(Psi_hat):
    def Psi_h(x):
        return get_u_h(Psi_hat, basis_fn)(x)
    return Psi_h

def get_B_h(Psi_hat):
    def B_h(x):
        R, φ, Z = x
        def Psi_h(x):
            return get_u_h(Psi_hat, basis_fn)(x)
        dPsi = grad(Psi_h)(x)
        return 1/R * jnp.cross(dPsi, jnp.array([0, 1, 0]))
    return B_h

def get_p_h(p_hat):
    def p_h(x):
        return get_u_h(p_hat, basis_fn)(x)
    return p_h

def get_F_h(Psi_hat, p_hat):
    def F_h(x):
        B_h = get_B_h(Psi_hat)
        p_h = get_p_h(p_hat)
        # j = curl B, but this is really a scalar quantity
        j = 1/μ0 * cyl_curl(B_h)(x)
        # DB = jacfwd(B_h)(x)
        # j = 1/μ0 * jnp.array([0.0, DB[0, 2] - DB[2, 0], 0.0])
        return jnp.cross(j, B_h(x)) - cyl_grad(p_h)(x)
    return F_h

def get_v(Psi_hat, p_hat):
    F_h = get_F_h(Psi_hat, p_hat)
    def v(x):
        return χ * (F_h)(x)
    return v

def get_Psi_dot(Psi_hat, p_hat):
    v = get_v(Psi_hat, p_hat)
    Psi_h = get_u_h(Psi_hat, basis_fn)
    def Psi_dot(x):
        return - jnp.dot(v(x), grad(Psi_h)(x))
    return Psi_dot

def get_p_dot(Psi_hat, p_hat):
    v = get_v(Psi_hat, p_hat)
    p_h = get_u_h(p_hat, basis_fn)
    def p_dot(x):
        return - jnp.dot(v(x), cyl_grad(p_h)(x)) - γ * p_h(x) * cyl_div(v)(x)
    return p_dot

# %%
B_h = jit(get_B_h(Psi_hat))
plt.quiver( x[::100,0], x[::100,2],
    vmap(mask(B_h))(x)[::100,0],
    vmap(mask(B_h))(x)[::100,2],
    color = 'black',
    )
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.show()

# %%
p_h = jit(get_p_h(p_hat))
plt.contourf(_R, _Z, vmap(p_h)(x).reshape(nx, nx).T)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()

# %%
plt.contourf(_R, _Z, vmap(mask(cyl_curl(B_h)))(x)[:,1].reshape(nx, nx).T, 100)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()

# %%
F_h = jit(get_F_h(Psi_hat, p_hat))
plt.quiver( x[::100,0], x[::100,2],
            vmap(mask(F_h))(x)[::100,0],
            vmap(mask(F_h))(x)[::100,2],
            color = 'black',
    )
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.show()

# %%
plt.contourf(_R, _Z, vmap(mask(cyl_div(F_h)))(x).reshape(nx, nx).T, 100)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()

# %%
_start = time.time()
p_dot = jit(get_p_dot(Psi_hat, p_hat))
_end = time.time()
print("Time to compile dp/dt: ", _end - _start)
_start = time.time()
p_dot_hat = jnp.linalg.solve(M, l2_proj(p_dot))
_end = time.time()
print("Time to project dp/dt: ", _end - _start)

# %%
_start = time.time()
Psi_dot = jit(get_Psi_dot(Psi_hat, p_hat))
_end = time.time()
print("Time to compile dΨ/dt: ", _end - _start)
_start = time.time()
Psi_dot_hat = jnp.linalg.solve(M, l2_proj(Psi_dot))
_end = time.time()
print("Time to project dΨ/dt: ", _end - _start)
    
### Time stepping
dt = 0.001

Psi_hats = [ Psi_hat ]
p_hats = [ p_hat ]
Energies = [ ]

for _ in range(100):
    Psi_dot = jit(get_Psi_dot(Psi_hat, p_hat))
    p_dot = jit(get_p_dot(Psi_hat, p_hat))
    v = jit(get_v(Psi_hat, p_hat))
    Energy = l2_product(v, v, J, x, w_q)
    print("|F|^2 = ", Energy)
    
    Psi_dot_hat = jnp.linalg.solve(M, l2_proj(Psi_dot))
    p_dot_hat = jnp.linalg.solve(M, l2_proj(p_dot))
    Psi_hat = Psi_hat + dt * Psi_dot_hat
    p_hat = p_hat + dt * p_dot_hat
    
    Psi_hats.append(Psi_hat)
    p_hats.append(p_hat)
    Energies.append(Energy)

# %%
B_h = get_B_h(Psi_hats[-1])
B_h_0 = get_B_h(Psi_hats[0])

plt.quiver( x[::100,0], x[::100,2],
    vmap(mask(B_h))(x)[::100,0],
    vmap(mask(B_h))(x)[::100,2],
    color = 'black',
    )
plt.quiver( x[::100,0], x[::100,2],
    vmap(mask(B_h_0))(x)[::100,0],
    vmap(mask(B_h_0))(x)[::100,2],
    color = 'cyan',
    )
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.show()

# %%
Psi_h = get_Psi_h(Psi_hats[-1])
Psi_h_0 = get_Psi_h(Psi_hats[0])
plt.contour(_R, _Z, vmap(Psi_restricted)(x).reshape(nx, nx).T,  10, colors='black', alpha = 0.5)
plt.contour(_R, _Z, vmap(Psi_h_0)(x).reshape(nx, nx).T,         10, colors='red', alpha = 0.5)
plt.contour(_R, _Z, vmap(Psi_h)(x).reshape(nx, nx).T,           10, colors='blue', alpha = 0.5)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()

# %%
p_h = get_p_h(p_hats[-1])
p_h_0 = get_p_h(Psi_hats[0])
plt.contour(_R, _Z, vmap(p)(x).reshape(nx, nx).T,  10, colors='black', alpha = 0.5)
plt.contour(_R, _Z, vmap(p_h_0)(x).reshape(nx, nx).T,         10, colors='red', alpha = 0.5)
plt.contour(_R, _Z, vmap(p_h)(x).reshape(nx, nx).T,           10, colors='blue', alpha = 0.5)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()
# %%

B_h = get_B_h(Psi_hats[-1])
B_h_0 = get_B_h(Psi_hats[0])

plt.quiver( x[::100,0], x[::100,2],
    vmap(mask(B_h))(x)[::100,0],
    vmap(mask(B_h))(x)[::100,2],
    color = 'black',
    )
plt.quiver( x[::100,0], x[::100,2],
    vmap(mask(B_h_0))(x)[::100,0],
    vmap(mask(B_h_0))(x)[::100,2],
    color = 'cyan',
    )
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.show()

# %%

# %%
import ipywidgets as widgets
from IPython.display import display

def update(i):
    p_h = (get_p_h(p_hats[i]))
    plt.contourf(_R, _Z, vmap(p_h)(x).reshape(nx, nx).T)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$Z$')
    plt.colorbar()
    plt.show()
    
i_slider = widgets.IntSlider(
    value=0,
    min=0,    
    max=len(p_hats)-1,   
    description='iteration',
    continuous_update=True
)
interactive = widgets.interactive(update, i=i_slider)
display(interactive)
# %%

def update(i):
    F_h = jit(get_F_h(Psi_hats[i], p_hats[i]))
    plt.quiver( x[::100,0], x[::100,2],
            vmap((F_h))(x)[::100,0],
            vmap((F_h))(x)[::100,2],
            color = 'black',
    )
    plt.xlabel(r'$R$')
    plt.ylabel(r'$Z$')
    plt.show()
    
i_slider = widgets.IntSlider(
    value=0,
    min=0,    
    max=len(p_hats)-1,   
    description='iteration',
    continuous_update=True
)
interactive = widgets.interactive(update, i=i_slider)
display(interactive)
# %%

plt.plot(Energies)
plt.xlabel("iteration")
plt.ylabel("Energy")
plt.yscale('log')
plt.show()
# %%
F_h = jit(get_F_h(Psi_hat, p_hat))
plt.contourf(_R, _Z, vmap((cyl_curl(F_h)))(x)[:, 1].reshape(nx, nx).T, 100)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()
# %%
F_h = jit(get_F_h(Psi_hat, p_hat))
plt.contourf(_R, _Z, vmap((cyl_div(F_h)))(x).reshape(nx, nx).T, 100)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.colorbar()
plt.show()
# %%

def update(i):
    Psi_h = (get_Psi_h(Psi_hats[i]))
    plt.contourf(_R, _Z, vmap(((Psi_h)))(x).reshape(nx, nx).T)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$Z$')
    plt.colorbar()
    plt.show()
    
i_slider = widgets.IntSlider(
    value=0,
    min=0,    
    max=len(Psi_hats)-1,   
    description='iteration',
    continuous_update=True
)
interactive = widgets.interactive(update, i=i_slider)
display(interactive)
# %%

def update(i):
    p_h = (get_p_h(p_hats[i]))
    plt.contourf(_R, _Z, vmap(((p_h)))(x).reshape(nx, nx).T)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$Z$')
    plt.colorbar()
    plt.show()
    
i_slider = widgets.IntSlider(
    value=0,
    min=0,    
    max=len(Psi_hats)-1,   
    description='iteration',
    continuous_update=True
)
interactive = widgets.interactive(update, i=i_slider)
display(interactive)

# %%

def ς(t):
    return 1 + 0.1 * jnp.sin(2*jnp.pi * t)
