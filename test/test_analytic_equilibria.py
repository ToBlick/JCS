import unittest
from mhd_equilibria.coordinate_transforms import *
from mhd_equilibria.operators import *
import numpy.testing as npt
from jax import numpy as jnp
from jax import vmap, grad
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

class AnalyticTests(unittest.TestCase):
    
    def test_onedim(self):
        # theta pinch (cylindrical coordinates)
        B0 = 1.0
        a = 1.0
        μ0 = 1.0
        β0 = 0.8
        p0 = 0.5 * β0 * B0**2 / μ0
        βhat = β0 / ( 1 + (1 - β0)**0.5)
        
        def B(x):
            r, θ, z = x
            ρ = r / a
            Bθ = 0.0
            Br = 0.0
            Bz = B0 * ( 1 - βhat * ( 1 - ρ**2 )**2 )
            return jnp.array([Br, Bθ, Bz])
        def p(x):
            r, θ, z = x
            ρ = r / a
            return 0.5 * B0**2 / μ0 * ( 1 - ( 1 - βhat * ( 1 - ρ**2 )**2 )**2 )
        def Jθ(x):
            r, θ, z = x
            ρ = r / a
            return - 4 * B0 / a / μ0 * βhat * ( 1 - ρ**2 ) * ρ
        
        def jcrossB(x):
            return jnp.cross(cyl_curl(B)(x), B(x))
        
        def F(x):
            return 1/μ0 * jcrossB(x) - cyl_grad(p)(x)
        
        nx = 10
        h = 1.0 / nx
        dθ = 2*jnp.pi / nx
        _ρ = jnp.linspace(h/2, 1.0 - h/2, nx-1)
        _θ = jnp.linspace(dθ/2, 2*jnp.pi - dθ/2, nx-1)
        _z = jnp.array([1.0])
        
        x = jnp.array(jnp.meshgrid(_ρ, _θ, _z)) # shape 3, n_x, n_x, 1
        print(x.shape)
        x = x.transpose(1, 2, 3, 0).reshape(1*(nx-1)**2, 3)
        
        npt.assert_allclose(vmap(cyl_div(B))(x), 0, atol=1e-9)
        npt.assert_allclose(vmap(cyl_curl(B))(x)[:,1], μ0 * vmap(Jθ)(x), rtol=1e-9)
        npt.assert_allclose(vmap(F)(x), 0, atol=1e-9)
        
    def test_solovev(self):
        # This is all in cylindrical coordinates
        
        κ = 1.5
        q = 1.5
        B0 = 1.0
        R0 = 1.0
        μ0 = 1.0
        
        c0 = B0 / (R0**2 * κ * q)
        c1 = B0 * ( κ**2 + 1) / ( R0**2 * κ * q )
        c2 = 0.0
        
        p0 = 1.8
        
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
        
        B = get_B(Psi)
        p = get_p(Psi)
        
        nx = 256
        hR = 1.2 / nx
        hZ = 1.8 / nx
        _R = jnp.linspace(0.2, 1.4, nx)
        _Φ = jnp.array([0.0])
        _Z = jnp.linspace(-0.9, 0.9, nx)
        
        x = jnp.array(jnp.meshgrid(_R, _Φ, _Z)) # shape 3, n_x, n_x, 1
        x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)
        
        divB =  cyl_div(B)
        curlB = cyl_curl(B)
        gradp = cyl_grad(p)
        
        def jcrossB(x):
            return jnp.cross(cyl_curl(B)(x), B(x))
        
        def F(x):
            return 1/μ0 * jcrossB(x) - cyl_grad(p)(x)
        
        def plot_mask(x):
            T = Psi(jnp.array([0.0, 0.0, 0.0])) - 1e-2
            return jax.lax.cond( Psi(x) < T, lambda x: 1.0, lambda x: jnp.nan, x)
        
        npt.assert_allclose(vmap(F)(x), 0, atol=1e-9)
        npt.assert_allclose(vmap(divB)(x), 0, atol=1e-9)
        
        B_masked = lambda x: B(x) * plot_mask(x)
        Psi_masked = lambda x: Psi(x) * plot_mask(x)
        p_masked = lambda x: p(x) * plot_mask(x)
        
        # plt.contour(_R, _Z, vmap(Psi_masked)(x).reshape(nx, nx).T, 10, colors='black')
        # plt.contourf(_R, _Z, (vmap(p_masked)(x)).reshape(nx, nx).T, 100)
        # plt.xlabel(r'$R$')
        # plt.ylabel(r'$Z$')
        # plt.colorbar()
        # plt.show()
    
        # plt.contour(_R, _Z, vmap(Psi_masked)(x).reshape(nx, nx).T, 10, colors='black', alpha = 0.5)
        # plt.quiver( x[::100,0], x[::100,3],
        #     vmap(B_masked)(x)[::100,0],
        #     vmap(B_masked)(x)[::100,2],
        #     color = 'black',
        #     )
        # plt.xlabel(r'$R$')
        # plt.ylabel(r'$Z$')
        # plt.show()
        
        # def Psi_perturbed(x):
        #     R, φ, Z = x
        #     return (Psi(x) + 1e-2 * jnp.sin(2*jnp.pi * R) * jnp.sin(2*jnp.pi * Z))
        
        # Psi_perturbed_masked = lambda x: Psi_perturbed(x) * plot_mask(x)
        
        # B_perturbed = get_B(Psi_perturbed)
        # p_perturbed = get_p(Psi_perturbed)
        
        # def F_perturbed(x):
        #     return 1/μ0 * jnp.cross(cyl_curl(B_perturbed)(x), B_perturbed(x)) - cyl_grad(p_perturbed)(x)
        
        # print("L2 error of force balance: ", jnp.sum(vmap(F)(x)**2) * hR * hZ)
        # print("L2 error of force balance, perturbed: ", jnp.sum(vmap(F_perturbed)(x)**2) * hR * hZ)
        
        # plt.contour(_R, _Z, vmap(Psi_masked)(x).reshape(nx, nx).T, 10, colors='black')
        # plt.contour(_R, _Z, vmap(Psi_perturbed_masked)(x).reshape(nx, nx).T, 10, colors='red')
        # plt.show()
    
    #TODO: There is some bug here I cannot find
    def test_highbeta(self):
        # toroidal beta ~ eps
        # poloidal beta ~ 1/eps
        # B_p/B_phi  ~ eps
        # q ~ 1
        ε = 0.01
        a = 1.0
        R0 = a / ε
        B0 = 1.0
        μ0 = 1.0
        
        A = 2.0
        C = 3.5

        βt = a**2 * A * C / ( 8 * R0 * B0**2 )
        qstar = 2 * B0 / A
        
        ν = βt * qstar**2 / ε
        
        def psi(x):
            r, θ, z = x
            ρ = r / a
            return 0.5 * a**2 * B0 / qstar * ( ρ**2 - 1 + ν * (ρ**3 - ρ) * jnp.cos(θ) )
        def B(x):
            r, θ, z = x
            ρ = r / a
            Bθ = ε * B0 / qstar * ( ρ + 0.5 * ν * (3*ρ**2 - 1) * jnp.cos(θ) )
            Br = - 0.5 * ε * B0 * ν / qstar * (ρ**2 - 1) * jnp.sin(θ)
            Bz = - B0 * (1 - ε*ρ*jnp.cos(θ) - βt*(1 - ρ**2)*(1 + ν*ρ*jnp.cos(θ)) )
            return jnp.array([Br, Bθ, Bz])
        def p(x):
            r, θ, z = x
            ρ = r / a
            return βt * B0**2 / μ0 * (1 - ρ**2) * (1 + ν*ρ*jnp.cos(θ))
        def Jz(x):
            r, θ, z = x
            ρ = r / a
            return 2 * B0 / R0 / μ0 / qstar * (1 + 2*ν*ρ*jnp.cos(θ)) 
        
        nx = 256
        h = 1.0 / nx
        dθ = 2*jnp.pi / nx
        _ρ = jnp.linspace(h/2, 1.0 - h/2, nx)
        _θ = jnp.linspace(dθ/2, 2*jnp.pi - dθ/2, nx)
        _z = jnp.array([1.0])
        
        x = jnp.array(jnp.meshgrid(_ρ, _θ, _z)) # shape 3, n_x, n_x, 1
        x = x.transpose(1, 2, 3, 0).reshape(1*(nx)**2, 3)
        
        tok_div =  get_tok_div(R0)
        tok_grad = get_tok_grad(R0)
        tok_curl = get_tok_curl(R0)
        
        divB = tok_div(B)
        curlB = tok_curl(B)
        gradp = tok_grad(p)
        gradpsi = tok_grad(psi)

        
        # print("integrated violation of divB: ", jnp.sum(vmap(divB)(x)**2) * h * dθ)
        # print("integrated violation of Force balance: ", jnp.sum(vmap(F)(x)) * h * dθ)
        
        # npt.assert_allclose(vmap(F)(x), 0, atol=1e-3)
        # npt.assert_allclose(vmap(divB)(x), 0, atol=1e-3)
        # npt.assert_allclose(vmap(curlB)(x)[:,2], vmap(Jz)(x), rtol=1e-3)
        
        # plt.contourf(_ρ, _θ, (vmap(curlB)(x)[:,2] - μ0 * vmap(Jz)(x)).reshape(nx, nx), 100)
        # plt.xlabel(r'$\rho$')
        # plt.ylabel(r'$\theta$')
        # plt.title(r'$\nabla \times B - \mu_0 j$')
        # plt.colorbar()
        # plt.show()
        
        # plt.contourf(_ρ, _θ, vmap(divB)(x).reshape(nx, nx), 100)
        # plt.xlabel(r'$\rho$')
        # plt.ylabel(r'$\theta$')
        # plt.title(r'$\nabla \cdot B$')
        # plt.colorbar()
        # plt.show()
        
        # print(jnp.sum(jnp.abs(vmap(divB)(x).reshape(nx, nx).T[:,0]      )))
        # print(jnp.sum(jnp.abs(vmap(divB)(x).reshape(nx, nx).T[:,nx//4]  )))
        # print(jnp.sum(jnp.abs(vmap(divB)(x).reshape(nx, nx).T[:,nx//2]  )))
        # print(jnp.sum(jnp.abs(vmap(divB)(x).reshape(nx, nx).T[:,3*nx//4])))
        # print(jnp.sum(jnp.abs(vmap(divB)(x).reshape(nx, nx).T[:,-1]     )))
        
        # plt.plot(_ρ, jnp.abs(vmap(divB)(x)).reshape(nx, nx).T[:,0]      , alpha=0.2)
        # plt.plot(_ρ, jnp.abs(vmap(divB)(x)).reshape(nx, nx).T[:,nx//4]  , alpha=0.2)
        # plt.plot(_ρ, jnp.abs(vmap(divB)(x)).reshape(nx, nx).T[:,nx//2]  , alpha=0.2)
        # plt.plot(_ρ, jnp.abs(vmap(divB)(x)).reshape(nx, nx).T[:,3*nx//4], alpha=0.2)
        # plt.plot(_ρ, jnp.abs(vmap(divB)(x)).reshape(nx, nx).T[:,-1]     , alpha=0.2)
        # plt.xlabel(r'$\rho$')
        # plt.title(r'$\nabla \cdot B$')
        # plt.yscale('log')
        # plt.show()
        
        # npt.assert_allclose(vmap(F)(x), 0, atol=1e-3)