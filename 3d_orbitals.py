import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.constants import physical_constants

#
def radial_function(n, l, r, a0):
    #Compute the normalized radial wavefunction using Laguerre polynomials
    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)
    p = 2 * r / (n * a0)
    
    # Normalization factor
    constant_factor = np.sqrt(
        ((2 / (n * a0)) ** 3 * sp.factorial(n - l - 1)) /
        (2 * n * (sp.factorial(n + l)))
    )
    
    return constant_factor * np.exp(-p / 2) * (p ** l) * laguerre(p)


#
def angular_function(m, l, theta, phi):
    #Associated Legendre polynomial
    legendre = sp.lpmv(abs(m), l, np.cos(theta))
    
    # Normalization factor 
    norm_factor = np.sqrt(
        ((2 * l + 1) / (4 * np.pi)) * (sp.factorial(l - abs(m)) / sp.factorial(l + abs(m)))
    )
    
    # Compute tesseral spherical harmonics
    if m < 0:
        return norm_factor * legendre * np.sin(abs(m) * phi) * np.sqrt(2) # imaginary component
    elif m > 0:
        return norm_factor * legendre * np.cos(m * phi) * np.sqrt(2) # real component
    else: 
        return norm_factor * legendre



#
def compute_wavefunction(n, l, m, grid_size, scale):
    # Scaled Bohr radius
    a0 = scale * physical_constants['Bohr radius'][0] * 1e+10
    
    # Define 3D grid (spanning -10 to 10 in AU containg grid_size points in each direction) 
    lin_space = np.linspace(-10, 10, grid_size)
    x, y, z = np.meshgrid(lin_space, lin_space, lin_space)
    
    # Convert Cartesian to Spherical
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-10  
    theta = np.arccos(z / r) # Polar angle
    phi = np.arctan2(y, x)  # Azimuthal angle

    # Compute radial and angular components
    R_nl = radial_function(n, l, r, a0)
    Y_lm = angular_function(m, l, theta, phi) 
    
    # Compute total wavefunction Ψ(n,l,m)
    psi = R_nl * Y_lm 
    psi = np.abs(psi)

    #return axes and probability density
    return x, y, z, psi ** 2  




#display raw function with matplotlib
def plot_3d_orbital(n, l, m, orbital_size=2.3, grid_size=100, threshold=0.005):

    orbital_size = orbital_size/ n

    x, y, z, prob_density = compute_wavefunction(n, l, m, grid_size, orbital_size)

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=prob_density.flatten(),
        isomin= threshold/n**2, # decrease threshold as n increases
        isomax=prob_density.max(),
        opacity=0.5, surface_count=3, colorscale="temps"
    ))
    
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    
    fig.show()


def get_obj():
    pass


plot_3d_orbital(n=3, l=2, m=-2)

