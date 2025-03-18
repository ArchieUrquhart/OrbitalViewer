import numpy as np
import scipy.special as sp
from scipy.constants import physical_constants
from skimage import measure
import fast_simplification as fs


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#calculates the radial term
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




#calculates the angular term (normalization factor and legendre)
def angular_function(m, l, theta, phi, real):
    #Associated Legendre polynomial
    legendre = sp.lpmv(abs(m), l, np.cos(theta))
    
    # Normalization factor 
    norm_factor = np.sqrt(
        ((2 * l + 1) / (4 * np.pi)) * (sp.factorial(l - abs(m)) / sp.factorial(l + abs(m)))
    )
    
    if real == True:
        # Compute tesseral spherical harmonics
        if m < 0:
            return norm_factor * legendre * np.sin(abs(m) * phi)  # imaginary component
        elif m > 0:
            return norm_factor * legendre * np.cos(m * phi)  # real component
        else: 
            return norm_factor * legendre / np.sqrt(2)
        
    else:
        #compute Laplace spherical harmonics
        return norm_factor * legendre * np.exp(1j * m * phi)





#calculate whole wave functions
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
    Y_lm = angular_function(m, l, theta, phi, False) 
    
    # Compute total wavefunction Ψ(n,l,m)
    psi = R_nl * Y_lm 
        

    #return axes and probability density array
    return x, y, z, [np.real(psi), np.abs(psi), np.imag(psi)]







#uses plotly to display the orbital to a local host page
def show_obj(verts,faces, prob_densities):
    fig = go.Figure(data=[go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], 
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    intensity=prob_densities[0],
                                    colorscale='temps',
                                    opacity=1)])

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                bgcolor='rgb(0, 0, 0)'),
                    margin=dict(l=0, r=0, b=0, t=0))
    fig.show()






#writes the output of marching cubes to 
def write_obj(filename, verts, faces, normals):
    with open(filename, "w") as file:
        file.write("# OBJ file\n")
        
        # Write vertices
        for v in verts:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write normals if provided
        if normals:
            for n in normals:
                file.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # Write faces
        for face in faces:
            file.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


        print(f"OBJ file '{filename}' written successfully.") 





#get obj file 
def get_obj(n, l, m, orbital_size=2, grid_size=200, threshold=0.007):

    orbital_size = orbital_size/ n
    threshold = threshold/ n**2

    x, y, z, prob_densities = compute_wavefunction(n, l, m, grid_size, orbital_size)
    
    #0: real values, 1: absolute values, 2: imaginary values
    prob_density = prob_densities[0] ** 2

    verts, faces, normals, values = measure.marching_cubes(prob_density, threshold)

    verts, faces = fs.simplify(verts, faces, 0.9)
    write_obj("orbital.obj",verts,faces, None)
    #show_obj(verts,faces,prob_density)




get_obj(n=3, l=2, m=0)

