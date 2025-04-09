import numpy as np
import scipy.special as sp
from scipy.constants import physical_constants
from skimage import measure
import fast_simplification as fs
import json



import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




#uses plotly to display the orbital to a local host page
def show_obj(verts,faces, prob_densities):
    fig = go.Figure(data=[go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], 
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    intensity=prob_densities,
                                    colorscale='temps',
                                    opacity=1)])

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                bgcolor='rgb(0, 0, 0)'),
                    margin=dict(l=0, r=0, b=0, t=0))
    fig.show()



#writes the output of marching cubes to obj file
def write_obj(filename, verts, faces, normals):
    with open(filename, "w") as file:
        file.write("# OBJ file\n")
        
        # Write vertices
        for v in verts:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write normals if provided
        if normals is not None:
            for n in normals:
                file.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # Write faces
        for face in faces:
            file.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


        print(f"OBJ file '{filename}' written successfully.") 


#reads obj file
def read_obj(file_path):
    vertices = []
    faces = []
    normals = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':  # Vertex
                vertices.append(tuple(map(float, parts[1:])))
            elif parts[0] == 'vn':  # Normal
                normals.append(tuple(map(float, parts[1:])))
            elif parts[0] == 'f':  # Face
                face = []
                for vert in parts[1:]:
                    indices = vert.split('/')
                    face.append(tuple(map(lambda x: int(x) if x else None, indices)))
                faces.append(face)
    
    return vertices, faces, normals




#reduce resolution of obj file from its given verts and faces
def simplify_obj_from_verts_faces(verts, faces,percent):
    percent = 1 - percent
    verts, faces = fs.simplify(verts, faces, percent)
    return verts, faces


#directly reduce resolution of an obj file
def simplify_obj(file,percent):
    percent = 1.0 - percent
    read_obj(file)
    verts, faces = fs.simplify(verts, faces, percent)
    write_obj(file,verts,faces,None)




#get data from a json file
def get_data(jsonFile):
    # Open and read the JSON file
    with open(jsonFile, 'r') as file:
        data = json.load(file)

    # Print the data
    print(data)

    #load data into variables
    n = data['n']
    l = data['l']
    m = data['m']

    grid_rez = data['grid_rez']
    mesh_decimation = data['mesh_dec']









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
    return x, y, z, psi






#get obj file from wave function 
def get_obj(n, l, m, orbital_size=2, grid_size=200, threshold=0.007):

    orbital_size = orbital_size / n
    threshold = threshold / n**2

    x, y, z, wave_function = compute_wavefunction(n, l, m, grid_size, orbital_size)

    # Compute probability density for marching cubes
    prob_density = np.real(wave_function) ** 2  

    # Extract surface from marching cubes
    verts, faces, normals, values = measure.marching_cubes(prob_density, threshold)

    # Simplify the mesh
    verts, faces = simplify_obj_from_verts_faces(verts, faces, 1)




    # Get imaginary component at surface vertices
    real_component = np.real(wave_function)
    
    # Normalize the imaginary values for coloring
    real_values = real_component[tuple(verts.T.astype(int))]
    real_values = (real_values - real_values.min()) / (real_values.max() - real_values.min() + 1e-100)



    # Get imaginary component at surface vertices
    both_component = np.abs(wave_function)
    
    # Normalize the imaginary values for coloring
    abs_values = both_component[tuple(verts.T.astype(int))]
    abs_values = (abs_values - abs_values.min()) / (abs_values.max() - abs_values.min()  + 1e-100)


    
    # Get imaginary component at surface vertices
    imag_component = np.imag(wave_function)
    
    # Normalize the imaginary values for coloring
    imag_values = imag_component[tuple(verts.T.astype(int))]
    imag_values = (imag_values - imag_values.min()) / (imag_values.max() - imag_values.min()  + 1e-100)
    


    # Save to OBJ
    write_obj("orbital.obj", verts, faces, None)

    # Show with color based on imaginary component
    show_obj(verts, faces, real_values)


get_obj(n=4, l=3 , m= 2)
