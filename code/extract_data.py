import numpy as np
import open3d as o3d

def extract_point_data(image):
    threshold = 1000  # threshold controls point density. destroys my kernel
    significant_points = np.argwhere(image > threshold)
    intensities = image[significant_points[:, 0], significant_points[:, 1], significant_points[:, 2]]
    point_data = np.column_stack((significant_points, intensities))
    return point_data

def extract_vertices(point_data):
    # Extract the x, y, z coordinates (first three columns)
    vertices = point_data[:, :3]
    return vertices

def extract_ply(file_path):
    vertices = []
    vertex_count = 0
    start_reading = False
    
    with open(file_path, 'r') as f:
        for line in f:
            if 'element vertex' in line:
                vertex_count = int(line.split()[-1])
            elif 'end_header' in line:
                start_reading = True
                continue
            elif start_reading and vertex_count > 0:
                try:
                    x, y, z = map(float, line.split()[:3])
                    vertices.append([x, y, z])
                    vertex_count -= 1
                except:
                    continue
                    
    return np.array(vertices) 
    
def extract_ply_o3(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    return vertices
