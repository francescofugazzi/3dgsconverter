"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
from .utility import *
from plyfile import PlyData, PlyElement
from collections import deque
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors
from .utility_functions import debug_print, init_worker

class BaseConverter:
    def __init__(self, data):
        self.data = data

    def extract_vertex_data(vertices, has_scal=True, has_rgb=False):
        """Extract and convert vertex data from a structured numpy array of vertices."""
        debug_print("[DEBUG] Executing 'extract_vertex_data' function...")
        converted_data = []
        
        # Determine the prefix to be used based on whether "scal_" should be included
        prefix = 'scal_' if has_scal else ''
        debug_print(f"[DEBUG] Prefix determined as: {prefix}")
        
        # Iterate over each vertex and extract the necessary attributes
        for vertex in vertices:
            entry = (
                vertex['x'], vertex['y'], vertex['z'],
                vertex['nx'], vertex['ny'], vertex['nz'],
                vertex[f'{prefix}f_dc_0'], vertex[f'{prefix}f_dc_1'], vertex[f'{prefix}f_dc_2'],
                *[vertex[f'{prefix}f_rest_{i}'] for i in range(45)],
                vertex[f'{prefix}opacity'],
                vertex[f'{prefix}scale_0'], vertex[f'{prefix}scale_1'], vertex[f'{prefix}scale_2'],
                vertex[f'{prefix}rot_0'], vertex[f'{prefix}rot_1'], vertex[f'{prefix}rot_2'], vertex[f'{prefix}rot_3']
            )
            
            # If the point cloud contains RGB data, append it to the entry
            if has_rgb:
                entry += (vertex['red'], vertex['green'], vertex['blue'])
            
            converted_data.append(entry)
        
        debug_print("[DEBUG] 'extract_vertex_data' function completed.")
        return converted_data

    def apply_density_filter(self, voxel_size=1.0, threshold_percentage=0.32):
        debug_print("[DEBUG] Executing 'apply_density_filter' function...")
        vertices = self.data['vertex'].data

        # Convert threshold_percentage into a ratio
        threshold_ratio = threshold_percentage / 100.0

        # Parallelized voxel counting
        voxel_counts = Utility.parallel_voxel_counting(vertices, voxel_size)

        threshold = int(len(vertices) * threshold_ratio)
        dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

        visited = set()
        max_cluster = set()
        for voxel in dense_voxels:
            if voxel not in visited:
                current_cluster = set()
                queue = deque([voxel])
                while queue:
                    current_voxel = queue.popleft()
                    visited.add(current_voxel)
                    current_cluster.add(current_voxel)
                    for neighbor in Utility.get_neighbors(current_voxel):
                        if neighbor in dense_voxels and neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)
                if len(current_cluster) > len(max_cluster):
                    max_cluster = current_cluster

        filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]
        new_vertex_element = PlyElement.describe(np.array(filtered_vertices, dtype=vertices.dtype), 'vertex')
        
        # Update the plydata elements and the internal self.data
        converted_data = self.data
        converted_data.elements = (new_vertex_element,) + converted_data.elements[1:]
        self.data = converted_data  # Update the internal data with the filtered data
        
        print(f"After density filter, retained {len(filtered_vertices)} out of {len(vertices)} vertices.")
        return self.data

    def remove_flyers(self, k=25, threshold_factor=10.5, chunk_size=50000):
        debug_print("[DEBUG] Executing 'remove_flyers' function...")

        # Extract vertex data from the current object's data
        vertices = self.data['vertex'].data
        
        # Display the number of input vertices
        debug_print(f"[DEBUG] Number of input vertices: {len(vertices)}")

        # Number of chunks
        num_chunks = len(vertices) // chunk_size + (len(vertices) % chunk_size > 0)
        masks = []

        # Create a pool of workers
        num_cores = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk_coords = np.vstack((vertices['x'][start_idx:end_idx], vertices['y'][start_idx:end_idx], vertices['z'][start_idx:end_idx])).T

                # Compute K-Nearest Neighbors for the chunk
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(chunk_coords)
                avg_distances = pool.map(Utility.knn_worker, [(coord, nbrs, k) for coord in chunk_coords])

                # Calculate the threshold for removal based on the mean and standard deviation of the average distances
                threshold = np.mean(avg_distances) + threshold_factor * np.std(avg_distances)

                # Create a mask for points to retain for this chunk
                mask = np.array(avg_distances) < threshold
                masks.append(mask)

        # Combine masks from all chunks
        combined_mask = np.concatenate(masks)

        # Generate a new PlyElement with the filtered vertices based on the combined mask
        new_vertex_element = PlyElement.describe(vertices[combined_mask], 'vertex')

        # Update the plydata elements and the internal self.data
        self.data.elements = (new_vertex_element,) + self.data.elements[1:]
        
        print(f"After removing flyers, retained {len(vertices[combined_mask])} out of {len(vertices)} vertices.")
        return self.data


    def define_dtype(self, has_scal, has_rgb=False):
        debug_print("[DEBUG] Executing 'define_dtype' function...")
        
        prefix = 'scalar_scal_' if has_scal else ''
        debug_print(f"[DEBUG] Prefix determined as: {prefix}")
        
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            (f'{prefix}f_dc_0', 'f4'), (f'{prefix}f_dc_1', 'f4'), (f'{prefix}f_dc_2', 'f4'),
            *[(f'{prefix}f_rest_{i}', 'f4') for i in range(45)],
            (f'{prefix}opacity', 'f4'),
            (f'{prefix}scale_0', 'f4'), (f'{prefix}scale_1', 'f4'), (f'{prefix}scale_2', 'f4'),
            (f'{prefix}rot_0', 'f4'), (f'{prefix}rot_1', 'f4'), (f'{prefix}rot_2', 'f4'), (f'{prefix}rot_3', 'f4')
        ]
        debug_print("[DEBUG] Main dtype constructed.")
        
        if has_rgb:
            dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            debug_print("[DEBUG] RGB fields added to dtype.")
        
        debug_print("[DEBUG] 'define_dtype' function completed.")
        return dtype, prefix
    
    def has_rgb(self):
        return 'red' in self.data['vertex'].data.dtype.names and 'green' in self.data['vertex'].data.dtype.names and 'blue' in self.data['vertex'].data.dtype.names
    
    def crop_by_bbox(self, min_x, min_y, min_z, max_x, max_y, max_z):
        # Perform cropping based on the bounding box
        self.data['vertex'].data = self.data['vertex'].data[
            (self.data['vertex'].data['x'] >= min_x) &
            (self.data['vertex'].data['x'] <= max_x) &
            (self.data['vertex'].data['y'] >= min_y) &
            (self.data['vertex'].data['y'] <= max_y) &
            (self.data['vertex'].data['z'] >= min_z) &
            (self.data['vertex'].data['z'] <= max_z)
        ]
        # Print the number of vertices after cropping
        print(f"Number of vertices after cropping: {len(self.data['vertex'].data)}")
        
        return self.data