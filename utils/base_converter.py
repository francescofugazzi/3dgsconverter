"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
import pandas as pd
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
        # Ensure self.data is a numpy structured array
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")
            
        vertices = self.data  # This assumes self.data is already a numpy structured array

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

        # Filter vertices to only include those in dense voxels
        filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]

        # Convert the filtered vertices list to a numpy structured array
        self.data = np.array(filtered_vertices, dtype=vertices.dtype)

        # Informative print statement
        print(f"After density filter, retained {len(self.data)} out of {len(vertices)} vertices.")

        # Since we're working with numpy arrays, just return self.data
        return self.data

    def remove_flyers(self, k=25, threshold_factor=10.5, chunk_size=50000):
        debug_print("[DEBUG] Executing 'remove_flyers' function...")

        # Ensure self.data is a numpy structured array
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")

        # Extract vertex data from the current object's data
        vertices = self.data
        num_vertices = len(vertices)
        
        # Display the number of input vertices
        debug_print(f"[DEBUG] Number of input vertices: {num_vertices}")
        
        # Adjust k based on the number of vertices
        k = max(3, min(k, num_vertices // 100))  # Example: ensure k is between 3 and 1% of the total vertices
        debug_print(f"[DEBUG] Adjusted k to: {k}")

        # Number of chunks
        num_chunks = (num_vertices + chunk_size - 1) // chunk_size  # Ceiling division
        masks = []

        # Create a pool of workers
        num_cores = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, num_vertices)  # Avoid going out of bounds
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

        # Apply the mask to the vertices and store the result in self.data
        self.data = vertices[combined_mask]
        
        print(f"After removing flyers, retained {np.count_nonzero(combined_mask)} out of {num_vertices} vertices.")
        return self.data

    @staticmethod
    def define_dtype(has_scal, has_rgb=False):
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
        return 'red' in self.data.dtype.names and 'green' in self.data.dtype.names and 'blue' in self.data.dtype.names

    def crop_by_bbox(self, min_x, min_y, min_z, max_x, max_y, max_z):
        # Perform cropping based on the bounding box
        self.data = self.data[
            (self.data['x'] >= min_x) &
            (self.data['x'] <= max_x) &
            (self.data['y'] >= min_y) &
            (self.data['y'] <= max_y) &
            (self.data['z'] >= min_z) &
            (self.data['z'] <= max_z)
        ]
        # Print the number of vertices after cropping
        debug_print(f"[DEBUG] Number of vertices after cropping: {len(self.data)}")

    @staticmethod
    def load_parquet(file_path):
            # Load the Parquet file into a DataFrame
            df = pd.read_parquet(file_path)
            
            # Define a mapping from the Parquet column names to the expected dtype names
            column_mapping = {
                'x': 'x',
                'y': 'y',
                'z': 'z',
                # Assuming 'nx', 'ny', 'nz' need to be created and set to 0
                'r_sh0': 'f_dc_0',
                'g_sh0': 'f_dc_1',
                'b_sh0': 'f_dc_2',
                'r_sh1': 'f_rest_0',
                'r_sh2': 'f_rest_1',
                'r_sh3': 'f_rest_2',
                'r_sh4': 'f_rest_3',
                'r_sh5': 'f_rest_4',
                'r_sh6': 'f_rest_5',
                'r_sh7': 'f_rest_6',
                'r_sh8': 'f_rest_7',
                'r_sh9': 'f_rest_8',
                'r_sh10': 'f_rest_9',
                'r_sh11': 'f_rest_10',
                'r_sh12': 'f_rest_11',
                'r_sh13': 'f_rest_12',
                'r_sh14': 'f_rest_13',
                'r_sh15': 'f_rest_14',
                'g_sh1': 'f_rest_15',
                'g_sh2': 'f_rest_16',
                'g_sh3': 'f_rest_17',
                'g_sh4': 'f_rest_18',
                'g_sh5': 'f_rest_19',
                'g_sh6': 'f_rest_20',
                'g_sh7': 'f_rest_21',
                'g_sh8': 'f_rest_22',
                'g_sh9': 'f_rest_23',
                'g_sh10': 'f_rest_24',
                'g_sh11': 'f_rest_25',
                'g_sh12': 'f_rest_26',
                'g_sh13': 'f_rest_27',
                'g_sh14': 'f_rest_28',
                'g_sh15': 'f_rest_29',
                'b_sh1': 'f_rest_30',
                'b_sh2': 'f_rest_31',
                'b_sh3': 'f_rest_32',
                'b_sh4': 'f_rest_33',
                'b_sh5': 'f_rest_34',
                'b_sh6': 'f_rest_35',
                'b_sh7': 'f_rest_36',
                'b_sh8': 'f_rest_37',
                'b_sh9': 'f_rest_38',
                'b_sh10': 'f_rest_39',
                'b_sh11': 'f_rest_40',
                'b_sh12': 'f_rest_41',
                'b_sh13': 'f_rest_42',
                'b_sh14': 'f_rest_43',
                'b_sh15': 'f_rest_44',
                'alpha': 'opacity',
                'cov_s0': 'scale_0',
                'cov_s1': 'scale_1',
                'cov_s2': 'scale_2',
                'cov_q3': 'rot_0',
                'cov_q0': 'rot_1',
                'cov_q1': 'rot_2',
                'cov_q2': 'rot_3',
            }

            for col in ['nx', 'ny', 'nz']:
                if col not in df.columns:
                    df[col] = 0.0

            # Rename the DataFrame columns according to the mapping
            df_renamed = df.rename(columns=column_mapping)

            # Fetch the dtype from BaseConverter
            dtype_list, _ = BaseConverter.define_dtype(has_scal=False, has_rgb=False)
            
            # Convert the dtype list to a structured dtype object
            dtype_structured = np.dtype(dtype_list)

            # Convert DataFrame to a structured array with the defined dtype
            structured_array = np.zeros(df_renamed.shape[0], dtype=dtype_structured)
            for name in dtype_structured.names:
                structured_array[name] = df_renamed[name].values if name in df_renamed.columns else 0

            return structured_array