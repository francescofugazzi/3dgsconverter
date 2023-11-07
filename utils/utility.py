"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
from .utility_functions import debug_print, init_worker

class Utility:
    @staticmethod
    def text_based_detect_format(file_path):
        debug_print("[DEBUG] Executing 'text_based_detect_format' function...")

        """Detect if the given file is in '3dgs' or 'cc' format."""
        with open(file_path, 'rb') as file:
            header_bytes = file.read(2048)  # Read the beginning to detect the format

        header = header_bytes.decode('utf-8', errors='ignore')

        if "property float f_dc_0" in header:
            debug_print("[DEBUG] Detected format: 3dgs")
            return "3dgs"
        elif "property float scal_f_dc_0" in header or "property float scalar_scal_f_dc_0" in header or "property float scalar_f_dc_0" in header:
            debug_print("[DEBUG] Detected format: cc")
            return "cc"
        else:
            return None

    @staticmethod
    def copy_data_with_prefix_check(source, target, possible_prefixes):
        debug_print("[DEBUG] Executing 'copy_data_with_prefix_check' function...")

        """
        Given two structured numpy arrays (source and target), copy the data from source to target.
        If a field exists in source but not in target, this function will attempt to find the field
        in target by adding any of the possible prefixes to the field name.
        """
        for name in source.dtype.names:
            if name in target.dtype.names:
                target[name] = source[name]
            else:
                copied = False
                for prefix in possible_prefixes:
                    # If the field starts with the prefix, try the field name without the prefix
                    if name.startswith(prefix):
                        stripped_name = name[len(prefix):]
                        if stripped_name in target.dtype.names:
                            target[stripped_name] = source[name]
                            copied = True
                            break
                    # If the field doesn't start with any prefix, try adding the prefix
                    else:
                        prefixed_name = prefix + name
                        if prefixed_name in target.dtype.names:
                            debug_print(f"[DEBUG] Copying data from '{name}' to '{prefixed_name}'")
                            target[prefixed_name] = source[name]
                            copied = True
                            break
                ##if not copied:
                ##    print(f"Warning: Field {name} not found in target.")

    @staticmethod
    def compute_rgb_from_vertex(vertices):
        debug_print("[DEBUG] Executing 'compute_rgb_from_vertex' function...")
        
        # Depending on the available field names, choose the appropriate ones
        if 'f_dc_0' in vertices.dtype.names:
            f_dc = np.column_stack((vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']))
        else:
            f_dc = np.column_stack((vertices['scalar_scal_f_dc_0'], vertices['scalar_scal_f_dc_1'], vertices['scalar_scal_f_dc_2']))
        
        colors = (f_dc + 1) * 127.5
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        
        debug_print("[DEBUG] RGB colors computed.")
        return colors

    @staticmethod
    def parallel_voxel_counting(vertices, voxel_size=1.0):
        debug_print("[DEBUG] Executing 'parallel_voxel_counting' function...")
        
        """Counts the number of points in each voxel in a parallelized manner."""
        num_processes = cpu_count()
        chunk_size = len(vertices) // num_processes
        chunks = [vertices[i:i + chunk_size] for i in range(0, len(vertices), chunk_size)]

        num_cores = max(1, multiprocessing.cpu_count() - 1)
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            results = pool.starmap(Utility.count_voxels_chunk, [(chunk, voxel_size) for chunk in chunks])

        # Aggregate results from all processes
        total_voxel_counts = {}
        for result in results:
            for k, v in result.items():
                if k in total_voxel_counts:
                    total_voxel_counts[k] += v
                else:
                    total_voxel_counts[k] = v

        debug_print(f"[DEBUG] Voxel counting completed with {len(total_voxel_counts)} unique voxels found.")
        return total_voxel_counts
    
    @staticmethod
    def count_voxels_chunk(vertices_chunk, voxel_size):
        debug_print("[DEBUG] Executing 'count_voxels_chunk' function for a chunk...")
        
        """Count the number of points in each voxel for a chunk of vertices."""
        voxel_counts = {}
        for vertex in vertices_chunk:
            voxel_coords = (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size))
            if voxel_coords in voxel_counts:
                voxel_counts[voxel_coords] += 1
            else:
                voxel_counts[voxel_coords] = 1
        
        debug_print(f"[DEBUG] Chunk processed with {len(voxel_counts)} voxels counted.")
        return voxel_counts
    
    @staticmethod
    def get_neighbors(voxel_coords):
        debug_print(f"[DEBUG] Getting neighbors for voxel: {voxel_coords}...")
        
        """Get the face-touching neighbors of the given voxel coordinates."""
        x, y, z = voxel_coords
        neighbors = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        return neighbors

    @staticmethod
    def knn_worker(args):
        debug_print(f"[DEBUG] Executing 'knn_worker' function for vertex: {args[0]}...")
        
        """Utility function for parallel KNN computation."""
        coords, tree, k = args
        coords = coords.reshape(1, -1)  # Reshape to a 2D array
        distances, _ = tree.kneighbors(coords)
        avg_distance = np.mean(distances[:, 1:])
        
        debug_print(f"[DEBUG] Average distance computed for vertex: {args[0]} is {avg_distance}.")
        return avg_distance
    