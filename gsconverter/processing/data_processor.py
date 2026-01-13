import numpy as np
from collections import deque
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors
from ..utils.utility_functions import debug_print, init_worker, status_print

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def apply_density_filter(self, voxel_size=1.0, threshold_percentage=0.32, sensitivity=None, keep_multicluster=False):
        debug_print("[DEBUG] Executing 'apply_density_filter' function...")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")
            
        # Map sensitivity if provided (0.0 - 1.0) -> (voxel_size, threshold)
        if sensitivity is not None:
             # Heuristic mapping:
             # Sens 0.1 (Lazy) -> Voxel 2.0, Thresh 0.1
             # Sens 0.5 (Normal) -> Voxel 1.0, Thresh 0.32
             # Sens 0.9 (Aggressive) -> Voxel 0.5, Thresh 1.0
             
             # Derived logic:
             voxel_size = 2.0 - (sensitivity * 1.8)
             voxel_size = max(0.1, voxel_size)
             
             # Threshold: 0.1 -> 1.0
             threshold_percentage = 0.1 + (sensitivity * 0.9)
             
        debug_print(f"Density Filter Params: Voxel={voxel_size:.4f}, Thresh={threshold_percentage:.4f}%, MultiCluster={keep_multicluster}")

            
        vertices = self.data
        
        # Quantize specific coordinates
        # Using numpy floor division for speed
        # Coords as Nx3 array
        coords = np.column_stack((vertices['x'], vertices['y'], vertices['z']))
        quantized_coords = np.floor(coords / voxel_size).astype(np.int64)
        
        # Unique voxels and counts
        # This replaces the manual loop and parallel_voxel_counting
        unique_voxels, inverse_indices, counts = np.unique(quantized_coords, axis=0, return_inverse=True, return_counts=True)
        
        debug_print(f"[DEBUG] Found {len(unique_voxels)} unique voxels.")
        
        # Threshold: Points per voxel based on percentage of total
        min_points = int(len(vertices) * (threshold_percentage / 100.0))
        
        # Filter voxels based on density
        dense_mask = counts >= min_points
        dense_voxel_indices = np.where(dense_mask)[0]
        
        if len(dense_voxel_indices) == 0:
            status_print("Warning: Density filter removed all points.")
            self.data = self.data[:0]
            return self.data
            
        dense_voxel_set = set(map(tuple, unique_voxels[dense_voxel_indices]))
        
        # BFS for clusters
        visited = set()
        clusters = []
        
        for voxel in dense_voxel_set:
            if voxel not in visited:
                current_cluster = set()
                queue = deque([voxel])
                visited.add(voxel)
                current_cluster.add(voxel)
                
                while queue:
                    curr = queue.popleft()
                    cx, cy, cz = curr
                    neighbors = [
                        (cx-1, cy, cz), (cx+1, cy, cz),
                        (cx, cy-1, cz), (cx, cy+1, cz),
                        (cx, cy, cz-1), (cx, cy, cz+1)
                    ]
                    
                    for n in neighbors:
                        if n in dense_voxel_set and n not in visited:
                            visited.add(n)
                            current_cluster.add(n)
                            queue.append(n)
                
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                    
        # Filter clusters
        if not clusters:
             self.data = self.data[:0]
             return self.data
             
        clusters.sort(key=len, reverse=True)
        max_len = len(clusters[0])
        
        min_cluster_size = max_len * 0.05 if keep_multicluster else max_len # 5% of largest if keeping multi, else just largest
        
        valid_voxels = set()
        kept_clusters = 0
        for c in clusters:
             if len(c) >= min_cluster_size:
                 valid_voxels.update(c)
                 kept_clusters += 1
                 if not keep_multicluster: break 
        


        # Mask original vertices
        is_in_cluster = np.array([tuple(v) in valid_voxels for v in unique_voxels])
        vertex_mask = is_in_cluster[inverse_indices]
        
        self.data = vertices[vertex_mask]
        status_print(f"Density Filter: Kept {kept_clusters} clusters (largest: {max_len} voxels).")
        status_print(f"After density filter, retained {len(self.data)} out of {len(vertices)} vertices.")
        return self.data

    def remove_flyers(self, k=25, threshold_factor=10.5, chunk_size=50000, intensity=None):
        debug_print("[DEBUG] Executing 'remove_flyers' function...")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")
            
        # Map intensity (1-10) -> (k, threshold_factor)
        if intensity is not None:
             # Intensity 1 (Weak): K=10, Factor=20.0
             # Intensity 5 (Medium): K=25, Factor=10.5
             # Intensity 10 (Strong): K=50, Factor=3.0
             
             # K mapping: 1->10, 10->50. 
             k = int(10 + (intensity - 1) * (40 / 9))
             
             # Factor mapping: 1->20.0, 10->3.0
             threshold_factor = 20.0 - (intensity - 1) * (17.0 / 9)
             
        debug_print(f"SOR Filter (Remove Flyers) Params: K={k}, Sigma={threshold_factor:.2f}")

        vertices = self.data
        coords = np.column_stack((vertices['x'], vertices['y'], vertices['z']))
        num_points = len(coords)

        # Try GPU Implementation
        from .gpu_ops import filter_sor_gpu, HAS_TAICHI
        if HAS_TAICHI:
            status_print("[SOR] Determining outliers on GPU (Taichi)...")
            try:
                gpu_mask = filter_sor_gpu(coords, k, threshold_factor, verbose=True)
                if gpu_mask is not None:
                     self.data = vertices[gpu_mask]
                     status_print(f"After removing flyers (GPU), retained {len(self.data)} out of {num_points} vertices.")
                     return self.data
            except Exception as e:
                debug_print(f"[SOR] GPU Failed: {e}. Fallback to CPU.")
        
        # Use Scipy cKDTree (CPU Fallback)
        from scipy.spatial import cKDTree
        
        # Build tree once
        status_print("[SOR] Building KDTree (CPU Fallback)...")
        tree = cKDTree(coords)
        
        # Query K neighbors
        debug_print("[DEBUG] Querying KNN...")
        all_mean_dists = np.zeros(num_points, dtype=np.float32)
        
        # Helper for batch processing
        for i in range(0, num_points, chunk_size):
            end = min(i + chunk_size, num_points)
            batch = coords[i:end]
            
            dists, _ = tree.query(batch, k=k+1, workers=cpu_count()-1)
            mean_dists = np.mean(dists[:, 1:], axis=1) # Exclude self at 0
            all_mean_dists[i:end] = mean_dists
            
        # Threshold
        global_mean = np.mean(all_mean_dists)
        global_std = np.std(all_mean_dists)
        threshold = global_mean + threshold_factor * global_std
        
        mask = all_mean_dists < threshold
        status_print(f"After removing flyers, retained {len(self.data)} out of {num_points} vertices.")
        return self.data

    def apply_alpha_filter(self, min_opacity_u8):
        debug_print(f"[DEBUG] Executing 'apply_alpha_filter' with min={min_opacity_u8}")
        
        if 'opacity' not in self.data.dtype.names:
            status_print("Warning: No opacity channel found. Alpha filter skipped.")
            return

        # Opacity is logit in internal representation: opacity = log(alpha / (1-alpha))
        # Filter where alpha_u8 >= min_opacity_u8
        
        # Compute threshold in Logit space to avoid full conversion
        # alpha = min / 255.0
        # logit = log(alpha / (1-alpha))
        
        limit = min_opacity_u8
        if limit <= 0: return
        if limit >= 255: 
             self.data = self.data[:0]; return

        alpha_thresh = limit / 255.0
        alpha_thresh = np.clip(alpha_thresh, 1e-6, 1.0-1e-6)
        logit_thresh = np.log(alpha_thresh / (1.0 - alpha_thresh))
        
        # Filter: keep if opacity >= logit_thresh
        mask = self.data['opacity'] >= logit_thresh
        original_len = len(self.data)
        self.data = self.data[mask]
        
        status_print(f"Alpha Filter (min {limit}): Retained {len(self.data)} out of {original_len} splats.")
        return self.data

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
        
        # Informative print statement
        status_print(f"After cropping, retained {len(self.data)} vertices.")

        return self.data

    def add_rgb_from_sh(self):
        debug_print("[DEBUG] Executing 'add_rgb_from_sh' function...")
        
        # Check if RGB already exists in dtype
        if 'red' in self.data.dtype.names:
            debug_print("[DEBUG] RGB fields already exist.")
            return

        # Compute RGB from SH (DC component)
        if 'f_dc_0' in self.data.dtype.names:
             f_dc = np.column_stack((self.data['f_dc_0'], self.data['f_dc_1'], self.data['f_dc_2']))
        # In case internal structure used scalar_ prefix (should not happen with new design but safe check)
        elif 'scalar_f_dc_0' in self.data.dtype.names:
             f_dc = np.column_stack((self.data['scalar_f_dc_0'], self.data['scalar_f_dc_1'], self.data['scalar_f_dc_2']))
        else:
            debug_print("[DEBUG] No SH DC components found, cannot compute RGB.")
            return

        # Using internal logic
        colors = self._compute_rgb_from_sh(self.data)
        
        if colors is None:
            return


        # Create new structured array with RGB
        # Extend the dtype
        new_dtype = self.data.dtype.descr + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        new_data = np.empty(len(self.data), dtype=new_dtype)
        
        for name in self.data.dtype.names:
            new_data[name] = self.data[name]
            
        new_data['red'] = colors[:, 0]
        new_data['green'] = colors[:, 1]
        new_data['blue'] = colors[:, 2]
        
        self.data = new_data
        debug_print("[DEBUG] RGB added to data.")

    def cap_sh_degree(self, degree):
        """
        Zeros out SH coefficients above the specified degree.
        Degree 0: Keep DC only (set all f_rest_* to 0)
        Degree 1: Keep first 9 f_rest_* coefficients
        Degree 2: Keep first 24 f_rest_* coefficients
        Degree 3: Keep all (no change)
        """
        if degree is None or degree >= 3:
            return self.data
            
        debug_print(f"[DEBUG] Capping SH degree to {degree}")
        
        # Mapping degree to starting index of coefficients to remove
        # Order 0: 0 AC (remove from index 0)
        # Order 1: 9 AC (remove from index 9)
        # Order 2: 24 AC (remove from index 24)
        degree_to_start_idx = {0: 0, 1: 9, 2: 24}
        start_idx = degree_to_start_idx.get(degree, 45)
        
        for i in range(start_idx, 45):
            f_name = f'f_rest_{i}'
            if f_name in self.data.dtype.names:
                self.data[f_name] = 0.0
                
        return self.data

    @staticmethod
    def _compute_rgb_from_sh(vertices):
        """
        Compute RGB from SH DC component (0-th order).
        Formula: RGB = 0.5 + C0 * SH
        """
        # 0.28209479177387814 is SH constant C0
        SH_C0 = 0.28209479177387814
        
        # Depending on the available field names, choose the appropriate ones
        if 'f_dc_0' in vertices.dtype.names:
            f_dc = np.column_stack((vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']))
        elif 'scalar_f_dc_0' in vertices.dtype.names:
             f_dc = np.column_stack((vertices['scalar_f_dc_0'], vertices['scalar_f_dc_1'], vertices['scalar_f_dc_2']))
        else:
             # CloudCompare scalar_ prefixes
             if 'scalar_scalar_f_dc_0' in vertices.dtype.names:
                  f_dc = np.column_stack((vertices['scalar_scalar_f_dc_0'], vertices['scalar_scalar_f_dc_1'], vertices['scalar_scalar_f_dc_2']))
             else:
                  return None
        
        # Convert SH to Linear RGB
        # RGB = 0.5 + SH * C0
        rgb_linear = 0.5 + f_dc * SH_C0
        
        # Clamp to 0..1 (Linear)
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        
        # Apply sRGB Gamma Correction
        # sRGB = x^(1/2.2)
        rgb_srgb = np.power(rgb_linear, 1.0/2.2)
        
        colors = (rgb_srgb * 255).astype(np.uint8)
        return colors

    def apply_auto_bbox(self):
        """
        No-op on the data itself (data defines the bbox), 
        but calculates and prints the tight bounding box of the remaining points.
        Useful for logging the result of filtering operations transparently.
        """
        debug_print("[DEBUG] Auto-Correction of Bounding Box (Calculating tight fit)...")
        if len(self.data) == 0:
            status_print("Auto-BBox: No points remaining. Bounding box is undefined.")
            return

        min_x = np.min(self.data['x'])
        min_y = np.min(self.data['y'])
        min_z = np.min(self.data['z'])
        
        max_x = np.max(self.data['x'])
        max_y = np.max(self.data['y'])
        max_z = np.max(self.data['z'])
        
        status_print(f"Auto-BBox Applied: [{min_x:.4f}, {min_y:.4f}, {min_z:.4f}] to [{max_x:.4f}, {max_y:.4f}, {max_z:.4f}]")
