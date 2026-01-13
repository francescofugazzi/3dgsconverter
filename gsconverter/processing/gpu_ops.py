import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

# Attempt to import Taichi
HAS_TAICHI = False
try:
    import os
    os.environ["TI_LOG_LEVEL"] = "error"
    import taichi as ti
    try:
        ti.init(arch=ti.gpu, offline_cache=True, log_level='error')
    except Exception as e:
        if "already initialized" not in str(e) and "Multi-threading" not in str(e):
             try:
                ti.init(arch=ti.cpu, log_level='error')
             except: pass

    HAS_TAICHI = True
except ImportError:
    pass

from sklearn.cluster import MiniBatchKMeans

def kmeans(data: np.ndarray, k: int, max_iter=10, tolerance=1e-4, use_gpu=True, verbose=False):
    N, D = data.shape
    
    if k >= N:
        return data.copy(), np.arange(N, dtype=np.int32)
        
    # Fallback if no Taichi or CPU requested
    if not HAS_TAICHI or not use_gpu:
        if verbose: 
            from ..utils.utility_functions import status_print
            status_print(f"[GPU_OPS] Fallback to Sklearn (CPU) for K={k}...")
        return _kmeans_sklearn(data, k, max_iter)

    try:
        return _kmeans_taichi(data, k, max_iter)
    except Exception as e:
        from ..utils.utility_functions import debug_print, status_print
        status_print(f"[GPU_OPS] Taichi Execution failed: {e}. Falling back to Sklearn (CPU).")
        debug_print(f"Taichi Execution failed: {e}. Falling back to Sklearn.")
        return _kmeans_sklearn(data, k, max_iter)

def _kmeans_sklearn(data, k, max_iter):
    bs = min(4096 * 4, len(data))
    km = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, batch_size=bs, n_init='auto', compute_labels=True)
    km.fit(data)
    return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.int32)

# --- Taichi Kernels (Module Level for Stability) ---

if HAS_TAICHI:
    @ti.kernel
    def k_means_assign(data: ti.types.ndarray(), centroids: ti.types.ndarray(), labels: ti.types.ndarray(), N: int, K: int, D: int):
        for i in range(N):
            min_dist = 1e20
            best_k = -1
            
            for c in range(K):
                dist = 0.0
                for dim in range(D):
                    diff = data[i, dim] - centroids[c, dim]
                    dist += diff * diff
                
                if dist < min_dist:
                    min_dist = dist
                    best_k = c
            
            labels[i] = best_k

    @ti.kernel
    def k_means_update(data: ti.types.ndarray(), centroids: ti.types.ndarray(), labels: ti.types.ndarray(), counts: ti.types.ndarray(), N: int, K: int, D: int):
        # Reset
        for c in range(K):
            for dim in range(D):
                centroids[c, dim] = 0.0
            counts[c] = 0
            
        # Accumulate
        for i in range(N):
            l = labels[i]
            for dim in range(D):
                ti.atomic_add(centroids[l, dim], data[i, dim])
            ti.atomic_add(counts[l], 1)
            
        # Average
        for c in range(K):
            cnt = counts[c]
            if cnt > 0:
                inv = 1.0 / float(cnt)
                for dim in range(D):
                    centroids[c, dim] *= inv

    @ti.kernel
    def sor_compute_mean_dists(pos: ti.types.ndarray(), 
                              cell_start: ti.types.ndarray(), 
                              cell_count: ti.types.ndarray(), 
                              mean_dists: ti.types.ndarray(),
                              bbox_min_x: float, bbox_min_y: float, bbox_min_z: float,
                              cell_size: float, hash_size: int,
                              N: int, K: int):
        for i in range(N):
            p_x = pos[i, 0]
            p_y = pos[i, 1]
            p_z = pos[i, 2]
            
            # Grid Index
            if cell_size > 1e-8:
                gx = int(ti.floor((p_x - bbox_min_x) / cell_size))
                gy = int(ti.floor((p_y - bbox_min_y) / cell_size))
                gz = int(ti.floor((p_z - bbox_min_z) / cell_size))
                
                # Stack for top-K distances (manual fixed size array)
                # Max K supported is 50
                dists = ti.Vector([1.0e10] * 50) 
                
                # Neighbor Search
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            nx = gx + dx
                            ny = gy + dy
                            nz = gz + dz
                            
                            # Spatial Hash
                            p1 = 73856093; p2 = 19349663; p3 = 83492791
                            h = ((nx * p1) ^ (ny * p2) ^ (nz * p3)) % hash_size
                            if h < 0: h += hash_size
                            
                            start = cell_start[h]
                            cnt = cell_count[h]
                            
                            if start != -1:
                                for j in range(start, start + cnt):
                                    # Distance sq
                                    op_x = pos[j, 0]
                                    op_y = pos[j, 1]
                                    op_z = pos[j, 2]
                                    
                                    diff_x = p_x - op_x
                                    diff_y = p_y - op_y
                                    diff_z = p_z - op_z
                                    d2 = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
                                    
                                    if d2 > 1.0e-12: # Skip self
                                        d = ti.sqrt(d2)
                                        
                                        # Insertion Sort into dists (asc)
                                        # Only if d < largest in heap (dists[K-1])
                                        if d < dists[K - 1]:
                                            # Find position
                                            ins_pos = K - 1
                                            while ins_pos > 0 and dists[ins_pos-1] > d:
                                                dists[ins_pos] = dists[ins_pos-1]
                                                ins_pos -= 1
                                            dists[ins_pos] = d
                
                # Compute Mean
                sum_d = 0.0
                valid_k = 0
                for ki in range(K):
                    val = dists[ki]
                    if val < 0.9e10:
                        sum_d += val
                        valid_k += 1
                
                if valid_k > 0:
                    mean_dists[i] = sum_d / float(valid_k)
                else:
                    mean_dists[i] = 0.0 # Isolated?
            else:
                 mean_dists[i] = 0.0

def _kmeans_taichi(data_np, K, max_iter=20):
    N, D = data_np.shape
    # Ensure types
    data_np = data_np.astype(np.float32)
    centroids_np = data_np[np.random.choice(N, K, replace=False)].astype(np.float32)
    labels_np = np.zeros(N, dtype=np.int32)
    counts_np = np.zeros(K, dtype=np.int32)
    
    for _ in range(max_iter):
        k_means_assign(data_np, centroids_np, labels_np, N, K, D)
        k_means_update(data_np, centroids_np, labels_np, counts_np, N, K, D)
        
    ti.sync()
    return centroids_np, labels_np

def filter_sor_gpu(data_np: np.ndarray, k: int = 25, threshold_factor: float = 1.0, verbose=False):
    if not HAS_TAICHI:
        return None
        
    N, D = data_np.shape
    if D != 3: raise ValueError("Requires 3D data")
    
    pos_np = data_np.astype(np.float32)
    
    # 1. Grid Setup (CPU)
    min_bound = np.min(pos_np, axis=0)
    max_bound = np.max(pos_np, axis=0)
    extent = max_bound - min_bound
    vol = np.prod(extent)
    if vol <= 0: vol = 1.0
    
    target_points_per_cell = 32
    avg_vol = max(1e-8, vol / N)
    cell_vol = avg_vol * target_points_per_cell
    cell_size = float(cell_vol ** (1.0/3.0))
    cell_size = max(cell_size, 1e-4) # Min cell size
    
    # Grid indices
    grid_indices = np.floor((pos_np - min_bound) / cell_size).astype(np.int32)
    
    hash_size = N
    p1, p2, p3 = 73856093, 19349663, 83492791
    # Check for overflow in python (arbitrary precision) but numpy int32 might overflow in calc before mod
    # Use int64 for hash calc
    gi_64 = grid_indices.astype(np.int64)
    hashed = ((gi_64[:,0] * p1) ^ (gi_64[:,1] * p2) ^ (gi_64[:,2] * p3)) % hash_size
    hashed = hashed.astype(np.int32)
    
    # Sort
    sort_order = np.argsort(hashed)
    sorted_pos = pos_np[sort_order]
    sorted_hashes = hashed[sort_order]
    
    # Cell Lookups
    unique, idx, counts = np.unique(sorted_hashes, return_index=True, return_counts=True)
    cell_start = np.full(hash_size, -1, dtype=np.int32)
    cell_count = np.zeros(hash_size, dtype=np.int32)
    
    cell_start[unique] = idx
    cell_count[unique] = counts
    
    # Prepare Output
    mean_dists = np.zeros(N, dtype=np.float32)
    
    # Run Kernel
    # K is capped at 50 in kernel
    actual_k = min(k, 50)
    
    sor_compute_mean_dists(sorted_pos, cell_start, cell_count, mean_dists,
                          float(min_bound[0]), float(min_bound[1]), float(min_bound[2]),
                          cell_size, hash_size, N, actual_k)
    ti.sync()
    
    # Unsort stats (mean_dists corresponds to sorted_pos)
    # Restore mean_dists to original order
    # mean_dists[i] is mean dist for point sort_order[i]
    # So to get original order:
    final_means = np.zeros(N, dtype=np.float32)
    final_means[sort_order] = mean_dists
    
    # Stats
    glob_mean = np.mean(final_means)
    glob_std = np.std(final_means)
    thresh = glob_mean + threshold_factor * glob_std
    
    return final_means < thresh
