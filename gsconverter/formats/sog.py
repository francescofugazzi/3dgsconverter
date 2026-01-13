import numpy as np
import os
import json
import zipfile
import tempfile
import io
import struct
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print, status_print
from ..processing import gpu_ops

# Dependencies check
try:
    from PIL import Image
    from sklearn.cluster import MiniBatchKMeans, KMeans
except ImportError:
    Image = None
    MiniBatchKMeans = None
    debug_print("[WARNING] Pillow or scikit-learn not found. SOG format requires them.")

class SogFormat(BaseFormat):
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading .sog file from {path}")
        if not Image:
             raise ImportError("Pillow is required to read .sog files. Please install it.")
        
        # SOG is a ZIP-bundled format containing WebP texture maps
        if not zipfile.is_zipfile(path):
             raise ValueError("SOG Format: Only ZIP-bundled .sog files are supported.")
             
        with zipfile.ZipFile(path, 'r') as zf:
            # Read meta.json
            with zf.open('meta.json') as f:
                meta = json.load(f)
            
            count = meta['count']
            
            # --- Decode Positions ---
            # means_l.webp, means_u.webp -> u16 -> float -> log_inv
            # Decode Positions from low/high byte WebP images
            
            def read_webp_to_flat(filename, channels=4, expected_count=None):
                if expected_count is None: expected_count = count
                with zf.open(filename) as f:
                    img = Image.open(f)
                    width, height = img.size
                    # Check mode
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    data = np.array(img).flatten()
                    # Data is packed [r, g, b, a]. Indices map to rows in the texture.
                    pixel_count = width * height
                    if pixel_count < expected_count:
                        raise ValueError(f"Image {filename} too small: {pixel_count} < {expected_count}")
                    return data[:expected_count*4] # Flattened
            
            # Positions
            means_l_data = read_webp_to_flat(meta['means']['files'][0])
            means_u_data = read_webp_to_flat(meta['means']['files'][1])
            
            # Reconstruct u16 positions from low/high byte textures
            # channel 0=X, 1=Y, 2=Z
            
            # Reshape to (N, 4)
            means_l = means_l_data.reshape(-1, 4)[:count]
            means_u = means_u_data.reshape(-1, 4)[:count]
            
            qx = means_l[:, 0].astype(np.uint16) | (means_u[:, 0].astype(np.uint16) << 8)
            qy = means_l[:, 1].astype(np.uint16) | (means_u[:, 1].astype(np.uint16) << 8)
            qz = means_l[:, 2].astype(np.uint16) | (means_u[:, 2].astype(np.uint16) << 8)
            
            # Transform normalized u16 to log-scale floats
            mins = meta['means']['mins']
            maxs = meta['means']['maxs']
            
            def inv_log_transform_norm(qv, idx):
                norm = qv / 65535.0
                log_val = norm * (maxs[idx] - mins[idx]) + mins[idx]
                # Inverse log transform logTransform(v) = sign(v) * log(|v|+1)
                return np.sign(log_val) * (np.exp(np.abs(log_val)) - 1.0)

            x = inv_log_transform_norm(qx, 0)
            y = inv_log_transform_norm(qy, 1)
            z = inv_log_transform_norm(qz, 2)
            
            # --- Scales ---
            # Scales are stored as indices into a 256-float codebook.
            
            scales_idx_data = read_webp_to_flat(meta['scales']['files'][0])
            scales_idx = scales_idx_data.reshape(-1, 4)[:count]
            
            scale_codebook = np.array(meta['scales']['codebook'], dtype=np.float32)
            
            s0_idx = scales_idx[:, 0]
            s1_idx = scales_idx[:, 1]
            s2_idx = scales_idx[:, 2]
            
            scale_0 = scale_codebook[s0_idx]
            scale_1 = scale_codebook[s1_idx]
            scale_2 = scale_codebook[s2_idx]
            
            # Unpack rotation components from quats.webp
            quats_data = read_webp_to_flat(meta['quats']['files'][0])
            quats_u8 = quats_data.reshape(-1, 4)[:count]
            
            q_rest = (quats_u8[:, :3].astype(np.float32) / 255.0 - 0.5) * 2.0
            
            # Recover max component index from Alpha
            # quats[...3] = 252 + maxComp
            max_comp_idx = quats_u8[:, 3] - 252
            
            # Reconstruct full quaternion (normalized sum of squares = 1)
            q_missing = np.sqrt(np.maximum(1.0 - np.sum(q_rest**2, axis=1), 0.0))
            
            # Scatter components back to their original slots based on the maximum component index
            # mapping: 0 -> [1,2,3], 1 -> [0,2,3], 2 -> [0,1,3], 3 -> [0,1,2]
            
            rot_0 = np.zeros(count, dtype=np.float32)
            rot_1 = np.zeros(count, dtype=np.float32)
            rot_2 = np.zeros(count, dtype=np.float32)
            rot_3 = np.zeros(count, dtype=np.float32)
            
            # Reconstruct quaternion components using vectorized mask
            for mc in range(4):
                mask = (max_comp_idx == mc)
                if not np.any(mask): continue
                c0 = q_rest[mask, 0]
                c1 = q_rest[mask, 1]
                c2 = q_rest[mask, 2]
                # The missing component is recovered under the assumption it was forced positive during packing
                cm = q_missing[mask]
                
                if mc == 0:
                    rot_0[mask] = cm; rot_1[mask] = c0; rot_2[mask] = c1; rot_3[mask] = c2
                elif mc == 1:
                    rot_0[mask] = c0; rot_1[mask] = cm; rot_2[mask] = c1; rot_3[mask] = c2
                elif mc == 2:
                    rot_0[mask] = c0; rot_1[mask] = c1; rot_2[mask] = cm; rot_3[mask] = c2
                elif mc == 3:
                    rot_0[mask] = c0; rot_1[mask] = c1; rot_2[mask] = c2; rot_3[mask] = cm

            # --- Colors (SH0) ---
            # sh0.webp: R,G,B are indices into codebook. A is Opacity.
            sh0_data = read_webp_to_flat(meta['sh0']['files'][0])
            sh0_raw = sh0_data.reshape(-1, 4)[:count]
            
            sh0_codebook = np.array(meta['sh0']['codebook'], dtype=np.float32)
            
            f_dc_0 = sh0_codebook[sh0_raw[:, 0]]
            f_dc_1 = sh0_codebook[sh0_raw[:, 1]]
            f_dc_2 = sh0_codebook[sh0_raw[:, 2]]
            
            # Convert linear Alpha [0, 255] stored in Alpha channel to Logit Opacity
            linear_alpha_sh = sh0_raw[:, 3].astype(np.float32) / 255.0
            linear_alpha_sh = np.clip(linear_alpha_sh, 1.0/255.0, 0.9999)
            opacity = -np.log((1.0 / linear_alpha_sh) - 1.0)
            
            # Decode SH AC coefficients if present in metadata
            has_sh = 'shN' in meta
            sh_acc_data = {}
            
            if has_sh:
                # Read Palette from shN_centroids.webp
                
                sh_bands = meta['shN']['bands']
                palette_size = meta['shN']['count']
                
                coeffs_per_band = [0, 9, 24, 45][sh_bands]
                coeffs_per_color = coeffs_per_band // 3
                
                # Image dims used in write:
                w_c = 64 * coeffs_per_band
                h_c = int(np.ceil(palette_size / 64))
                centroids_pixel_count = w_c * h_c
                
                centroids_raw = read_webp_to_flat(meta['shN']['files'][0], channels=4, expected_count=centroids_pixel_count) 
                # Raw is (W*H*4).
                
                # Reconstruct Palette (Indices)
                # Map image layout back to palette structure (Palette, 3, CoeffsPerColor)
                centroids_indices_reshaped = np.zeros((palette_size, 3, coeffs_per_color), dtype=np.uint8)
                
                # Image dims used in write:
                w_c = 64 * coeffs_per_band
                # h_c derived from image size usually
                
                # Extract palette indices from interleaved centroids image
                for i in range(palette_size):
                    for j in range(coeffs_per_color):
                        row_img = i // 64
                        col_img = (i % 64) * coeffs_per_color + j
                        idx_flat = (row_img * w_c + col_img) * 4
                          
                        r_idx = centroids_raw[idx_flat + 0]
                        g_idx = centroids_raw[idx_flat + 1]
                        b_idx = centroids_raw[idx_flat + 2]
                        
                        centroids_indices_reshaped[i, 0, j] = r_idx
                        centroids_indices_reshaped[i, 1, j] = g_idx
                        centroids_indices_reshaped[i, 2, j] = b_idx
                          
                # Map indices to Codebook values
                codebook_sh = np.array(meta['shN']['codebook'], dtype=np.float32)
                
                # Palette (shape: Palette, 3, CoeffsPerColor)
                palette = codebook_sh[centroids_indices_reshaped]
                
                # Flatten palette to (Palette, CoeffsPerBand)
                palette_flat = palette.reshape(palette_size, -1)
                
                # Read Labels (Indices into Palette)
                labels_raw = read_webp_to_flat(meta['shN']['files'][1], channels=4)
                labels_raw = labels_raw.reshape(-1, 4)[:count]
                
                # Reconstruct u16
                labels = labels_raw[:, 0].astype(np.uint16) | (labels_raw[:, 1].astype(np.uint16) << 8)
                
                # Map labels to palette to get per-point SH coefficients
                sh_values = palette_flat[labels]
                
                # Assign to corresponding f_rest_i columns
                for idx, col_name in enumerate([f'f_rest_{i}' for i in range(coeffs_per_band)]):
                     sh_acc_data[col_name] = sh_values[:, idx]
                
            # Build Result
            
            # Determine degree for structure definition
            deg_for_struct = 0
            if has_sh and 'shN' in meta and 'bands' in meta['shN']:
                 deg_for_struct = meta['shN']['bands']
                 
            standard_dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=False, sh_degree=deg_for_struct)
            out = np.zeros(count, dtype=standard_dtype)
            out['x'] = x; out['y'] = y; out['z'] = z
            out['scale_0'] = scale_0; out['scale_1'] = scale_1; out['scale_2'] = scale_2
            out['rot_0'] = rot_0; out['rot_1'] = rot_1; out['rot_2'] = rot_2; out['rot_3'] = rot_3
            out['f_dc_0'] = f_dc_0; out['f_dc_1'] = f_dc_1; out['f_dc_2'] = f_dc_2
            out['opacity'] = opacity
            
            # Populate SH AC
            for k, v in sh_acc_data.items():
                if k in out.dtype.names:
                    out[k] = v
            
            return out

    def write(self, data: np.ndarray, path: str, **kwargs):
        
        N = len(data)
        if not Image: # MiniBatchKMeans is replaced by gpu_ops.kmeans
             raise ImportError("Pillow is required to write .sog files.")
             
        debug_print(f"[DEBUG] Writing .sog file to {path}")
        debug_print(f"[DEBUG] Input data dtype: {data.dtype.names}")
        
        N = len(data)
        width = int(np.ceil(np.sqrt(N) / 4) * 4)
        height = int(np.ceil(N / width / 4) * 4)
        
        # Lexsort order for spatial locality (improves chunked clustering and WebP compression)
        # sort by z, y, x
        indices = np.lexsort((data['z'], data['y'], data['x']))
        data_s = data[indices]

        # Prepare ZIP bundle (stored, as WebP provides its own compression)
        zf = zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED)
        
        def write_webp(filename, array_flat, w=width, h=height):
            # array_flat is (W*H*4) uint8
            img = Image.frombytes('RGBA', (w, h), array_flat.tobytes())
            bio = io.BytesIO()
            # Lossless webp, method=1 for speed (default 4 is slow, 6 is max)
            img.save(bio, format='WEBP', lossless=True, quality=100, method=1)
            zf.writestr(filename, bio.getvalue())
        
        # --- Positions ---
        def log_transform(v):
            return np.sign(v) * np.log(np.abs(v) + 1.0)
        
        lx = log_transform(data_s['x'])
        ly = log_transform(data_s['y'])
        lz = log_transform(data_s['z'])
        
        mins = [np.min(lx), np.min(ly), np.min(lz)]
        maxs = [np.max(lx), np.max(ly), np.max(lz)]
        
        # Normalize to 0-65535
        def norm_u16(v, i):
            n = (v - mins[i]) / (maxs[i] - mins[i])
            return np.clip(n * 65535, 0, 65535).astype(np.uint16)
            
        ux = norm_u16(lx, 0)
        uy = norm_u16(ly, 1)
        uz = norm_u16(lz, 2)
        
        # Store positions in low (means_l) and high (means_u) byte textures
        means_l = np.full((height * width, 4), 255, dtype=np.uint8)
        means_u = np.full((height * width, 4), 255, dtype=np.uint8)
        
        # Map sorted splats to texture grid
        means_l[:N, 0] = ux & 0xff
        means_l[:N, 1] = uy & 0xff
        means_l[:N, 2] = uz & 0xff
        
        means_u[:N, 0] = ux >> 8
        means_u[:N, 1] = uy >> 8
        means_u[:N, 2] = uz >> 8
        
        write_webp('means_l.webp', means_l)
        write_webp('means_u.webp', means_u)
        
        # --- Rotations ---
        q = np.column_stack((data_s['rot_0'], data_s['rot_1'], data_s['rot_2'], data_s['rot_3']))
        # Normalize
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        
        # Max component
        max_idx = np.abs(qn).argmax(axis=1)
        # Correct max_idx shape
        max_val = np.take_along_axis(qn, max_idx[:, None], axis=1).flatten()
        
        # Flip sign if max is negative (enforce positive hemisphere)
        sign_flip = np.sign(max_val).reshape(-1, 1)
        qn *= sign_flip
        
        # Scale by sqrt(2)
        qn *= np.sqrt(2.0)
        
        # Pack
        # Slots mapping depends on max_idx
        quats = np.full((height * width, 4), 255, dtype=np.uint8) # Default 255
        
        # Permutation logic
        # 0: 1,2,3
        # 1: 0,2,3
        # 2: 0,1,3
        # 3: 0,1,2
        
        # Vectorized Packing
        
        # We need to extract the 3 non-max components for each Splat.
        # This is tricky because the "missing" index varies per row.
        # But there are only 4 cases (max_idx = 0, 1, 2, 3).
        # We can handle each case with a mask.
        
        c0 = np.zeros(N, dtype=np.uint8)
        c1 = np.zeros(N, dtype=np.uint8)
        c2 = np.zeros(N, dtype=np.uint8)
        
        def quantize_vec(v):
             return np.clip((v * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # Case 0: Max=0 (W), keep 1,2,3 (X,Y,Z)
        m0 = (max_idx == 0)
        if np.any(m0):
             c0[m0] = quantize_vec(qn[m0, 1])
             c1[m0] = quantize_vec(qn[m0, 2])
             c2[m0] = quantize_vec(qn[m0, 3])
             
        # Case 1: Max=1 (X), keep 0,2,3 (W,Y,Z)
        m1 = (max_idx == 1)
        if np.any(m1):
             c0[m1] = quantize_vec(qn[m1, 0])
             c1[m1] = quantize_vec(qn[m1, 2])
             c2[m1] = quantize_vec(qn[m1, 3])
             
        # Case 2: Max=2 (Y), keep 0,1,3 (W,X,Z)
        m2 = (max_idx == 2)
        if np.any(m2):
             c0[m2] = quantize_vec(qn[m2, 0])
             c1[m2] = quantize_vec(qn[m2, 1])
             c2[m2] = quantize_vec(qn[m2, 3])
             
        # Case 3: Max=3 (Z), keep 0,1,2 (W,X,Y)
        m3 = (max_idx == 3)
        if np.any(m3):
             c0[m3] = quantize_vec(qn[m3, 0])
             c1[m3] = quantize_vec(qn[m3, 1])
             c2[m3] = quantize_vec(qn[m3, 2])
             
        quats[:N, 0] = c0
        quats[:N, 1] = c1
        quats[:N, 2] = c2
        quats[:N, 3] = 252 + max_idx.astype(np.uint8)
            
        write_webp('quats.webp', quats)
        
        # --- Scales (Clustered) ---
        # Flatten S0, S1, S2
        s_data = np.concatenate([data_s['scale_0'], data_s['scale_1'], data_s['scale_2']])
        # KMeans 1D
        status_print("Clustering Scales...")
        
        # Subsample for fit
        fit_data = s_data
        if len(s_data) > 50000:
             idx = np.random.choice(len(s_data), 50000, replace=False)
             fit_data = s_data[idx]
             
        c, _ = gpu_ops.kmeans(fit_data.reshape(-1, 1), 256, max_iter=20)
        scale_codebook = sorted(c.flatten())
        
        # Remap labels because KMeans labels are generated in arbitrary order
        
        scale_codebook_arr = np.array(scale_codebook)
        def quantize_to_codebook(vals, cb):
              # Use binary search to find the closest codebook entry
             if len(cb) == 1: return np.zeros_like(vals, dtype=np.uint8)
             idx = np.searchsorted(cb, vals)
             idx = np.clip(idx, 0, len(cb)-1)
             # Check left neighbor
             left = np.maximum(idx - 1, 0)
             d_idx = np.abs(vals - cb[idx])
             d_left = np.abs(vals - cb[left])
             use_left = d_left < d_idx
             idx[use_left] = left[use_left]
             return idx.astype(np.uint8)
             
        l_s0 = quantize_to_codebook(data_s['scale_0'], scale_codebook_arr)
        l_s1 = quantize_to_codebook(data_s['scale_1'], scale_codebook_arr)
        l_s2 = quantize_to_codebook(data_s['scale_2'], scale_codebook_arr)
        
        scales_img = np.zeros((height * width, 4), dtype=np.uint8)
        scales_img[:N, 0] = l_s0
        scales_img[:N, 1] = l_s1
        scales_img[:N, 2] = l_s2
        scales_img[:N, 3] = 255
        
        write_webp('scales.webp', scales_img)
        
        # --- Colors / SH0 (Clustered) ---
        # Cluster flatten F_DC
        dc_data = np.concatenate([data_s['f_dc_0'], data_s['f_dc_1'], data_s['f_dc_2']])
        status_print("Clustering Colors...")
        
        fit_data_c = dc_data
        if len(dc_data) > 50000:
             idx = np.random.choice(len(dc_data), 50000, replace=False)
             fit_data_c = dc_data[idx]
             
        c, _ = gpu_ops.kmeans(fit_data_c.reshape(-1, 1), 256, max_iter=20)
        color_codebook = sorted(c.flatten())
        color_codebook_arr = np.array(color_codebook)
        
        l_dc0 = quantize_to_codebook(data_s['f_dc_0'], color_codebook_arr)
        l_dc1 = quantize_to_codebook(data_s['f_dc_1'], color_codebook_arr)
        l_dc2 = quantize_to_codebook(data_s['f_dc_2'], color_codebook_arr)
        
        # Opacity: Logit -> Sigmoid -> 0-255
        op_sig = 1.0 / (1.0 + np.exp(-data_s['opacity']))
        op_u8 = np.clip(op_sig * 255, 0, 255).astype(np.uint8)
        
        sh0_img = np.zeros((height * width, 4), dtype=np.uint8)
        sh0_img[:N, 0] = l_dc0
        sh0_img[:N, 1] = l_dc1
        sh0_img[:N, 2] = l_dc2
        sh0_img[:N, 3] = op_u8
        
        write_webp('sh0.webp', sh0_img)
        
        # --- SH AC (shN) ---
        # Detect active SH bands
        sh_bands = 0
        if 'f_rest_0' in data.dtype.names:
             # Naive count first
             count_sh = 0
             for i in range(45):
                 if f'f_rest_{i}' in data.dtype.names:
                      count_sh += 1
                      
             if count_sh >= 45: sh_bands = 3
             elif count_sh >= 24: sh_bands = 2
             elif count_sh >= 9: sh_bands = 1
             
             # Smart Detection: Downgrade if content suggests lower degree works
             if sh_bands > 0:
                  last_active_idx = -1
                  max_poss_idx = {3: 44, 2: 23, 1: 8}[sh_bands]
                  
                  # Check content (on sorted data data_s, though order doesn't matter for non-zero check)
                  for i in range(max_poss_idx, -1, -1):
                      fn = f'f_rest_{i}'
                      if fn in data.dtype.names and np.any(data_s[fn] != 0):
                           last_active_idx = i
                           break
                  
                  if last_active_idx >= 24: sh_bands = 3
                  elif last_active_idx >= 9: sh_bands = 2
                  elif last_active_idx >= 0: sh_bands = 1
                  else: sh_bands = 0
                  
             debug_print(f"[DEBUG] SOG Write: Effective SH Bands detected: {sh_bands}")
             
        shN_meta = None
        if sh_bands > 0:
             coeffs_per_band = [0, 9, 24, 45][sh_bands]
             sh_names = [f'f_rest_{i}' for i in range(coeffs_per_band)]
             
             # Extract SH data
             # Shape (N, coeffs_per_band)
             sh_data_flat = np.column_stack([data_s[name] for name in sh_names]).astype(np.float32)
             
             # Determine Palette Size based on Compression Level (Quality)
             # Default to 0 (Max Quality)
             comp_level = kwargs.get('compression_level', 0)
             try: comp_level = int(comp_level)
             except: comp_level = 0
             
             status_print(f"SOG Write Quality Level: {comp_level} (0=Max, 9=Min)")
             
             official_standard_k = min(64, 2 ** int(np.floor(np.log2(N / 1024)))) * 1024
             
             # Adjust target codebook size based on compression level
             
             if comp_level <= 3: target_k = min(65536, official_standard_k)
             elif comp_level <= 6: target_k = min(16384, official_standard_k)
             else: target_k = min(4096, official_standard_k)
             
             target_k = max(256, target_k)
             

             status_print(f"SH Clustering: K={target_k}, Points={N}. Strategy: {'GPU' if gpu_ops.HAS_TAICHI else 'CPU (Fallback)'}")
             
             # Chunked Strategy: Divide and conquer for large K
             num_chunks = max(1, min(64, N // 1024))
             chunk_size = int(np.ceil(N / num_chunks))
             k_per_chunk = max(16, int(np.ceil(target_k / num_chunks)))
             
             all_centroids = []
             all_labels = []
             
             from tqdm import tqdm
             with tqdm(total=num_chunks, desc="Clustering Segments", leave=False, bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}') as pbar_cluster:
                 for i in range(num_chunks):
                     start = i * chunk_size
                     end = min((i + 1) * chunk_size, N)
                     if start >= end: break
                     
                     chunk_data = sh_data_flat[start:end]
                     this_k = min(len(chunk_data), k_per_chunk)
                     
                     c, l = gpu_ops.kmeans(chunk_data, this_k, max_iter=10)
                     
                     l_offset = l + sum([len(ac) for ac in all_centroids])
                     all_centroids.append(c)
                     all_labels.append(l_offset)
                     pbar_cluster.update(1)

             centroids = np.vstack(all_centroids)
             labels = np.concatenate(all_labels)
             palette_size = len(centroids)
             
             # 2. Codebook Clustering (Scalar Quantization of Centroids)
             # Flatten centroids and cluster them into 256 scalars
             centroids_flat = centroids.flatten().reshape(-1, 1)
             
             status_print("Clustering SH Centroids into Codebook...")
             # Always use simple scalar KMeans for codebook (small data)
             kmeans_cb = MiniBatchKMeans(n_clusters=256, n_init='auto').fit(centroids_flat)
             codebook = sorted(kmeans_cb.cluster_centers_.flatten())
             codebook_arr = np.array(codebook)
             
             # Quantize centroids into indices mapping to the codebook
             centroids_indices = quantize_to_codebook(centroids.flatten(), codebook_arr)
             
             # Pack Centroids Image
             w_c = 64 * coeffs_per_band
             h_c = int(np.ceil(palette_size / 64))
             centroids_pixel_count = w_c * h_c
             
             centroids_img = np.full((centroids_pixel_count, 4), 255, dtype=np.uint8)
             
             coeffs_per_color = coeffs_per_band // 3
             centroids_indices_reshaped = centroids_indices.reshape(palette_size, 3, coeffs_per_color)
             
             # map palette to texture image layout
             # map palette to texture image layout
             # Original loop logic: idx_flat = i * coeffs_per_color + j
             # Data in centroids_indices_reshaped is (P, 3, C)
             # We want (P, C, 3) flattened to (P*C, 3) for the image
             
             coeffs_pixel_data = centroids_indices_reshaped.transpose(0, 2, 1).reshape(-1, 3)
             num_valid_pixels = len(coeffs_pixel_data)
             
             # Ensure source and dest shapes match
             centroids_img[:num_valid_pixels, :3] = coeffs_pixel_data
                       
             write_webp('shN_centroids.webp', centroids_img, w_c, h_c)
             
             # Pack Labels Image
             # labels is (N,) uint indices into palette.
             labels_img = np.zeros((height * width, 4), dtype=np.uint8)
             labels_u16 = labels.astype(np.uint16)
             
             labels_img[:N, 0] = labels_u16 & 0xff
             labels_img[:N, 1] = labels_u16 >> 8
             labels_img[:N, 2] = 0
             labels_img[:N, 3] = 255
             
             write_webp('shN_labels.webp', labels_img)
             
             shN_meta = {
                 "count": int(palette_size),
                 "bands": int(sh_bands),
                 "codebook": [float(c) for c in codebook],
                 "files": ["shN_centroids.webp", "shN_labels.webp"]
             }
        
        # Meta JSON
        meta = {
            "version": 2,
            "asset": {"generator": "gsconverter-sog"},
            "count": N,
            "means": {
                "mins": [float(m) for m in mins],
                "maxs": [float(m) for m in maxs],
                "files": ["means_l.webp", "means_u.webp"]
            },
            "scales": {
                "codebook": [float(c) for c in scale_codebook],
                "files": ["scales.webp"]
            },
            "quats": {
                "files": ["quats.webp"]
            },
            "sh0": {
                "codebook": [float(c) for c in color_codebook],
                "files": ["sh0.webp"]
            }
        }
        
        if shN_meta:
             meta["shN"] = shN_meta
             
        zf.writestr('meta.json', json.dumps(meta))
        zf.close()
        status_print(f"SOG write completed to {path}. {N} points bundled.")
