import numpy as np
import struct
import os
import io
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print

class KSplatFormat(BaseFormat):
    MAGIC_MAJOR = 0
    MAGIC_MINOR = 1
    HEADER_SIZE = 4096
    SECTION_HEADER_SIZE = 1024

    def __init__(self):
        super().__init__()
        self.metadata = {}

    @staticmethod
    def _unpack_at(data, fmt, offset):
        return struct.unpack_from(fmt, data, offset)[0]

    @staticmethod
    def _linear_u8_to_logit(val_u8):
        alpha = val_u8.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 1e-7, 1.0 - 1e-7)
        return np.log(alpha / (1.0 - alpha))

    def read(self, path: str) -> np.ndarray:
        debug_print(f"[DEBUG] Reading .ksplat file from {path}")
        with open(path, 'rb') as f:
            header_data = f.read(self.HEADER_SIZE)
            
            # versionMajor: byte 0, versionMinor: byte 1
            v_major = header_data[0]
            v_minor = header_data[1]
            if v_major != self.MAGIC_MAJOR or v_minor != self.MAGIC_MINOR:
                 debug_print(f"[DEBUG] Warning: KSplat version mismatch. Expected {self.MAGIC_MAJOR}.{self.MAGIC_MINOR}, got {v_major}.{v_minor}")
            
            # maxSectionCount: offset 4 (u32), sectionCount: offset 8 (u32)
            max_section_count = self._unpack_at(header_data, 'I', 4)
            section_count = self._unpack_at(header_data, 'I', 8)
            
            # maxSplatCount: offset 12 (u32), splatCount: offset 16 (u32)
            max_splat_count = self._unpack_at(header_data, 'I', 12)
            splat_count = self._unpack_at(header_data, 'I', 16)
            
            # compressionLevel: offset 20 (u16)
            compression_level = self._unpack_at(header_data, 'H', 20)
            
            # minShCoeff: offset 36 (f32), maxShCoeff: offset 40 (f32)
            min_sh = self._unpack_at(header_data, 'f', 36)
            max_sh = self._unpack_at(header_data, 'f', 40)

            debug_print(f"[DEBUG] KSplat: v{v_major}.{v_minor}, Splats={splat_count}, Compression={compression_level}")
            
            self.metadata = {
                'v_major': v_major,
                'v_minor': v_minor,
                'splat_count': splat_count,
                'compression_level': compression_level,
                'min_sh': min_sh,
                'max_sh': max_sh,
                'sections': []
            }

            # Read section headers
            section_headers = []
            for i in range(max_section_count):
                section_data = f.read(self.SECTION_HEADER_SIZE)
                if not section_data: break
                
                s_splat_count = self._unpack_at(section_data, 'I', 0)
                s_max_splat_count = self._unpack_at(section_data, 'I', 4)
                bucket_size = self._unpack_at(section_data, 'I', 8)
                bucket_count = self._unpack_at(section_data, 'I', 12)
                bucket_block_size = self._unpack_at(section_data, 'f', 16)
                bucket_storage_size_bytes = self._unpack_at(section_data, 'H', 20)
                compression_scale_range = self._unpack_at(section_data, 'I', 24)
                if compression_scale_range == 0 and compression_level >= 1:
                    compression_scale_range = 32767
                
                storage_size_bytes = self._unpack_at(section_data, 'I', 28)
                full_bucket_count = self._unpack_at(section_data, 'I', 32)
                partially_filled_bucket_count = self._unpack_at(section_data, 'I', 36)
                sh_degree = self._unpack_at(section_data, 'H', 40)
                
                section_info = {
                    'splatCount': s_splat_count,
                    'maxSplatCount': s_max_splat_count,
                    'bucketSize': bucket_size,
                    'bucketCount': bucket_count,
                    'bucketBlockSize': bucket_block_size,
                    'bucketStorageSizeBytes': bucket_storage_size_bytes,
                    'compressionScaleRange': compression_scale_range,
                    'storageSizeBytes': storage_size_bytes,
                    'fullBucketCount': full_bucket_count,
                    'partiallyFilledBucketCount': partially_filled_bucket_count,
                    'shDegree': sh_degree
                }
                section_headers.append(section_info)
                self.metadata['sections'].append(section_info)

            payload_data = f.read()
            payload_offset = 0
            
            all_splats = []
            
            for s_header in section_headers:
                # 1. Partially filled bucket meta data (u32 array)
                pfb_count = s_header['partiallyFilledBucketCount']
                pfb_lengths = []
                if pfb_count > 0:
                    pfb_lengths = np.frombuffer(payload_data[payload_offset:payload_offset + pfb_count * 4], dtype=np.uint32)
                    payload_offset += pfb_count * 4
                
                # 2. Buckets array (3x f32 per bucket)
                b_count = s_header['bucketCount']
                bucket_centers = []
                if b_count > 0:
                    bucket_centers = np.frombuffer(payload_data[payload_offset:payload_offset + b_count * 12], dtype=np.float32).reshape(-1, 3)
                    payload_offset += b_count * 12
                
                # 3. Splat data
                n_splats = s_header['splatCount']
                sh_deg = s_header['shDegree']
                
                if compression_level == 0:
                    bytes_per_center, bytes_per_scale, bytes_per_rot, bytes_per_color, sh_item_size = 12, 12, 16, 4, 4
                    sh_dtype = np.float32
                else: 
                    bytes_per_center, bytes_per_scale, bytes_per_rot, bytes_per_color = 6, 6, 8, 4
                    if compression_level == 1:
                        sh_item_size, sh_dtype = 2, np.float16
                    else: # Level 2
                        sh_item_size, sh_dtype = 1, np.uint8
                
                sh_count = 0
                if sh_deg == 1: sh_count = 9
                elif sh_deg == 2: sh_count = 24
                
                bytes_per_splat = bytes_per_center + bytes_per_scale + bytes_per_rot + bytes_per_color + (sh_count * sh_item_size)
                
                section_splat_data = payload_data[payload_offset : payload_offset + n_splats * bytes_per_splat]
                payload_offset += s_header['maxSplatCount'] * bytes_per_splat
                
                # Decompress splat data and map to buckets (Level 1+)
                bucket_assignments = []
                if compression_level >= 1:
                    fb_count = s_header['fullBucketCount']
                    b_size = s_header['bucketSize']
                    for b_idx in range(fb_count):
                        bucket_assignments.extend([b_idx] * b_size)
                    for pfb_idx in range(pfb_count):
                        bucket_assignments.extend([fb_count + pfb_idx] * pfb_lengths[pfb_idx])
                    bucket_assignments = bucket_assignments[:n_splats]
                
                # Vectorized Reader - Decoding Logic
                
                # Define dtype based on compression level
                raw_dtype = []
                
                if compression_level == 0:
                     # 12+12+16+4 = 44 bytes base
                     raw_dtype = [
                        ('pos', '3f4'),
                        ('scale', '3f4'),
                        ('rot', '4f4'),
                        ('color', '4u1')
                     ]
                     sh_offset_bytes = 44
                else:
                     # 6+6+8+4 = 24 bytes base
                     raw_dtype = [
                        ('pos', '3u2'),
                        ('scale', '3u2'),
                        ('rot', '4u2'),
                        ('color', '4u1')
                     ]
                     sh_offset_bytes = 24
                
                # Read raw data into structured array
                # Calculate total bytes per splat for verification
                bytes_per_splat_calc = sh_offset_bytes + (sh_count * sh_item_size)
                
                # IMPORTANT: We extract the raw byte arrays for each component.
                # Since SH counts can vary by section (though usually consistent), we handle dynamic dtypes.
                
                if sh_count > 0:
                     if compression_level == 0:
                         raw_dtype.append(('sh', f'{sh_count}f4'))
                     elif compression_level == 1:
                         raw_dtype.append(('sh', f'{sh_count}f2')) # f2 is float16
                     else:
                         raw_dtype.append(('sh', f'{sh_count}u1'))

                raw_splats = np.frombuffer(section_splat_data, dtype=np.dtype(raw_dtype))
                
                # --- Unpack Position ---
                if compression_level == 0:
                    pos = raw_splats['pos']
                else:
                    b_indices = np.array(bucket_assignments, dtype=np.int32)
                    centers = bucket_centers[b_indices]
                    
                    sf = (s_header['bucketBlockSize'] / 2.0) / s_header['compressionScaleRange']
                    sr = s_header['compressionScaleRange']
                    
                    pos_u = raw_splats['pos'].astype(np.float32)
                    pos = (pos_u - sr) * sf + centers
                    
                # --- Unpack Scale ---
                if compression_level == 0:
                    scales = raw_splats['scale']
                else:
                    # Cast uint16 to float16 then float32
                    # Note: KSplat stores log-scales directly.
                    s_u = raw_splats['scale']
                    scales = s_u.view(np.float16).astype(np.float32)
                    
                # --- Unpack Rotation ---
                if compression_level == 0:
                    rots = raw_splats['rot']
                else:
                    r_u = raw_splats['rot'].astype(np.float32)
                    rots = ((r_u - 32767.5) / 32767.5) * 1.41421356
                    
                # --- Unpack Color & Opacity ---
                colors_u8 = raw_splats['color']
                rgba_f = colors_u8.astype(np.float32) / 255.0
                
                # f_dc conversion
                f_dc = (rgba_f[:, :3] - 0.5) / 0.28209479177387814
                opacity_logit = self._linear_u8_to_logit(colors_u8[:, 3])
                
                # --- Construct s_data ---
                s_data = {
                    'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2],
                    'scale_0': scales[:, 0], 'scale_1': scales[:, 1], 'scale_2': scales[:, 2],
                    # Rotation Order: KSplat stores (w, x, y, z)
                    # We map this directly to our internal structure.
                    'nw': rots[:, 0], 'nx': rots[:, 1], 'ny': rots[:, 2], 'nz': rots[:, 3],
                    
                    'f_dc_0': f_dc[:, 0], 'f_dc_1': f_dc[:, 1], 'f_dc_2': f_dc[:, 2],
                    'opacity': opacity_logit
                }
                
                # --- Unpack SH ---
                if sh_count > 0:
                    sh_raw = raw_splats['sh']
                    sh_coeffs = None
                    if compression_level == 0:
                        sh_coeffs = sh_raw
                    elif compression_level == 1:
                        sh_coeffs = sh_raw.astype(np.float32)
                    else:
                        s_u = sh_raw.astype(np.float32)
                        sh_coeffs = (s_u - 128.0) / 128.0
                        
                    for k in range(sh_coeffs.shape[1]):
                        s_data[f'f_rest_{k}'] = sh_coeffs[:, k]
                
                all_splats.append(s_data)
                debug_print(f"[DEBUG] Processed section with {n_splats} splats.")

            # Consolidate sections
            debug_print("[DEBUG] Consolidating KSplat sections...")
            
            # Determine global SH degree for the structure
            # Use the degree from the first section (assuming consistency) or max
            global_sh_degree = 3
            if self.metadata.get('sections'):
                 global_sh_degree = max(s['shDegree'] for s in self.metadata['sections'])
            
            if not all_splats:
                dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=False, sh_degree=global_sh_degree)
                return np.zeros(0, dtype=dtype)
                
            total_splats = sum(len(s['x']) for s in all_splats)
            
            # Use standard dtype from structures.py with correct SH degree
            dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=False, sh_degree=global_sh_degree)
            data = np.zeros(total_splats, dtype=dtype)
            
            current_idx = 0
            for section in all_splats:
                 n = len(section['x'])
                 end = current_idx + n
                 
                 data['x'][current_idx:end] = section['x']
                 data['y'][current_idx:end] = section['y']
                 data['z'][current_idx:end] = section['z']
                 data['scale_0'][current_idx:end] = section['scale_0']
                 data['scale_1'][current_idx:end] = section['scale_1']
                 data['scale_2'][current_idx:end] = section['scale_2']
                 # Note: Vectorized reader constructed 'nw', 'nx', 'ny', 'nz'
                 data['rot_0'][current_idx:end] = section['nw']
                 data['rot_1'][current_idx:end] = section['nx']
                 data['rot_2'][current_idx:end] = section['ny']
                 data['rot_3'][current_idx:end] = section['nz']
                 
                 data['f_dc_0'][current_idx:end] = section['f_dc_0']
                 data['f_dc_1'][current_idx:end] = section['f_dc_1']
                 data['f_dc_2'][current_idx:end] = section['f_dc_2']
                 data['opacity'][current_idx:end] = section['opacity']
                 
                 # SH
                 # Calculate limit based on global degree
                 n_coeffs = 3 * ((global_sh_degree + 1)**2 - 1)
                 for k in range(n_coeffs):
                     key = f'f_rest_{k}'
                     if key in section and key in data.dtype.names:
                         data[key][current_idx:end] = section[key]
                 
                 current_idx += n
                 
            return data

    def write(self, data: np.ndarray, path: str, compression_level: int = 0, **kwargs):
        compression_level = int(compression_level)
        
        # User requested parameters
        requested_sh_level = kwargs.get('sh_level')
        if requested_sh_level is not None:
            requested_sh_level = int(requested_sh_level)
            
        bucket_size = kwargs.get('bucket_size')
        if bucket_size is None: bucket_size = 256
        bucket_size = int(bucket_size)
        
        bucket_block_size = kwargs.get('block_size')
        if bucket_block_size is None: bucket_block_size = 5.0
        bucket_block_size = float(bucket_block_size)

        debug_print(f"[DEBUG] Writing .ksplat file to {path}")
        debug_print(f"[DEBUG] Compression={compression_level}, BucketSize={bucket_size}, BlockSize={bucket_block_size}")
        
        N = len(data)
        
        # Decide SH degree
        sh_degree = 0
        has_sh1 = False
        for j in range(0, 9):
            if f'f_rest_{j}' in data.dtype.names:
                if not np.all(data[f'f_rest_{j}'] == 0):
                    has_sh1 = True
                    break
        if has_sh1:
            sh_degree = 1
            has_sh2 = False
            for j in range(9, 24):
                if f'f_rest_{j}' in data.dtype.names:
                    if not np.all(data[f'f_rest_{j}'] == 0):
                        has_sh2 = True
                        break
            if has_sh2:
                sh_degree = 2
        
        # Cap SH degree (KSplat officially supports up to 2)
        if sh_degree > 2:
            debug_print(f"[DEBUG] Capping SH degree from {sh_degree} to 2 for KSplat")
            sh_degree = 2
            
        # User override for sh_level
        if requested_sh_level is not None:
            if requested_sh_level < sh_degree:
                debug_print(f"[DEBUG] Capping SH degree from {sh_degree} to user-requested {requested_sh_level}")
                sh_degree = requested_sh_level
        
        header = bytearray(self.HEADER_SIZE)
        header[0] = self.MAGIC_MAJOR
        header[1] = self.MAGIC_MINOR
        struct.pack_into('<I', header, 4, 1) # maxSectionCount
        struct.pack_into('<I', header, 8, 1) # sectionCount
        struct.pack_into('<I', header, 12, N) # maxSplatCount
        struct.pack_into('<I', header, 16, N) # splatCount
        struct.pack_into('<H', header, 20, compression_level)
        
        min_sh, max_sh = -2.0, 2.0
        struct.pack_into('<f', header, 36, min_sh)
        struct.pack_into('<f', header, 40, max_sh)

        section_header = bytearray(self.SECTION_HEADER_SIZE)
        struct.pack_into('<I', section_header, 0, N) 
        struct.pack_into('<I', section_header, 4, N) 
        
        compression_scale_range = 32767

        if compression_level >= 1:
            struct.pack_into('<I', section_header, 8, bucket_size)
            struct.pack_into('<I', section_header, 12, (N + bucket_size - 1) // bucket_size)
            struct.pack_into('<f', section_header, 16, bucket_block_size)
            struct.pack_into('<H', section_header, 20, 12) 
            struct.pack_into('<I', section_header, 24, compression_scale_range)

        if compression_level == 0:
            bytes_per_center, bytes_per_scale, bytes_per_rot, bytes_per_color, sh_item_size = 12, 12, 16, 4, 4
        else:
            bytes_per_center, bytes_per_scale, bytes_per_rot, bytes_per_color = 6, 6, 8, 4
            sh_item_size = 2 if compression_level == 1 else 1

        sh_count = 9 if sh_degree == 1 else (24 if sh_degree == 2 else 0)
        bytes_per_splat = bytes_per_center + bytes_per_scale + bytes_per_rot + bytes_per_color + (sh_count * sh_item_size)
        
        full_bucket_count = N // bucket_size
        pfb_count = 1 if (N % bucket_size) != 0 else 0
        bucket_count = full_bucket_count + pfb_count
        
        storage_size = (pfb_count * 4) + (bucket_count * 12 if compression_level >= 1 else 0) + (N * bytes_per_splat)
        struct.pack_into('<I', section_header, 28, storage_size)
        struct.pack_into('<I', section_header, 32, full_bucket_count)
        struct.pack_into('<I', section_header, 36, pfb_count)
        struct.pack_into('<H', section_header, 40, sh_degree)

        payload_parts = []
        if pfb_count > 0:
            payload_parts.append(struct.pack('<I', N % bucket_size))
            
        # Prepare quantized position data (Interleaved XYZ)
        x = data['x']
        y = data['y']
        z = data['z']
        
        bucket_centers_flat = []
        
        if compression_level >= 1:
            # Create indices for reduceat
            indices = np.arange(0, N, bucket_size)
            
            # Calculate min/max for each bucket
            min_x = np.minimum.reduceat(x, indices)
            max_x = np.maximum.reduceat(x, indices)
            min_y = np.minimum.reduceat(y, indices)
            max_y = np.maximum.reduceat(y, indices)
            min_z = np.minimum.reduceat(z, indices)
            max_z = np.maximum.reduceat(z, indices)
            
            # Calculate centers
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            cz = (min_z + max_z) / 2.0
            
            # Interleave into (N_buckets, 3) and flatten to bytes
            centers = np.column_stack((cx, cy, cz)).astype(np.float32)
            payload_parts.append(centers.tobytes())
            
            # Expand centers to splat level for quantization
            expanded_cx = np.repeat(cx, np.diff(np.append(indices, N)))
            expanded_cy = np.repeat(cy, np.diff(np.append(indices, N)))
            expanded_cz = np.repeat(cz, np.diff(np.append(indices, N)))
            
            sf_inv = compression_scale_range / (bucket_block_size / 2.0)
            
            # Quantize positions
            qx = np.clip(np.round((x - expanded_cx) * sf_inv) + compression_scale_range, 0, 65535).astype(np.uint16)
            qy = np.clip(np.round((y - expanded_cy) * sf_inv) + compression_scale_range, 0, 65535).astype(np.uint16)
            qz = np.clip(np.round((z - expanded_cz) * sf_inv) + compression_scale_range, 0, 65535).astype(np.uint16)
            
            pos_data = np.column_stack((qx, qy, qz)).tobytes()
        else:
            pos_data = np.column_stack((x, y, z)).astype(np.float32).tobytes()

        # Prepare scale data
        sx, sy, sz = np.exp(data['scale_0']), np.exp(data['scale_1']), np.exp(data['scale_2'])
        if compression_level == 0:
            scale_data = np.column_stack((sx, sy, sz)).astype(np.float32).tobytes()
        else:
            scale_data = np.column_stack((sx, sy, sz)).astype(np.float16).tobytes()

        # Prepare rotation data
        rw, rx, ry, rz = data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']
        if compression_level == 0:
            rot_data = np.column_stack((rw, rx, ry, rz)).astype(np.float32).tobytes()
        else:
            rot_data = np.column_stack((rw, rx, ry, rz)).astype(np.float16).tobytes()

        # --- Vectorized Color/Opacity Processing ---
        SH_C0 = 0.28209479177387814
        r = np.clip((0.5 + SH_C0 * data['f_dc_0']) * 255, 0, 255).astype(np.uint8)
        g = np.clip((0.5 + SH_C0 * data['f_dc_1']) * 255, 0, 255).astype(np.uint8)
        b_val = np.clip((0.5 + SH_C0 * data['f_dc_2']) * 255, 0, 255).astype(np.uint8)
        a = np.clip((1 / (1 + np.exp(-data['opacity']))) * 255, 0, 255).astype(np.uint8)
        color_data = np.column_stack((r, g, b_val, a)).tobytes()

        # Prepare Spherical Harmonics (SH) data
        sh_data = b''
        if sh_count > 0:
            sh_indices = [f'f_rest_{j}' for j in range(sh_count)]
            sh_matrix = np.stack([data[k] for k in sh_indices], axis=1) # (N, sh_count)
            
            if compression_level == 0:
                sh_data = sh_matrix.astype(np.float32).tobytes()
            elif compression_level == 1:
                sh_data = sh_matrix.astype(np.float16).tobytes()
            else:
                sh_quant = np.clip((sh_matrix - min_sh) / (max_sh - min_sh) * 255, 0, 255).astype(np.uint8)
                sh_data = sh_quant.tobytes()

        # Interleave attributes per splat and build payload
        # Construct a structured array to easily write in interleaved order.
        
        # dtypes for components
        dt_pos = np.float32 if compression_level == 0 else np.uint16
        dt_scale = np.float32 if compression_level == 0 else np.float16
        dt_rot = np.float32 if compression_level == 0 else np.float16
        dt_sh = np.float32 if compression_level == 0 else (np.float16 if compression_level == 1 else np.uint8)
        
        struct_dtype_list = [
            ('pos', dt_pos, (3,)),
            ('scale', dt_scale, (3,)),
            ('rot', dt_rot, (4,)),
            ('color', np.uint8, (4,))
        ]
        if sh_count > 0:
            struct_dtype_list.append(('sh', dt_sh, (sh_count,)))

        interleaved_data = np.zeros(N, dtype=struct_dtype_list)
        
        # Fill structured array
        if compression_level >= 1:
             interleaved_data['pos'] = np.column_stack((qx, qy, qz))
        else:
             interleaved_data['pos'] = np.column_stack((x, y, z))
             
        interleaved_data['scale'] = np.column_stack((sx, sy, sz)).astype(dt_scale)
        interleaved_data['rot'] = np.column_stack((rw, rx, ry, rz)).astype(dt_rot)
        interleaved_data['color'] = np.column_stack((r, g, b_val, a))
        
        if sh_count > 0:
            if compression_level == 2:
                interleaved_data['sh'] = np.clip((sh_matrix - min_sh) / (max_sh - min_sh) * 255, 0, 255).astype(np.uint8)
            else:
                interleaved_data['sh'] = sh_matrix.astype(dt_sh)
        
        # Add interleaved splat data to payload
        payload_parts.append(interleaved_data.tobytes())

        with open(path, 'wb') as f:
            f.write(header)
            f.write(section_header)
            for part in payload_parts:
                f.write(part)
        
        debug_print(f"KSplat (Level {compression_level}) write completed. {N} points.")
