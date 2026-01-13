import numpy as np
import struct
from .base import BaseFormat
from ..utils.utility_functions import debug_print

class CompressedPlyFormat(BaseFormat):
    """
    Implementation of the Compressed PLY format as defined in splat-transform.
    Uses chunk-based quantization (256 splats) and bit-packing.
    """
    
    CHUNK_SIZE = 256

    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading Compressed PLY file from {path}")
        from plyfile import PlyData
        plydata = PlyData.read(path)
        
        # Verify it's a compressed PLY (must have chunk element)
        if 'chunk' not in plydata:
            debug_print("[WARNING] File does not contain 'chunk' element, falling back to standard PLY read.")
            from .ply_3dgs import Ply3DGSFormat
            return Ply3DGSFormat().read(path, **kwargs)
            
        chunks = plydata['chunk'].data
        vertices = plydata['vertex'].data
        num_splats = len(vertices)
        
        # Check SH
        sh_data = plydata['sh'].data if 'sh' in plydata else None
        sh_names = []
        if sh_data is not None:
            sh_names = list(sh_data.dtype.names)
            
        # Determine SH Degree for metadata
        # Default to 0 (DC) since standard PLY always has color/DC
        max_sh_deg = 0
        if sh_names:
             n_coeffs = len(sh_names)
             if n_coeffs >= 45: max_sh_deg = 3
             elif n_coeffs >= 24: max_sh_deg = 2
             elif n_coeffs >= 9: max_sh_deg = 1
        
        self.metadata = {
             'count': num_splats,
             'sh_degree': max_sh_deg,
             'chunks': len(chunks)
        }
            
        # Prepare output data
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        if sh_names:
            dtype_list.extend([(n, 'f4') for n in sh_names])
            
        data = np.zeros(num_splats, dtype=np.dtype(dtype_list))
        
        # Unpack
        for i in range(len(chunks)):
            start = i * self.CHUNK_SIZE
            end = min(start + self.CHUNK_SIZE, num_splats)
            if start >= num_splats: break
            
            chunk = chunks[i]
            v_chunk = vertices[start:end]
            
            # Position
            px, py, pz = self._unpack_and_denormalize_11_10_11(
                v_chunk['packed_position'],
                [chunk['min_x'], chunk['min_y'], chunk['min_z']],
                [chunk['max_x'], chunk['max_y'], chunk['max_z']]
            )
            data['x'][start:end] = px
            data['y'][start:end] = py
            data['z'][start:end] = pz
            
            # Rotation
            r0, r1, r2, r3 = self._unpack_quaternions(v_chunk['packed_rotation'])
            data['rot_0'][start:end] = r0
            data['rot_1'][start:end] = r1
            data['rot_2'][start:end] = r2
            data['rot_3'][start:end] = r3
            
            # Scale
            sx, sy, sz = self._unpack_and_denormalize_11_10_11(
                v_chunk['packed_scale'],
                [chunk['min_scale_x'], chunk['min_scale_y'], chunk['min_scale_z']],
                [chunk['max_scale_x'], chunk['max_scale_y'], chunk['max_scale_z']]
            )
            data['scale_0'][start:end] = sx
            data['scale_1'][start:end] = sy
            data['scale_2'][start:end] = sz
            
            # Color & Opacity
            cr, cg, cb, a = self._unpack_and_denormalize_8888(
                v_chunk['packed_color'],
                [chunk['min_r'], chunk['min_g'], chunk['min_b']],
                [chunk['max_r'], chunk['max_g'], chunk['max_b']]
            )
            
            # Convert Color back to f_dc
            SH_C0 = 0.28209479177387814
            data['f_dc_0'][start:end] = (cr - 0.5) / SH_C0
            data['f_dc_1'][start:end] = (cg - 0.5) / SH_C0
            data['f_dc_2'][start:end] = (cb - 0.5) / SH_C0
            
            # Convert Opacity back to Logit (inverse sigmoid)
            a_clamped = np.clip(a, 1e-6, 1.0 - 1e-6)
            data['opacity'][start:end] = np.log(a_clamped / (1.0 - a_clamped))
            
            # SH
            if sh_data is not None:
                v_sh = sh_data[start:end]
                for name in sh_names:
                    # (uchar / 256.0 - 0.5) * 8.0
                    data[name][start:end] = (v_sh[name] / 256.0 - 0.5) * 8.0
                    
        return data

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        debug_print(f"[DEBUG] Writing Compressed PLY file to {path}")
        
        # 1. Morton Sort for spatial locality
        indices = np.arange(len(data), dtype=np.uint32)
        self._sort_morton_order(data, indices)
        sorted_data = data[indices]
        
        num_splats = len(sorted_data)
        num_chunks = (num_splats + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        
        # 2. Prepare Chunks and Vertex attributes
        # Detect actual active SH degree from content
        # We must check sorted_data because that's what we are writing, although checking data would be same result
        
        all_sh_names = [n for n in data.dtype.names if n.startswith('f_rest_')]
        last_active_idx = -1
        target_degree = 0
        
        if all_sh_names:
            # Check backwards from 44 down to 0 to find the last non-zero coefficient
            for i in range(44, -1, -1):
                fname = f'f_rest_{i}'
                if fname in data.dtype.names:
                    # Check if any value is non-zero
                    # Use a small epsilon or exact zero? SH coeffs are floats.
                    # Exact zero is likely fine for "empty" columns, but let's safely assume non-zero check.
                    if np.any(sorted_data[fname] != 0):
                        last_active_idx = i
                        break
            
            # Determine degree
            if last_active_idx >= 24: target_degree = 3
            elif last_active_idx >= 9: target_degree = 2
            elif last_active_idx >= 0: target_degree = 1
            else: target_degree = 0
            
        # Select names corresponding to the target degree
        needed_coeffs = 0
        if target_degree == 3: needed_coeffs = 45
        elif target_degree == 2: needed_coeffs = 24
        elif target_degree == 1: needed_coeffs = 9
        
        sh_names = [f'f_rest_{i}' for i in range(needed_coeffs) if f'f_rest_{i}' in data.dtype.names]
        num_sh = len(sh_names)
        
        debug_print(f"[DEBUG] Compressed PLY SH Detection: Max Index={last_active_idx}, Degree={target_degree}, Coeffs Count={num_sh}")
        
        chunk_data = np.zeros(num_chunks, dtype=[
            ('min_x', 'f4'), ('min_y', 'f4'), ('min_z', 'f4'),
            ('max_x', 'f4'), ('max_y', 'f4'), ('max_z', 'f4'),
            ('min_scale_x', 'f4'), ('min_scale_y', 'f4'), ('min_scale_z', 'f4'),
            ('max_scale_x', 'f4'), ('max_scale_y', 'f4'), ('max_scale_z', 'f4'),
            ('min_r', 'f4'), ('min_g', 'f4'), ('min_b', 'f4'),
            ('max_r', 'f4'), ('max_g', 'f4'), ('max_b', 'f4')
        ])
        
        vertex_data = np.zeros(num_splats, dtype=[
            ('packed_position', 'u4'),
            ('packed_rotation', 'u4'),
            ('packed_scale', 'u4'),
            ('packed_color', 'u4')
        ])
        
        if num_sh > 0:
            sh_dtype = np.dtype([(n, 'u1') for n in sh_names])
            sh_data_arr = np.zeros(num_splats, dtype=sh_dtype)
        
        # Linearize SH DC to Colors (SH_C0 = 0.28209479177387814)
        SH_C0 = 0.28209479177387814
        r = sorted_data['f_dc_0'] * SH_C0 + 0.5
        g = sorted_data['f_dc_1'] * SH_C0 + 0.5
        b = sorted_data['f_dc_2'] * SH_C0 + 0.5
        
        # Opacity (Sigmoid)
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        opacity = sigmoid(sorted_data['opacity'])

        # 3. Process Chunks
        for i in range(num_chunks):
            start = i * self.CHUNK_SIZE
            end = min(start + self.CHUNK_SIZE, num_splats)
            chunk_slice = sorted_data[start:end]
            
            # Bounds calculation
            cx = chunk_slice['x']; cy = chunk_slice['y']; cz = chunk_slice['z']
            sx = np.clip(chunk_slice['scale_0'], -20, 20)
            sy = np.clip(chunk_slice['scale_1'], -20, 20)
            sz = np.clip(chunk_slice['scale_2'], -20, 20)
            cr = r[start:end]; cg = g[start:end]; cb = b[start:end]
            
            c_min_p = [cx.min(), cy.min(), cz.min()]
            c_max_p = [cx.max(), cy.max(), cz.max()]
            c_min_s = [sx.min(), sy.min(), sz.min()]
            c_max_s = [sx.max(), sy.max(), sz.max()]
            c_min_c = [cr.min(), cg.min(), cb.min()]
            c_max_c = [cr.max(), cg.max(), cb.max()]
            
            chunk_data[i] = (
                *c_min_p, *c_max_p,
                *c_min_s, *c_max_s,
                *c_min_c, *c_max_c
            )
            
            # Quantize and Pack Chunk Vertices
            p_pos = self._normalize_and_pack_11_10_11(cx, cy, cz, c_min_p, c_max_p)
            p_rot = self._pack_quaternions(chunk_slice['rot_0'], chunk_slice['rot_1'], chunk_slice['rot_2'], chunk_slice['rot_3'])
            p_scale = self._normalize_and_pack_11_10_11(sx, sy, sz, c_min_s, c_max_s)
            p_color = self._normalize_and_pack_8888(cr, cg, cb, opacity[start:end], c_min_c, c_max_c)
            
            vertex_data['packed_position'][start:end] = p_pos
            vertex_data['packed_rotation'][start:end] = p_rot
            vertex_data['packed_scale'][start:end] = p_scale
            vertex_data['packed_color'][start:end] = p_color
            
            # SH AC Quantization
            if num_sh > 0:
                for name in sh_names:
                    sh_val = (chunk_slice[name] / 8.0 + 0.5)
                    sh_data_arr[name][start:end] = np.clip(sh_val * 256, 0, 255).astype(np.uint8)

        # 4. Write PLY
        self._write_ply_file(path, chunk_data, vertex_data, sh_data_arr if num_sh > 0 else None)
        debug_print(f"Compressed PLY write completed. {num_splats} points in {num_chunks} chunks.")

    def _sort_morton_order(self, data, indices):
        x = data['x']; y = data['y']; z = data['z']
        
        def encode_morton3(ix, iy, iz):
            def part_1_by_2(n):
                n &= 0x000003ff
                n = (n ^ (n << 16)) & 0xff0000ff
                n = (n ^ (n << 8)) & 0x0300f00f
                n = (n ^ (n << 4)) & 0x030c30c3
                n = (n ^ (n << 2)) & 0x09249249
                return n
            return (part_1_by_2(iz) << 2) | (part_1_by_2(iy) << 1) | part_1_by_2(ix)

        def recursive_sort(idxs):
            if len(idxs) <= 1: return
            
            cx = x[idxs]; cy = y[idxs]; cz = z[idxs]
            mx, Mx = cx.min(), cx.max()
            my, My = cy.min(), cy.max()
            mz, Mz = cz.min(), cz.max()
            
            xlen = Mx - mx; ylen = My - my; zlen = Mz - mz
            if xlen == 0 and ylen == 0 and zlen == 0: return
            
            xmul = 1024.0 / xlen if xlen > 0 else 0
            ymul = 1024.0 / ylen if ylen > 0 else 0
            zmul = 1024.0 / zlen if zlen > 0 else 0
            
            ix = (np.clip((cx - mx) * xmul, 0, 1023)).astype(np.uint32)
            iy = (np.clip((cy - my) * ymul, 0, 1023)).astype(np.uint32)
            iz = (np.clip((cz - mz) * zmul, 0, 1023)).astype(np.uint32)
            
            codes = encode_morton3(ix, iy, iz)
            order = np.argsort(codes)
            idxs[:] = idxs[order]
            
            sorted_codes = codes[order]
            diff = np.where(sorted_codes[1:] != sorted_codes[:-1])[0] + 1
            starts = np.insert(diff, 0, 0)
            ends = np.append(diff, len(idxs))
            
            for start, end in zip(starts, ends):
                if end - start > 256:
                    recursive_sort(idxs[start:end])

        recursive_sort(indices)

    def _normalize_and_pack_11_10_11(self, x, y, z, mins, maxs):
        def normalize(v, v_min, v_max, bits):
            if v_max - v_min < 1e-5: return np.zeros_like(v, dtype=np.uint32)
            t = (1 << bits) - 1
            norm = (v - v_min) / (v_max - v_min)
            return np.clip(np.floor(norm * t + 0.5), 0, t).astype(np.uint32)
        nx = normalize(x, mins[0], maxs[0], 11)
        ny = normalize(y, mins[1], maxs[1], 10)
        nz = normalize(z, mins[2], maxs[2], 11)
        return (nx << 21) | (ny << 11) | nz

    def _normalize_and_pack_8888(self, r, g, b, a, mins, maxs):
        def normalize(v, v_min, v_max):
            if v_max - v_min < 1e-5: return np.zeros_like(v, dtype=np.uint32)
            norm = (v - v_min) / (v_max - v_min)
            return np.clip(np.floor(norm * 255 + 0.5), 0, 255).astype(np.uint32)
        nr = normalize(r, mins[0], maxs[0])
        ng = normalize(g, mins[1], maxs[1])
        nb = normalize(b, mins[2], maxs[2])
        na = np.clip(np.floor(a * 255 + 0.5), 0, 255).astype(np.uint32)
        return (nr << 24) | (ng << 16) | (nb << 8) | na

    def _pack_quaternions(self, r0, r1, r2, r3):
        quats = np.stack([r0, r1, r2, r3], axis=-1)
        norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats /= (norm + 1e-10)
        abs_quats = np.abs(quats)
        largest = np.argmax(abs_quats, axis=-1)
        signs = np.sign(quats[np.arange(len(quats)), largest])
        quats *= signs[:, None]
        
        SQRT2_2 = 0.7071067811865476
        def pack_unorm(v, bits):
            t = (1 << bits) - 1
            return np.clip(np.floor((v * SQRT2_2 + 0.5) * t + 0.5), 0, t).astype(np.uint32)

        res = largest.astype(np.uint32)
        for i in range(4):
            is_not_largest = (largest != i)
            packed_comp = pack_unorm(quats[:, i], 10)
            res = np.where(is_not_largest, (res << 10) | packed_comp, res)
        return res

    def _unpack_and_denormalize_11_10_11(self, packed, mins, maxs):
        nx = (packed >> 21) & 0x7FF
        ny = (packed >> 11) & 0x3FF
        nz = packed & 0x7FF
        def denormalize(nv, v_min, v_max, bits):
            t = (1 << bits) - 1
            return (nv / t) * (v_max - v_min) + v_min
        return (denormalize(nx, mins[0], maxs[0], 11), denormalize(ny, mins[1], maxs[1], 10), denormalize(nz, mins[2], maxs[2], 11))

    def _unpack_and_denormalize_8888(self, packed, mins, maxs):
        nr = (packed >> 24) & 0xFF
        ng = (packed >> 16) & 0xFF
        nb = (packed >> 8) & 0xFF
        na = packed & 0xFF
        def denormalize(nv, v_min, v_max):
            return (nv / 255.0) * (v_max - v_min) + v_min
        return (denormalize(nr, mins[0], maxs[0]), denormalize(ng, mins[1], maxs[1]), denormalize(nb, mins[2], maxs[2]), na / 255.0)

    def _unpack_quaternions(self, packed):
        largest = packed >> 30
        v0 = (packed >> 20) & 0x3FF
        v1 = (packed >> 10) & 0x3FF
        v2 = packed & 0x3FF
        SQRT2_2 = 0.7071067811865476
        def denormalize_quat(nv):
            return (nv / 1023.0 - 0.5) / SQRT2_2
        dv0 = denormalize_quat(v0)
        dv1 = denormalize_quat(v1)
        dv2 = denormalize_quat(v2)
        missing = np.sqrt(np.clip(1.0 - (dv0**2 + dv1**2 + dv2**2), 0, 1))
        q = np.zeros((len(packed), 4), dtype=np.float32)
        masks = [largest == i for i in range(4)]
        q[masks[0], 0] = missing[masks[0]]; q[masks[0], 1] = dv0[masks[0]]; q[masks[0], 2] = dv1[masks[0]]; q[masks[0], 3] = dv2[masks[0]]
        q[masks[1], 1] = missing[masks[1]]; q[masks[1], 0] = dv0[masks[1]]; q[masks[1], 2] = dv1[masks[1]]; q[masks[1], 3] = dv2[masks[1]]
        q[masks[2], 2] = missing[masks[2]]; q[masks[2], 0] = dv0[masks[2]]; q[masks[2], 1] = dv1[masks[2]]; q[masks[2], 3] = dv2[masks[2]]
        q[masks[3], 3] = missing[masks[3]]; q[masks[3], 0] = dv0[masks[3]]; q[masks[3], 1] = dv1[masks[3]]; q[masks[3], 2] = dv2[masks[3]]
        return q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    def _write_ply_file(self, path, chunk_data, vertex_data, sh_data):
        from plyfile import PlyData, PlyElement
        elements = [PlyElement.describe(chunk_data, 'chunk'), PlyElement.describe(vertex_data, 'vertex')]
        if sh_data is not None:
            elements.append(PlyElement.describe(sh_data, 'sh'))
        PlyData(elements, text=False, byte_order='<').write(path)
