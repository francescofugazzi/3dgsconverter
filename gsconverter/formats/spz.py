
import numpy as np
import struct
import zlib
import gzip
import io
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print, status_print

class SpzFormat(BaseFormat):
    MAGIC = 0x5053474e
    
    # Constants from official implementation
    COLOR_SCALE = 0.15
    SQRT1_2 = 0.707106781186547524401
    
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading .spz file from {path}")
        
        with open(path, 'rb') as f:
            file_data = f.read()
            
        # Check if file starts with GZIP magic (1f 8b)
        is_gzip = (len(file_data) > 2 and file_data[0] == 0x1f and file_data[1] == 0x8b)
        
        raw_data = file_data
        if is_gzip:
            raw_data = gzip.decompress(file_data)
        
        if len(raw_data) < 16:
             raise ValueError("Decompressed SPZ data too short for header")
             
        header_bytes = raw_data[:16]
        body_data = raw_data[16:]
        
        magic, version, num_points, sh_degree, fractional_bits, flags, reserved = struct.unpack('<IIIBBBB', header_bytes)
        
        if magic != self.MAGIC:
             raise ValueError(f"Invalid SPZ magic number: {hex(magic)}")
        
        if version < 1 or version > 3:
             raise ValueError(f"Unsupported SPZ version: {version}")
             
        debug_print(f"[DEBUG] SPZ Header: Ver={version}, N={num_points}, SH={sh_degree}, Bits={fractional_bits}")

        return self._read_body(body_data, version, num_points, sh_degree, fractional_bits)

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        num_points = len(data)
        sh_degree = 0
        
        # Naive check based on columns
        if 'f_rest_0' in data.dtype.names:
            if 'f_rest_44' in data.dtype.names: sh_degree = 3
            elif 'f_rest_23' in data.dtype.names: sh_degree = 2
            elif 'f_rest_8' in data.dtype.names: sh_degree = 1
            
        # Refine by checking active content (Smart SH Detection)
        if sh_degree > 0:
            last_active_idx = -1
            # Check backwards from max possible index based on naive degree
            max_idx = {3: 44, 2: 23, 1: 8}[sh_degree]
            
            for i in range(max_idx, -1, -1):
                fname = f'f_rest_{i}'
                # Only check if exists (it should, based on naive check)
                if fname in data.dtype.names:
                    if np.any(data[fname] != 0):
                        last_active_idx = i
                        break
            
            # Update degree based on actual content
            if last_active_idx >= 24: sh_degree = 3
            elif last_active_idx >= 9: sh_degree = 2
            elif last_active_idx >= 0: sh_degree = 1
            else: sh_degree = 0
            
        debug_print(f"[DEBUG] SPZ Write: Detected effective SH degree {sh_degree} (from content).")
        
        # Data is stored in the source's coordinate system (standard identity mapping)
        
        # 2. Fractional Bits (Standard 12)
        fractional_bits = 12
        
        # 3. Pack Body
        packed_body = self._pack_v3(data, num_points, sh_degree, fractional_bits)
            
        # 4. Header
        magic = self.MAGIC
        version = 3
        flags = 1 # FlagAntialiased
        reserved = 0
        header = struct.pack('<IIIBBBB', magic, version, num_points, sh_degree, fractional_bits, flags, reserved)
        
        # 5. Compress and Write
        full_payload = header + packed_body
        comp_level = kwargs.get('compression_level', 0)
        compressed = gzip.compress(full_payload, compresslevel=comp_level)
        
        with open(path, 'wb') as f:
            f.write(compressed)
            
        status_print(f"Native SPZ (v3, no-flip, lvl={comp_level}) export completed. {num_points} points.")

    def _pack_v3(self, data, N, sh_deg, frac_bits):
        stream = io.BytesIO()
        
        # 1. Positions (N * 9 bytes) - Interleaved XYZ (3 bytes each)
        scale = (1 << frac_bits)
        # Use Identity (RDF) coordinate system
        coords = np.round(np.column_stack((data['x'] * scale, data['y'] * scale, data['z'] * scale))).astype(np.int32)
        pos_bytes = np.zeros((N, 9), dtype=np.uint8)
        pos_bytes[:, 0:3] = np.column_stack((coords[:,0] & 0xFF, (coords[:,0] >> 8) & 0xFF, (coords[:,0] >> 16) & 0xFF))
        pos_bytes[:, 3:6] = np.column_stack((coords[:,1] & 0xFF, (coords[:,1] >> 8) & 0xFF, (coords[:,1] >> 16) & 0xFF))
        pos_bytes[:, 6:9] = np.column_stack((coords[:,2] & 0xFF, (coords[:,2] >> 8) & 0xFF, (coords[:,2] >> 16) & 0xFF))
        stream.write(pos_bytes.tobytes())
        
        # 2. Alpha (N bytes) - Sigmoid applied
        if 'opacity' in data.dtype.names:
             # Alpha in SPZ is sigmoid(alpha_ply) * 255
             alpha = (1.0 / (1.0 + np.exp(-np.clip(data['opacity'], -20, 20))) * 255.0).astype(np.uint8)
        else:
             alpha = np.full(N, 255, dtype=np.uint8)
        stream.write(alpha.tobytes())
        
        # 3. Colors (N * 3 bytes) - Interleaved RGB (SH DC)
        # Official: Enc = (SH_DC * 0.15 + 0.5) * 255
        if 'f_dc_0' in data.dtype.names:
            r = np.clip((data['f_dc_0'] * self.COLOR_SCALE + 0.5) * 255.0, 0, 255).astype(np.uint8)
            g = np.clip((data['f_dc_1'] * self.COLOR_SCALE + 0.5) * 255.0, 0, 255).astype(np.uint8)
            b = np.clip((data['f_dc_2'] * self.COLOR_SCALE + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            r = g = b = np.full(N, 128, dtype=np.uint8)
        stream.write(np.column_stack((r, g, b)).astype(np.uint8).tobytes())
        
        # 4. Scales (N * 3 bytes) - Interleaved XYZ
        sx = np.clip((data['scale_0'] + 10.0) * 16.0, 0, 255).astype(np.uint8)
        sy = np.clip((data['scale_1'] + 10.0) * 16.0, 0, 255).astype(np.uint8)
        sz = np.clip((data['scale_2'] + 10.0) * 16.0, 0, 255).astype(np.uint8)
        stream.write(np.column_stack((sx, sy, sz)).astype(np.uint8).tobytes())
        
        # 5. Rotations (N * 4 bytes) - Packed uint32
        # Pack rotation using identity (no flips) to match official behavior
        packed_rot = self._pack_rot_v3(data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3'], N)
        stream.write(packed_rot.tobytes())
        
        # 6. SH (N * shDim * 3 bytes) - Interleaved RGB in binary
        sh_dim = self._dim_for_degree(sh_deg)
        if sh_dim > 0:
            # Assume planar grouping in source (R0..R14, G0..G14, B0..B14)
            sh_r = [data[f'f_rest_{i}'] for i in range(sh_dim)]
            sh_g = [data[f'f_rest_{i+15}'] for i in range(sh_dim)]
            sh_b = [data[f'f_rest_{i+30}'] for i in range(sh_dim)]
            
            # Interleave them for binary: R0, G0, B0, R1, G1, B1...
            interleaved_list = []
            for i in range(sh_dim):
                interleaved_list.extend([sh_r[i], sh_g[i], sh_b[i]])
            sh_pts = np.column_stack(interleaved_list)
            
            def quant_sh(val, bits):
                bs = 1 << (8 - bits)
                q = np.round(val * 128.0 + 128.0).astype(np.int32)
                return np.clip((q + bs // 2) // bs * bs, 0, 255).astype(np.uint8)
            
            q_sh = np.zeros_like(sh_pts, dtype=np.uint8)
            q_sh[:, :9] = quant_sh(sh_pts[:, :9], 5)
            if sh_dim > 3:
                q_sh[:, 9:] = quant_sh(sh_pts[:, 9:], 4)
            stream.write(q_sh.tobytes())
        
        return stream.getvalue()

    def _read_body(self, raw, version, N, sh_deg, frac_bits):
        ptr = 0
        # Respect the file's SH degree when creating the struct
        dtype_list, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=True, sh_degree=sh_deg)
        out = np.zeros(N, dtype=dtype_list)
        
        # 1. Positions
        if version == 1:
            # Float16 positions (6 bytes per point)
            sz_pos = N * 3 * 2
            pos_raw = np.frombuffer(raw, dtype=np.float16, count=N*3, offset=ptr).reshape(N, 3).astype(np.float32)
            ptr += sz_pos
            out['x'], out['y'], out['z'] = pos_raw[:,0], pos_raw[:,1], pos_raw[:,2]
        else:
            # 24-bit fixed point (9 bytes per point)
            sz_pos = N * 3 * 3
            pos_raw = np.frombuffer(raw, dtype=np.uint8, count=sz_pos, offset=ptr).reshape(N, 3, 3)
            ptr += sz_pos
            b0, b1, b2 = pos_raw[:,:,0].astype(np.int32), pos_raw[:,:,1].astype(np.int32), pos_raw[:,:,2].astype(np.int32)
            i32 = b0 | (b1 << 8) | (b2 << 16)
            i32[(i32 & 0x800000) != 0] |= -16777216
            val = i32.astype(np.float32) / (1 << frac_bits)
            out['x'], out['y'], out['z'] = val[:,0], val[:,1], val[:,2]
        
        # 2. Alpha (always 1 byte)
        out['opacity'] = self._linear_u8_to_logit(np.frombuffer(raw, dtype=np.uint8, count=N, offset=ptr))
        ptr += N
        
        # 3. Colors (always 3 bytes)
        sz_col = N * 3
        col_data = np.frombuffer(raw, dtype=np.uint8, count=sz_col, offset=ptr).reshape(N, 3)
        ptr += sz_col
        dc_r = (col_data[:,0].astype(np.float32) / 255.0 - 0.5) / self.COLOR_SCALE
        dc_g = (col_data[:,1].astype(np.float32) / 255.0 - 0.5) / self.COLOR_SCALE
        dc_b = (col_data[:,2].astype(np.float32) / 255.0 - 0.5) / self.COLOR_SCALE
        out['f_dc_0'], out['f_dc_1'], out['f_dc_2'] = dc_r, dc_g, dc_b
        
        # Helper RGB
        SH_C0 = 0.28209479177387814
        out['red'] = np.clip((0.5 + SH_C0 * dc_r) * 255.0, 0, 255).astype(np.uint8)
        out['green'] = np.clip((0.5 + SH_C0 * dc_g) * 255.0, 0, 255).astype(np.uint8)
        out['blue'] = np.clip((0.5 + SH_C0 * dc_b) * 255.0, 0, 255).astype(np.uint8)
        
        # 4. Scales (always 3 bytes)
        sz_s = N * 3
        s_data = np.frombuffer(raw, dtype=np.uint8, count=sz_s, offset=ptr).reshape(N, 3)
        ptr += sz_s
        out['scale_0'], out['scale_1'], out['scale_2'] = s_data[:,0]/16.0-10.0, s_data[:,1]/16.0-10.0, s_data[:,2]/16.0-10.0
        
        # 5. Rotations
        if version >= 3:
            # 4-byte packed uint32 (Smallest Three)
            rot_packed = np.frombuffer(raw, dtype=np.uint32, count=N, offset=ptr)
            ptr += N * 4
            out['rot_0'], out['rot_1'], out['rot_2'], out['rot_3'] = self._unpack_rot_v3(rot_packed, N)
        else:
            # 3-byte legacy (First Three: X, Y, Z)
            sz_rot = N * 3
            rot_raw = np.frombuffer(raw, dtype=np.uint8, count=sz_rot, offset=ptr).reshape(N, 3)
            ptr += sz_rot
            out['rot_0'], out['rot_1'], out['rot_2'], out['rot_3'] = self._unpack_rot_legacy(rot_raw, N)
        
        # 6. SH AC
        sh_dim = self._dim_for_degree(sh_deg)
        if sh_dim > 0:
            sz_sh = N * sh_dim * 3
            sh_raw = np.frombuffer(raw, dtype=np.uint8, count=sz_sh, offset=ptr).reshape(N, sh_dim, 3)
            # sh_raw is interleaved: [N, shDim, RGB]
            sh_unq = (sh_raw.astype(np.float32) - 128.0) / 128.0
            
            # De-interleave back to Grouped for PLY: R0..Rk, G0..Gk, B0..Bk
            # The offsets depend on the number of coefficients per channel (sh_dim)
            for j in range(sh_dim):
                out[f'f_rest_{j}'] = sh_unq[:, j, 0]             # R_j
                out[f'f_rest_{j+sh_dim}'] = sh_unq[:, j, 1]      # G_j
                out[f'f_rest_{j+2*sh_dim}'] = sh_unq[:, j, 2]    # B_j
        return out

    def _unpack_rot_legacy(self, raw, N):
        """Unpacks 'First Three' quaternions (X, Y, Z) for version < 3."""
        # raw is [N, 3] uint8
        xyz = raw.astype(np.float32) / 127.5 - 1.0
        # Compute W
        s2 = np.sum(xyz**2, axis=1)
        w = np.sqrt(np.maximum(0.0, 1.0 - s2))
        # SPZ Result indices: 0:X, 1:Y, 2:Z, 3:W
        # Standard 3DGS/PLY order: [W, X, Y, Z]
        return w, xyz[:,0], xyz[:,1], xyz[:,2]

    def _dim_for_degree(self, degree):
        return {0: 0, 1: 3, 2: 8, 3: 15}.get(degree, 0)

    def _unpack_rot_v3(self, packed, N):
        """Unpacks 'Smallest Three' quaternions with XYZW ordering. m_idx in bits 30..31."""
        idx = (packed >> 30) & 0x3
        v0raw = (packed >> 20) & 0x3FF # Slot 0: Lowest index
        v1raw = (packed >> 10) & 0x3FF # Slot 1: Mid index
        v2raw = packed & 0x3FF         # Slot 2: Highest index
        
        def unq(c):
            mag = c & 0x1FF
            neg = (c >> 9) & 0x1
            return (mag.astype(np.float32) / 511.0) * self.SQRT1_2 * (1.0 - 2.0 * neg)
        
        v0, v1, v2 = unq(v0raw), unq(v1raw), unq(v2raw)
        s2 = v0**2 + v1**2 + v2**2
        mVal = np.sqrt(np.maximum(0.0, 1.0 - s2))
        
        rW, rX, rY, rZ = np.zeros(N, 'f4'), np.zeros(N, 'f4'), np.zeros(N, 'f4'), np.zeros(N, 'f4')
        for i in range(4):
            m = (idx == i)
            if not np.any(m): continue
            # Slots are indices NOT in [i]
            # If i=0: slots are 1,2,3 -> v0=q1, v1=q2, v2=q3
            # If i=3: slots are 0,1,2 -> v0=q0, v1=q1, v2=q2
            others = [j for j in range(4) if j != i]
            comps = [rX, rY, rZ, rW]
            comps[i][m] = mVal[m]
            comps[others[0]][m] = v0[m]
            comps[others[1]][m] = v1[m]
            comps[others[2]][m] = v2[m]
        return rW, rX, rY, rZ

    def _pack_rot_v3(self, rW, rX, rY, rZ, N):
        """Packs quaternions using 'Smallest Three' with XYZW order. m_idx in bits 0..1."""
        norm = np.sqrt(rW*rW + rX*rX + rY*rY + rZ*rZ + 1e-9)
        # SPZ Result indices: 0:X, 1:Y, 2:Z, 3:W
        R = np.column_stack((rX/norm, rY/norm, rZ/norm, rW/norm))
        max_idx = np.argmax(np.abs(R), axis=1)
        
        packed = (max_idx.astype(np.uint32) << 30)
        should_neg = R[np.arange(N), max_idx] < 0
        scale = 511.0 / self.SQRT1_2
        
        # Map components to 10-bit slots
        # Packing logic: if max_idx=0: 1->slot0(20-29), 2->slot1(10-19), 3->slot2(0-9)
        # General: iterate i=0..3. if i != iLargest, push into next slot.
        # This means:
        # slot 0: lowest index (that's not max)
        # slot 1: middle index (that's not max)
        # slot 2: highest index (that's not max)
        
        # Loop logic pushes components into 10-bit slots
        # This pushes earlier components to HIGHER bits.
        # Lowest index component (first i in loop) ends up in bits 20-29.
        # Highest index component (last i in loop) ends up in bits 0-9.
        
        slots = np.zeros((N, 4), dtype=np.int32) - 1
        for i in range(N):
            curr = 0
            for j in range(4):
                if max_idx[i] != j:
                    slots[i, j] = curr # 0, 1, or 2
                    curr += 1
        
        for j in range(4):
            m = (max_idx != j)
            if not np.any(m): continue
            val = R[m, j]
            # Negate components if max component is negative to enforce positive hemisphere
            negbit = (np.not_equal(val < 0, should_neg[m])).astype(np.uint32)
            mag = np.clip((np.abs(val) * scale + 0.5), 0, 511).astype(np.uint32)
            
            # Bits: [idx: 30-31] [slot0: 20-29] [slot1: 10-19] [slot2: 0-9]
            slot = slots[m, j].astype(np.uint32)
            shift = (2 - slot) * 10 
            component = ((negbit << 9) | mag).astype(np.uint32)
            packed[m] |= (component << shift)
        return packed

    def _linear_u8_to_logit(self, u8):
        v = u8.astype(np.float32) / 255.0
        v = np.clip(v, 1e-7, 1.0 - 1e-7)
        return np.log(v / (1.0 - v))
