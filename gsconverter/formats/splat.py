import numpy as np
import struct
import os
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print

class SplatFormat(BaseFormat):
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading .splat file from {path}")
        
        file_size = os.path.getsize(path)
        
        # Standard 32-byte splat format: Pos(12) + Scale(12) + Color(4) + Quad(4)
        splat_size = 32
        
        if file_size % splat_size != 0:
            debug_print(f"[WARNING] File size {file_size} is not a multiple of {splat_size}. Truncation may occur.")
            
        num_splats = file_size // splat_size
        debug_print(f"[DEBUG] Estimated {num_splats} splats based on file size.")
        
        # Define dtype for reading raw binary (32 bytes)
        raw_dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('opacity', 'u1'),
            ('rot_0', 'u1'), ('rot_1', 'u1'), ('rot_2', 'u1'), ('rot_3', 'u1')
        ])
        
        # Read data
        data = np.fromfile(path, dtype=raw_dtype)
        
        # Convert to Internal Standard Structure
        standard_dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=True, sh_degree=0)
        converted_data = np.zeros(num_splats, dtype=standard_dtype)
        
        converted_data['x'] = data['x']
        converted_data['y'] = data['y']
        converted_data['z'] = data['z']
        
        # Convert linear scale to log scale, ensuring stability for non-positive values
        s0 = np.maximum(data['scale_0'], 1e-6)
        s1 = np.maximum(data['scale_1'], 1e-6)
        s2 = np.maximum(data['scale_2'], 1e-6)
        converted_data['scale_0'] = np.log(s0)
        converted_data['scale_1'] = np.log(s1)
        converted_data['scale_2'] = np.log(s2)
        
        # Rotation (Uint8 Quantized -> Float32)
        # Mapping: uint8 = (val * 128 + 128)  =>  val = (uint8 - 128) / 128.0
        r0 = (data['rot_0'].astype(np.float32) - 128) / 128.0
        r1 = (data['rot_1'].astype(np.float32) - 128) / 128.0
        r2 = (data['rot_2'].astype(np.float32) - 128) / 128.0
        r3 = (data['rot_3'].astype(np.float32) - 128) / 128.0
        
        # Re-normalize quaternion
        norms = np.sqrt(r0**2 + r1**2 + r2**2 + r3**2)
        norms = np.maximum(norms, 1e-6) # Avoid div by zero
        converted_data['rot_0'] = r0 / norms
        converted_data['rot_1'] = r1 / norms
        converted_data['rot_2'] = r2 / norms
        converted_data['rot_3'] = r3 / norms
        
        # Color & Opacity
        # Convert linear opacity (0-255) to logit
        linear_alpha = data['opacity'].astype(np.float32) / 255.0
        linear_alpha = np.clip(linear_alpha, 1.0/255.0, 0.9999) # Clip for logit stability
        converted_data['opacity'] = -np.log((1.0 / linear_alpha) - 1.0)
        
        # RGB -> SH DC
        # SH_C0 * DC + 0.5 = RGB [0-1]
        # DC = (RGB - 0.5) / SH_C0
        SH_C0 = 0.28209479177387814
        converted_data['f_dc_0'] = (data['red'].astype(np.float32) / 255.0 - 0.5) / SH_C0
        converted_data['f_dc_1'] = (data['green'].astype(np.float32) / 255.0 - 0.5) / SH_C0
        converted_data['f_dc_2'] = (data['blue'].astype(np.float32) / 255.0 - 0.5) / SH_C0
        
        debug_print(f"[DEBUG] Loaded {num_splats} points from .splat")
        return converted_data

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        debug_print(f"[DEBUG] Writing .splat file to {path}")
        
        # Check for color data or SH DC to compute RGB
        if 'red' not in data.dtype.names and 'f_dc_0' not in data.dtype.names:
            pass
        
        N = len(data)
        
        # Sort splats by volume/opacity metric (matches official standards)
        scale_sum = data['scale_0'] + data['scale_1'] + data['scale_2']
        opacity_term = 1.0 / (1.0 + np.exp(-data['opacity']))
        metric = np.exp(scale_sum) * opacity_term # simplified since 1/(1+exp(-op)) is sigmoid(op) aka linear alpha
        
        # Sort indices descending (largest/most visible first?)
        # antimatter15 sorts by: -np.exp(...) -> Ascending of negative = Descending of positive
        sorted_indices = np.argsort(-metric)
        
        # Reorder data
        data_sorted = data[sorted_indices]
        
        # --- Preparation for Write ---
        
        # Positions (Float32)
        pos_data = np.column_stack((data_sorted['x'], data_sorted['y'], data_sorted['z'])).astype(np.float32).tobytes()
        
        # Scales (Log -> Linear Float32)
        scales = np.exp(np.column_stack((data_sorted['scale_0'], data_sorted['scale_1'], data_sorted['scale_2'])))
        scale_data = scales.astype(np.float32).tobytes()
        
        # Rotations (Float32 -> Uint8 Quantized)
        # Formula: (val * 128 + 128).clip(0, 255)
        # Need to normalize first?
        r0, r1 = data_sorted['rot_0'], data_sorted['rot_1']
        r2, r3 = data_sorted['rot_2'], data_sorted['rot_3']
        # Normalize
        norms = np.sqrt(r0**2 + r1**2 + r2**2 + r3**2)
        r0 /= norms
        r1 /= norms
        r2 /= norms
        r3 /= norms
        
        rot_u8 = np.column_stack((
            np.clip(r0 * 128 + 128, 0, 255),
            np.clip(r1 * 128 + 128, 0, 255),
            np.clip(r2 * 128 + 128, 0, 255),
            np.clip(r3 * 128 + 128, 0, 255)
        )).astype(np.uint8)
        rot_data = rot_u8.tobytes()
        
        # Use pre-computed RGB if available, otherwise compute from SH DC
        SH_C0 = 0.28209479177387814
        if 'f_dc_0' in data_sorted.dtype.names:
            r = np.clip((0.5 + SH_C0 * data_sorted['f_dc_0']) * 255, 0, 255).astype(np.uint8)
            g = np.clip((0.5 + SH_C0 * data_sorted['f_dc_1']) * 255, 0, 255).astype(np.uint8)
            b_val = np.clip((0.5 + SH_C0 * data_sorted['f_dc_2']) * 255, 0, 255).astype(np.uint8)
        else:
            # Fallback (unlikely given GS structure)
            r = data_sorted['red']
            g = data_sorted['green']
            b_val = data_sorted['blue']
            
        # Opacity
        a = np.clip((1.0 / (1.0 + np.exp(-data_sorted['opacity']))) * 255, 0, 255).astype(np.uint8)
        
        color_data = np.column_stack((r, g, b_val, a)).tobytes()
        
        # Construct structured array for optimized packing
        out_dtype = np.dtype([
            ('pos', 'f4', (3,)),
            ('scale', 'f4', (3,)),
            ('color', 'u1', (4,)),
            ('rot', 'u1', (4,))
        ])
        
        out_arr = np.zeros(N, dtype=out_dtype)
        out_arr['pos'] = np.column_stack((data_sorted['x'], data_sorted['y'], data_sorted['z']))
        out_arr['scale'] = scales
        out_arr['color'] = np.column_stack((r, g, b_val, a))
        out_arr['rot'] = rot_u8
        
        with open(path, 'wb') as f:
            f.write(out_arr.tobytes())
            
        debug_print(f".splat write completed. {N} splats sorted and packed.")
