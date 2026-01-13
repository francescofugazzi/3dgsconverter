import numpy as np
from plyfile import PlyData, PlyElement
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print, status_print

class Ply3DGSFormat(BaseFormat):
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading 3DGS PLY file from {path}")
        plydata = PlyData.read(path)
        
        if 'vertex' not in plydata:
            raise ValueError("PLY file does not contain 'vertex' element")
            
        # Store non-vertex elements (e.g., extrinsic, intrinsic, cameras)
        self.extra_elements = [el for el in plydata.elements if el.name != 'vertex']
            
        vertices = plydata['vertex'].data
        source_names = vertices.dtype.names
        
        # 1. Identify source prefix
        source_prefix = ""
        if 'scalar_f_dc_0' in source_names:
            source_prefix = "scalar_"
            if 'scalar_scal_f_dc_0' in source_names:
                 source_prefix = "scalar_scal_"
        elif 'scal_f_dc_0' in source_names:
            source_prefix = "scal_"
            
        # 2. Identify Extra Fields
        # Standard fields (potential names in file)
        std_base_names = GaussianStruct.get_standard_order(has_rgb=True)
        # std_base_names includes nx, ny, nz
        
        std_source_names = {source_prefix + name for name in std_base_names} | set(std_base_names)
        extra_fields = []
        for name in source_names:
            if name not in std_source_names:
                # Keep original type
                extra_fields.append((name, vertices.dtype[name].str))
        
        # 3. Define Internal Dtype (normalized, no prefix)
        has_rgb = 'red' in source_names
        internal_dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=has_rgb, extra_fields=extra_fields)
        converted_data = np.zeros(len(vertices), dtype=internal_dtype)
        
        # 4. Map Fields
        for name_entry in internal_dtype:
            target_name = name_entry[0]
            
            # Try normalized name first (x, y, z, nx, ny, nz, red, etc.)
            if target_name in source_names:
                converted_data[target_name] = vertices[target_name]
            else:
                # Try prefixed name (for SH, etc.)
                source_name = source_prefix + target_name
                if source_name in source_names:
                    converted_data[target_name] = vertices[source_name]
                    
        return converted_data

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        debug_print(f"[DEBUG] Writing 3DGS PLY file to {path}")
        
        has_rgb = 'red' in data.dtype.names
        std_order = GaussianStruct.get_standard_order(has_rgb=has_rgb)
        
        # Determine if SH cropping is required (Degree < 3)
        crop_sh = kwargs.get('crop_sh', False)
        if crop_sh:
            last_idx = -1
            for i in range(44, -1, -1):
                f_name = f'f_rest_{i}'
                if f_name in data.dtype.names and np.any(data[f_name] != 0):
                    last_idx = i
                    break
            debug_print(f"[DEBUG] crop_sh=True. Detected last active SH index: {last_idx}")
            std_order = [n for n in std_order if not (n.startswith('f_rest_') and int(n.split('_')[-1]) > last_idx)]
        
        actual_fields = data.dtype.names
        output_dtype_list = []
        
        # 1. Standard Fields in Order
        for name in std_order:
            if name in actual_fields:
                output_dtype_list.append((name, data.dtype[name].str))
            else:
                # Pad missing standard fields with 0s unless cropped
                if name.startswith('f_rest_'):
                    if not crop_sh:
                        output_dtype_list.append((name, 'f4'))
                elif name in ['nx', 'ny', 'nz']:
                    output_dtype_list.append((name, 'f4'))
                # Required attributes (positions, DC, etc.) are padded with 0 if missing
        
        # 2. Extra Fields at the end (truly unknown ones)
        # Exclude standard names to avoid re-adding cropped attributes as extras
        full_std_names = set(GaussianStruct.get_standard_order(has_rgb=True)) | {'nx', 'ny', 'nz'}
        for name in actual_fields:
            if name not in std_order and name not in full_std_names:
                output_dtype_list.append((name, data.dtype[name].str))
                
        output_dtype = np.dtype(output_dtype_list)
        output_data = np.zeros(len(data), dtype=output_dtype)
        
        # Copy values
        for name in output_dtype.names:
            if name in data.dtype.names:
                output_data[name] = data[name]
            
        el = PlyElement.describe(output_data, 'vertex')
        
        # Add extra elements if provided
        elements_to_write = [el]
        extra_elements = kwargs.get('extra_elements', [])
        if extra_elements:
             elements_to_write.extend(extra_elements)
             status_print(f"Maintained {len(extra_elements)} extra PLY elements.")

        PlyData(elements_to_write, byte_order='<').write(path)
        status_print(f"3DGS PLY write completed. {len(data)} points.")
