import numpy as np
from plyfile import PlyData, PlyElement
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print

class PlyCCFormat(BaseFormat):
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading CC PLY file from {path}") # The original line was kept as the requested change would cause a NameError.
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
        elif 'scalar_scal_f_dc_0' in source_names:
            source_prefix = "scalar_scal_"
            
        # 2. Identify Extra Fields
        std_base_names = GaussianStruct.get_standard_order(has_rgb=True)
        std_base_names.extend(['nx', 'ny', 'nz'])
        
        std_source_names = {source_prefix + name for name in std_base_names} | set(std_base_names)
        extra_fields = []
        for name in source_names:
            if name not in std_source_names:
                # Strip 'scalar_' prefix from extra properties to normalize for internal use
                internal_name = name
                if name.startswith('scalar_'):
                    internal_name = name[7:]
                extra_fields.append((internal_name, vertices.dtype[name].str))
        
        # 3. Define Internal Dtype
        has_rgb = 'red' in source_names
        internal_dtype, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=has_rgb, extra_fields=extra_fields)
        converted_data = np.zeros(len(vertices), dtype=internal_dtype)
        
        # 4. Map Fields
        for name_entry in internal_dtype:
            target_name = name_entry[0]
            
            # Try Direct Match
            if target_name in source_names:
                converted_data[target_name] = vertices[target_name]
            else:
                # Try Prefixed (e.g. f_dc_0 -> scalar_f_dc_0 or extra_prop -> scalar_extra_prop)
                source_name = source_prefix + target_name
                if source_name in source_names:
                    converted_data[target_name] = vertices[source_name]
                elif f"scalar_{target_name}" in source_names:
                    converted_data[target_name] = vertices[f"scalar_{target_name}"]
        
        return converted_data

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        debug_print(f"[DEBUG] Writing CC PLY file to {path}")
        
        has_rgb = 'red' in data.dtype.names
        std_order = GaussianStruct.get_standard_order(has_rgb=has_rgb)
        crop_sh = kwargs.get('crop_sh', False)
        if crop_sh:
            last_idx = -1
            for i in range(44, -1, -1):
                f_name = f'f_rest_{i}'
                if f_name in data.dtype.names and np.any(data[f_name]):
                    last_idx = i
                    break
            std_order = [n for n in std_order if not (n.startswith('f_rest_') and int(n.split('_')[-1]) > last_idx)]
        
        actual_fields = data.dtype.names
        output_dtype_list = []
        mapping = {}
        
        # 1. Standard Fields in Order
        # Spatial/RGB fields DON'T get scalar_ prefix.
        # SH, Opacity, Scale, Rot DO get scalar_ prefix.
        spatial_fields = {'x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue'}
        
        for name in std_order:
            if name in actual_fields:
                out_name = name if name in spatial_fields else f"scalar_{name}"
                output_dtype_list.append((out_name, data.dtype[name].str))
                mapping[name] = out_name
            else:
                # Padding
                if name.startswith('f_rest_'):
                    if not crop_sh:
                        out_name = f"scalar_{name}"
                        output_dtype_list.append((out_name, 'f4'))
                elif name in ['nx', 'ny', 'nz']:
                    output_dtype_list.append((name, 'f4'))
        
        # 2. Extra Fields at the end (get scalar_ prefix block)
        # Exclude standard names (even if not in std_order due to cropping)
        full_std_names = set(GaussianStruct.get_standard_order(has_rgb=True)) | {'nx', 'ny', 'nz'}
        for name in actual_fields:
            if name not in std_order and name not in full_std_names:
                out_name = f"scalar_{name}"
                output_dtype_list.append((out_name, data.dtype[name].str))
                mapping[name] = out_name
                
        output_dtype = np.dtype(output_dtype_list)
        output_data = np.zeros(len(data), dtype=output_dtype)
        
        # Copy values
        for orig_name, out_name in mapping.items():
            output_data[out_name] = data[orig_name]
            
        el = PlyElement.describe(output_data, 'vertex')
        
        # Add extra elements if provided
        elements_to_write = [el]
        extra_elements = kwargs.get('extra_elements', [])
        if extra_elements:
             elements_to_write.extend(extra_elements)
             # debug_print because ply_cc typically uses debug_print, keep consistent usage or use status_print if preferred? 
             # Let's use debug_print as per file convention, or standard print if we want visibility. 
             # Given ply_3dgs used status_print, let's stick to debug_print here as CC format is secondary/specialized.
             # Actually, user wants to know it "maintains". 
             print(f"Maintained {len(extra_elements)} extra PLY elements.")

        PlyData(elements_to_write, byte_order='<').write(path)
        debug_print(f"CloudCompare PLY write completed. {len(data)} points.")
