import numpy as np
import pandas as pd
from .base import BaseFormat
from ..structures import GaussianStruct
from ..utils.utility_functions import debug_print, status_print

class ParquetFormat(BaseFormat):
    def read(self, path: str, **kwargs) -> np.ndarray:
        debug_print(f"[DEBUG] Reading Parquet file from {path}")
        df = pd.read_parquet(path)
        
        # Map Parquet column names to internal standard attributes
        
        column_mapping = {
            'x': 'x', 'y': 'y', 'z': 'z',
            'r_sh0': 'f_dc_0', 'g_sh0': 'f_dc_1', 'b_sh0': 'f_dc_2',
            'alpha': 'opacity',
            'cov_s0': 'scale_0', 'cov_s1': 'scale_1', 'cov_s2': 'scale_2',
            'cov_q3': 'rot_0', 'cov_q0': 'rot_1', 'cov_q1': 'rot_2', 'cov_q2': 'rot_3',
        }
        
        # Generate mapping for SH coefficients (r_sh1-15, g_sh1-15, b_sh1-15)
        
        # 15 coeffs per channel
        
        rest_idx = 0
        for channel in ['r', 'g', 'b']:
            for i in range(1, 16):
                col_name = f'{channel}_sh{i}'
                dest_name = f'f_rest_{rest_idx}'
                column_mapping[col_name] = dest_name
                rest_idx += 1
                
        # Ensure normals exist or create them
        for col in ['nx', 'ny', 'nz']:
            if col not in df.columns:
                df[col] = 0.0
                
        # Rename columns to match internal standard
        # Filter columns that are in our mapping or standard
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_renamed = df.rename(columns=rename_dict)
        
        # Prepare standard dtype
        # Check if RGB columns are available
        has_rgb = 'red' in df_renamed.columns
        standard_dtype_list, _ = GaussianStruct.define_dtype(has_scal=False, has_rgb=has_rgb)
        final_dtype = np.dtype(standard_dtype_list)
        converted_data = np.zeros(len(df), dtype=final_dtype)
        
        for name in final_dtype.names:
            if name in df_renamed.columns:
                converted_data[name] = df_renamed[name].values
            else:
                converted_data[name] = 0 # Fill missing with 0
                
        return converted_data

    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        debug_print(f"[DEBUG] Writing Parquet file to {path}")
        
        # 1. Reverse Mapping (Internal -> Parquet)
        reverse_mapping = {
            'x': 'x', 'y': 'y', 'z': 'z',
            'rot_0': 'cov_q3', 'rot_1': 'cov_q0', 'rot_2': 'cov_q1', 'rot_3': 'cov_q2',
            'scale_0': 'cov_s0', 'scale_1': 'cov_s1', 'scale_2': 'cov_s2',
            'opacity': 'alpha',
            'f_dc_0': 'r_sh0', 'f_dc_1': 'g_sh0', 'f_dc_2': 'b_sh0',
            'nx': 'nx', 'ny': 'ny', 'nz': 'nz'
        }
        
        # Map f_rest to channel_shX (Inria order 0-14 R, 15-29 G, 30-44 B)
        # Taichi expects r_sh1...15 etc.
        for i in range(15):
            reverse_mapping[f'f_rest_{i}'] = f'r_sh{i+1}'
            reverse_mapping[f'f_rest_{15+i}'] = f'g_sh{i+1}'
            reverse_mapping[f'f_rest_{30+i}'] = f'b_sh{i+1}'
            
        # 2. Define Strict Column Order (Taichi Standard)
        column_order = ['x', 'y', 'z']
        if 'nx' in data.dtype.names: 
            column_order.extend(['nx', 'ny', 'nz'])
            
        column_order.extend(['cov_q0', 'cov_q1', 'cov_q2', 'cov_q3'])
        column_order.extend(['cov_s0', 'cov_s1', 'cov_s2'])
        column_order.append('alpha')
        
        # SH order: channel by channel
        for channel in ['r', 'g', 'b']:
            for i in range(16):
                column_order.append(f'{channel}_sh{i}')
        
        # 3. Process DataFrame
        df = pd.DataFrame(data)
        
        # Rename available columns
        rename_dict = {k: v for k, v in reverse_mapping.items() if k in df.columns}
        df_renamed = df.rename(columns=rename_dict)
        
        # Filter existing ones and Apply Order
        existing_order = [c for c in column_order if c in df_renamed.columns]
        
        # Add any truly "extra" properties that weren't in mapping
        all_target_names = set(reverse_mapping.values())
        extra_targets = [c for c in df_renamed.columns if c not in all_target_names]
        final_order = existing_order + extra_targets
        
        df_final = df_renamed[final_order]
        
        # Save to parquet
        df_final.to_parquet(path)
        status_print(f"Parquet write completed. {len(df_final)} rows.")
