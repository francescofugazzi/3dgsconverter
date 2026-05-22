import numpy as np
from .utils.utility_functions import debug_print

class GaussianStruct:
    @staticmethod
    def sh_coeff_count(sh_degree):
        """Return the total number of f_rest coefficients for a given SH degree."""
        if sh_degree is None or sh_degree < 0:
            return 0
        return 3 * ((sh_degree + 1) ** 2 - 1)

    @staticmethod
    def infer_sh_degree_from_names(field_names):
        """
        Infer the highest SH degree supported by a field name list.
        Accepts both canonical names (f_rest_*) and scalar-prefixed variants.
        """
        max_idx = -1
        if field_names is None:
            iterable = []
        else:
            iterable = field_names

        for name in iterable:
            if name.startswith('f_rest_'):
                try:
                    max_idx = max(max_idx, int(name.split('_')[-1]))
                except ValueError:
                    continue
            elif name.startswith('scalar_f_rest_'):
                try:
                    max_idx = max(max_idx, int(name.split('_')[-1]))
                except ValueError:
                    continue

        if max_idx >= 71:
            return 4
        if max_idx >= 44:
            return 3
        if max_idx >= 23:
            return 2
        if max_idx >= 8:
            return 1
        return 0

    @staticmethod
    def get_standard_order(has_rgb=False, sh_degree=3):
        """
        Returns the standard 3DGS attribute names in strict order.
        """
        order = [
            'x', 'y', 'z', 'nx', 'ny', 'nz',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            *[f'f_rest_{i}' for i in range(GaussianStruct.sh_coeff_count(sh_degree))],
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]
        if has_rgb:
            order.extend(['red', 'green', 'blue'])
        return order

    @staticmethod
    def define_dtype(has_scal=False, has_rgb=False, extra_fields=None, sh_degree=3):
        """
        Defines the structured numpy dtype for 3DGS data, optionally including extra fields.
        """
        debug_print("[DEBUG] Executing 'define_dtype' function...")
        
        prefix = 'scalar_scal_' if has_scal else ''
        
        # Calculate number of rest coefficients.
        # Degree 0: 0
        # Degree 1: 9
        # Degree 2: 24
        # Degree 3: 45
        # Degree 4: 72
        n_coeffs = GaussianStruct.sh_coeff_count(sh_degree)
        
        # Standard attributes
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            (f'{prefix}f_dc_0', 'f4'), (f'{prefix}f_dc_1', 'f4'), (f'{prefix}f_dc_2', 'f4'),
            *[(f'{prefix}f_rest_{i}', 'f4') for i in range(n_coeffs)],
            (f'{prefix}opacity', 'f4'),
            (f'{prefix}scale_0', 'f4'), (f'{prefix}scale_1', 'f4'), (f'{prefix}scale_2', 'f4'),
            (f'{prefix}rot_0', 'f4'), (f'{prefix}rot_1', 'f4'), (f'{prefix}rot_2', 'f4'), (f'{prefix}rot_3', 'f4')
        ]
        
        if has_rgb:
            dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        # Add extra fields (preserving their original types)
        if extra_fields:
            for field_name, field_type in extra_fields:
                # Avoid duplicates
                if not any(d[0] == field_name for d in dtype):
                    dtype.append((field_name, field_type))
        
        return dtype, prefix
