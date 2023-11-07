"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
from .format_3dgs import Format3dgs
from .format_cc import FormatCC
from .format_parquet import FormatParquet
from utils.utility_functions import debug_print
from .data_processing import process_data  # Place this import statement at the top with other imports

def convert(data, source_format, target_format, **kwargs):
    debug_print(f"[DEBUG] Starting conversion from {source_format} to {target_format}...")
    
    if source_format == "3dgs":
        converter = Format3dgs(data)
    elif source_format == "cc":
        converter = FormatCC(data)
    elif source_format == "parquet":
        converter = FormatParquet(data)
    else:
        raise ValueError("Unsupported source format")
    
    # Apply optional pre-processing steps using process_data (newly added)
    process_data(converter, bbox=kwargs.get("bbox"), apply_density_filter=kwargs.get("density_filter"), remove_flyers=kwargs.get("remove_flyers"))

    # RGB processing
    if source_format == "cc":
        if kwargs.get("process_rgb", False) and converter.has_rgb():
            print("Error: Source CC file already contains RGB data. Conversion stopped.")
            return None
        debug_print("[DEBUG] Adding or ignoring RGB for CC data...")
        converter.add_or_ignore_rgb(process_rgb=kwargs.get("process_rgb", False))

    # Conversion operations
    process_rgb_flag = kwargs.get("process_rgb", False)
    if source_format == "3dgs" and target_format == "cc":
        debug_print("[DEBUG] Converting 3DGS to CC...")
        return converter.to_cc(process_rgb=process_rgb_flag)
    elif source_format == "cc" and target_format == "3dgs":
        debug_print("[DEBUG] Converting CC to 3DGS...")
        return converter.to_3dgs()
    elif source_format == "parquet" and target_format == "cc":
        debug_print("[DEBUG] Converting Parquet to CC...")
        return converter.to_cc(process_rgb=process_rgb_flag)
    elif source_format == "parquet" and target_format == "3dgs":
        debug_print("[DEBUG] Converting Parquet to 3DGS...")
        return converter.to_3dgs()
    elif source_format == "3dgs" and target_format == "3dgs":
        debug_print("[DEBUG] Applying operations on 3DGS data...")
        if not any(kwargs.values()):  # If no flags are provided
            print("[INFO] No flags provided. The conversion will not happen as the output would be identical to the input.")
            return data['vertex'].data
        else:
            return converter.to_3dgs()
    elif source_format == "cc" and target_format == "cc":
        debug_print("[DEBUG] Applying operations on CC data...")
        converted_data = converter.to_cc()
        if isinstance(converted_data, np.ndarray):
            return converted_data
        else:
            return data['vertex'].data
    else:
        raise ValueError("Unsupported conversion")