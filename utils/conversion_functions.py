import numpy as np
from .format_3dgs import Format3dgs
from .format_cc import FormatCC
from utils.utility_functions import debug_print

def convert(data, source_format, target_format, **kwargs):
    debug_print(f"[DEBUG] Starting conversion from {source_format} to {target_format}...")
    
    if source_format == "3dgs":
        converter = Format3dgs(data)
    elif source_format == "cc":
        converter = FormatCC(data)
    else:
        raise ValueError("Unsupported source format")

    # Apply optional operations
    if kwargs.get("bbox"):
        min_x, min_y, min_z, max_x, max_y, max_z = kwargs.get("bbox")
        print("Cropping by bounding box...")
        converter.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
    if kwargs.get("density_filter"):
        # Unpack the density filter values
        voxel_size, threshold_percentage = kwargs.get("density_filter")
        print(f"Applying density filter with voxel size: {voxel_size} and threshold percentage: {threshold_percentage}...")
        converter.apply_density_filter(voxel_size=voxel_size, threshold_percentage=threshold_percentage)
    if kwargs.get("remove_flyers"):
        # Unpack the remove_flyers values
        k, threshold_factor = kwargs.get("remove_flyers")
        print(f"Removing flyers with k: {k} and threshold factor: {threshold_factor}...")
        converter.remove_flyers(k=k, threshold_factor=threshold_factor)

    # RGB processing
    if source_format == "3dgs" and target_format == "cc":
        if kwargs.get("process_rgb", False):
            debug_print("[DEBUG] Computing RGB for 3DGS data...")
            # No need to explicitly call a function here, as the RGB computation is part of the to_cc() method.
        else:
            debug_print("[DEBUG] Ignoring RGB for 3DGS data...")
            converter.ignore_rgb()
    elif source_format == "cc":
        if kwargs.get("process_rgb", False) and converter.has_rgb():
            print("Error: Source CC file already contains RGB data. Conversion stopped.")
            return None
        debug_print("[DEBUG] Adding or ignoring RGB for CC data...")
        converter.add_or_ignore_rgb(process_rgb=kwargs.get("process_rgb", False))
    elif source_format == "3dgs" and target_format == "3dgs":
        debug_print("[DEBUG] Ignoring RGB for 3DGS to 3DGS conversion...")
        converter.ignore_rgb()

    # Conversion operations
    process_rgb_flag = kwargs.get("process_rgb", False)
    if source_format == "3dgs" and target_format == "cc":
        debug_print("[DEBUG] Converting 3DGS to CC...")
        return converter.to_cc(process_rgb=process_rgb_flag)
    elif source_format == "cc" and target_format == "3dgs":
        debug_print("[DEBUG] Converting CC to 3DGS...")
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