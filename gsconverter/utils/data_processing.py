"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

from .utility_functions import debug_print

def process_data(data_object, bbox=None, apply_density_filter=None, remove_flyers=None):
    # Crop the data based on the bounding box if specified
    if bbox:
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        data_object.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
        debug_print("[DEBUG] Bounding box cropped.")
        
    # Apply density filter if parameters are provided
    if apply_density_filter:
        # Unpack parameters, applying default values if not all parameters are given
        voxel_size, threshold_percentage = (apply_density_filter + [1.0, 0.32])[:2]  # Defaults to 1.0 and 0.32 if not provided
        data_object.apply_density_filter(voxel_size=float(voxel_size), threshold_percentage=float(threshold_percentage))
        debug_print("[DEBUG] Density filter applied.")

    # Remove flyers if parameters are provided
    if remove_flyers:
        # Example: expecting remove_flyers to be a list or tuple like [k, threshold_factor]
        # Provide default values if necessary
        k, threshold_factor = (remove_flyers + [25, 1.0])[:2]  # Defaults to 25 and 1.0 if not provided
        data_object.remove_flyers(k=int(k), threshold_factor=float(threshold_factor))
        debug_print("[DEBUG] Flyers removed.")