"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

from .utility_functions import debug_print

def process_data(data_object, bbox=None, apply_density_filter=False, remove_flyers=False):
    # Crop the data based on the bounding box if specified
    if bbox:
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        data_object.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
        debug_print("[DEBUG] Bounding box cropped.")
        
    # Apply density filter if required
    if apply_density_filter:
        data_object.data = data_object.apply_density_filter()
        debug_print("[DEBUG] Density filter applied.")

    # Remove flyers if required
    if remove_flyers:
        data_object.data = data_object.remove_flyers()
        debug_print("[DEBUG] Flyers removed.")