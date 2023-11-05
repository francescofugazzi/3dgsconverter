"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
from .base_converter import BaseConverter
from .utility_functions import debug_print
from .utility import Utility
from . import config

class Format3dgs(BaseConverter):
    def to_cc(self, bbox=None, apply_density_filter=False, remove_flyers=False, process_rgb=True):
        debug_print("[DEBUG] Starting conversion from 3DGS to CC...")

        # Crop the data based on the bounding box if specified
        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            self.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
            debug_print("[DEBUG] Bounding box cropped.")
        
        # Apply density filter if required
        if apply_density_filter:
            self.apply_density_filter()
            debug_print("[DEBUG] Density filter applied.")

        # Remove flyers if required
        if remove_flyers:
            self.remove_flyers()
            debug_print("[DEBUG] Flyers removed.")

        # Load vertices from the provided data
        vertices = self.data['vertex'].data
        debug_print(f"[DEBUG] Loaded {len(vertices)} vertices.")

        # Check if RGB processing is required
        if process_rgb:
            debug_print("[DEBUG] RGB processing is enabled.")

            # Compute RGB values for the vertices
            rgb_values = Utility.compute_rgb_from_vertex(vertices)

            if rgb_values is not None:
                # Define a new data type for the vertices that includes RGB
                new_dtype, prefix = self.define_dtype(has_scal=True, has_rgb=True)

                # Create a new numpy array with the new data type
                converted_data = np.zeros(vertices.shape, dtype=new_dtype)

                # Copy the vertex data to the new numpy array
                Utility.copy_data_with_prefix_check(vertices, converted_data, [prefix])

                # Add the RGB values to the new numpy array
                converted_data['red'] = rgb_values[:, 0]
                converted_data['green'] = rgb_values[:, 1]
                converted_data['blue'] = rgb_values[:, 2]

                debug_print("RGB processing completed.")
            else:
                debug_print("[DEBUG] RGB computation failed. Skipping RGB processing.")
                process_rgb = False

        if not process_rgb:
            debug_print("[DEBUG] RGB processing is skipped.")

            # Define a new data type for the vertices without RGB
            new_dtype, prefix = self.define_dtype(has_scal=True, has_rgb=False)

            # Create a new numpy array with the new data type
            converted_data = np.zeros(vertices.shape, dtype=new_dtype)

            # Copy the vertex data to the new numpy array
            Utility.copy_data_with_prefix_check(vertices, converted_data, [prefix])

        # For now, we'll just return the converted_data for the sake of this integration
        debug_print("[DEBUG] Conversion from 3DGS to CC completed.")
        return converted_data

    def to_3dgs(self, bbox=None, apply_density_filter=False, remove_flyers=False):
        debug_print("[DEBUG] Starting conversion from 3DGS to 3DGS...")

        # Crop the data based on the bounding box if specified
        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            self.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
            debug_print("[DEBUG] Bounding box cropped.")
        
        # Apply density filter if required
        if apply_density_filter:
            self.apply_density_filter()
            debug_print("[DEBUG] Density filter applied.")

        # Remove flyers if required
        if remove_flyers:
            self.remove_flyers()
            debug_print("[DEBUG] Flyers removed.")

        # Load vertices from the updated data after all filters
        vertices = self.data['vertex'].data
        debug_print(f"[DEBUG] Loaded {len(vertices)} vertices.")

        # Create a new structured numpy array for 3DGS format
        dtype_3dgs = self.define_dtype(has_scal=False, has_rgb=False)  # Define 3DGS dtype without any prefix
        converted_data = np.zeros(vertices.shape, dtype=dtype_3dgs)

        # Use the helper function to copy the data from vertices to converted_data
        Utility.copy_data_with_prefix_check(vertices, converted_data, ["", "scal_", "scalar_", "scalar_scal_"])

        debug_print("[DEBUG] Data copying completed.")
        debug_print("\\n[DEBUG] Sample of converted data (first 5 rows):")
        if config.DEBUG:
            for i in range(5):
                debug_print(converted_data[i])

        debug_print("[DEBUG] Conversion from 3DGS to 3DGS completed.")
        return converted_data
    
    def ignore_rgb(self):
        debug_print("[DEBUG] Checking RGB for 3DGS data...")

        # Initialize converted_data to the original vertex data
        converted_data = self.data['vertex'].data

        # Check if RGB is present
        if self.has_rgb():
            # Define a new data type for the data that excludes RGB
            new_dtype = Utility.define_dtype(has_scal=True, has_rgb=False)
            
            # Create a new numpy array with the new data type
            converted_data_without_rgb = np.zeros(self.data['vertex'].data.shape, dtype=new_dtype)
            
            # Copy the data to the new numpy array, excluding RGB
            Utility.copy_data_with_prefix_check(self.data['vertex'].data, converted_data_without_rgb, exclude=['red', 'green', 'blue'])
            
            converted_data = converted_data_without_rgb  # Update the converted data
            debug_print("[DEBUG] RGB removed from data.")
        else:
            debug_print("[DEBUG] Data does not have RGB or RGB removal is skipped.")
        
        # For now, we'll just return the converted_data for the sake of this integration
        debug_print("[DEBUG] RGB check for 3DGS data completed.")
        return converted_data