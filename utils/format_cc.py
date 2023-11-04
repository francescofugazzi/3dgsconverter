import numpy as np
from .base_converter import BaseConverter
from .utility import Utility
from .utility_functions import debug_print
from . import config

class FormatCC(BaseConverter):
    def to_3dgs(self, bbox=None, apply_density_filter=False, remove_flyers=False):
        debug_print("[DEBUG] Starting conversion from CC to 3DGS...")

        # Crop the data based on the bounding box if specified
        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            self.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
            debug_print("[DEBUG] Bounding box cropped.")
        
        # Apply density filter if required
        if apply_density_filter:
            self.data = self.apply_density_filter()
            debug_print("[DEBUG] Density filter applied.")

        # Remove flyers if required
        if remove_flyers:
            self.data = self.remove_flyers()
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

        debug_print("[DEBUG] Conversion from CC to 3DGS completed.")
        return converted_data


    def to_cc(self, bbox=None, apply_density_filter=False, remove_flyers=False, process_rgb=False):
        debug_print("[DEBUG] Processing CC data...")

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

        # Check if RGB processing is required
        if process_rgb and not self.has_rgb():
            self.add_rgb()
            debug_print("[DEBUG] RGB added to data.")
        else:
            debug_print("[DEBUG] RGB processing is skipped or data already has RGB.")
        
        converted_data = self.data
        
        # For now, we'll just return the converted_data for the sake of this integration
        debug_print("[DEBUG] CC data processing completed.")
        return converted_data

    def add_or_ignore_rgb(self, process_rgb=True):
        debug_print("[DEBUG] Checking RGB for CC data...")

        # If RGB processing is required and if RGB is not present
        if process_rgb and not self.has_rgb():
            # Compute RGB values for the data
            rgb_values = Utility.compute_rgb_from_vertex(self.data)
            
            # Define a new data type for the data that includes RGB
            new_dtype = Utility.define_dtype(has_scal=True, has_rgb=True)
            
            # Create a new numpy array with the new data type
            converted_data = np.zeros(self.data.shape, dtype=new_dtype)
            
            # Copy the data to the new numpy array
            Utility.copy_data_with_prefix_check(self.data, converted_data)
            
            # Add the RGB values to the new numpy array
            converted_data['red'] = rgb_values[:, 0]
            converted_data['green'] = rgb_values[:, 1]
            converted_data['blue'] = rgb_values[:, 2]

            
            self.data = converted_data  # Update the instance's data with the new data
            debug_print("[DEBUG] RGB added to data.")
        else:
            debug_print("[DEBUG] RGB processing is skipped or data already has RGB.")
            converted_data = self.data  # If RGB is not added or skipped, the converted_data is just the original data.

        # Return the converted_data
        debug_print("[DEBUG] RGB check for CC data completed.")
        return converted_data