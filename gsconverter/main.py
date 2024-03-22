"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import argparse
import os
import sys
import numpy as np
from .utils.utility import Utility
from .utils.conversion_functions import convert
from plyfile import PlyData, PlyElement
from multiprocessing import Pool
from .utils import config
from .utils.utility_functions import init_worker
from .utils.argument_actions import DensityFilterAction, RemoveFlyersAction, AboutAction
from .utils.base_converter import BaseConverter

__version__ = '0.2'

def main():
    print(f"3D Gaussian Splatting Converter: {__version__}")
    
    parser = argparse.ArgumentParser(description="Convert between standard 3D Gaussian Splat and Cloud Compare formats.")
    
    # Arguments for input and output
    parser.add_argument("--input", "-i", required=True, help="Path to the source point cloud file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the converted point cloud file.")
    parser.add_argument("--target_format", "-f", choices=["3dgs", "cc"], required=True, help="Target point cloud format.")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug prints.")
    parser.add_argument('--about', action=AboutAction, help='Show copyright and license info')

    # Other flags
    parser.add_argument("--rgb", action="store_true", help="Add RGB values to the output file based on f_dc values (only applicable when converting to Cloud Compare format).")
    parser.add_argument("--bbox", nargs=6, type=float, metavar=('minX', 'minY', 'minZ', 'maxX', 'maxY', 'maxZ'), help="Specify the 3D bounding box to crop the point cloud.")
    parser.add_argument("--density_filter", nargs='*', action=DensityFilterAction, help="Filter the points to keep only regions with higher point density. Optionally provide 'voxel_size' and 'threshold_percentage' as two numbers (e.g., --density_filter 0.5 0.25). If no numbers are provided, defaults of 1.0 and 0.32 are used.")
    parser.add_argument("--remove_flyers", nargs='*', action=RemoveFlyersAction, help="Remove flyers based on k-nearest neighbors. Requires two numbers: 'k' (number of neighbors) and 'threshold_factor'.")
    
    args = parser.parse_args()
    
    config.DEBUG = args.debug

    # Check and append ".ply" extension if absent
    if not args.output.lower().endswith('.ply'):
        args.output += '.ply'

    # Now check if the file exists after potentially appending the extension
    if os.path.exists(args.output):
        user_response = input(f"File {args.output} already exists. Do you want to overwrite it? (y/N): ").lower()
        if user_response != 'y':
            print("Operation aborted by the user.")
            return

    # Detect the format of the input file
    if args.input.lower().endswith('.parquet'):
        source_format = 'parquet'
    else:
        source_format = Utility.text_based_detect_format(args.input)

    if not source_format:
        print("The provided file is not a recognized 3D Gaussian Splat point cloud format.")
        return

    print(f"Detected source format: {source_format}")
    
    # Check if --rgb flag is set for conversions involving 3dgs as target
    if args.target_format == "3dgs" and args.rgb:
        if source_format == "3dgs":
            print("Error: --rgb flag is not applicable for 3dgs to 3dgs conversion.")
            return
        elif source_format == "parquet":
            print("Error: --rgb flag is not applicable for parquet to 3dgs conversion.")
            return
        else:
            print("Error: --rgb flag is not applicable for cc to 3dgs conversion.")
            return

    # Check for RGB flag and format conditions
    if source_format == "cc" and args.target_format == "cc" and args.rgb:
        if 'red' in PlyData.read(args.input)['vertex']._property_lookup:
            print("Error: Source CC file already contains RGB data. Conversion stopped.")
            return

    # Read the data from the input file based on detected format
    if source_format == 'parquet':
        structured_data = BaseConverter.load_parquet(args.input)
        
        print(f"Number of vertices: {len(structured_data)}")
    else:
        data = PlyData.read(args.input)
        if isinstance(data, PlyData) and 'vertex' in data:
            print(f"Number of vertices in the header: {len(data['vertex'].data)}")
            structured_data = data['vertex'].data
        else:
            print("Error: Data format is not PlyData with a 'vertex' field.")
            return

    try:
        with Pool(initializer=init_worker) as pool:
            # If the bbox argument is provided, extract its values
            bbox_values = args.bbox if args.bbox else None
            
            # If the data is a structured array from a Parquet file, pass it directly
            if source_format == 'parquet':
                data_to_convert = structured_data
            else:
                # For PlyData, access the vertex data
                data_to_convert = data['vertex'].data
            
            # Call the convert function and pass the data to convert
            converted_data = convert(data_to_convert, source_format, args.target_format, process_rgb=args.rgb, density_filter=args.density_filter, remove_flyers=args.remove_flyers, bbox=bbox_values, pool=pool)
            
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
        sys.exit(-1)
        
    # Check if the conversion actually happened and save the result
    if isinstance(converted_data, np.ndarray):
        # Save the converted data to the output file
        PlyData([PlyElement.describe(converted_data, 'vertex')], byte_order='=').write(args.output)
        print(f"Conversion completed and saved to {args.output}.")
    else:
        print("Conversion was skipped.")

if __name__ == "__main__":
    main()