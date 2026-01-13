"""
3D Gaussian Splatting Converter
Copyright (c) 2026 Francesco Fugazzi (@franzipol)

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import argparse
import os
import sys
from .converter import Converter
from .utils import config


__version__ = '0.8'

class AboutAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(f"3D Gaussian Splatting Converter v{__version__}")
        print("Copyright (c) 2026 Francesco Fugazzi (@franzipol)")
        print("Project: https://github.com/francescofugazzi/3dgsconverter")
        print("License: MIT")
        parser.exit()

def check_source_extras(path):
    # Lightweight check for extra PLY elements
    try:
        if path.lower().endswith('.ply'):
            from plyfile import PlyData
            # Read header only? plyfile reads header on init but let's be safe
            # PlyData.read parses the file. For large files this might be slow if it reads body.
            # However, standard PlyData.read reads everything. 
            # We can use a text-based scan for "element" lines that are not "vertex" or "face"?
            # Or accept the overhead. Let's try text scan for speed.
            img_elems = ['extrinsic', 'intrinsic', 'camera', 'image_size', 'frame', 'disparity', 'color_space', 'version']
            with open(path, 'rb') as f:
                 header = b""
                 while True:
                     line = f.readline()
                     header += line
                     if b"end_header" in line:
                         break
            header_str = header.decode('utf-8', errors='ignore')
            for line in header_str.split('\n'):
                 if line.startswith('element'):
                     parts = line.split()
                     if len(parts) >= 2:
                         name = parts[1]
                         if name not in ['vertex', 'face']: # Standard elements
                             return True
    except:
        pass
    return False

def report_info(input_path, converter_obj=None):
    import numpy as np
    abs_path = os.path.abspath(input_path)
    print(f"\n{'-'*60}")
    print(f"File: {abs_path}")
    
    try:
        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")
        
        # If converter_obj is not provided, create a dummy one to load source
        if converter_obj is None:
            from .converter import Converter
            converter_obj = Converter(abs_path, "dummy_out.ply", "3dgs")
            data = converter_obj.load_source_only()
        else:
            data = converter_obj.data
            
        if hasattr(converter_obj, 'source_handler') and converter_obj.source_format == 'ksplat':
            handler = converter_obj.source_handler
            meta = getattr(handler, 'metadata', {})
            if meta:
                print(f"KSplat Version: {meta.get('v_major')}.{meta.get('v_minor')}")
                print(f"Compression Level: {meta.get('compression_level')}")
                if meta.get('compression_level', 0) >= 1 and meta.get('sections'):
                    s0 = meta['sections'][0]
                    print(f"Bucket Size: {s0.get('bucketSize')}")
                    print(f"Block Size: {s0.get('bucketBlockSize')}")
                if 'min_sh' in meta:
                    print(f"SH Range: [{meta['min_sh']:.2f}, {meta['max_sh']:.2f}]")
        
        if converter_obj.source_format == 'compressed_ply':
            from plyfile import PlyData
            ply = PlyData.read(abs_path)
            num_chunks = len(ply['chunk'].data)
            print(f"Quantization: Chunk-based (256 splats/chunk)")
            print(f"Chunks: {num_chunks:,}")
            print(f"Position/Scale Packing: 11-10-11 bit")
            print(f"Rotation Packing: 2-10-10-10 bit")
            print(f"Color Packing: 8-8-8-8 bit")
            if 'sh' in ply:
                print(f"SH Quantization: 8-bit ([-4, 4] range)")

        # Scan for Extra Elements
        if hasattr(converter_obj, 'source_handler') and hasattr(converter_obj.source_handler, 'extra_elements'):
            extras = [el.name for el in converter_obj.source_handler.extra_elements]
            if extras:
                print(f"Extra Elements: {', '.join(extras)}")

        print(f"Format Detected: {converter_obj.source_format.upper()}")
        N = len(data)
        print(f"Points: {N:,}")
        
        # Bounding Box
        if 'x' in data.dtype.names:
            min_pos = np.array([data['x'].min(), data['y'].min(), data['z'].min()])
            max_pos = np.array([data['x'].max(), data['y'].max(), data['z'].max()])
            print(f"Bounds Min: [{min_pos[0]:.4f}, {min_pos[1]:.4f}, {min_pos[2]:.4f}]")
            print(f"Bounds Max: [{max_pos[0]:.4f}, {max_pos[1]:.4f}, {max_pos[2]:.4f}]")
        
        # Attributes Scan
        fields = data.dtype.names
        has_rgb = 'red' in fields
        has_opacity = 'opacity' in fields
        has_scale = 'scale_0' in fields
        has_rot = 'rot_0' in fields
        
        attrs = []
        if has_rgb: attrs.append("RGB")
        if has_opacity: attrs.append("Opacity")
        if has_scale: attrs.append("Scale")
        if has_rot: attrs.append("Rotation")
        print(f"Attributes: {', '.join(attrs)}")
        
        # SH Analysis
        # SH Analysis
        raw_fields = fields
        header_msg = "None"
        active_msg = "None"

        if input_path.lower().endswith('.ply'):
            try:
                from plyfile import PlyData
                pd = PlyData.read(input_path)
                
                is_compressed = 'chunk' in pd
                
                if is_compressed:
                    # Compressed PLY Logic: SH data is in 'sh' element (if present)
                    if 'sh' in pd:
                        sh_props = [p.name for p in pd['sh'].properties]
                        n_sh = len(sh_props)
                        deg = 0
                        if n_sh >= 45: deg = 3
                        elif n_sh >= 24: deg = 2
                        elif n_sh >= 9: deg = 1
                        
                        header_msg = f"Degree {deg} ({n_sh} coeffs)"
                        active_msg = f"Degree {deg}"
                    else:
                        header_msg = "Degree 0 (DC)"
                        active_msg = "Degree 0"
                        
                elif 'vertex' in pd:
                    raw_fields = pd['vertex'].data.dtype.names
                    
                    sh_cols = [p.name for p in pd['vertex'].properties if p.name.startswith('f_rest_') or p.name.startswith('scalar_f_rest_')]
                    
                    if not sh_cols:
                         # Check if DC exists at least
                        if any(p.name.startswith('f_dc_') for p in pd['vertex'].properties) or 'red' in pd['vertex'].properties:
                             header_msg = "Degree 0 (DC)"
                             active_msg = "Degree 0"
                    else:
                        # Determine Max Degree in Header
                        max_degree = 0
                        if len(sh_cols) >= 45: max_degree = 3
                        elif len(sh_cols) >= 24: max_degree = 2
                        elif len(sh_cols) >= 9: max_degree = 1
                        
                        header_msg = f"Degree {max_degree} ({len(sh_cols)} coeffs)"

                        is_deg3_active = False
                        is_deg2_active = False
                        is_deg1_active = False
                        
                        def get_f_val(idx):
                            name = f'f_rest_{idx}'
                            if name in data.dtype.names:
                                return data[name]
                            if f'scalar_{name}' in data.dtype.names:
                                return data[f'scalar_{name}']
                            return None

                        if max_degree >= 3:
                            for i in range(24, 45):
                                val = get_f_val(i)
                                if val is not None and np.any(val):
                                    is_deg3_active = True
                                    break
                        
                        if max_degree >= 2:
                            for i in range(9, 24):
                                val = get_f_val(i)
                                if val is not None and np.any(val):
                                    is_deg2_active = True
                                    break
                                    
                        if max_degree >= 1:
                            for i in range(0, 9):
                                val = get_f_val(i)
                                if val is not None and np.any(val):
                                    is_deg1_active = True
                                    break

                        eff_deg = 0
                        if is_deg1_active: eff_deg = 1
                        if is_deg2_active: eff_deg = 2
                        if is_deg3_active: eff_deg = 3
                        
                        active_msg = f"Degree {eff_deg}"
                        if eff_deg < max_degree:
                            active_msg += " (Cropped/Zeroed)"

            except Exception as e:
                # Handle plyfile errors gracefully
                print(f"Warning: Could not parse PLY header for SH analysis: {e}")
                pass 
        
        # Fallback/Generic SH analysis if not handled by specific PLY logic
        if header_msg == "None" and active_msg == "None":
            sh_cols_raw = [f for f in raw_fields if 'f_rest_' in f]
            dc_cols_raw = [f for f in raw_fields if 'f_dc_' in f]

            if dc_cols_raw:
                header_msg = "DC (Degree 0)"
                active_msg = "DC"
            
            if sh_cols_raw:
                count_raw = len(sh_cols_raw)
                max_degree_raw = 0
                if count_raw >= 45: max_degree_raw = 3
                elif count_raw >= 24: max_degree_raw = 2
                elif count_raw >= 9: max_degree_raw = 1
                
                if max_degree_raw > 0:
                    header_msg = f"Degree {max_degree_raw} ({count_raw} coeffs)"
                elif dc_cols_raw:
                    header_msg = "Degree 0 (DC only)"
                
                # Analyze Content using existing logic for message consistency
                active_msg = header_msg # Estimate

        print(f"SH Headers: {header_msg}")
        print(f"SH Content: {active_msg}")

    except Exception as e:
        print(f"Error reading info for {input_path}: {e}")
    print(f"3D Gaussian Splatting Converter: {__version__}")
    
def main():
    parser = argparse.ArgumentParser(description="Universal 3D Gaussian Splatting Converter. Supports: 3DGS (.ply), CloudCompare (.ply), KSplat (.ksplat), Splat (.splat), SPZ (.spz), SOG (.sog), Parquet (.parquet), Compressed PLY (.ply).")
    
    # Arguments for input and output
    parser.add_argument("--input", "-i", required=True, help="Path to the source point cloud file.")
    parser.add_argument("--output", "-o", help="Path to save the converted point cloud file.")
    parser.add_argument("--target_format", "-f", help="Target point cloud format (3dgs, cc, ksplat, splat, spz, sog, parquet, compressed_ply).")
    parser.add_argument("--info", "-I", action="store_true", help="Print file metadata and statistics without converting")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug prints.")
    parser.add_argument('--about', action=AboutAction, help='Show copyright and license info')
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing output file.")
    
    # Other flags (pass-through)
    parser.add_argument("--rgb", action="store_true", help="Add RGB values to the output file based on f_dc values (useful for formats needing explicit RGB like CC, SOG, SPZ, Parquet).")
    parser.add_argument("--bbox", nargs=6, type=float, metavar=('minX', 'minY', 'minZ', 'maxX', 'maxY', 'maxZ'), help="Specify the 3D bounding box to crop the point cloud.")
    parser.add_argument("--auto_bbox", action="store_true", help="Automatically calculate and apply a tight bounding box to the filtered data before saving.")
    parser.add_argument("--extra_elements", action="store_true", help="Preserve extra PLY elements (like camera extrinsic/intrinsic) when converting between 3DGS/CC formats.")
    
    # Explicit Advanced Params (replaces old list actions)
    # Explicit Advanced Params (Hidden from Help but Functional)
    parser.add_argument("--density_voxel_size", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--density_threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--sor_k", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--sor_sigma", type=float, help=argparse.SUPPRESS)
    
    parser.add_argument("--crop_sh", action="store_true", help="Crop SH coefficients to only those present in the source (disables 45-coeff padding).")
    
    # Advanced Compression & SH Defaults
    # Note: sh_level is generic but KSplat specifically uses unique packing. 
    parser.add_argument("--sh_level", type=int, help="Target SH degree (0-3). Automatically capped by source data and target format limits.")
    parser.add_argument("--bucket_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--block_size", type=float, help=argparse.SUPPRESS)
    
    # Advanced Filtering (User-Friendly Sliders)
    parser.add_argument("--density_sensitivity", type=float, help="Density filter sensitivity (0.0-1.0). Higher values merge more points to reduce file size. (0.1=High Detail, 0.9=High Compression).")
    parser.add_argument("--sor_intensity", type=float, help="SOR filter intensity (1.0-10.0). Higher values remove more 'floating' noise artifacts. (1.0=Safe/Conservative, 10.0=Aggressive Cleaning).")
    parser.add_argument("--min_opacity", type=int, help="Minimum opacity threshold (0-255) to keep a splat.")
    parser.add_argument("--keep_multicluster", action="store_true", help="If set, density filter keeps multiple dense clusters instead of just the largest one.")
    
    # Compression & Format Specifics
    parser.add_argument("--compression_level", type=int, default=0, help="Compression level (0-9). Format specific: KSplat (0=Float32, 1=Block/f16, 2+=Block/u8-SH), SPZ (Gzip level), SOG (0-3=64k, 4-6=16k, 7-9=4k Palette).")
    
    # Parse arguments
    args = parser.parse_args()

    
    config.DEBUG = args.debug
    
    # --- ARGUMENT VALIDATION ---
    if args.density_sensitivity is not None:
        if not (0.0 <= args.density_sensitivity <= 1.0):
            print(f"Error: --density_sensitivity must be between 0.0 and 1.0. Got {args.density_sensitivity}.")
            return

    if args.sor_intensity is not None:
        if not (1.0 <= args.sor_intensity <= 10.0):
            print(f"Error: --sor_intensity must be between 1.0 and 10.0. Got {args.sor_intensity}.")
            return
            
    if args.min_opacity is not None:
        if not (0 <= args.min_opacity <= 255):
             print(f"Error: --min_opacity must be between 0 and 255. Got {args.min_opacity}.")
             return
             
    if args.compression_level < 0 or args.compression_level > 9:
         print(f"Error: --compression_level must be between 0 and 9. Got {args.compression_level}.")
         return

    # --- INFO MODE ---
    if args.info:
        import glob
        import numpy as np
        
        input_files = glob.glob(args.input)
        if not input_files:
            print(f"Error: No input files found matching '{args.input}'")
            return

        for input_path in input_files:
            report_info(input_path)
        return

    # --- CONVERSION MODE ---
    if not args.target_format:
        if args.info: return
        parser.error("--target_format is required for conversion mode.")

    # Early Validation
    valid_formats = ['3dgs', 'cc', 'parquet', 'splat', 'ksplat', 'spz', 'sog', 'compressed_ply']
    if args.target_format.lower() not in valid_formats:
         print(f"Error: Unknown target format '{args.target_format}'. Supported: {', '.join(valid_formats)}")
         return

    # Auto-Output Logic
    if not args.output:
        base_name, input_ext = os.path.splitext(args.input)
        
        # Determine extension based on target format
        ext_map = {
            '3dgs': '.ply', 'cc': '.ply', 'compressed_ply': '.ply',
            'sog': '.sog', 'splat': '.splat', 'ksplat': '.ksplat', 
            'spz': '.spz', 'parquet': '.parquet'
        }
        target_ext = ext_map.get(args.target_format, '.' + args.target_format)
        
        # Automatic Suffix Logic
        suffix = ""
        if input_ext.lower() == target_ext.lower():
             # Avoid self-overwrite by adding format-specific suffix
             if args.target_format == 'cc': suffix = "_cc"
             elif args.target_format == 'compressed_ply': suffix = "_compressed"
             elif args.target_format == '3dgs': suffix = "_3dgs"
             else: suffix = "_processed"
             
        args.output = f"{base_name}{suffix}{target_ext}"
        print(f"Auto-Output: Destination set to {args.output}")

    # No-Op / Redundant Conversion Check
    # If source extension matches target extension AND no filters are applied, warn user.
    # We can approximate source format by extension for this early check.
    in_ext = os.path.splitext(args.input)[1].lower()
    
    # Check if filters are active
    filters_active = any([
        args.density_voxel_size, args.density_threshold,
        args.sor_k, args.sor_sigma,
        args.crop_sh,
        args.sh_level is not None,
        args.min_opacity,
        args.keep_multicluster,
        # Compression > 0 is an action. Converting format is also an action.
        # This check prevents PLY->PLY no-ops.
    ])
    
    # Filter Activation Check
    # Logic: If extensions match (e.g. .ply -> .ply) and no filters/compression are active, warn the user.
    # Exception: Specialized variants (CC, Compressed) should still process.
    
    # Check for presence of extra elements in source
    has_source_extras = check_source_extras(args.input)
    
    # "Stripping" is an action. If source has extras and valid flag is NOT set, we are stripping.
    is_stripping_action = has_source_extras and not args.extra_elements
    
    # "Maintaining" is a no-op if inputs are same.
    # So if has_source_extras AND args.extra_elements, that doesn't count as an active filter (it preserves status quo).
    # But filters_active logic below just lists flags. 
    # args.extra_elements is a flag. 
    # If I set filters_active = ... or args.extra_elements, that implies maintaining is an ACTION.
    # User logic: 
    # - Input(Extras) -> Output(Default): Proceed (Stripping)
    # - Input(Extras) -> Output(Maintain): Block (No-Op)
    
    # So we DO NOT add args.extra_elements to filters_active list. 
    # Instead we treat 'is_stripping_action' as a filter.
    
    filters_active = any([
        args.density_voxel_size, args.density_threshold,
        args.sor_k, args.sor_sigma,
        args.crop_sh,
        args.sh_level is not None,
        args.min_opacity,
        args.keep_multicluster,
        # Compression > 0 is an action. Converting format is also an action.
        # This check prevents PLY->PLY no-ops.
        is_stripping_action
    ])
    
    # No-Op / Redundant Conversion Check
    # Prevent consuming resources if the operation is a no-op (e.g. 3DGS PLY -> 3DGS PLY with no filters).
    # Specialized PLY variants (CC, Compressed) are considered distinct formats.
    
    is_same_extension = (in_ext == os.path.splitext(args.output)[1].lower())
    is_generic_target = (args.target_format == '3dgs')
    
    # Blocking condition:
    # If same extension, generic target, no filters active, no compression, not forced.
    # AND crucially: if we ARE maintaining extras (args.extra_elements=True), that confirms status quo, so it should block.
    # (Since we didn't add args.extra_elements to filters_active, it remains False in the Maintain case).
    
    if is_same_extension and is_generic_target and not filters_active and args.compression_level == 0 and not args.force:
        print("\n[INFO] Target is generic 3DGS PLY (same as input extension) and no filters are active.")
        if args.extra_elements and has_source_extras:
            print("       (You are maintaining extra elements, so the output would be identical to input).")
        print("       Refer to --help to apply filters (density, SOR, SH reduction) or remove --extra_elements to strip data.")
        print("       Operation aborted to prevent redundant processing.")
        return

    # Auto-Extension Logic: Ensures output has correct extension (if user didn't provide it)
    if not os.path.splitext(args.output)[1]:
        ext_map = {
            '3dgs': '.ply', 'cc': '.ply', 'compressed_ply': '.ply',
            'sog': '.sog', 'splat': '.splat', 'ksplat': '.ksplat', 
            'spz': '.spz', 'parquet': '.parquet'
        }
        target_ext = ext_map.get(args.target_format, '.' + args.target_format)
        args.output += target_ext
        print(f"Auto-Extension: Appended extension, new output: {args.output}")

    # Ensure directory existence for output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Overwrite Safety mechanism
    if os.path.exists(args.output) and not args.force:
        print(f"Warning: Output file '{args.output}' already exists.")
        confirm = input("Overwrite? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return

    try:
        # 1. Report Input Info
        print("\n>>> SOURCE FILE INFO")
        report_info(args.input)

        # 2. Convert
        converter = Converter(args.input, args.output, args.target_format)
        
        # Explicitly pass all filter arguments
        converter.run(
            density_voxel_size=args.density_voxel_size,
            density_threshold=args.density_threshold,
            density_sensitivity=args.density_sensitivity,
            keep_multicluster=args.keep_multicluster,
            sor_k=args.sor_k,
            sor_sigma=args.sor_sigma,
            sor_intensity=args.sor_intensity,
            min_opacity=args.min_opacity,
            bbox=args.bbox,
            rgb=args.rgb,
            sh_level=args.sh_level,
            bucket_size=args.bucket_size,
            block_size=args.block_size,
            crop_sh=args.crop_sh,
            auto_bbox=args.auto_bbox,
            compression_level=args.compression_level,
            maintain_extra_elements=args.extra_elements
        )


        # 3. Report Output Info
        print("\n>>> TARGET FILE INFO")
        report_info(args.output)

    except Exception as e:
        print(f"Error: {e}")
        if config.DEBUG:
            raise e
        sys.exit(1)

if __name__ == "__main__":
    main()
