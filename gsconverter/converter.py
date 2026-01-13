import os
from .formats.ply_3dgs import Ply3DGSFormat
from .formats.ply_cc import PlyCCFormat
from .formats.parquet import ParquetFormat
from .formats.splat import SplatFormat
from .formats.spz import SpzFormat
from .formats.sog import SogFormat
from .formats.ksplat import KSplatFormat
from .formats.compressed_ply import CompressedPlyFormat
from .processing.data_processor import DataProcessor
from .utils.utility_functions import debug_print, status_print
class Converter:
    def __init__(self, input_path, output_path, target_format):
        self.input_path = input_path
        self.output_path = output_path
        self.target_format = target_format.lower()
        
        # Early Validation
        valid_formats = ['3dgs', 'cc', 'parquet', 'splat', 'ksplat', 'spz', 'sog', 'compressed_ply']
        if self.target_format not in valid_formats:
             raise ValueError(f"Unknown target format '{self.target_format}'. Supported: {', '.join(valid_formats)}")
             
        self.data = None
        self.source_format = None
        self.source_handler = None

    def _detect_format(self, path):
        if path.lower().endswith('.parquet'):
            return 'parquet'
        # Check for other extensions
        if path.lower().endswith('.splat'):
            return 'splat'
        if path.lower().endswith('.ksplat'):
            return 'ksplat'
        if path.lower().endswith('.spz'):
            return 'spz'
        if path.lower().endswith('.sog'):
            return 'sog'
            
        # If .ply, use content detection
        return self._text_based_detect(path)

    def _text_based_detect(self, file_path):
        debug_print("[DEBUG] Executing '_text_based_detect'...")
        
        try:
            with open(file_path, 'rb') as file:
                header_bytes = file.read(2048)  # Read the beginning to detect the format
                
            header = header_bytes.decode('utf-8', errors='ignore')
    
            if "element chunk" in header:
                return "compressed_ply"
            elif "property float f_dc_0" in header:
                return "3dgs"
            elif "property float scal_f_dc_0" in header or "property float scalar_scal_f_dc_0" in header or "property float scalar_f_dc_0" in header:
                return "cc"
        except Exception as e:
            debug_print(f"[DEBUG] Error identifying PLY flavor: {e}")
            
        return None
        
    def load_source_only(self):
        """Loads the source file and determines format without converting."""
        self.source_format = self._detect_format(self.input_path)
        if not self.source_format:
             raise ValueError("Could not detect source format")
             
        debug_print(f"[DEBUG] Detected source format: {self.source_format}")
        self.source_handler = self._get_format_handler(self.source_format)
        self.data = self.source_handler.read(self.input_path)
        return self.data

    def _get_format_handler(self, format_name):
        if format_name == '3dgs':
            return Ply3DGSFormat()
        elif format_name == 'cc':
            return PlyCCFormat()
        elif format_name == 'parquet':
            return ParquetFormat()
        elif format_name == 'splat':
            return SplatFormat()
        elif format_name == 'spz':
            return SpzFormat()
        elif format_name == 'sog':
            return SogFormat()
        elif format_name == 'ksplat':
            return KSplatFormat()
        elif format_name == 'compressed_ply':
            return CompressedPlyFormat()
        # Fallback/Errors
        raise ValueError(f"Unsupported format: {format_name}")

    def run(self, **kwargs):
        from tqdm import tqdm
        debug_print(f"[DEBUG] Starting conversion: {self.input_path} -> {self.output_path} ({self.target_format})")
        
        # Define progress milestones
        # 1. Detect & Read (30%)
        # 2. Process/Filter (30%)
        # 3. Write (40%)
        
        with tqdm(total=100, desc="Converting", bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}') as pbar:
        
            # 1. Detect Source Format
            source_format = self._detect_format(self.input_path)
            if not source_format:
                raise ValueError("Could not detect source format")
            debug_print(f"[DEBUG] Detected source format: {source_format}")
            pbar.update(5)
                
            # 2. Read
            pbar.set_description("Reading Source")
            self.source_handler = self._get_format_handler(source_format)
            self.data = self.source_handler.read(self.input_path)
            pbar.update(25)
            
            # Detect Source SH Degree
            # Detect Source SH Degree (scan content for padded formats)
            import numpy as np
            sh_cols = [f for f in self.data.dtype.names if f.startswith('f_rest_')]
            source_sh_degree = 0
            
            # Default based on columns
            if len(sh_cols) >= 45: source_sh_degree = 3
            elif len(sh_cols) >= 24: source_sh_degree = 2
            elif len(sh_cols) >= 9: source_sh_degree = 1
            
            # Refine by checking content (ignoring zero-padding from Ply3DGSFormat)
            if source_sh_degree > 0:
                last_active_idx = -1
                # Check backwards
                max_idx = {3: 44, 2: 23, 1: 8}[source_sh_degree]
                
                # Check blocks for efficiency? Or just loop. 45 checks is trivial.
                for i in range(max_idx, -1, -1):
                    col = f'f_rest_{i}'
                    if col in self.data.dtype.names:
                        if np.any(self.data[col] != 0):
                            last_active_idx = i
                            break
                
                if last_active_idx >= 24: source_sh_degree = 3
                elif last_active_idx >= 9: source_sh_degree = 2
                elif last_active_idx >= 0: source_sh_degree = 1
                else: source_sh_degree = 0
            
            # 3. Process
            pbar.set_description("Processing")
            processor = DataProcessor(self.data)
            
            # --- SH Capping Logic ---
            # Format Limits
            self.format_max_sh = {
            '3dgs': 3,
            'cc': 3,
            'parquet': 3,
            'ksplat': 2, # KSplat officially max degree 2
            'splat': 0,
            'spz': 3, # SPZ supports up to 3? Actually v3 spec allows SH.
            'sog': 3,
            'compressed_ply': 3
            }
            
            target_limit = self.format_max_sh.get(self.target_format, 3)
            requested_sh = kwargs.get('sh_level')
            
            # Start matching source to avoid upscaling
            final_sh_degree = source_sh_degree
            
            if requested_sh is not None:
                # Check: Requested vs Format Limit
                if requested_sh > target_limit:
                    status_print(f"Warning: Requested SH degree {requested_sh} exceeds limit for '{self.target_format}' ({target_limit}). Capping to {target_limit}.")
                
                # Check: Requested vs Source Data
                if requested_sh > source_sh_degree:
                    status_print(f"Warning: Requested SH degree {requested_sh} exceeds source data degree ({source_sh_degree}). Capping to {source_sh_degree}.")
                
                final_sh_degree = min(final_sh_degree, requested_sh)
                
            # Enforce Format Limit
            final_sh_degree = min(final_sh_degree, target_limit)
            
            if final_sh_degree < source_sh_degree:
                 status_print(f"SH Reduction: Source degree {source_sh_degree} -> Target degree {final_sh_degree}")
            
            processor.cap_sh_degree(final_sh_degree)
            pbar.update(5)

            # Filters
            pbar.set_description("Filtering")
                 
            # Filters - Manual Bounding Box (ROI)
            # Applied first to reduce processing load
            if kwargs.get('bbox'):
                bbox = kwargs['bbox']
                processor.crop_by_bbox(*bbox)

            # Filters - Alpha
            min_opacity = kwargs.get('min_opacity')
            if min_opacity is not None and min_opacity > 0:
                 processor.apply_alpha_filter(min_opacity)
                 
            # Filters - Density
            # Trigger if explicit params are set OR sensitivity is provided
            # New Explicit Params: density_voxel_size, density_threshold
            d_voxel = kwargs.get('density_voxel_size')
            d_thresh = kwargs.get('density_threshold')
            d_sens = kwargs.get('density_sensitivity')
            
            if (d_voxel is not None and d_thresh is not None) or d_sens is not None:
                 # Default values
                 voxel_size = 1.0 if d_voxel is None else float(d_voxel)
                 threshold = 0.32 if d_thresh is None else float(d_thresh)
                 
                 multi = kwargs.get('keep_multicluster', False)
                 processor.apply_density_filter(voxel_size, threshold, sensitivity=d_sens, keep_multicluster=multi)
                 
            pbar.update(10)

            # Filters - SOR (Flyers)
            # New Explicit Params: sor_k, sor_sigma
            s_k = kwargs.get('sor_k')
            s_sigma = kwargs.get('sor_sigma')
            s_intensity = kwargs.get('sor_intensity')
            
            if (s_k is not None and s_sigma is not None) or s_intensity is not None:
                 pbar.set_description("Filtering (SOR)")
                 # Default values
                 k = 25 if s_k is None else int(s_k)
                 threshold = 10.5 if s_sigma is None else float(s_sigma)
                       
                 processor.remove_flyers(k, threshold, intensity=s_intensity)
                 
            pbar.update(10)
            
            if kwargs.get('auto_bbox'):
                processor.apply_auto_bbox()
                
            # Auto-RGB: Check if target format REQUIRES RGB and if it's missing
            # CC, KSplat, Splat strongly imply RGB usage.
            # Validate RGB requirements for specific formats
            formats_needing_rgb = ['cc', 'splat', 'ksplat', 'sogs', 'sog'] 
            
            has_rgb = 'red' in self.data.dtype.names
            force_rgb = kwargs.get('rgb', False)
            
            if (self.target_format in formats_needing_rgb and not has_rgb) or force_rgb:
                if not has_rgb:
                    status_print(f"Target format '{self.target_format}' requires RGB. Auto-calculating from SH...")
                    processor.add_rgb_from_sh()
                elif force_rgb:
                    pass 
            
            pbar.update(5)
                
            # Update data from processor
            self.data = processor.data
            
            # 4. Write
            # 4. Write
            pbar.set_description(f"Writing {self.target_format.upper()}")
            
            # Handle Extra Elements logic
            # If maintain_extra_elements is set, retrieve from source handler
            if kwargs.get('maintain_extra_elements', False):
                if hasattr(self.source_handler, 'extra_elements') and self.source_handler.extra_elements:
                    kwargs['extra_elements'] = self.source_handler.extra_elements
                    
                    # WARN if target format does not support extra elements
                    # Currently only standard PLY writers (3dgs, cc) support appending raw elements.
                    # compressed_ply is a ply but uses a strict schema, likely won't support raw extras unless updated.
                    # Let's verify compressed_ply support later or assume only 3dgs/cc for now.
                    supported_extra_formats = ['3dgs', 'cc']
                    if self.target_format not in supported_extra_formats:
                        status_print(f"Warning: Target format '{self.target_format}' does not support preserving extra elements. These will be ignored.")

                else:
                    status_print("Warning: --extra_elements passed but no extra elements found in source.")  
            else:
                 # Ensure we don't pass the flag down if it confuses writers (though they look for 'extra_elements')
                 if hasattr(self.source_handler, 'extra_elements') and self.source_handler.extra_elements:
                     count = len(self.source_handler.extra_elements)
                     status_print(f"Stripping {count} extra PLY elements (use --extra_elements to preserve).")

            target_handler = self._get_format_handler(self.target_format)
            target_handler.write(self.data, self.output_path, **kwargs)
            
            pbar.update(40)
            pbar.refresh()
            pbar.set_description("Completed")
        
        status_print(f"Conversion completed: Saved to {self.output_path}")
