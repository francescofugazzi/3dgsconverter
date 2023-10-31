import argparse
import numpy as np
import multiprocessing
import sys
import signal
import os
from plyfile import PlyData, PlyElement
#from tqdm import tqdm
from collections import deque
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors

DEBUG = False

class Utility:
    @staticmethod
    def text_based_detect_format(file_path):
        debug_print("[DEBUG] Executing 'text_based_detect_format' function...")

        """Detect if the given file is in '3dgs' or 'cc' format."""
        with open(file_path, 'rb') as file:
            header_bytes = file.read(2048)  # Read the beginning to detect the format

        header = header_bytes.decode('utf-8', errors='ignore')

        if "property float f_dc_0" in header:
            debug_print("[DEBUG] Detected format: 3dgs")
            return "3dgs"
        elif "property float scal_f_dc_0" in header or "property float scalar_scal_f_dc_0" in header or "property float scalar_f_dc_0" in header:
            debug_print("[DEBUG] Detected format: cc")
            return "cc"
        else:
            return None

    @staticmethod
    def copy_data_with_prefix_check(source, target, possible_prefixes):
        debug_print("[DEBUG] Executing 'copy_data_with_prefix_check' function...")

        """
        Given two structured numpy arrays (source and target), copy the data from source to target.
        If a field exists in source but not in target, this function will attempt to find the field
        in target by adding any of the possible prefixes to the field name.
        """
        for name in source.dtype.names:
            if name in target.dtype.names:
                target[name] = source[name]
            else:
                copied = False
                for prefix in possible_prefixes:
                    # If the field starts with the prefix, try the field name without the prefix
                    if name.startswith(prefix):
                        stripped_name = name[len(prefix):]
                        if stripped_name in target.dtype.names:
                            target[stripped_name] = source[name]
                            copied = True
                            break
                    # If the field doesn't start with any prefix, try adding the prefix
                    else:
                        prefixed_name = prefix + name
                        if prefixed_name in target.dtype.names:
                            debug_print(f"[DEBUG] Copying data from '{name}' to '{prefixed_name}'")
                            target[prefixed_name] = source[name]
                            copied = True
                            break
                if not copied:
                    print(f"Warning: Field {name} not found in target.")

    @staticmethod
    def compute_rgb_from_vertex(vertices):
        debug_print("[DEBUG] Executing 'compute_rgb_from_vertex' function...")
        
        # Depending on the available field names, choose the appropriate ones
        if 'f_dc_0' in vertices.dtype.names:
            f_dc = np.column_stack((vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']))
        else:
            f_dc = np.column_stack((vertices['scal_f_dc_0'], vertices['scal_f_dc_1'], vertices['scal_f_dc_2']))
        
        colors = (f_dc + 1) * 127.5
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        
        debug_print("[DEBUG] RGB colors computed.")
        return colors

    @staticmethod
    def parallel_voxel_counting(vertices, voxel_size=1.0):
        debug_print("[DEBUG] Executing 'parallel_voxel_counting' function...")
        
        """Counts the number of points in each voxel in a parallelized manner."""
        num_processes = cpu_count()
        chunk_size = len(vertices) // num_processes
        chunks = [vertices[i:i + chunk_size] for i in range(0, len(vertices), chunk_size)]

        num_cores = max(1, multiprocessing.cpu_count() - 1)
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            results = pool.starmap(Utility.count_voxels_chunk, [(chunk, voxel_size) for chunk in chunks])

        # Aggregate results from all processes
        total_voxel_counts = {}
        for result in results:
            for k, v in result.items():
                if k in total_voxel_counts:
                    total_voxel_counts[k] += v
                else:
                    total_voxel_counts[k] = v

        debug_print(f"[DEBUG] Voxel counting completed with {len(total_voxel_counts)} unique voxels found.")
        return total_voxel_counts
    
    @staticmethod
    def count_voxels_chunk(vertices_chunk, voxel_size):
        debug_print("[DEBUG] Executing 'count_voxels_chunk' function for a chunk...")
        
        """Count the number of points in each voxel for a chunk of vertices."""
        voxel_counts = {}
        for vertex in vertices_chunk:
            voxel_coords = (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size))
            if voxel_coords in voxel_counts:
                voxel_counts[voxel_coords] += 1
            else:
                voxel_counts[voxel_coords] = 1
        
        debug_print(f"[DEBUG] Chunk processed with {len(voxel_counts)} voxels counted.")
        return voxel_counts
    
    @staticmethod
    def get_neighbors(voxel_coords):
        debug_print(f"[DEBUG] Getting neighbors for voxel: {voxel_coords}...")
        
        """Get the face-touching neighbors of the given voxel coordinates."""
        x, y, z = voxel_coords
        neighbors = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        return neighbors

    @staticmethod
    def knn_worker(args):
        debug_print(f"[DEBUG] Executing 'knn_worker' function for vertex: {args[0]}...")
        
        """Utility function for parallel KNN computation."""
        coords, tree, k = args
        coords = coords.reshape(1, -1)  # Reshape to a 2D array
        distances, _ = tree.kneighbors(coords)
        avg_distance = np.mean(distances[:, 1:])
        
        debug_print(f"[DEBUG] Average distance computed for vertex: {args[0]} is {avg_distance}.")
        return avg_distance
    
    @staticmethod
    def some_other_utility_function():
        ...

class BaseConverter:
    def __init__(self, data):
        self.data = data

    def extract_vertex_data(vertices, has_scal=True, has_rgb=False):
        """Extract and convert vertex data from a structured numpy array of vertices."""
        debug_print("[DEBUG] Executing 'extract_vertex_data' function...")
        converted_data = []
        
        # Determine the prefix to be used based on whether "scal_" should be included
        prefix = 'scal_' if has_scal else ''
        debug_print(f"[DEBUG] Prefix determined as: {prefix}")
        
        # Iterate over each vertex and extract the necessary attributes
        for vertex in vertices:
            entry = (
                vertex['x'], vertex['y'], vertex['z'],
                vertex['nx'], vertex['ny'], vertex['nz'],
                vertex[f'{prefix}f_dc_0'], vertex[f'{prefix}f_dc_1'], vertex[f'{prefix}f_dc_2'],
                *[vertex[f'{prefix}f_rest_{i}'] for i in range(45)],
                vertex[f'{prefix}opacity'],
                vertex[f'{prefix}scale_0'], vertex[f'{prefix}scale_1'], vertex[f'{prefix}scale_2'],
                vertex[f'{prefix}rot_0'], vertex[f'{prefix}rot_1'], vertex[f'{prefix}rot_2'], vertex[f'{prefix}rot_3']
            )
            
            # If the point cloud contains RGB data, append it to the entry
            if has_rgb:
                entry += (vertex['red'], vertex['green'], vertex['blue'])
            
            converted_data.append(entry)
        
        debug_print("[DEBUG] 'extract_vertex_data' function completed.")
        return converted_data

    def apply_density_filter(self, voxel_size=1.0, threshold_percentage=0.32):
        debug_print("[DEBUG] Executing 'apply_density_filter' function...")
        vertices = self.data['vertex'].data

        # Convert threshold_percentage into a ratio
        threshold_ratio = threshold_percentage / 100.0

        # Parallelized voxel counting
        voxel_counts = Utility.parallel_voxel_counting(vertices, voxel_size)

        threshold = int(len(vertices) * threshold_ratio)
        dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

        visited = set()
        max_cluster = set()
        for voxel in dense_voxels:
            if voxel not in visited:
                current_cluster = set()
                queue = deque([voxel])
                while queue:
                    current_voxel = queue.popleft()
                    visited.add(current_voxel)
                    current_cluster.add(current_voxel)
                    for neighbor in Utility.get_neighbors(current_voxel):
                        if neighbor in dense_voxels and neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)
                if len(current_cluster) > len(max_cluster):
                    max_cluster = current_cluster

        filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]
        new_vertex_element = PlyElement.describe(np.array(filtered_vertices, dtype=vertices.dtype), 'vertex')
        
        # Update the plydata elements and the internal self.data
        converted_data = self.data
        converted_data.elements = (new_vertex_element,) + converted_data.elements[1:]
        self.data = converted_data  # Update the internal data with the filtered data
        
        print(f"After density filter, retained {len(filtered_vertices)} out of {len(vertices)} vertices.")

    def remove_flyers(self, k=25, threshold_factor=10.5, chunk_size=50000):
        debug_print("[DEBUG] Executing 'remove_flyers' function...")

        # Extract vertex data from the current object's data
        vertices = self.data['vertex'].data
        
        # Display the number of input vertices
        debug_print(f"[DEBUG] Number of input vertices: {len(vertices)}")

        # Number of chunks
        num_chunks = len(vertices) // chunk_size + (len(vertices) % chunk_size > 0)
        masks = []

        # Create a pool of workers
        num_cores = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk_coords = np.vstack((vertices['x'][start_idx:end_idx], vertices['y'][start_idx:end_idx], vertices['z'][start_idx:end_idx])).T

                # Compute K-Nearest Neighbors for the chunk
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(chunk_coords)
                avg_distances = pool.map(Utility.knn_worker, [(coord, nbrs, k) for coord in chunk_coords])

                # Calculate the threshold for removal based on the mean and standard deviation of the average distances
                threshold = np.mean(avg_distances) + threshold_factor * np.std(avg_distances)

                # Create a mask for points to retain for this chunk
                mask = np.array(avg_distances) < threshold
                masks.append(mask)

        # Combine masks from all chunks
        combined_mask = np.concatenate(masks)

        # Generate a new PlyElement with the filtered vertices based on the combined mask
        new_vertex_element = PlyElement.describe(vertices[combined_mask], 'vertex')

        # Update the plydata elements and the internal self.data
        self.data.elements = (new_vertex_element,) + self.data.elements[1:]
        
        print(f"After removing flyers, retained {len(vertices[combined_mask])} out of {len(vertices)} vertices.")


    def define_dtype(self, has_scal, has_rgb=False):
        debug_print("[DEBUG] Executing 'define_dtype' function...")
        
        prefix = 'scalar_scal_' if has_scal else ''
        debug_print(f"[DEBUG] Prefix determined as: {prefix}")
        
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            (f'{prefix}f_dc_0', 'f4'), (f'{prefix}f_dc_1', 'f4'), (f'{prefix}f_dc_2', 'f4'),
            *[(f'{prefix}f_rest_{i}', 'f4') for i in range(45)],
            (f'{prefix}opacity', 'f4'),
            (f'{prefix}scale_0', 'f4'), (f'{prefix}scale_1', 'f4'), (f'{prefix}scale_2', 'f4'),
            (f'{prefix}rot_0', 'f4'), (f'{prefix}rot_1', 'f4'), (f'{prefix}rot_2', 'f4'), (f'{prefix}rot_3', 'f4')
        ]
        debug_print("[DEBUG] Main dtype constructed.")
        
        if has_rgb:
            dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            debug_print("[DEBUG] RGB fields added to dtype.")
        
        debug_print("[DEBUG] 'define_dtype' function completed.")
        return dtype, prefix
    
    def has_rgb(self):
        return 'red' in self.data['vertex'].data.dtype.names and 'green' in self.data['vertex'].data.dtype.names and 'blue' in self.data['vertex'].data.dtype.names

class Format3dgs(BaseConverter):
    def to_cc(self, apply_density_filter=False, remove_flyers=False, process_rgb=True):
        debug_print("[DEBUG] Starting conversion from 3DGS to CC...")

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
        if process_rgb and self.has_rgb():
            debug_print("[DEBUG] RGB processing is enabled.")
            
            # Compute RGB values for the vertices
            rgb_values = Utility.compute_rgb_from_vertex(vertices)
            
            # Define a new data type for the vertices that includes RGB
            new_dtype, prefix = self.define_dtype(has_scal=True, has_rgb=process_rgb)
            
            # Create a new numpy array with the new data type
            converted_data = np.zeros(vertices.shape, dtype=new_dtype)
            
            # Copy the vertex data to the new numpy array
            Utility.copy_data_with_prefix_check(vertices, converted_data, [prefix])
            
            # Add the RGB values to the new numpy array
            converted_data['red'] = rgb_values[:, 0]
            converted_data['green'] = rgb_values[:, 1]
            converted_data['blue'] = rgb_values[:, 2]
            
            print("RGB processing completed.")
        else:
            debug_print("[DEBUG] RGB processing is skipped.")
            
            # Define a new data type for the vertices without RGB
            new_dtype, prefix = self.define_dtype(has_scal=True, has_rgb=process_rgb)
            
            # Create a new numpy array with the new data type
            converted_data = np.zeros(vertices.shape, dtype=new_dtype)
            
            # Copy the vertex data to the new numpy array
            Utility.copy_data_with_prefix_check(vertices, converted_data, [prefix])

        # For now, we'll just return the converted_data for the sake of this integration
        debug_print("[DEBUG] Conversion from 3DGS to CC completed.")
        return converted_data

    def to_3dgs(self, apply_density_filter=False, remove_flyers=False):
        debug_print("[DEBUG] Starting conversion from 3DGS to 3DGS...")

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
        if DEBUG:
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

class FormatCC(BaseConverter):
    def to_3dgs(self, apply_density_filter=False, remove_flyers=False):
        debug_print("[DEBUG] Starting conversion from CC to 3DGS...")

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
        if DEBUG:
            for i in range(5):
                debug_print(converted_data[i])

        debug_print("[DEBUG] Conversion from CC to 3DGS completed.")
        return converted_data


    def to_cc(self, apply_density_filter=False, remove_flyers=False, process_rgb=False):
        debug_print("[DEBUG] Processing CC data...")

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
            converted_data['red'] = rgb_values[0]
            converted_data['green'] = rgb_values[1]
            converted_data['blue'] = rgb_values[2]
            
            self.data = converted_data  # Update the instance's data with the new data
            debug_print("[DEBUG] RGB added to data.")
        else:
            debug_print("[DEBUG] RGB processing is skipped or data already has RGB.")
            converted_data = self.data  # If RGB is not added or skipped, the converted_data is just the original data.

        # Return the converted_data
        debug_print("[DEBUG] RGB check for CC data completed.")
        return converted_data

def convert(data, source_format, target_format, **kwargs):
    debug_print(f"[DEBUG] Starting conversion from {source_format} to {target_format}...")
    
    if source_format == "3dgs":
        converter = Format3dgs(data)
    elif source_format == "cc":
        converter = FormatCC(data)
    else:
        raise ValueError("Unsupported source format")

    # Apply optional operations
    if kwargs.get("density_filter"):
        print("Applying density filter...")
        converter.apply_density_filter()
    if kwargs.get("remove_flyers"):
        print("Removing flyers...")
        converter.remove_flyers()

    # RGB processing
    if source_format == "3dgs":
        debug_print("[DEBUG] Ignoring RGB for 3DGS data...")
        converter.ignore_rgb()
    elif source_format == "cc":
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
    elif source_format == "3dgs" and target_format == "3dgs":
        debug_print("[DEBUG] Applying operations on 3DGS data...")
        if not any(kwargs.values()):  # If no flags are provided
            print("[INFO] No flags provided. The conversion will not happen as the output would be identical to the input.")
            return data['vertex'].data
        else:
            return converter.to_3dgs()
    elif source_format == "cc" and target_format == "cc":
        ddebug_print("[DEBUG] Applying operations on CC data...")
        converted_data = converter.to_cc()
        if isinstance(converted_data, np.ndarray):
            return converted_data
        else:
            return data['vertex'].data
    else:
        raise ValueError("Unsupported conversion")

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def debug_print(message):
    if DEBUG:
        print(message)

def main():
    parser = argparse.ArgumentParser(description="Convert between standard 3D Gaussian Splat and Cloud Compare formats.")
    
    # Arguments for input and output
    parser.add_argument("--input", "-i", required=True, help="Path to the source point cloud file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the converted point cloud file.")
    parser.add_argument("--target_format", "-f", choices=["3dgs", "cc"], required=True, help="Target point cloud format.")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug prints.")

    # Other flags
    parser.add_argument("--rgb", action="store_true", help="Add RGB values to the output file based on f_dc values (only applicable when converting to Cloud Compare format).")
    parser.add_argument("--density_filter", action="store_true", help="Filter the points to keep only regions with higher point density.")
    parser.add_argument("--remove_flyers", action="store_true", help="Remove flyer points that are distant from the main cloud.")
    
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug

    if os.path.exists(args.output):
        user_response = input(f"File {args.output} already exists. Do you want to overwrite it? (y/N): ").lower()
        if user_response != 'y':
            print("Operation aborted by the user.")
            return

    # Detect the format of the input file
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
        else:
            print("Error: --rgb flag is not applicable for cc to 3dgs conversion.")
            return

    # Check for RGB flag and format conditions
    if source_format == "cc" and args.target_format == "cc" and args.rgb:
        if 'red' in PlyData.read(args.input)['vertex']._property_lookup:
            print("Error: Source CC file already contains RGB data. Conversion stopped.")
            return

    # Read the data from the input file
    data = PlyData.read(args.input)

    # Print the number of vertices in the header
    print(f"Number of vertices in the header: {len(data['vertex'].data)}")
    
    try:
        with Pool(initializer=init_worker) as pool:
            # Call the convert function
            converted_data = convert(data, source_format, args.target_format, process_rgb=args.rgb, density_filter=args.density_filter, remove_flyers=args.remove_flyers, pool=pool)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
        sys.exit(-1)
        
    # Check if the conversion actually happened and save the result
    if isinstance(converted_data, np.ndarray):
        # Check and append ".ply" extension if absent
        if not args.output.lower().endswith('.ply'):
            args.output += '.ply'
        # Save the converted data to the output file
        PlyData([PlyElement.describe(converted_data, 'vertex')], byte_order='=').write(args.output)
        print(f"Conversion completed and saved to {args.output}.")
    else:
        print("Conversion was skipped.")

if __name__ == "__main__":
    main()
