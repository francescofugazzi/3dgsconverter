# 3D Gaussian Splatting Converter

[![GitHub Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/francescofugazzi)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/franzipol)

A versatile, high-performance tool for converting between various 3D Gaussian Splatting formats. It supports advanced filtering, GPU acceleration, and universal N-to-N conversion between all supported formats.

## Key Features

- **Universal Conversion**: Convert **any** supported format to **any** other format (e.g., `.ply` to `.ksplat`, `.spz` to `.ply`, `.sog` to `.sog`).
- **Cross-Platform**: Fully compatible with **Windows**, **Linux**, and **macOS** (Apple Silicon & Intel).
- **GPU Acceleration**: Built-in GPU support via **Taichi Lang** for extremely fast K-Means clustering (SOG/KSplat compression) and Statistical Outlier Removal (SOR). Supports NVIDIA CUDA, Apple Metal, and Vulkan/Integrated GPUs.
- **Advanced Filtering**:
    - **Region of Interest**: Crop using Manual Bounding Box (Runs first for speed).
    - **Alpha/Opacity**: Remove transparent splats.
    - **Density Control**: Remove sparse areas while keeping dense clusters (Multi-cluster support).
    - **Outlier Removal**: Statistical Outlier Removal (SOR) to clean up "flyers".
- **Compression Control**: Balance quality and file size with `--compression_level` (0 = Max Quality).
- **Format Agnostic**: Filters apply to the intermediate raw data, so they work on ANY input format.
- **Custom Data Preservation**: Use `--extra_elements` to maintain non-standard PLY properties (like camera parameters) when converting between PLY formats.

## Supported Formats & Peculiarities

| Format | Extension | Max SH | Features & Parameters |
| :--- | :--- | :--- | :--- |
| **3DGS PLY** | `.ply` | **3** | **Standard Format**. Full float32. <br>• `--sh_level`: Caps SH degree. |
| **Cloud Compare** | `.ply` | **3** | **Viewer Compatible**. Forces RGB generation. <br>• `--rgb`: Auto-converts SH to RGB. |
| **Compressed PLY** | `.ply` | **3** | **High Compression**. Chunk-based quantization stored in standard PLY container. |
| **Splat** | `.splat` | **0** | **Legacy Web**. RGB only (No SH). |
| **KSplat** | `.ksplat` | **2** | **Streaming/Web**. Hierarchical quantization. <br>• `compression_level` 0 (F32), 1 (F16), ≥2 (Quantized) |
| **SPZ** | `.spz` | **3** | **Fast Loading**. Gzip compressed. <br>• `compression_level`: Controls Gzip level (0-9). |
| **SOG** | `.sog` | **3** | **Web/Mobile Optimized**. GPU texture compression. <br>• `compression_level`: Codebook size control. |
| **Parquet** | `.parquet` | **3** | **Data Analysis**. Columnar storage. Pandas compatible. |

## Installation

It is recommended to use **Conda** to manage dependencies.

1.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n gsconv python=3.10
    conda activate gsconv
    ```
    *(Or use a local `.venv` if preferred)*

2.  **Install the Package:**

    *Option A: Quick Install (via pip)*
    ```bash
    pip install git+https://github.com/francescofugazzi/3dgsconverter.git
    ```

    *Option B: Manual Development Install (Clone)*
    ```bash
    git clone https://github.com/francescofugazzi/3dgsconverter.git
    cd 3dgsconverter
    pip install -r requirements.txt
    ```

    *Note: To enable GPU acceleration, ensure `taichi` is installed (`pip install taichi`). It is included in `requirements.txt`.*

## Usage

Basic syntax:
```bash
3dgsconverter -i <input_file> -f <target_format> [-o <output_file>] [options]
```
*(Aliases: `gsconverter`, `3dgsconv`, `gsconv`)*
*Note: `-o` is optional. If omitted, the output filename is derived from the input with the correct extension.*
*Use `--force` to overwrite existing files without confirmation.*

### Common Examples

**1. Basic Conversion (PLY to KSplat)**
```bash
3dgsconverter -i input.ply -f ksplat
# Creates input.ksplat automatically
```

**2. High-Quality Compression (SOG)**
```bash
# Level 1 = High Quality. Level 10 = High Compression.
3dgsconverter -i input.ply -o output.sog -f sog --compression_level 1
```

**3. Cleaning a Point Cloud (Filters)**
```bash
# Remove invisible points and outliers
3dgsconverter -i raw.ply -o clean.ply -f 3dgs --min_opacity 5 --sor_intensity 8
```

**4. Extracting a Region of Interest**
```bash
# Crop BEFORE filtering (Attributes: minX minY minZ maxX maxY maxZ)
3dgsconverter -i scene.spz -o crop.ply -f 3dgs --bbox -2 -2 -2 2 2 2
```

### Parameter Reference

#### General
-   `-i, --input`: Path to source file.
-   `-o, --output`: Path to destination file (Optional).
-   `-f, --format`: Target format (`3dgs`, `cc`, `ksplat`, `splat`, `sog`, `spz`, `parquet`, `compressed_ply`).
-   `--force`: Overwrite existing output file without prompting.
-   `--rgb`: Force RGB generation from SH (for supported formats).
-   `--extra_elements`: Preserves non-standard PLY elements (like `extrinsic`, `intrinsic`) when converting between 3DGS/CC formats.
-   `--sh_level`: Target SH degree (0-3). Tool **never** upscales; keeps source degree if lower.
-   `--compression_level`: 0-10. Controls quality/size ratio for supported formats (SOG, KSplat, SPZ).
    -   **KSplat**: Set to `2` for quantization (experimental levels >2 map to 2).

#### Advanced Filters
Filters run in this specific order for efficiency:

1.  **Manual BBox** (`--bbox`): Crops space first.
2.  **Alpha Filter** (`--min_opacity <0-255>`): Discards points with opacity < threshold.
3.  **Density Filter**:
    -   `--density_sensitivity <0.0-1.0>`: Simple slider. 0.0 = Lazy (keeps more), 1.0 = Aggressive (removes sparse halos).
    -   `--keep_multicluster`: If set, keeps *all* dense clusters. Default: keeps only the largest main cluster.
4.  **SOR (Flyers)**:
    -   `--sor_intensity <1-10>`: Simple slider. 1 = Weak (keeps more), 10 = Strong (removes all outliers).
5.  **Auto BBox** (`--auto_bbox`):
    -   Calculates the minimal bounding box of the remaining valid points before saving, ensuring tight packing.

## Technical Details

Built with **Taichi Lang** for high-performance GPU computing.
-   **Parallelism**: Voxels and SOR are processed in parallel on the GPU.
-   **Differentiable Design**: While currently used for conversion, the core is differentiable, allowing for future optimization and training applications.
-   **No Upscale Rule**: The converter strictly respects the source SH degree. If you convert a File with SH=1 to a format supporting SH=3, the output will remain SH=1 (padded with zeros if necessary, but never "hallucinated").

## Credits

This project builds upon the incredible work of the 3D Gaussian Splatting community. Special thanks to:

-   **[Inria (GraphDeco)](https://github.com/graphdeco-inria/gaussian-splatting)**: The original 3D Gaussian Splatting paper and implementation.
-   **[Antimatter15](https://github.com/antimatter15/splat)**: For the `.splat` format and the first WebGL viewer.
-   **[mkkellogg](https://github.com/mkkellogg/GaussianSplats3D)**: For the `.ksplat` format and 3DGS-Three.js integration.
-   **[Niantic Labs](https://github.com/nianticlabs/spz)**: For the `.spz` format definition.
-   **[PlayCanvas](https://github.com/playcanvas/supersplat)**: For the `.sog` format, `.ply` chunk-quantization schema, and SuperSplat viewer.

## Refactoring & Breaking Changes (Type-Check Safe)

This project has undergone a complete refactoring to modularize the codebase and improve type safety. As a result, some legacy command-line arguments have been deprecated and replaced with more explicit or user-friendly alternatives.

### Replaced Commands
| Legacy Command | New Equivalent | Description |
| :--- | :--- | :--- |
| `--density_filter [v, t]` | `--density_sensitivity` (Slider) | **Effect**: Reduces file size by merging overlapping points.<br>**Practical**: 0.1 keeps detail, 0.9 aggressively simplifies. |
| `--remove_flyers [k, s]` | `--sor_intensity` (Slider) | **Effect**: Cleans up floating noise/artifacts.<br>**Practical**: 1.0 is safe, 10.0 is aggressive. |

### New Features
*   **Format-Specific Compression**: The `--compression_level` flag adapts to the target format:
    *   **KSplat**:
        *   `0`: No compression (Float32 w/ SH). Max quality.
        *   `1`: Block compression (Uint16 Pos, Float16 Scale/Rot/SH). Good balance.
        *   `2+`: Aggressive Block compression (Uint16 Pos, Float16 Scale/Rot, Uint8 SH). Smallest size, lossy SH.
    *   **SPZ**: Controls Gzip compression effort (1-9). This is **lossless** for the data itself, affecting only file size and save time.
    *   **SOG**: Controls SH Palette Quality (0=Max Quality, 9=Min Quality). Note: Texture compression is always Lossless WebP.

*   **User-Friendly Sliders**: New `--density_sensitivity` (0.0-1.0) and `--sor_intensity` (1.0-10.0) abstract complex parameters away for easier use.
*   **Modular Architecture**: The monolith code has been split into `formats/`, `processing/`, and `utils/` for better maintainability and extensibility. 
