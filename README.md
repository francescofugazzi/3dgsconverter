# 3D Gaussian Splatting Converter

A tool for converting 3D Gaussian Splatting `.ply` and `.parquet` files into a format suitable for Cloud Compare and vice-versa. Enhance your point cloud editing with added functionalities like RGB coloring, density filtering, and flyer removal.

## Features

- **Format Conversion**: Seamlessly switch between 3DGS `.ply` and Cloud Compare-friendly `.ply` formats. Now also `.parquet` is supported as input file.
- **RGB Coloring**: Add RGB values to your point cloud for better visualization and editing in Cloud Compare.
- **Density Filtering**: Focus on the dense regions of your point cloud by removing sparse data.
- **Flyer Removal**: Get rid of unwanted outliers or floating points in your dataset. Especially useful when combined with the density filter due to its intensive nature.
- **Bounding box cropping**: command for cropping point clouds to focus on specific regions.

## Installation

There are two ways to install the 3D Gaussian Splatting Converter:

**1. Direct Installation via pip**:

Directly install the app from GitHub using pip. This method is straightforward and recommended for most users.

  ```bash
  pip install git+https://github.com/francescofugazzi/3dgsconverter.git
  ```

**2. Installation by Cloning the Repository:**:

If you prefer to clone the repository and install from the source, follow these steps:

  ```bash
  git clone https://github.com/francescofugazzi/3dgsconverter
  cd 3dgsconverter
  pip install .
  ```

## Usage

Here are some basic examples to get you started:

**1. Conversion from 3DGS to Cloud Compare format with RGB addition**:

   ```bash
   gsconverter -i input_3dgs.ply -o output_cc.ply -f cc --rgb
   ```

**2. Conversion from Cloud Compare format back to 3DGS:**:

   ```bash
   gsconverter -i input_cc.ply -o output_3dgs.ply -f 3dgs
   ```

**3. Applying Density Filter during conversion:**:

   ```bash
   gsconverter -i input_3dgs.ply -o output_cc.ply -f cc --density_filter
   ```

**4. Applying Density Filter and Removing floaters during conversion:**:

   ```bash
   gsconverter -i input_3dgs.ply -o output_cc.ply -f cc --density_filter --remove_flyers
   ```

For a full list of parameters and their descriptions, you can use the `-h` or `--help` argument:

```bash
gsconverter -h
```

## Debug Information

For detailed insights pass the `--debug` flag (or `-d` for short) when executing the script.

## Contribute

Feel free to open issues or PRs if you have suggestions or improvements for this tool!
