# 3D Gaussian Splatting Converter

A tool for converting 3D Gaussian Splatting `.ply` files into a format suitable for Cloud Compare and vice-versa. Enhance your point cloud editing with added functionalities like RGB coloring, density filtering, and flyer removal.

## Features

- **Format Conversion**: Seamlessly switch between 3DGS `.ply` and Cloud Compare-friendly `.ply` formats.
- **RGB Coloring**: Add RGB values to your point cloud for better visualization and editing in Cloud Compare.
- **Density Filtering**: Focus on the dense regions of your point cloud by removing sparse data.
- **Flyer Removal**: Get rid of unwanted outliers or floating points in your dataset. Especially useful when combined with the density filter due to its intensive nature.

## Usage

Here are some basic examples to get you started:

**1. Conversion from 3DGS to Cloud Compare format with RGB addition**:

   ```bash
   python 3dgsconverter.py -i input_3dgs.ply -o output_cc.ply -f cc --rgb
   ```

**2. Conversion from Cloud Compare format back to 3DGS:**:

   ```bash
   python 3dgsconverter.py -i input_cc.ply -o output_3dgs.ply -f 3dgs
   ```

**3. Applying Density Filter during conversion:**:

   ```bash
   python 3dgsconverter.py -i input_3dgs.ply -o output_cc.ply -f cc --density_filter
   ```

**4. Applying Density Filter and Removing floaters during conversion:**:

   ```bash
   python 3dgsconverter.py -i input_3dgs.ply -o output_cc.ply -f cc --density_filter --remove fliers
   ```

For a full list of parameters and their descriptions, you can use the `-h` or `--help` argument:

```bash
python 3dgsconverter.py -h
```

## Debug Information

For detailed insights pass the `--debug` flag (or `-d` for short) when executing the script.

## Contribute

Feel free to open issues or PRs if you have suggestions or improvements for this tool!
