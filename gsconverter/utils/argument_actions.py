"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import argparse

class DensityFilterAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values:
            if len(values) != 2:
                parser.error("--density_filter requires two numbers: voxel_size and threshold_percentage.")
            try:
                values = [float(v) for v in values]
            except ValueError:
                parser.error("Both arguments for --density_filter must be numbers.")
        else:
            values = [1.0, 0.32]  # Default values if none are provided
        setattr(args, self.dest, values)
        
class RemoveFlyersAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values:
            if len(values) != 2:
                parser.error("--remove_flyers requires two numbers: 'k' for the number of neighbors and 'threshold_factor' for the multiplier of the standard deviation.")
            try:
                values = [float(v) for v in values]
            except ValueError:
                parser.error("Both arguments for --remove_flyers must be numbers.")
        else:
            values = [25, 10.5]  # Default values if none are provided
        setattr(args, self.dest, values)
        
class AboutAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(AboutAction, self).__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        copyright_info = """
        3D Gaussian Splatting Converter
        Copyright (c) 2023 Francesco Fugazzi

        This software is released under the MIT License.
        For more information about the license, please see the LICENSE file.
        """
        print(copyright_info)
        parser.exit()  # Exit after displaying the information.
