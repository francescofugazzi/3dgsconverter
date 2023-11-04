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