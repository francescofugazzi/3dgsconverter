"""
3D Gaussian Splatting Converter
Copyright (c) 2026 Francesco Fugazzi (@franzipol)

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import signal
from . import config

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def debug_print(*args, **kwargs):
    if config.DEBUG:
        status_print(*args, **kwargs)

def status_print(*args, **kwargs):
    """
    Prints a message to the console in a way that plays nicely with tqdm progress bars.
    Always prints, unlike debug_print which respects the DEBUG flag.
    """
    try:
        from tqdm import tqdm
        # tqdm.write automatically handles formatting
        tqdm.write(" ".join(map(str, args)), **kwargs)
    except ImportError:
        print(*args, **kwargs)