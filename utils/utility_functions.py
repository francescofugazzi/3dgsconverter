"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import signal
from . import config

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def debug_print(message):
    if config.DEBUG:
        print(message)