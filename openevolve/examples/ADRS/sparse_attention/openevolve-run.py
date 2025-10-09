#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import torch
import torch.multiprocessing as mp

import sys
from openevolve.cli import main

if __name__ == "__main__":
    mp.set_start_method('spawn')
    sys.exit(main())