''' Main file for generating MT dataset '''

import os
import sys

import bpy

# to import own modules
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from gen_options import GenOptions

# parse arguments
try:
    args = sys.argv[sys.argv.index('--') + 1:]
except ValueError:
    args = []
opt = GenOptions().parse(args)
