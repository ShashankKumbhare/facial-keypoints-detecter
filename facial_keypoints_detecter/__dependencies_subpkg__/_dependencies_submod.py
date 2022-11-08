
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/__dependencies_subpkg__/_dependencies_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter.__dependencies_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter.__dependencies_subpkg__._dependencies_submod
# ==================================================================================================================================
# >>
"""
This submodule imports all the required 3rd party dependency packages/libraries for
the package which are then shared across all the package modules & submodules. All
the 3rd party dependency packages are imported here at one place and any other
dependencies are not to be imported in any module or submodule other than this
submodule.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('classic')
style.use('seaborn-white')
from matplotlib.ticker import MultipleLocator
import math
import os
import sys
import inspect
import glob                                 # library for loading images from a directory
import matplotlib.image as mpimg
import cv2
from IPython.display import Markdown, display, Latex
from textwrap import wrap
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
import imutils
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter.__dependencies_subpkg__._dependencies_submod
# ==================================================================================================================================
