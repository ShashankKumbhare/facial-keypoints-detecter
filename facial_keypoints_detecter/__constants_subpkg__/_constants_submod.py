
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/__constants_subpkg__/_constants_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter.__constants_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter.__constants_subpkg__._constants_submod
# ==================================================================================================================================
# >>
"""
This submodule stores all the required constants for the package.
These constants will then be shared across all the package modules & submodules.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> CLASS >> Struct
# ==================================================================================================================================
# >>
class Struct:
    pass
# <<
# ==================================================================================================================================
# END << CLASS << Struct
# ==================================================================================================================================


# ==================================================================================================================================
# START >> CONSTANTS >> strings related
# ==================================================================================================================================
# >>
tab2    = "  "
tab4    = "    "
newline = "\n"
DEFAULT_DEFAULT_STR = "default"
# <<
# ==================================================================================================================================
# END << CONSTANTS << strings related
# ==================================================================================================================================


# ==================================================================================================================================
# START >> CONSTANTS >> _plots_subpkg related
# ==================================================================================================================================
# >>
DEFAULT_FIGSIZE        = 3.33
DEFAULT_FIGSIZE_OUTPUT = 10
DEFAULT_FIGSIZESCALE   = 1
DEFAULT_NAME_IMAGE     = ""
DEFAULT_CMAP           = "gray" # "viridis"
DEFAULT_ALPHA          = 0.4
DEFAULT_KEYPTS_MARKER_SHAPE = "."
DEFAULT_KEYPTS_MARKER_SIZE  = 20
DEFAULT_KEYPTS_MARKER_COLOR_GT   = "m"
DEFAULT_KEYPTS_MARKER_COLOR_PRED = "#00FF00" # "#00CFAF" # "#9578FF"
DEFAULT_SIZE_BOX_FACE_DETECTED = 5
DEFAULT_N_FILTERS_TO_PLOT      = 10
# <<
# ==================================================================================================================================
# END << CONSTANTS << _plots_subpkg related
# ==================================================================================================================================


# ==================================================================================================================================
# START >> CONSTANTS >> __data_subpkg__ related
# ==================================================================================================================================
# >>
# <<
# ==================================================================================================================================
# END << CONSTANTS << __data_subpkg__ related
# ==================================================================================================================================


# ==================================================================================================================================
# START >> CONSTANTS >> _preprocessing_subpkg related
# ==================================================================================================================================
# >>
DEFAULT_PREPROCESS_SIZE_RESCALE    = 250
DEFAULT_PREPROCESS_SIZE_RANDOMCROP = 224
DEFAULT_PREPROCESS_SCALING_MEAN    = 100.0
DEFAULT_PREPROCESS_SCALING_SQRT    = 50.0
DEFAULT_PREPROCESS_ROTATE_ANGLE    = 10
# <<
# ==================================================================================================================================
# END << CONSTANTS << _preprocessing_subpkg related
# ==================================================================================================================================


# ==================================================================================================================================
# START >> CONSTANTS >> Model related
# ==================================================================================================================================
# >>
DEFAULT_CRITERION   = nn.SmoothL1Loss # nn.MSELoss
DEFAULT_OPTIMIZER   = optim.Adam      # optim.SGD
DEFAULT_LR          = 0.001
DEFAULT_NUM_WORKERS = 1
DEFAULT_N_EPOCHS    = 10
DEFAULT_BATCH_SIZE  = 10
DEFAULT_SHUFFLE     = True
DEFAULT_PADDING     = 70
DEFAULT_N_BATCH_TO_PRINT_LOSS = 20
path_file = os.path.abspath((inspect.stack()[0])[1])
path_dir  = os.path.dirname(path_file)
DEFAULT_FILE_MODEL_HARR_CASCADE = f"{path_dir}/detector_architectures/haarcascade_frontalface_default.xml"
FACE_HARR_CASCADE               = cv2.CascadeClassifier(DEFAULT_FILE_MODEL_HARR_CASCADE)
DEFAULT_FILE_FKD_NET_MODEL      = f"{path_dir}/saved_models/saved_model_4_adam_relu.pt"
DEFAULT_COLOR_BOX_DETECTED_FACE = (0, 255, 255) # (0, 207, 175) # (238, 120, 255) # (255, 0, 0)
DEFAULT_HARR_SCALE_FACTOR       = 1.3
DEFAULT_HARR_MIN_NEIGHBOURS     = 5
DEFAULT_FILE_FILTERS_SUNGLASSES = f"{path_dir}/filters/sunglasses.png"
DEFAULT_FILE_FILTERS_MOUSTACHE  = f"{path_dir}/filters/moustache.png"
# <<
# ==================================================================================================================================
# END << CONSTANTS << Model related
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter.__constants_subpkg__._constants_submod
# ==================================================================================================================================
