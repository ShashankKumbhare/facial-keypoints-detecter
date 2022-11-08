
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/__init__.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a __init__ file for python package 'facial_keypoint_detecter'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> PACKAGE >> facial_keypoint_detecter
# ==================================================================================================================================
# >>s
"""
PACKAGE description PACKAGE description PACKAGE description PACKAGE description
PACKAGE description PACKAGE description.
"""

__version__  = '1.0.0'
_name_pkg    = __name__.partition(".")[0]
print("")
print(f"==========================================================================")
print(f"Importing package '{_name_pkg}'...")
print(f"==========================================================================")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# MODULES >>
from .                        import auxil
from .                        import plots
from .                        import data
from .                        import model
from .                        import preprocessing
# ELEMENTS >>
# from .datasets                import datasets, FacialKeypointsDataset
# from .model                   import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================

print(f"==========================================================================")
print(f"Package '{_name_pkg}' imported sucessfully !!")
print(f"==========================================================================")
print(f"version {__version__}")
print("")

# <<
# ==================================================================================================================================
# END << PACKAGE << facial_keypoint_detecter
# ==================================================================================================================================
