
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/__data_subpkg__/_datasets_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter.__data_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter.__data_subpkg__._datasets_submod
# ==================================================================================================================================
# >>
"""
This submodule is created to manage all the datasets related variables.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
from .._preprocessing_subpkg   import *
from ._load_dataset_submod     import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["datasets"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> DATASET
# ==================================================================================================================================
# >>
datasets       = Struct()
datasets.train = Struct()
datasets.test  = Struct()

# Assigning dataset directories >>
path_file = os.path.abspath((inspect.stack()[0])[1])
path_dir  = os.path.dirname(path_file)
datasets.train._dir = f"{path_dir}/dataset_train"
datasets.test._dir  = f"{path_dir}/dataset_test"

# Loading raw datasets >>
datasets.train.raw = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_train.csv",
                                             root_dir  = f"{path_dir}/dataset_train",
                                             transform = None )

datasets.test.raw  = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_test.csv",
                                             root_dir  = f"{path_dir}/dataset_test",
                                             transform = None )

# Pre-process raw datasets >>
transform = Compose( [ Rescale(DEFAULT_PREPROCESS_SIZE_RESCALE)
                     , RandomCrop(DEFAULT_PREPROCESS_SIZE_RANDOMCROP)
                     , Normalize()
                     , ToTensor() ] )

datasets.train.preprocessed = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_train.csv",
                                                      root_dir  = f"{path_dir}/dataset_train",
                                                      transform = transform )

datasets.test.preprocessed  = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_test.csv",
                                                      root_dir  = f"{path_dir}/dataset_test",
                                                      transform = transform )


# Augmented data: Ratated anti-clockwise >>
transform_rotate_anti_clockwise = Compose(  [ Rescale(DEFAULT_PREPROCESS_SIZE_RESCALE)
                                            , RandomCrop(DEFAULT_PREPROCESS_SIZE_RANDOMCROP)
                                            , Normalize()
                                            , Rotate(DEFAULT_PREPROCESS_ROTATE_ANGLE)
                                            , Rescale(DEFAULT_PREPROCESS_SIZE_RANDOMCROP)
                                            , ToTensor() ] )

datasets.train.augmented_rotate_left = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_train.csv",
                                                               root_dir  = f"{path_dir}/dataset_train",
                                                               transform = transform_rotate_anti_clockwise )

datasets.test.augmented_rotate_left  = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_test.csv",
                                                               root_dir  = f"{path_dir}/dataset_test",
                                                               transform = transform_rotate_anti_clockwise )

# Augmented data: Ratated clockwise >>
transform_rotate_clockwise = Compose(   [ Rescale(DEFAULT_PREPROCESS_SIZE_RESCALE)
                                        , RandomCrop(DEFAULT_PREPROCESS_SIZE_RANDOMCROP)
                                        , Normalize()
                                        , Rotate(-DEFAULT_PREPROCESS_ROTATE_ANGLE)
                                        , Rescale(DEFAULT_PREPROCESS_SIZE_RANDOMCROP)
                                        , ToTensor() ] )

datasets.train.augmented_rotate_right = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_train.csv",
                                                                root_dir  = f"{path_dir}/dataset_train",
                                                                transform = transform_rotate_clockwise )

datasets.test.augmented_rotate_right  = FacialKeypointsDataset( csv_file  = f"{path_dir}/keypoints_frames_test.csv",
                                                                root_dir  = f"{path_dir}/dataset_test",
                                                                transform = transform_rotate_clockwise )

# Overall combined dataset >>
datasets.train.combined = ConcatDataset( [ datasets.train.preprocessed
                                         , datasets.train.augmented_rotate_left
                                         , datasets.train.augmented_rotate_right ] )

datasets.test.combined = ConcatDataset(  [ datasets.test.preprocessed
                                         , datasets.test.augmented_rotate_left
                                         , datasets.test.augmented_rotate_right ] )

# <<
# ==================================================================================================================================
# END << DATASET
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter.__data_subpkg__._datasets_submod
# ==================================================================================================================================
