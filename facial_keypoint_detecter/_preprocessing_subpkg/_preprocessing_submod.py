
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/_template_subpkg/_template_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter._template_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter._template_subpkg._template_submod
# ==================================================================================================================================
# >>
"""
This submodule is created for the pre-processing of the dataset.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
# from ..__data_subpkg__         import *
from .._plots_subpkg           import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["Normalize", "Rescale", "RandomCrop", "ToTensor", "Compose"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================

Compose = transforms.Compose

# ==================================================================================================================================
# START >> CLASS >> Normalize
# ==================================================================================================================================
# >>
class Normalize:
    
    """
    ================================================================================
    START >> DOC >> Normalize
    ================================================================================
        
        GENERAL INFO
        ============
            
            Convert a color image to grayscale and normalize the color range to [0,1].
        
        PARAMETERS
        ==========
            
            None
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << Normalize
    ================================================================================
    """
    
    def __call__(self, sample):
        
        """
        ============================================================================
        START >> DOC >> __call__
        ============================================================================
            
            GENERAL INFO
            ============
                
                Converts the input color image to grayscale and normalizes the color
                range to [0,1].
            
            PARAMETERS
            ==========
                
                sample <dict>
                    
                    Dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
            
            RETURNS
            =======
                
                sample_normalized <dict>
                    
                    Normalized sample dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
        
        ============================================================================
        END << DOC << __call__
        ============================================================================
        """
        
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy   = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        # Converting image to grayscale >>
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        
        # Scaling values range from [0, 255] to [0, 1] >>
        image_copy =  image_copy / 255.0
        
        # Scaling keypoints to be centered around 0 with a range of [-1, 1] >>
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0
        
        # Creating sample_normalized dictionary >>
        sample_normalized = {'image': image_copy, 'keypoints': key_pts_copy}
        
        return sample_normalized
    
# <<
# ==================================================================================================================================
# END << CLASS << Normalize
# ==================================================================================================================================



# ==================================================================================================================================
# START >> CLASS >> Rescale
# ==================================================================================================================================
# >>
class Rescale:
    
    """
    ================================================================================
    START >> DOC >> Rescale
    ================================================================================
        
        GENERAL INFO
        ============
            
            Rescale the image in a sample to a given size.
        
        PARAMETERS
        ==========
            
            output_size <tuple or int>
                
                Desired output size.
                If tuple, output is matched to output_size.
                If int, smaller of image edges is matched to output_size maintining aspect ratio.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << Rescale
    ================================================================================
    """
    
    # ==============================================================================================================================
    # START >> METHOD >> __init__
    # ==============================================================================================================================
    # >>
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    # <<
    # ==============================================================================================================================
    # END << METHOD << __init__
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> __call__
    # ==============================================================================================================================
    # >>
    def __call__(self, sample):
        
        """
        ============================================================================
        START >> DOC >> __call__
        ============================================================================
            
            GENERAL INFO
            ============
                
                Rescales the image in the input sample to a given size.
            
            PARAMETERS
            ==========
                
                sample <dict>
                    
                    Dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
            
            RETURNS
            =======
                
                sample_rescaled <dict>
                    
                    Scaled sample dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
        
        ============================================================================
        END << DOC << __call__
        ============================================================================
        """
        
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        image_rescaled = cv2.resize(image, (new_w, new_h))
        
        # Scaling the pts, too >>
        key_pts_rescaled = key_pts * [new_w / w, new_h / h]
        
        # Creating sample_scaled dictionary >>
        sample_rescaled = {'image': image_rescaled, 'keypoints': key_pts_rescaled}
        
        return sample_rescaled
    # <<
    # ==============================================================================================================================
    # END << METHOD << __call__
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << Rescale
# ==================================================================================================================================



# ==================================================================================================================================
# START >> CLASS >> RandomCrop
# ==================================================================================================================================
# >>
class RandomCrop:
    
    """
    ================================================================================
    START >> DOC >> RandomCrop
    ================================================================================
        
        GENERAL INFO
        ============
            
            Crops randomly the image in an input sample.
        
        PARAMETERS
        ==========
            
            output_size <tuple or int>
                
                Desired output size.
                If tuple, output is matched to output_size.
                If int, square crop is made.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << RandomCrop
    ================================================================================
    """
    
    # ==============================================================================================================================
    # START >> METHOD >> __init__
    # ==============================================================================================================================
    # >>
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    # <<
    # ==============================================================================================================================
    # END << METHOD << __init__
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> __call__
    # ==============================================================================================================================
    # >>
    def __call__(self, sample):
        
        """
        ============================================================================
        START >> DOC >> __call__
        ============================================================================
            
            GENERAL INFO
            ============
                
                Crops randomly the image in the input sample to a given size.
            
            PARAMETERS
            ==========
                
                sample <dict>
                    
                    Dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
            
            RETURNS
            =======
                
                sample_cropped <dict>
                    
                    Cropped sample dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
        
        ============================================================================
        END << DOC << __call__
        ============================================================================
        """
        
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top  = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[ top : top  + new_h
                     , left: left + new_w ]

        key_pts = key_pts - [left, top]
        
        # Creating sample_cropped dictionary >>
        sample_cropped = {'image': image, 'keypoints': key_pts}
        
        return sample_cropped
    # <<
    # ==============================================================================================================================
    # END << METHOD << __call__
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << RandomCrop
# ==================================================================================================================================



# ==================================================================================================================================
# START >> CLASS >> ToTensor
# ==================================================================================================================================
# >>
class ToTensor:
    
    """
    ================================================================================
    START >> DOC >> ToTensor
    ================================================================================
        
        GENERAL INFO
        ============
            
            Convert ndarrays in sample to Tensors.
        
        PARAMETERS
        ==========
            
            None
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << ToTensor
    ================================================================================
    """
    
    # ==============================================================================================================================
    # START >> METHOD >> __call__
    # ==============================================================================================================================
    # >>
    def __call__(self, sample):
        
        """
        ============================================================================
        START >> DOC >> __call__
        ============================================================================
            
            GENERAL INFO
            ============
                
                Converts ndarrays in the input sample to Tensors.
            
            PARAMETERS
            ==========
                
                sample <dict>
                    
                    Dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
            
            RETURNS
            =======
                
                sample_tensor <dict>
                    
                    Tensor sample dictionary with,
                    'image'    : <np.array> [rgb image of shape (n_row, n_col, 3).]
                    'keypoints': <np.array> [keypoints of shape (n_keypoints, 2).]
        
        ============================================================================
        END << DOC << __call__
        ============================================================================
        """
        
        image, key_pts = sample['image'], sample['keypoints']
        
        # if image has no grayscale color channel, add one >>
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # Swapping color axis because >>
        # numpy image: H x W x C
        # torch image: C X H X W
        image_transposed = image.transpose((2, 0, 1))
        
        # Creating sample_cropped dictionary >>
        sample_tensor = {'image': torch.from_numpy(image_transposed), 'keypoints': torch.from_numpy(key_pts)}
        
        return sample_tensor
    # <<
    # ==============================================================================================================================
    # END << METHOD << __call__
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << ToTensor
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> _template_submod_func
# ==================================================================================================================================
# >>
def _template_submod_func   ( p_p_p_p_1 = ""
                            , p_p_p_p_2 = ""
                            ) :
    
    """
    ================================================================================
    START >> DOC >> _template_submod_func
    ================================================================================
        
        GENERAL INFO
        ============
            
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
        
        PARAMETERS
        ==========
            
            p_p_p_p_1 <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
            
            p_p_p_p_2 <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
        
        RETURNS
        =======
            
            r_r_r_r <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
    
    ================================================================================
    END << DOC << _template_submod_func
    ================================================================================
    """
    
    _name_func = inspect.stack()[0][3]
    print(f"This is a print from '{_name_func}'{p_p_p_p_1}{p_p_p_p_2}.")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << _template_submod_func
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter._template_subpkg._template_submod
# ==================================================================================================================================
