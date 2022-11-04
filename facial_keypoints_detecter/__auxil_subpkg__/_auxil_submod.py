
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/__auxil_subpkg__/_auxil_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter.__auxil_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter.__auxil_subpkg__._auxil_submod
# ==================================================================================================================================

# >>
"""
This submodule contains some auxiliary functions being used in rest of the modules
and submodules.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__constants_subpkg__    import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["detect_faces"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> detect_faces
# ==================================================================================================================================
# >>
def detect_faces( file_image
                , plot_enabled = False
                , figsizeScale = DEFAULT_FIGSIZESCALE
                ) :
    
    """
    ================================================================================
    START >> DOC >> detect_faces
    ================================================================================
        
        GENERAL INFO
        ============
            
            Detects faces in the input images using HAAR-cascade classifier for
            frontal faces.
        
        PARAMETERS
        ==========
            
            file_image <str>
                    
                    File path of the input image.
            
            plot_enabled <bool>
                    
                    When enabled plots the detected facial keypoints.
        
        RETURNS
        =======
            
            faces <np.ndarray>
                
                Numpy array of 4 points for each face detected indicating top-left
                corner x & y pos, width and height of bounding rectangles of shape
                (n_faces, 2).
    
    ================================================================================
    END << DOC << detect_faces
    ================================================================================
    """
    
    # Loading in color image for face detection >>
    image_bgr = cv2.imread(file_image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Running the haar cascade classifier for detecting frontal faces >>
    faces = FACE_HARR_CASCADE.detectMultiScale(image_rgb, scaleFactor=1.3, minNeighbors=5)
    
    # Making a copy of the original image to plot detections on >>
    image_with_detections = image_rgb.copy()
    
    # Looping over the detected faces, mark the image where each face is found >>
    if plot_enabled:
        for (x,y,w,h) in faces:
            # Drawing a rectangle around each detected face >>
            cv2.rectangle( image_with_detections,(x,y),(x+w,y+h), (150, 120, 255), DEFAULT_SIZE_BOX_FACE_DETECTED )
        _ = plt.figure( figsize = (figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE) )
        plt.imshow(image_with_detections)
    
    return faces
# <<
# ==================================================================================================================================
# END << FUNCTION << detect_faces
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
# END << SUBMODULE << facial_keypoint_detecter.__auxil_subpkg__._auxil_submod
# ==================================================================================================================================
