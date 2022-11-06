
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/_template_subpkg/_plots_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter._plots_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter._plots_subpkg._plots_submod
# ==================================================================================================================================
# >>
"""
This submodule is created for the visualization and analyzsis of the dataset.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["plot_keypoints", "plot_output", "plot_features"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> show_keypoints
# ==================================================================================================================================
# >>
def plot_keypoints  ( image
                    , keypoints_gt   = None
                    , keypoints_pred = None
                    , cmap           = None
                    , axes           = None
                    , figsizeScale   = DEFAULT_FIGSIZESCALE
                    , title_enabled  = True
                    , title          = DEFAULT_NAME_IMAGE
                    ) :
    
    """
    ================================================================================
    START >> DOC >> plot_keypoints
    ================================================================================
        
        GENERAL INFO
        ============
            
            Plots image with keypoints.
        
        PARAMETERS
        ==========
            
            image <np.array>
                
                Numpy array of rgb image of shape (H, W, 3).
            
            keypoints_gt <np.array>
                
                Numpy array of ground truth keypoints of shape (n_keypoints, 2).
            
            keypoints_pred <np.array>
                
                Numpy array of predicted keypoints of shape (n_keypoints, 2).
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_keypoints
    ================================================================================
    """
    
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize = (figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE))
    # else:
    #     plt.gcf().set_size_inches( (figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE) )
    
    axes.imshow(image, cmap = cmap)
    
    if keypoints_gt is not None:
        axes.scatter(keypoints_gt[:, 0],   keypoints_gt[:, 1],   s=DEFAULT_KEYPTS_MARKER_SIZE, marker=DEFAULT_KEYPTS_MARKER_SHAPE, c=DEFAULT_KEYPTS_MARKER_COLOR_GT)
    
    if keypoints_pred is not None:
        axes.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1], s=DEFAULT_KEYPTS_MARKER_SIZE, marker=DEFAULT_KEYPTS_MARKER_SHAPE, c=DEFAULT_KEYPTS_MARKER_COLOR_PRED)
    
    if title_enabled:
        axes.set_title(title)
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_keypoints
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> plot_output
# ==================================================================================================================================
# >>
def plot_output ( images
                , keypoints_preds = None
                , keypoints_gts   = None
                , batch_size      = 10
                , figsizeScale    = DEFAULT_FIGSIZESCALE
                ) :
    
    """
    ================================================================================
    START >> DOC >> plot_output
    ================================================================================
        
        GENERAL INFO
        ============
            
            Displays a set of images and their ground truth/predicted keypoints.
            This function's main role is to take batches of image and keypoint data
            (the input and output of CNN), and transform them into numpy images and
            un-normalized keypoints (x, y) for normal display.
            The un-transformation process turns keypoints and images into numpy
            arrays from Tensors, and it undoes the keypoint normalization done in
            the Normalize() transform; it's assumed that transformations were
            applied when loaded test data.
        
        PARAMETERS
        ==========
            
            images <torch.Tensor>
                
                Tensor of images of torch.Size([n_images, 1, H, W]).
            
            keypoints_preds <torch.Tensor>
                
                Tensor of predicted keypoints of torch.Size([n_images, n_keypoints, 2]).
            
            keypoints_gts <torch.Tensor>
                
                Tensor of ground truth keypoints of torch.Size([n_images, n_keypoints, 2]).
            
            batch_size <int>
                
                Number of images to plot.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_output
    ================================================================================
    """
    
    if len(images.shape) == 3: images = images.unsqueeze(0)
    if keypoints_preds is not None and len(keypoints_preds.shape) == 2: keypoints_preds = keypoints_preds.unsqueeze(0)
    if keypoints_gts   is not None and len(keypoints_gts.shape)   == 2: keypoints_gts   = keypoints_gts.unsqueeze(0)
    
    # Deciding the no. of rows and columns of the grid plot >>
    n_plots = min(len(images), batch_size)
    if n_plots < 11:
        n_col = 5 if n_plots >= 6 else n_plots
        # n_col = n_plots
    else:
        n_col = int( np.ceil(np.sqrt(n_plots)) )
    n_row = int( np.ceil(n_plots/n_col) )
    
    # Creating plot axes >>
    figsize    = (figsizeScale*n_col*DEFAULT_FIGSIZE, figsizeScale*n_row*DEFAULT_FIGSIZE)
    f, axes    = plt.subplots(n_row, n_col, figsize = figsize)
    
    # Loop for all images along the grid >>
    i_image = 0
    for i in range(n_row):
        
        if n_col == 1:
            ax = axes
        else:
            ax = axes[i] if n_row != 1 else axes
        
        for j in range(n_col):
            
            if n_plots == 1:
                ax_curr = ax
            else:
                ax_curr = ax[j]
            
            # Un-transforming the image data >>
            image = images[i_image].data             # get the image from its wrapper
            image = image.numpy()                    # convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
            
            # Un-transforming the predicted key_pts data >>
            key_pts_pred = None
            if keypoints_preds is not None:
                # Un-transforming the predicted key_pts data >>
                key_pts_pred = keypoints_preds[i_image].data
                key_pts_pred = key_pts_pred.numpy()
                # Undoing normalization of keypoints >>
                key_pts_pred = key_pts_pred*DEFAULT_PREPROCESS_SCALING_SQRT + DEFAULT_PREPROCESS_SCALING_MEAN
            
            # Un-transforming the ground truth key_pts data >>
            key_pts_gt = None
            if keypoints_gts is not None:
                key_pts_gt = keypoints_gts[i_image].data
                key_pts_gt = key_pts_gt.numpy()
                # Undoing normalization of keypoints >>
                key_pts_gt = key_pts_gt*DEFAULT_PREPROCESS_SCALING_SQRT + DEFAULT_PREPROCESS_SCALING_MEAN
            
            # Plotting all keypoints >>
            plot_keypoints(np.squeeze(image), key_pts_gt, key_pts_pred, cmap = "gray", axes = ax_curr)
            
            i_image = i_image + 1
            if i_image >= n_plots:
                break

    plt.show()
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_output
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> plot_features
# ==================================================================================================================================
# >>
def plot_features   ( image_gray_tensor
                    , weights
                    , n_features_to_plot = DEFAULT_N_FILTERS_TO_PLOT
                    , figsizeScale       = DEFAULT_FIGSIZESCALE
                    ) :
    
    """
    ================================================================================
    START >> DOC >> plot_features
    ================================================================================
        
        GENERAL INFO
        ============
            
            Plots convolution filter-kernels and filter-maps.
        
        PARAMETERS
        ==========
            
            image_gray_tensor <torch.Tensor>
                
                Tensor of gray image of torch.Size([1, H, W]).
            
            weights <torch.Tensor>
                
                Kernels of convolution filters of shape torch.Size([n_filters, 1, k, k]).
                i.e. (net.conv.weight.data)
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_features
    ================================================================================
    """
    
    image_numpy = image_gray_tensor.numpy()[0]
    weights     = weights.numpy()
    
    n_filters = weights.shape[0]
    
    # Deciding the no. of rows and columns of the grid plot >>
    n_plots = min(2*n_filters, 2*n_features_to_plot)
    n_col = n_plots if n_plots < 11 else 12
    n_row = int( np.ceil(n_plots/n_col) )
    
    # Creating plot axes >>
    figsize    = (figsizeScale*n_col*DEFAULT_FIGSIZE, figsizeScale*n_row*DEFAULT_FIGSIZE)
    f, axes    = plt.subplots(n_row, n_col, figsize = figsize)
    
    # Loop for all images along the grid >>
    i_filter = 0
    for i in range(n_row):
        
        if n_row == 1:
            ax = axes
        else:
            ax = axes[i]
            
        for j in range(0,n_col,2):
            
            # Plotting filter >>
            filter = weights[i_filter][0]
            ax[j].imshow(filter, cmap='gray')
            ax[j].set_title(f"filter #{i_filter}")
            
            # Plotting feature map >>
            image_filtered = cv2.filter2D(image_numpy, -1, filter)
            ax[j+1].imshow(image_filtered, cmap='gray')
            ax[j+1].set_title(f"feature map #{i_filter}")
            
            i_filter = i_filter + 1
            if i_filter >= min(n_filters, n_features_to_plot):
                break
    
    f.tight_layout()
    plt.show()
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_features
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
# END << SUBMODULE << facial_keypoint_detecter._plots_subpkg._plots_submod
# ==================================================================================================================================
