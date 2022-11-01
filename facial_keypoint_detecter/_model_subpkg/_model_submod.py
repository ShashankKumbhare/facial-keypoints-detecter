
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : facial_keypoint_detecter/_model_subpkg/_model_submod.py
# Author      : Shashank Kumbhare
# Date        : 30/10/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'facial_keypoint_detecter._model_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> facial_keypoint_detecter._model_subpkg._model_submod
# ==================================================================================================================================
# >>
"""
This module is created for the main facial-keypoint-detecter model class `Net`.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
from ..__data_subpkg__         import *
from .._plots_subpkg           import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["Net"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> CLASS >> Net
# ==================================================================================================================================
# >>
class Net(nn.Module):
    
    # ==============================================================================================================================
    # START >> METHOD >> __init__
    # ==============================================================================================================================
    # >>
    def __init__(self):
        
        super(Net, self).__init__()
        
        # --------------------------------------------------------------------------------------------------------------------------
        # Maxpool layer >>
        # In     : C x H   x W
        # Out    : C x W/2 x H/2
        self.pool  = nn.MaxPool2d( kernel_size = 2    # kernel : 2 x 2
                                 , stride      = 2 )  # stride : 2
        # --------------------------------------------------------------------------------------------------------------------------
        # # Maxpool layer 2 >>
        # # In     : C x H   x W
        # # Out    : C x W/4 x H/4
        # self.pool2 = nn.MaxPool2d( kernel_size = 4    # kernel : 4 x 4
        #                          , stride      = 4 )  # stride : 4
        # # --------------------------------------------------------------------------------------------------------------------------
        # Conv-1 >>
        self.conv1 = nn.Conv2d( in_channels  = 1      # In    :  1 x 224 x 224
                              , out_channels = 32     # Out   : 32 x 220 x 220   [ output size = (W-F)/S + 1 = (224-5)/1 + 1 = 220 ]
                              , kernel_size  = 3 )    # kernel:  3 x 3           [ 32 filters, n_parameters = (3*3*1+1)*32 = 320 ]
                                       # Out after max-polling: 32 x 110 x 110
        # --------------------------------------------------------------------------------------------------------------------------
        # Conv-2 >>
        self.conv2 = nn.Conv2d( in_channels  = 32     # In    : 32 x 110 x 110
                              , out_channels = 64     # Out   : 64 x 108 x 108   [ output size = (W-F)/S + 1 = (110-3)/1 + 1 = 108 ]
                              , kernel_size  = 3 )    # kernel:  3 x 3           [ 64 filters, n_parameters = (3*3*32+1)*64 = 18496 ]
                                       # Out after max-polling: 64 x 54 x 54
        # --------------------------------------------------------------------------------------------------------------------------
        # Conv-3 >>
        self.conv3 = nn.Conv2d( in_channels  = 64     # In    :  64 x 54 x 54
                              , out_channels = 128    # Out   : 128 x 52 x 52   [ output size = (W-F)/S + 1 = (54-3)/1 + 1 = 52 ]
                              , kernel_size  = 3 )    # kernel:   3 x 3         [ 128 filters, n_parameters = (3*3*64+1)*128 = 73856 ]
                                       # Out after max-pooling: 128 x 26 x 26
                                      # Out after max-pooling2: 128 x 13 x 13
        # --------------------------------------------------------------------------------------------------------------------------
        # # Conv-4 >>
        # self.conv4 = nn.Conv2d( in_channels  = 128    # In    : 128 x 26 x 26
        #                       , out_channels = 256    # Out   : 256 x 24 x 24   [ output size = (W-F)/S + 1 = (26-3)/1 + 1 = 24 ]
        #                       , kernel_size  = 3 )    # kernel:   3 x 3         [ 256 filters, n_parameters = (3*3*128+1)*256 = 295168 ]
        #                                # Out after max-polling: 256 x 12 x 12
        # # --------------------------------------------------------------------------------------------------------------------------
        # # Conv-5 >>
        # self.conv5 = nn.Conv2d( in_channels  = 256    # In    : 256 x 12 x 12
        #                       , out_channels = 512    # Out   : 512 x 12 x 12   [ output size = (W-F)/S + 1 = (12-1)/1 + 1 = 12 ]
        #                       , kernel_size  = 1 )    # kernel:   1 x 1         [ 512 filters, n_parameters = (1*1*256+1)*512 = 131584 ]
        #                                # Out after max-polling: 512 x 6  x 6
        # # --------------------------------------------------------------------------------------------------------------------------
        # # Bacth normalization layers >>
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # --------------------------------------------------------------------------------------------------------------------------
        # Fully-connected layers:
        self.fc      = nn.Linear( in_features  = 128*26*26, out_features = 136 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        
        # self.fc1      = nn.Linear( in_features  = 128*13*13, out_features = 100 )   # [ n_parameters = (128*13*13)*100 =  ]
        # self.fc2      = nn.Linear( in_features  = 100,       out_features = 200 )   # [ n_parameters =         100*200 = 20000 ]
        # self.fc3      = nn.Linear( in_features  = 200,       out_features = 136 )   # [ n_parameters =         200*136 = 27200 ]
        
        # self.fc1      = nn.Linear( in_features  = 512*6*6, out_features = 100 )   # [ n_parameters = (512*6*6)*1024 = 1857600 ]
        # self.fc2      = nn.Linear( in_features  =  100,    out_features = 200 )   # [ n_parameters =      100*200   =   20000 ]
        # self.fc3      = nn.Linear( in_features  =  200,    out_features = 136 )   # [ n_parameters =      200*136   =   27200 ]
        # --------------------------------------------------------------------------------------------------------------------------
        # Dropout layers:
        # self.drop_conv1 = nn.Dropout( p = 0.2 )
        # self.drop_conv2 = nn.Dropout( p = 0.2 )
        # self.drop_conv3 = nn.Dropout( p = 0.2 )
        # self.drop_conv4 = nn.Dropout( p = 0.2 )
        # self.drop_conv5 = nn.Dropout( p = 0.2 )
        # self.drop_fc1   = nn.Dropout( p = 0.5 )
        # self.drop_fc2   = nn.Dropout( p = 0.5 )
        
        self.drop   = nn.Dropout( p = 0.5 )
        
        # --------------------------------------------------------------------------------------------------------------------------
    
    # <<
    # ==============================================================================================================================
    # END << METHOD << __init__
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> forward
    # ==============================================================================================================================
    # >>
    def forward( self, x):
        
        """
        ============================================================================
        START >> DOC >> forward
        ============================================================================
            
            GENERAL INFO
            ============
                
                Forward pass the image through the cnn model to get the predicted,
                output keypoints.
            
            PARAMETERS
            ==========
                
                x <torch.Tensor>
                    
                    Tensor of image of torch.Size([1, 1, H, W]).
            
            RETURNS
            =======
                
                x <np.array>
                
                Numpy array of predicted keypoints of shape (n_keypoints, 2).
        
        ============================================================================
        END << DOC << forward
        ============================================================================
        """
        
        # Conv/relu + pool + dropout layers >>
        
        # LeakyRelu
        # In:  224 x 244 x 1
        x = self.pool( F.elu( self.conv1(x) ) ) # out: 32  x 110 x 110
        x = self.pool( F.elu( self.conv2(x) ) ) # out: 64  x 54  x 54
        x = self.pool( F.elu( self.conv3(x) ) ) # out: 128 x 26  x 26
        # x = self.pool( F.elu( self.conv4(x) ) ) # out: 256 x 12  x 12
        # x = self.pool( F.elu( self.conv5(x) ) ) # out: 512 x 6   x 6
        
        # x = self.pool( F.elu( self.bn1( self.conv1(x) ) ) ) # out: 32  x 110 x 110
        # x = self.pool( F.elu( self.bn2( self.conv2(x) ) ) ) # out: 64  x 54  x 54
        # x = self.pool( F.elu( self.bn3( self.conv3(x) ) ) ) # out: 128 x 26  x 26
        # x = self.pool( F.elu( self.bn4( self.conv4(x) ) ) ) # out: 256 x 12  x 12
        # x = self.pool( F.elu( self.bn5( self.conv5(x) ) ) ) # out: 512 x 6   x 6
        
        # x = self.drop_conv1( self.pool( F.elu( self.bn1( self.conv1(x) ) ) ) ) # out: 32  x 110 x 110
        # x = self.drop_conv2( self.pool( F.elu( self.bn2( self.conv2(x) ) ) ) ) # out: 64  x 54  x 54
        # x = self.drop_conv3( self.pool( F.elu( self.bn3( self.conv3(x) ) ) ) ) # out: 128 x 26  x 26
        # x = self.drop_conv4( self.pool( F.elu( self.bn4( self.conv4(x) ) ) ) ) # out: 256 x 12  x 12
        # x = self.drop_conv5( self.pool( F.elu( self.bn5( self.conv5(x) ) ) ) ) # out: 512 x 6   x 6
        
        # # In:  224 x 244 x 1
        # x = self.pool(  F.elu( self.conv1(x) ) ) # out: 32  x 110 x 110
        # x = self.pool(  F.elu( self.conv2(x) ) ) # out: 64  x 54  x 54
        # # x = self.pool(  F.elu( self.conv3(x) ) ) # out: 128 x 26  x 26
        # x = self.pool2( F.elu( self.conv3(x) ) ) # out: 128 x 13  x 13
        
        # Prep for linear layer: this line of code is the equivalent of Flatten in Keras >>
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        
        # Linear layers with dropout in between >>
        x = self.fc(x)
        
        # x = F.relu( self.fc1(x) )
        # x = F.relu( self.fc2(x) )
        # x = self.fc3(x)
        
        # x = F.elu( self.fc1(x) )
        # x = self.fc2(x)
        # x = self.fc3(x)
        
        # A modified x, having gone through all the layers of the model >>
        return x
    # <<
    # ==============================================================================================================================
    # END << METHOD << forward
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> load_model
    # ==============================================================================================================================
    # >>
    def load_model  ( self
                    , f = None
                    ) :
        
        """
        ============================================================================
        START >> DOC >> load_model
        ============================================================================
            
            GENERAL INFO
            ============
                
                Copies parameters and buffers from :attr:`state_dict` into
                this module and its descendants by loading an object saved with
                :func:`torch.save` from the input file.
            
            PARAMETERS
            ==========
                
                file <str>
                    
                    A file-like object, or a string or os.PathLike object containing
                    a file name.
            
            RETURNS
            =======
                
                self
                
                    Loded model.
        
        ============================================================================
        END << DOC << load_model
        ============================================================================
        """
        
        self.load_state_dict(torch.load(f))
        self.eval()
        
        return self
    # <<
    # ==============================================================================================================================
    # END << METHOD << load_model
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _template_method
    # ==============================================================================================================================
    # >>
    def _template_method    ( self
                            , x = None
                            ) :
        
        """
        ============================================================================
        START >> DOC >> _template_method
        ============================================================================
            
            GENERAL INFO
            ============
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t
            
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
        
        ============================================================================
        END << DOC << _template_method
        ============================================================================
        """
        
        print(x)
        
        return None
    # <<
    # ==============================================================================================================================
    # END << METHOD << _template_method
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << Net
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter._model_subpkg._model_submod
# ==================================================================================================================================
