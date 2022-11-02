
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
        
        self.criterion = None
        self.optimizer = None
        
        
        # In: C x H x W
        # --------------------------------------------------------------------------------------------------------------------------
        # Maxpool layer >>
        self.pool  = nn.MaxPool2d( kernel_size = 2    # kernel : 2 x 2
                                 , stride      = 2 )  # stride : 2               [ Out: C x W/2 x H/2,   output size = W/2 ]
        # --------------------------------------------------------------------------------------------------------------------------
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
        # # Bacth normalization layers >>
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # --------------------------------------------------------------------------------------------------------------------------
        # Fully-connected layers:
        self.fc      = nn.Linear( in_features  = 128*26*26, out_features = 136 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        # --------------------------------------------------------------------------------------------------------------------------
        # self.drop   = nn.Dropout( p = 0.5 )
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
                
                x <torch.Tensor>
                
                Tensor of predicted keypoints of torch.Size([n_images, n_keypoints, 2]).
        
        ============================================================================
        END << DOC << forward
        ============================================================================
        """
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Conv/relu + pool + dropout layers >>
        
        # LeakyRelu
        # In:  224 x 244 x 1
        x = self.pool( F.elu( self.conv1(x) ) ) # out: 32  x 110 x 110
        x = self.pool( F.elu( self.conv2(x) ) ) # out: 64  x 54  x 54
        x = self.pool( F.elu( self.conv3(x) ) ) # out: 128 x 26  x 26
        
        # x = self.pool( F.elu( self.bn1( self.conv1(x) ) ) ) # out: 32  x 110 x 110
        # x = self.pool( F.elu( self.bn2( self.conv2(x) ) ) ) # out: 64  x 54  x 54
        # x = self.pool( F.elu( self.bn3( self.conv3(x) ) ) ) # out: 128 x 26  x 26
        
        # Prep for linear layer: this line of code is the equivalent of Flatten in Keras >>
        x = x.view(x.size(0), -1)
        # x = self.drop(x)
        
        # Linear layers with dropout in between >>
        x = self.fc(x)
        
        # Reshaping to batch_size x 68 x 2 pts
        x = x.view(x.size()[0], 68, -1)
        
        # Returns a modified x, having gone through all the layers of the model >>
        return x
    # <<
    # ==============================================================================================================================
    # END << METHOD << forward
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> sample_output
    # ==============================================================================================================================
    # >>
    def sample_output   ( self
                        , data_loader = None
                        ) :
        
        """
        ============================================================================
        START >> DOC >> sample_output
        ============================================================================
            
            GENERAL INFO
            ============
                
                Extract the image and ground truth keypoints from a sample and
                forward pass the image through the cnn model to get the predicted,
                output keypoints.
                
                Note: This function test how the network performs on the first batch
                      of test data. It returns the images, the transformed images,
                      the predicted keypoints (produced by the model), and the
                      ground truth keypoints.
            
            PARAMETERS
            ==========
                
                data_loader <torch.utils.data.dataloader.DataLoader>
                    
                    An iterable over the given dataset.
            
            RETURNS
            =======
                
                images <torch.Tensor>
                    
                    Tensor of images of torch.Size([n_images, 1, H, W]).
                
                keypoints_preds <torch.Tensor>
                    
                    Tensor of predicted keypoints of torch.Size([n_images, n_keypoints, 2]).
                
                keypoints_gts <torch.Tensor>
                    
                    Tensor of ground truth keypoints of torch.Size([n_images, n_keypoints, 2]).
        
        ============================================================================
        END << DOC << sample_output
        ============================================================================
        """
        
        # Iterating through the test dataset >>
        for i, sample in enumerate(data_loader):
            
            # Getting sample data: images and ground truth keypoints >>
            images        = sample['image']
            keypoints_gts = sample['keypoints']
            
            # Converting images to FloatTensors >>
            images = images.type(torch.FloatTensor)
            print(type(images))
            print(images.shape)
            
            # Forward pass to get net output >>
            keypoints_preds = self(images)
            
            # break after first image is tested >>
            if i == 0:
                return images, keypoints_preds, keypoints_gts
    # <<
    # ==============================================================================================================================
    # END << METHOD << sample_output
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
                
                    Loded model from input file.
        
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
    # START >> METHOD >> train
    # ==============================================================================================================================
    # >>
    def train   ( self
                , n_epochs     = 10
                , train_loader = 0
                ) :
        
        """
        ============================================================================
        START >> DOC >> train
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
        END << DOC << train
        ============================================================================
        """
        
        # Preparing the cnn model for training >>
        # net.train()
        if self.criterion is None: self.criterion = nn.SmoothL1Loss()
        if self.optimizer is None: self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        # Looping over the dataset multiple times >>
        list_loss = []
        for epoch in range(n_epochs):
            
            # Training on batches of data >>
            running_loss = 0.0
            for batch_i, data in enumerate(train_loader):
                
                # Getting the input images and their corresponding labels >>
                images  = data['image']
                key_pts = data['keypoints']
                
                # Flatten pts >>
                key_pts = key_pts.view(key_pts.size(0), -1)
                
                # Converting variables to floats for regression loss >>
                key_pts = key_pts.type(torch.FloatTensor)
                images  = images.type(torch.FloatTensor)
                
                # forward pass to get outputs
                output_pts = self.forward(images)
                output_pts = output_pts.view(output_pts.size(0), -1)
                
                # calculate the loss between predicted and target keypoints
                loss = self.criterion(output_pts, key_pts)
                
                # zero the parameter (weight) gradients
                self.optimizer.zero_grad()
                
                # backward pass to calculate the weight gradients
                loss.backward()
                
                # update the weights
                self.optimizer.step()
                
                # Printing loss statistics >>
                # to convert loss into a scalar and add it to the running_loss, use .item()
                running_loss = running_loss + loss.item()
                if batch_i % 20 == 19:    # print every 20 batches
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                    list_loss.append(running_loss)
                    running_loss = 0.0
                
        print('Finished Training')
        
        return list_loss
    # <<
    # ==============================================================================================================================
    # END << METHOD << train
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
