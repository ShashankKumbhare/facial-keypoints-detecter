
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
        # Fully-connected layers >>
        self.fc       = nn.Linear( in_features  = 128*26*26, out_features = 500 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        self.fc2      = nn.Linear( in_features  =       500, out_features = 136 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        # --------------------------------------------------------------------------------------------------------------------------
        # Dropout layer >>
        self.drop   = nn.Dropout( p = 0.4 )
        # --------------------------------------------------------------------------------------------------------------------------
        
        # # Initializatiing with custom weights >>
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # Initializatizing weights from He/Kaiming Uniform distribution for Convolutional layers >>
        #         # Note: - He/Kaiming weight initialization is an initialization method for neural networks that takes into account the
        #         #         non-linearity of activation functions, such as ReLU activations.
        #         #       - A proper initialization method should avoid reducing or magnifying the magnitudes of input signals
        #         #         exponentially. Using a derivation they work out that the condition to stop this happening is:
        #         #             1/2*n*Var[w] = 1
        #         #       - This implies an initialization scheme of:
        #         #             w = N(0,2/n)
        #         #       - Biases are initialized be 0 and the weights at each layer are initialized as U[-bound, bound],
        #         #             where a is given by,
        #         #               bound = gain * sqrt( / (fan_mode) ).
        #         m.weight = I.kaiming_normal_(m.weight, a=0)
        #
        #     elif isinstance(m, nn.Linear):
        #         # Initializatiing weights from Xavier/Glorot Uniform distribution for Fully-connected layers >>
        #         # Note: - Xavier Initialization, or Glorot Initialization, is an initialization scheme for neural networks.
        #         #       - Biases are initialized be 0 and the weights at each layer are initialized as U[-a, a],
        #         #             where a is given by,
        #         #               a = gain * sqrt(6 / (fan_in+fan_out) ).
        #         #       - Xavier initialization was proposed by Glorot and Bengio. They point out that the signal must flow properly
        #         #         both forward and backward without dying. They stated that: For the signal to flow properly we need the
        #         #         variance of outputs of each layer to be equal to the variance of its input.
        #         #       - Xavier initialization works well with the Sigmoid activation function.
        #         m.weight = I.xavier_normal_(m.weight, gain=1)
        # # Note:
        # # - The Uniform distribution works very well when the Sigmoid activation function is used.
        # # - Xavier/Glorot initialization works well with the Sigmoid and Tanh activation function.
        # # - The He/Kaiming weight initialization technique works very well with the ReLU activation function.
        
        # --------------------------------------------------------------------------------------------------------------------------
        
        self.spec               = Struct()
        self.spec.criterion     = DEFAULT_CRITERION
        self.spec.optimizer     = DEFAULT_OPTIMIZER
        self.spec.learning_rate = DEFAULT_LR
        self.spec.dataset_train = datasets.train.preprocessed
        self.spec.dataset_test  = datasets.test.preprocessed
        self.spec.n_epochs      = DEFAULT_N_EPOCHS
        self.spec.batch_size    = DEFAULT_BATCH_SIZE
        self.spec.shuffle       = DEFAULT_SHUFFLE
        self.spec.num_workers   = DEFAULT_NUM_WORKERS
        
        # --------------------------------------------------------------------------------------------------------------------------
        
        # self.load_model(DEFAULT_FILE_FKD_NET_MODEL)
        
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
        # In:  224 x 244 x 1
        x = self.pool( F.elu( self.conv1(x) ) ) # out: 32  x 110 x 110
        x = self.pool( F.elu( self.conv2(x) ) ) # out: 64  x 54  x 54
        x = self.pool( F.elu( self.conv3(x) ) ) # out: 128 x 26  x 26
        
        # x = self.pool( F.elu( self.bn1( self.conv1(x) ) ) ) # out: 32  x 110 x 110
        # x = self.pool( F.elu( self.bn2( self.conv2(x) ) ) ) # out: 64  x 54  x 54
        # x = self.pool( F.elu( self.bn3( self.conv3(x) ) ) ) # out: 128 x 26  x 26
        
        # Preparing for linear layer >>
        # Note: This line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # Linear layers with dropout in between >>
        x = F.relu( self.fc(x) )
        x = self.drop(x)
        x = self.fc2(x)
        
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
        
        # self.eval()
        
        # Iterating through the test dataset >>
        for i, sample in enumerate(data_loader):
            
            # Getting sample data: images and ground truth keypoints >>
            images        = sample['image']
            keypoints_gts = sample['keypoints']
            
            # Converting images to FloatTensors >>
            images = images.type(torch.FloatTensor)
            
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
    # START >> METHOD >> train_model
    # ==============================================================================================================================
    # >>
    def train_model(self):
        
        """
        ============================================================================
        START >> DOC >> train_model
        ============================================================================
            
            GENERAL INFO
            ============
                
                Trains the model.
            
            PARAMETERS
            ==========
                
                None
            
            RETURNS
            =======
                
                list_loss <list>
                    
                    A list of losses for epochs.
        
        ============================================================================
        END << DOC << train_model
        ============================================================================
        """
        
        # Preparing the cnn model for training >>
        self.train()
        criterion = self.spec.criterion()
        optimizer = self.spec.optimizer(self.parameters(), lr = self.spec.learning_rate)
        data_loader = DataLoader( self.spec.dataset_train
                                , batch_size  = self.spec.batch_size
                                , shuffle     = self.spec.shuffle
                                , num_workers = self.spec.num_workers )
        
        # Looping over the dataset multiple times >>
        list_loss      = []
        loss_per_epoch = 0.0
        for epoch in range(self.spec.n_epochs):
            
            # Training on batches of data >>
            running_loss   = 0.0
            for batch_i, data in enumerate(data_loader):
                
                # Getting the input images and their corresponding labels >>
                images  = data['image']
                key_pts = data['keypoints']
                
                # Flatten pts >>
                key_pts = key_pts.view(key_pts.size(0), -1)
                
                # Converting variables to floats for regression loss >>
                key_pts = key_pts.type(torch.FloatTensor)
                images  = images.type(torch.FloatTensor)
                
                # Forward pass to get outputs >>
                output_pts = self.forward(images)
                output_pts = output_pts.view(output_pts.size(0), -1)
                
                # Calculate the loss between predicted and target keypoints >>
                loss = criterion(output_pts, key_pts)
                
                # Zero the parameter (weight) gradients >>
                optimizer.zero_grad()
                
                # Backward pass to calculate the weight gradients >>
                loss.backward()
                
                # Updating the weights >>
                optimizer.step()
                
                # Printing loss statistics >>
                # to convert loss into a scalar and add it to the running_loss, use .item()
                running_loss   = running_loss   + loss.item()
                loss_per_epoch = loss_per_epoch + loss.item()
                if batch_i % DEFAULT_N_BATCH_TO_PRINT_LOSS == (DEFAULT_N_BATCH_TO_PRINT_LOSS-1):    # print every 20 batches
                    print(f"Epoch: {epoch + 1}, Batch: {batch_i+1}, Avg. {self.spec.criterion.__name__} Loss: {running_loss/20}")
                    running_loss = 0.0
                
            list_loss.append(loss_per_epoch)
            loss_per_epoch = 0.0
        
        print('Finished Training')
        
        return list_loss
    # <<
    # ==============================================================================================================================
    # END << METHOD << train_model
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> load_model
    # ==============================================================================================================================
    # >>
    def load_model  ( self
                    , file_model = None
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
                
                file_model <str>
                    
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
        
        self.load_state_dict(torch.load(file_model))
        self.eval()
        
        return self
    # <<
    # ==============================================================================================================================
    # END << METHOD << load_model
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> detect_facial_keypoints
    # ==============================================================================================================================
    # >>
    def detect_facial_keypoints ( self
                                , file_image
                                , file_model   = "default"
                                , plot_enabled = False
                                , figsizeScale    = DEFAULT_FIGSIZESCALE
                                , padding      = DEFAULT_PADDING
                                ) :
        
        """
        ============================================================================
        START >> DOC >> detect_facial_keypoints
        ============================================================================
            
            GENERAL INFO
            ============
                
                Detects the facial keypoints in the iunput image.
            
            PARAMETERS
            ==========
                
                file_image <str>
                    
                    File path of the input image.
                
                file_model <str>
                    
                    A file-like object, or a string or os.PathLike object containing
                    a file name.
                    If "default", a saved model in this package will be used.
                
                plot_enabled <bool>
                    
                    When enabled plots the detected facial keypoints.
            
            RETURNS
            =======
                
                keypoints <list>
                    
                    List of length n_images containing tensors of predicted keypoints
                    of torch.Size([1, n_keypoints, 2]).
                
                images <list>
                    
                    List of length n_images containing tensors of face-images of
                    torch.Size([1, H, W])
        
        ============================================================================
        END << DOC << detect_facial_keypoints
        ============================================================================
        """
        
        # Detecting faces with HAAR-cascade classifier for frontal faces >>
        faces = detect_faces(file_image)
        
        # Loading in color image for face detection >>
        image_bgr = cv2.imread(file_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        if file_model != "default":
            self.load_model(file_model)
        # else:
        #     self.load_model(DEFAULT_FILE_FKD_NET_MODEL)
        
        image = np.copy(image_rgb)
        
        # Including a padding to extract face as HAAR classifier's bounding box, crops sections of the face
        images, keypoints = [], []
        
        # Looping over the detected faces >>
        if plot_enabled:
            len_faces = len(faces)
            fig, axes = plt.subplots(1, len_faces, figsize = (len_faces*figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE))
        
        for i, (x,y,w,h) in enumerate(faces):
            
            # Selecting the region of interest that is the face in the image >>
            roi = image[ max(y-padding, 0) : y+h+padding
                       , max(x-padding, 0) : x+w+padding ]
            
            # Rescaling the detected face to be the expected square size for CNN >>
            roi_rescaled = cv2.resize(roi, (DEFAULT_PREPROCESS_SIZE_RANDOMCROP, DEFAULT_PREPROCESS_SIZE_RANDOMCROP))
            
            # Converting the face region from RGB to grayscale >>
            roi_gray = cv2.cvtColor(roi_rescaled, cv2.COLOR_RGB2GRAY)
            
            # Normalizing the grayscale image so that its color range falls in [0,1] instead of [0,255] >>
            roi_normed = (roi_gray / 255.0 ).astype(np.float32)
            
            # Reshaping the numpy image shape (H x W x C) into a torch image shape (C x H x W) >>
            roi = roi_normed
            if len(roi.shape) == 2:
                roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            roi_transposed = roi.transpose((2, 0, 1))
            
            # Converting to torch array >>
            roi_torch = torch.from_numpy(roi_transposed)
            images.append(roi_torch)
            output_pts = self.forward(roi_torch)
            keypoints.append(output_pts)
            
            # Displaying each detected face and the corresponding keypoints >>
            if plot_enabled:
                # plot_output(roi_torch, output_pts)
                axes_curr = axes if len_faces == 1 else axes[i]
                key_pts_pred = output_pts.data
                key_pts_pred = key_pts_pred.numpy()
                # Undoing normalization of keypoints >>
                key_pts_pred = key_pts_pred[0]*DEFAULT_PREPROCESS_SCALING_SQRT + DEFAULT_PREPROCESS_SCALING_MEAN
                plot_keypoints(image = roi, keypoints_pred = key_pts_pred, cmap = "gray", axes = axes_curr)
        
        return keypoints, images
    # <<
    # ==============================================================================================================================
    # END << METHOD << detect_facial_keypoints
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
