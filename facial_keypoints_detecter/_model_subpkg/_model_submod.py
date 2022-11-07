
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
        self.fc       = nn.Linear( in_features  = 128*26*26, out_features = 136 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        # self.fc2      = nn.Linear( in_features  =       500, out_features = 136 )  # [ n_parameters = (512*6*6)*136 = 2506752 ]
        # --------------------------------------------------------------------------------------------------------------------------
        # Dropout layer >>
        # self.drop   = nn.Dropout( p = 0.4 )
        # --------------------------------------------------------------------------------------------------------------------------
        
        # Initializatiing with custom weights >>
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     # Initializatizing weights from He/Kaiming distribution for Convolutional layers >>
            #     # Note: - He/Kaiming weight initialization is an initialization scheme for neural networks that takes into account
            #     #         the non-linearity of activation functions, such as ReLU or ELU activations.
            #     #       - A proper initialization method should avoid reducing or magnifying the magnitudes of input signals
            #     #         exponentially. Using a derivation they work out that the condition to stop this happening is:
            #     #             1/2*n*Var[w] = 1
            #     #       - This implies an initialization scheme of:
            #     #             w = U(-1/sqrt(n), 1/sqrt(n))
            #     #                    or
            #     #             w = N(0,2/n)
            #     #               where,
            #     #                   n = no. of input/wights or no. activations/outputs
            #     #       - Biases are initialized be 0 and the weights at each layer are initialized as U[-a, a] or N[0, std**2],
            #     #             where,
            #     #                 a   = gain * sqrt( 3 / fan_mode )
            #     #                 std = gain * sqrt( 1 / fan_mode )
            #     # m.weight = I.kaiming_uniform_(m.weight, a = 0, nonlinearity='relu')  # 'a' is not same as 'a' above
            #     # m.weight = I.kaiming_normal_ (m.weight, a = 0, nonlinearity='relu')  # 'a' is not same as 'a' above
            #
            # elif isinstance(m, nn.Linear):
            #     # Initializatiing weights from Xavier/Glorot distribution for Fully-connected layers >>
            #     # Note: - Xavier Initialization, or Glorot Initialization is an initialization scheme for neural networks.
            #     #       - Xavier initialization was proposed by Glorot and Bengio. They point out that the signal must flow properly
            #     #         both forward and backward without dying. They stated that: For the signal to flow properly we need the
            #     #         variance of outputs of each layer to be equal to the variance of its input.
            #     #       - Biases are initialized be 0 and the weights at each layer are initialized as U[-a, a] or N[0, std**2],
            #     #             where,
            #     #                 a   = gain * sqrt( 6 / (fan_in + fan_out)) )
            #     #                 std = gain * sqrt( 2 / (fan_in + fan_out)) )
            #     # m.weight = I.xavier_uniform_(m.weight, gain=0.6)
            #     # m.weight = I.xavier_normal_ (m.weight, gain=3)
        
        # Note:
        # - The He/Kaiming weight initialization technique works well with the ReLU activation function.
        # - Xavier/Glorot weight  initialization technique works well with the Sigmoid and Tanh activation function.
        
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
        
        self.apps                         = Struct()
        self.apps.detect_faces            = self._detect_faces
        self.apps.detect_facial_keypoints = self._detect_facial_keypoints
        self.apps.apply_glasses           = self._apply_glasses
        self.apps.apply_face_blur         = self._apply_face_blur
        
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
        # x = F.relu( self.fc(x) )
        # x = self.drop(x)
        x = self.fc(x)
        # x = self.fc2(x)
        
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
        list_loss  = []
        loss_epoch = 0.0
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
                running_loss = running_loss + loss.item()
                loss_epoch   = loss_epoch   + loss.item()
                if batch_i % DEFAULT_N_BATCH_TO_PRINT_LOSS == (DEFAULT_N_BATCH_TO_PRINT_LOSS-1):    # print every 20 batches
                    print(f"Epoch: {epoch + 1}, Batch: {batch_i+1}, Avg. {self.spec.criterion.__name__} Loss: {running_loss/DEFAULT_N_BATCH_TO_PRINT_LOSS}")
                    running_loss = 0.0
            
            list_loss.append(loss_epoch/batch_i)
            print(f"Epoch: {epoch + 1} Complete! {self.spec.criterion.__name__} Loss: {loss_epoch/batch_i}")
            loss_epoch = 0.0
        
        print('Finished Training')
        
        return list_loss
    # <<
    # ==============================================================================================================================
    # END << METHOD << train_model
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> save_model
    # ==============================================================================================================================
    # >>
    def save_model  ( self
                    , path_file = None
                    ) :
        
        """
        ============================================================================
        START >> DOC >> save_model
        ============================================================================
            
            GENERAL INFO
            ============
                
                Saves model parameters in a '.pt' file.
            
            PARAMETERS
            ==========
                
                path_file <str>
                    
                    Path/name of the file to create and save model parameters.
            
            RETURNS
            =======
                
                None
        
        ============================================================================
        END << DOC << save_model
        ============================================================================
        """
        
        # Saving model parameters to the input path >>
        torch.save(self.state_dict(), path_file)
        
        return None
    # <<
    # ==============================================================================================================================
    # END << METHOD << save_model
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
    # START >> METHOD >> _detect_faces
    # ==============================================================================================================================
    # >>
    def _detect_faces   ( self
                        , file_image
                        , plot_enabled = False
                        , figsizeScale = DEFAULT_FIGSIZESCALE
                        ) :
        
        """
        ============================================================================
        START >> DOC >> _detect_faces
        ============================================================================
            
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
        
        ============================================================================
        END << DOC << _detect_faces
        ============================================================================
        """
        
        # Loading in color image for face detection >>
        image_bgr = cv2.imread(file_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Running the haar cascade classifier for detecting frontal faces >>
        faces = FACE_HARR_CASCADE.detectMultiScale(image_rgb, scaleFactor=DEFAULT_HARR_SCALE_FACTOR, minNeighbors=DEFAULT_HARR_MIN_NEIGHBOURS)
        
        # Making a copy of the original image to plot detections on >>
        image_with_detections = image_rgb.copy()
        
        # Looping over the detected faces, mark the image where each face is found >>
        if plot_enabled:
            for (x,y,w,h) in faces:
                # Drawing a rectangle around each detected face >>
                cv2.rectangle( image_with_detections,(x,y),(x+w,y+h), DEFAULT_COLOR_BOX_DETECTED_FACE, DEFAULT_SIZE_BOX_FACE_DETECTED )
            _ = plt.figure( figsize = (figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE) )
            plt.imshow(image_with_detections)
        
        return faces
    # <<
    # ==============================================================================================================================
    # END << METHOD << _detect_faces
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _detect_facial_keypoints
    # ==============================================================================================================================
    # >>
    def _detect_facial_keypoints( self
                                , file_image
                                , file_model   = "default"
                                , plot_enabled = False
                                , figsizeScale = DEFAULT_FIGSIZESCALE
                                ) :
        
        """
        ============================================================================
        START >> DOC >> _detect_facial_keypoints
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
                    torch.Size([1, H, W]).
        
        ============================================================================
        END << DOC << _detect_facial_keypoints
        ============================================================================
        """
        
        # Detecting faces with HAAR-cascade classifier for frontal faces >>
        faces = self._detect_faces(file_image)
        
        # Loading in color image for face detection >>
        image_bgr = cv2.imread(file_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Loading model if file_model is provided >>
        if file_model != "default":
            self.load_model(file_model)
        # else:
        #     self.load_model(DEFAULT_FILE_FKD_NET_MODEL)
        self.eval()
        
        image = np.copy(image_rgb)
        
        # Including a padding to extract face as HAAR classifier's bounding box, crops sections of the face
        images, keypoints = [], []
        
        # Looping over the detected faces >>
        if plot_enabled:
            len_faces = len(faces)
            fig, axes = plt.subplots(1, len_faces, figsize = (len_faces*figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE))
        
        for i, (x,y,w,h) in enumerate(faces):
            
            # Selecting the region of interest that is the face in the image >>
            helf_width_roi = int(max(w,h)*0.35)
            # roi = image[ max(y-padding, 0) : y+h+padding
            #            , max(x-padding, 0) : x+w+padding ]
            
            roi = image[ max(y-helf_width_roi, 0) : y+h+helf_width_roi
                       , max(x-helf_width_roi, 0) : x+w+helf_width_roi ]
            
            # Rescaling the detected face to be the expected square size for CNN >>
            roi_rescaled = cv2.resize(roi, (DEFAULT_PREPROCESS_SIZE_RANDOMCROP, DEFAULT_PREPROCESS_SIZE_RANDOMCROP))
            
            # Converting the face region from RGB to grayscale >>
            roi_gray = cv2.cvtColor(roi_rescaled, cv2.COLOR_RGB2GRAY)
            
            # Normalizing the grayscale image so that its color range falls in [0,1] instead of [0,255] >>
            roi_normed     = (roi_gray     / 255.0 ).astype(np.float32)
            roi_normed_rgb = (roi_rescaled / 255.0 ).astype(np.float32)
            
            # Reshaping the numpy image shape (H x W x C) into a torch image shape (C x H x W) >>
            roi = roi_normed
            if len(roi.shape) == 2:
                roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            roi_transposed     = roi.transpose((2, 0, 1))
            roi_transposed_rgb = roi_normed_rgb
            
            # Converting to torch array >>
            roi_torch     = torch.from_numpy(roi_transposed)
            roi_torch_rgb = torch.from_numpy(roi_transposed_rgb)
            images.append(roi_torch_rgb)
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
                plot_keypoints(image = roi_transposed_rgb, keypoints_pred = key_pts_pred, cmap = "gray", axes = axes_curr)
        
        plt.show()
        
        return keypoints, images
    # <<
    # ==============================================================================================================================
    # END << METHOD << _detect_facial_keypoints
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _apply_glasses
    # ==============================================================================================================================
    # >>
    def _apply_glasses  ( self
                        , file_image
                        , file_sunglasses = "default"
                        , file_model      = "default"
                        , figsizeScale    = DEFAULT_FIGSIZESCALE
                        ) :
        
        """
        ============================================================================
        START >> DOC >> _apply_glasses
        ============================================================================
            
            GENERAL INFO
            ============
                
                Applies sunglasses filter to faces.
            
            PARAMETERS
            ==========
                
                file_image <str>
                    
                    File path of the input image.
                
                file_sunglasses <str>
                    
                    File path of the sunglasses.
                
                file_model <str>
                    
                    A file-like object, or a string or os.PathLike object containing
                    a file name.
                    If "default", a saved model in this package will be used.
            
            RETURNS
            =======
                
                None
        
        ============================================================================
        END << DOC << _apply_glasses
        ============================================================================
        """
        
        # Loading sunglasses if file_sunglasses is provided >>
        if file_sunglasses == "default":
            file_sunglasses = DEFAULT_FILE_FILTERS_SUNGLASSES
        sunglasses = cv2.imread(file_sunglasses, cv2.IMREAD_UNCHANGED)
        
        # Loading model if file_model is provided >>
        if file_model != "default":
            self.load_model(file_model)
            self.eval()
        
        # Detecting facial keypoints >>
        keypoints, faces = self._detect_facial_keypoints(file_image)
        
        # Looping over the detected faces >>
        len_faces = len(faces)
        fig, axes = plt.subplots(1, len_faces, figsize = (len_faces*figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE))
        
        for j, (face, keypoint) in enumerate(zip(faces, keypoints)):
            
            # Untransform image and points >>
            face    = face.numpy()
            key_pts = keypoint.detach().numpy()[0]*DEFAULT_PREPROCESS_SCALING_SQRT+DEFAULT_PREPROCESS_SCALING_MEAN
            
            # Copy of the face image for overlay
            face_copy = np.copy(face)
            
            # Assigning location to top-left location for sunglasses to go >>
            # no. 17 = edge of left eyebrow
            x = int(key_pts[17, 0]) - 5
            y = int(key_pts[17, 1]) + 1
            
            # Assigning height and width of sunglasses >>
            # w = left to right eyebrow edges
            w = int(abs(key_pts[17,0] - key_pts[26,0])+7)
            # h: length of nose
            h = int(abs(key_pts[27,1] - key_pts[34,1])+5)
            
            # Resizing sunglasses >>
            # new_sunglasses = np.copy(sunglasses)
            new_sunglasses = cv2.resize(sunglasses, (w, h), interpolation = cv2.INTER_CUBIC)
            
            # Get region of interest on the face to change >>
            roi_color = face_copy[y:y+h,x:x+w]
            
            # Finding all non-transparent pts >>
            ind = np.argwhere(new_sunglasses[:,:,3] > 0)
            
            # Replacing the original image pixel with that of the new_sunglasses for each non-transparent point >>
            for i in range(3):
                roi_color[ind[:,0],ind[:,1],i] = new_sunglasses[ind[:,0],ind[:,1],i]/255
            
            # set the area of the image to the changed region with sunglasses
            face_copy[y:y+h,x:x+w] = roi_color
            
            # Plotting face with sunglasses >>
            axes_curr = axes if len_faces == 1 else axes[j]
            plot_keypoints(image = face_copy, axes = axes_curr)
        
        # plt.show()
        
        return keypoints, faces
    # <<
    # ==============================================================================================================================
    # END << METHOD << _apply_glasses
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _apply_face_blur
    # ==============================================================================================================================
    # >>
    def _apply_face_blur( self
                        , file_image
                        , file_model     = "default"
                        , figsizeScale   = DEFAULT_FIGSIZESCALE
                        ) :
        
        """
        ============================================================================
        START >> DOC >> _apply_face_blur
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
            
            RETURNS
            =======
                
                None
        
        ============================================================================
        END << DOC << _apply_face_blur
        ============================================================================
        """
        
        # Loading model if file_model is provided >>
        if file_model != "default":
            self.load_model(file_model)
            self.eval()
        
        # Detecting facial keypoints >>
        keypoints, faces = self._detect_facial_keypoints(file_image)
        
        # Looping over the detected faces >>
        len_faces = len(faces)
        fig, axes = plt.subplots(1, len_faces, figsize = (len_faces*figsizeScale*DEFAULT_FIGSIZE, figsizeScale*DEFAULT_FIGSIZE))
        
        for j, (face, keypoint) in enumerate(zip(faces, keypoints)):
            
            # Untransform image and points >>
            face    = face.numpy()
            key_pts = keypoint.detach().numpy()[0]*DEFAULT_PREPROCESS_SCALING_SQRT+DEFAULT_PREPROCESS_SCALING_MEAN
            
            # Copy of the face image for overlay
            face_copy = np.copy(face)
            
            # Assigning location to top-left location for moustache to go >>
            # np. 3 = edge of left eyebrow
            x = int(key_pts[0,  0]+2)
            y = int(key_pts[19, 1]-10)
            
            # Assigning height and width of moustache >>
            w = int(abs(key_pts[0, 0] - key_pts[16,0])-4)
            h = int(abs(key_pts[19,1] - key_pts[ 8,1]))
            
            # Kernel for blurring >>
            kernal = np.ones((h,w), dtype=np.float32)/(h*w)
            face_copy[y:y+h,x:x+w] = cv2.filter2D(face_copy[y:y+h,x:x+w], -1, kernal)
            
            # Plotting face with moustache >>
            axes_curr = axes if len_faces == 1 else axes[j]
            plot_keypoints(image = face_copy, axes = axes_curr)
        
        # plt.show()
        
        return keypoints, faces
    # <<
    # ==============================================================================================================================
    # END << METHOD << _apply_face_blur
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << Net
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << facial_keypoint_detecter._model_subpkg._model_submod
# ==================================================================================================================================
