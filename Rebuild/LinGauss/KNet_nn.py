#################################################################
### Build the Blockdiagram and the Neural Network as a Module ###
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as func # for ReLU, linear, MSE loss

# Module is the Base class for all neural network modules.
class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        # super() let's you avoid referring to the base class explicitly.
        # More info at https://realpython.com/python-super/.
        super().__init__()
        # Use a GPU if possible, the CPU otherwise.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############

    # Initiate the values for KNet.
    # Done in KNet_build.py for architecture #2.
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    # The KGain is calculated in a Neural Network, 
    # which we set up in here.
    # We have the following order of Layers for arch #1:
    # lin -> ReLU -> GRU -> lin -> ReLU -> lin
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        # bias: If set to False, the layer will not learn an additive bias. Default: True.
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        # Set negative values to zero.
        self.KG_relu1 = torch.nn.ReLU()

        # Keep in mind that both linear and ReLU are functions,
        # so they need an argument when used.

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension / Output Dim of GRU
        # m^2+n^2 is the joint dimensionality of the tracked moments Sigma and S.
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        # Tbh I don't really see the point of using a RNN if seq_len=1.
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # # batch_first – If True, then the input and output tensors are 
        # # provided as (batch, seq, feature) instead of (seq, batch, feature).
        # # Default: False
        # batch_first = False

        # # dropout – If non-zero, introduces a Dropout layer on the outputs 
        # # of each GRU layer except the last layer, with dropout probability equal to dropout. 
        # The Dropout Layer zeroes randomly some elements of the input tensor during training.
        # # Default: 0
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # Is now done directly in the KGain_step.
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        # Actually apply the GRU to the input seq.
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        # Apply lin transf on GRU output.
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        # Zero out negative values.
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        # Again, use a lin transf.
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    # F and H are used to calculate the state and observations resp., 
    # aswell as their estimates in the KF and KNet.
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    # Set some initial values for x_t|t-1, x_t|t and state process posterior.
    def InitSequence(self, M1_0):

        # x_t|t-1
        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        # x_t|t (or x_t-1|t-1 later)
        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        # Tbh I have no idea what this is supposed to be, or where it is even used.
        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        # This is never really used for anything.
        # sp_prior = F * sp_post
        self.state_process_prior_0 = torch.matmul(self.F, self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        # This is never even used.
        # obs_process = H * sp_prior
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        # x_t-1|t-2 = x_t|t-1
        self.m1x_prev_prior = self.m1x_prior
        # x_t|t-1 = F * x_t-1|t-1
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        # Keep in mind that this is an estimate.
        # y_t|t-1 = H * x_t|t-1
        self.m1y = torch.matmul(self.H, self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in x prior
        # Feature 4: dx = x_t|t - x_t|t-1, but t = t-1.
        # dx = x_t|t-1 - sp_prior, no idea what this is.
        #dm1x = self.m1x_prior - self.state_process_prior_0
        # Feature 4: dx = x_t-1|t-1 - x_t-1|t-2
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        # squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: dy = y_t - y_t|t-1
        # Keep in mind that y_t is a given observation.
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # Feature 1: dy~ = y_t - y_t-1

        # Feature 3: dx~ = x_t|t - x_t-1|t-1

        # So if we wanted to implement feature 1+3 would need 
        # to store y_t-1 and x_t-1|t-1.

        # KGain Net Input
        # Here use the features 2+4 as input for the NN.
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        # This step involves going through the NN.
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    # Do one step of the x_t|t estimation.
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation step
        # dy = y_t - y_t|t-1, similar to Feature 2.
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        # This is actually not the Innovation, but a handy substep.
        # x_t|t = xt|t-1 + KGain * dy
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    # Go through the NN to get our KGain.
    # Again, the order of the Layers for arch #1 is:
    # lin -> ReLU -> GRU -> lin -> ReLU -> lin
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    # This function is inherited from the base class Module, 
    # and forward defines the computation performed at every call.
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        # We want to estimate x_t|t for a given y_t.
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    # The weight matrix is initialized?
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data