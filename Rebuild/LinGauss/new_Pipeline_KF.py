#######################
# Pipeline for the KF #
#######################
# Pipeline for KF and therefore also KNet.
# Mainly Trains and Tests the KF and KNet.
# Apparently also Plots smth.
# This is all in the separate files 
# 'KNet_train.py' and 'KNet_test' for the new architecture.

import torch
import torch.nn as nn
import random
from new_Plot import Plot

from Error_Cov import Calc_Error_Cov

class Pipeline_KF:

    def __init__(self, Time, folderName, modelName):
        # super might be unnecessary since Pipeline it not a subclass.
        super().__init__()
        self.Time = Time
        # Might have to take out '/' if already included in the folder name.
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    # Saves self in PipelineName to a disk file.
    def save(self):
        torch.save(self, self.PipelineName)

    ################
    # Set SS Model #
    ################
    # Set the State Space model for the Pipeline.
    # Eg, KNet.SystemModel(F,q,H,r,T,T_test,'Name')
    # Ie, x_t = F * x_t-1 + v_t, y_t = x_t-1 + w_t
    def setssModel(self, ssModel):
        self.ssModel = ssModel

    ################
    # Set NN Model #
    ################
    # Set the model, ie, the neural network model.
    # Eg, the KNet_nn.KalmanNetNN, which is a subclass of torch.nn.Module
    # Ie, calc priors, estimate KGain, and calculate posteriors.
    def setModel(self, model):
        self.model = model

    ###########################
    # Set Training Parameters #
    ###########################
    # The learning rate controls how quickly the model is adapted to the problem. 
    # Smaller learning rates require more training epochs given the smaller changes made to the weights each update, 
    # whereas larger learning rates result in rapid changes and require fewer training epochs.
    # The learning rate is usually between 0 and 1.
    # WeightDecay: Loss = ||y - y_estimate||^2 + wd * sum(weights^2)
    # LearningRate: w(t) = w(t-1) - lr * dLoss / dw
    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        # nn.MSELoss has 2 modes: mean and sum. Default is mean.
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        # Adam is similar to SGD.
        # Remember that model is from a subclass of torch.nn.Module.
        # model.parameter(): Returns an iterator over module parameters. 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
    
    ###########################
    # Neural Network Training #
    ###########################
    # Trains and Validates the NN.
    # N_E: Number of Epochs.
    # N_CV: Number of Cross Validations.
    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):

        self.N_E = n_Examples
        self.N_CV = n_CV

        # Tensor to save Validation Loss per batch
        MSE_cv_linear_batch = torch.empty([self.N_CV])
        # Tensor to save avg Validation Loss per epoch
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        # Tensor to save Training Loss per batch
        MSE_train_linear_batch = torch.empty([self.N_B])
        # Tensor to save avg Training Loss per epoch
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        # Optimum loss that gets update, ie, we start high and go lower.
        self.MSE_cv_dB_opt = 1000
        # Index of the optimal loss.
        self.MSE_cv_idx_opt = 0

        # Go through all Epochs.
        for ti in range(0, self.N_Epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            # N_CV: Number of Validations
            for j in range(0, self.N_CV):
                # Inputs
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)
                # Targets
                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                # Array to save KG in, required for the calculation of the error covariance.
                # KG has the dimension m * n.
                self.KG_out_cv = torch.empty(self.ssModel.m, self.ssModel.n, self.ssModel.T)
                # Go through all Trajectories
                for t in range(0, self.ssModel.T):
                    # Calls the NN, similar to model.forward, 
                    # but one shouldn't use model.forward directly (smth with memory).
                    [x_out_cv[:, t], self.KG_out_cv[:, :, t]] = self.model(y_cv[:, t])

                # Compute Training Loss
                # .item extracts the value as python float.
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()

            # Average
            # Avg loss over the batches of an epoch.
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            # Saves the optimal loss found so far.
            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################
            # This is still the same Epoch ti as the Validation before.

            # Training Mode
            self.model.train()

            # Init Hidden State
            # The weight matrix is initialized.
            self.model.init_hidden()

            # Sums up the Losses in a batch. At the beginning we have no loss.
            Batch_Optimizing_LOSS_sum = 0

            # Go through all batches during the training.
            for j in range(0, self.N_B):
                # Pick a random Epoch.
                n_e = random.randint(0, self.N_E - 1)

                # Take the Inputs of that Epoch.
                y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                # Empty array for training outputs and KGain.
                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                self.KG_out_training = torch.empty(self.ssModel.m, self.ssModel.n, self.ssModel.T)
                # Train the weights for T Trajectories.
                for t in range(0, self.ssModel.T):
                    # Calls the NN, similar to model.forward, 
                    # but one shouldn't use model.forward directly (smth with memory).
                    [x_out_training[:, t], self.KG_out_training[:, :, t]] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])
                MSE_train_linear_batch[j] = LOSS.item()

                # Sum up the Losses.
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            # Avg loss over the batches of an epoch.
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            #                parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            # Print avg loss in training and validation in this Epoch.
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            # Print avg loss difference compared to last Epoch.
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            # Print optimal loss so far.
            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    ##########################
    # Neural Network Testing #
    ##########################
    # N_T: Number of Tests.
    def NNTest(self, n_Test, test_input, test_target):

        self.N_T = n_Test

        # Empty array for test loss in each Test.
        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # Empty array for KGain in order to calc P+.
        self.KG_out_test = torch.empty(self.ssModel.m, self.ssModel.n, self.ssModel.T_test, self.N_T)

        # MSE LOSS Function
        # ||target - target_estimated||^2
        loss_fn = nn.MSELoss(reduction='mean')

        # model is the NN model, ie, here the KalmanNetNN from KNet_nn.py.
        self.model = torch.load(self.modelFileName)

        # Cross Validation Mode
        self.model.eval()

        # Disables gradient calculation.
        torch.no_grad()

        # Iterate through all tests.
        for j in range(0, self.N_T):

            # Take the input for the test.
            y_mdl_tst = test_input[j, :, :]

            # Initiate the Sequence with x_t = m1x_0.
            self.model.InitSequence(self.ssModel.m1x_0)

            # Empty array for the outputs
            # self.ssModel.T seems suspect. Should probably be self.ssModel.T_test @@@@@@
            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test)

            # Calculate the outputs with the current input for all Trajectories.
            # T: Number of Trajectories.
            for t in range(0, self.ssModel.T):
                [x_out_test[:, t], self.KG_out_test[:, :, t, j]] = self.model(y_mdl_tst[:, t])

            # Calculate the MSE loss between the output and the target over all Trajectories.
            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()

        # Average
        # Calculate the avg loss over all Tests.
        # Problem: MSE_test_linear_arr is infinite. x_out_test is infinite after a few steps.
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")

    ###############
    # Plot Losses #
    ###############
    # Plot Losss of KNet in Train, CV and Test, and KF.
    # MSE_KF_linear_arr: Loss of the KF
    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        # Comment out NNPlot_Hist until distsplot if fixed.
        # self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    ##############################
    # Posterior Error Covariance #
    ##############################
    # Calculate the Posterior Error Covariance.
    # This require K, R and H.

    def Err_Cov(self, R, H):
        # Only calculating P+ for Testing phase for now, as there the Sigma is already available.
        # P_plus_cv = torch.empty(self.ssModel.m, self.ssModel.m, self.ssModel.T)
        # P_plus_train = torch.empty(self.ssModel.m, self.ssModel.m, self.ssModel.T)
        self.P_plus_test = torch.empty(self.ssModel.m, self.ssModel.m, self.ssModel.T, self.N_T)
        
        # Iterate through all tests. Train and CV need to Iterate throught the Epochs additionally.
        for j in range(0, self.N_T):
            for t in range(0, self.ssModel.T):
            #     P_plus_cv[:, :, t] = Calc_Error_Cov(H, R, self.KG_out_cv[:, : ,t], self.ssModel.m, self.ssModel.n)
            #     P_plus_train[:, :, t] = Calc_Error_Cov(H, R, self.KG_out_train[:, : ,t], self.ssModel.m, self.ssModel.n)
                self.P_plus_test[:, :, t, j] = Calc_Error_Cov(H, R, self.KG_out_test[:, : ,t], self.ssModel.m, self.ssModel.n)

        # Get Sigma from KFTest and calc MSE then plot.
        # Can get Sigma rom lin_gauss so can be argument. Or make fct in Plot.py <-- probably easier
        # So only calc p+ here and return back to lin_gauss

    def Plot_Err_Cov(self, Sigma):
        # Calculate MSE of Sigma and P_plus.
        MSE_err_cov_test = torch.empty(self.N_T)
        for j in range(0, self.N_T):
            MSE_err_cov_test[j] = self.loss_fn(Sigma[:,:,:,j], self.P_plus_test[:,:,:,j])
        # Avg
        MSE_err_cov_test_avg = torch.mean(MSE_err_cov_test)
        # dB
        MSE_err_cov_test_dB_avg = 10 * torch.log10(MSE_err_cov_test_avg)

        # Plot MSE
        # @@@@@ Make this (shouldnt take too long)
        self.Plot.NNPlot_err_cov(self.P_plus_test, Sigma)