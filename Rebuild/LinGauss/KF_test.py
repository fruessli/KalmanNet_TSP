##############
# KF Testing #
##############
# Calculates the MSE of the KF.

import torch
import torch.nn as nn

# System Model of the KF.
from KNet_KF import KalmanFilter
# N_T: Number of Trajectories.
from KalmanNet_data import N_T

def KFTest(SysModel, test_input, test_target):

    # LOSS
    # sum( ||x - x_estimate||^2 ). The loss for the 2nd moment is not calculated.
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    # Array for saving the MSEs of the Trajectories.
    MSE_KF_linear_arr = torch.empty(N_T)

    # Create Object of the Class KalmanFiler in KNet_KF.
    KF = KalmanFilter(SysModel)
    # Set Starting Values for the KF.
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    # Iterate through the Trajetories.
    for j in range(0, N_T):

        # Generate a Sequence of x_t|t and Sigma_t|t for t in [1,T].
        # T = KF.T_test: Sequence length.
        KF.GenerateSequence(test_input[j, :, :], KF.T_test)

        # Calculate the loss over the sequence. Save the loss in the array.
        MSE_KF_linear_arr[j] = loss_fn(KF.x, test_target[j, :, :]).item()
        #MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()

    # Avg loss over all Trajectories.
    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")

    # Return the loss as array, it's avg, and it's avg as dB.
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]