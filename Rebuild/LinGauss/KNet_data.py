###############################################
# Data-Design, -Size, -Generation and -Loader #
###############################################

import torch
import math

# Couldn't really figure what this does.
# os.environ: A mapping object representing the string environment. 
# For example, environ['HOME'] is the pathname of your home directory
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
# Number of generated Sequences
N_E = 1000

# Number of Cross Validation Examples
# Number of Validation Sequences
N_CV = 100

# Number of Testing Sequences
N_T = 100

# Sequence Length
T = 20
# Sequence Length during Testing
T_test = 1000

#################
## Design #10 ###
#################
# Design of F and H.
# Uncommented the desired size.
F10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

############
## 2 x 2 ###
############
m = 2
n = 2
F = F10[0:m, 0:m]
H = torch.eye(2)
m1_0 = torch.tensor([[0.0], [0.0]]).to(cuda0)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(cuda0)


#############
### 5 x 5 ###
#############
# m = 5
# n = 5
# F = F10[0:m, 0:m]
# H = H10[0:n, 10-m:10]
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(cuda0)
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

##############
## 10 x 10 ###
##############
# m = 10
# n = 10
# F = F10[0:m, 0:m]
# H = H10
# m1_0 = torch.zeros(m, 1).to(cuda0)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(cuda0)

###############################
# Data-Generation and -Loader #
###############################
# T is the Trajectory length of the Sequences, 
# ie, y_0 to y_T is calculated.
def DataGen(SysModel_data, fileName, T, T_test):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    # N_E Sequences get generated.
    SysModel_data.GenerateBatch(N_E, T)
    # input = y
    training_input = SysModel_data.Input
    # target = x
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    # N_CV Sequences get generated.
    SysModel_data.GenerateBatch(N_CV, T)
    # input = y
    cv_input = SysModel_data.Input
    # target = x
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    # Generate N_T Sequences for Testing
    SysModel_data.GenerateBatch(N_T, T_test)
    # input = y
    test_input = SysModel_data.Input
    # target = x
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    # Saves all inputs and targets in fileName.
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):
    # Load the previously saved inputs and targets at fileName.
    # They are first loaded into the CPU 
    # and then moved to the device they were saved from.
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader_GPU(fileName):
    # torch.utils.data.DataLoader should not be confused with the DataLoader above.
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # Default: False
    # However this enables fast data transfer to CUDA-enabled GPUs.
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    training_input = training_input.squeeze()
    training_target = training_target.squeeze()
    cv_input = cv_input.squeeze()
    cv_target =cv_target.squeeze()
    test_input = test_input.squeeze()
    test_target = test_target.squeeze()
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]