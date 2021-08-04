# from KalmanNet_lor_partial import DatafolderName
import torch
import torch.nn as nn

from KNet_sysmdl import SystemModel
from KNet_nn import KalmanNetNN

from KNet_data import DataGen, DataLoader, DataLoader_GPU
from KNet_data import N_E, N_CV, N_T, T, T_test
from KNet_data import m, n, F, H, m1_0, m2_0

from KF_test import KFTest

from new_Pipeline_KF import Pipeline_KF as Pipeline

from datetime import datetime

from new_Plot import Plot

from new_filing_paths import path_model
import sys
sys.path.insert(1, path_model)
# What do I need from parameters?
# from new_parameters import 
from new_model import f, h

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

##############################
# Location of generated Data #
##############################
DataFolderName = path_model + 'data_gen/'
data_gen = 'data_gen.pt'

##################
# Generate Error #
##################
# Generate r and q for the error.
# r = 1, q = 1
r2 = torch.tensor([1])
r = torch.sqrt(r2)
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

########
# Core #
########
# May run multiple times for different error statistics.
for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   
   #########
   # Model #
   #########
   
   # Initiate Model
   sys_model = SystemModel(F, q[rindex], H, r[rindex], T, T_test, 'LinGauss')
   sys_model.InitSequence(m1_0, m2_0)

   ##########################
   # Generate and load data #
   ##########################
   print("Start Data Gen")
   DataGen(sys_model, DataFolderName + data_gen, T, T_test)

   print("Start Data Load")
   # print(data_gen[rindex])
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(DataFolderName + data_gen)

   ##########################
   # Evaluate Kalman Filter #
   ##########################
   print("Evaluate Kalman Filter")
   # MSE_KF_linear_arr: Loss of the KF for each Trajectory
   # MSE_KF_linear_avg: Avg Loss over all Trajectories
   # MSE_KF_dB_avg: Avg Loss over all Trajectories in dB
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)

   ##########################
   ### KalmanNet Pipeline ###
   ##########################

   # Pipeline (Time, Folder Name, Model Name)
   KNet_Pipeline = Pipeline(strTime, "KNet", "KalmanNet")
   # Eg, x_t = F * x_t-1 + v_t, y_t = x_t-1 + w_t
   KNet_Pipeline.setssModel(sys_model)
   # The KGain is calculated in a Neural Network
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model)
   KNet_Pipeline.setModel(KNet_model)

   KNet_Pipeline.setTrainingParams(n_Epochs=30, n_Batch=50, learningRate=5E-4, weightDecay=5E-6)
   KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   KNet_Pipeline.NNTest(N_T, test_input, test_target)
   KNet_Pipeline.PlotTrain_KF(MSE_KF_linear_arr, MSE_KF_dB_avg)
   KNet_Pipeline.save()