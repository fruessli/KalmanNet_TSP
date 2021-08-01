import torch
import torch.nn as nn

from KNet_sysmdl import SystemModel
from KNet_data import DataGen, DataLoader, DataLoader_GPU
from KNet_data import N_E, N_CV, N_T, T, T_test
from KNet_data import m, n, F, H, m1_0, m2_0
from KalmanNet_nn import KalmanNetNN

from datetime import datetime
# 1st I need a plot function ...
# from Plot import Plot

from filing_paths import path_model
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