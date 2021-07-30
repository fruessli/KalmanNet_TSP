import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from KalmanNet_sysmdl import SystemModel
from KalmanNet_data import DataGen,DataLoader,DataLoader_GPU
from KalmanNet_data import N_E, N_CV, N_T
from KalmanNet_data import m, n, F, H, m1_0, m2_0

from KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate, h_nonlinear

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

offset = 0
DatafolderName = '../Simulations/Lorenz_Atractor/data/v0_smallT_NT1000' + '/'
data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
# [true_sequence] = data_gen_file['All Data']

r2 = torch.tensor([1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

r2optdB = torch.tensor([-1])
ropt = torch.sqrt(10**(-r2optdB/10))
print("Searched optimal 1/r2 [dB]: ", 10 * torch.log10(1/ropt**2))

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['r0q0_T20.pt']#,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
EKFResultName = 'EKF_nonLinearh_rq00_T20' 
for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   #Model
   sys_model = SystemModel(F, q[rindex], H, r[rindex], T, T_test, m, n,"Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

   #Generate and load data DT case
   print("Start Data Gen")
   # T = 2000
   DataGen(sys_model, DatafolderName + dataFileName[rindex], T, T_test)
   print("Data Load")
   print(dataFileName[rindex])
   [train_input_long, train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[rindex],map_location=cuda0)  
   print("trainset long:",train_target_long.size())
   # T = 100
   # [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   # print("trainset chopped:",train_target.size())
   print("testset:",test_target.size())
   print("cvset:",cv_target.size())