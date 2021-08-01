import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#########################
### Design Parameters ###
#########################
# These parameters are not needed for lin gauss, 
# since they are already in KNet_data.
# m = 3
# n = 3
# variance = 0
# m1x_0 = torch.ones(m, 1) 
# m1x_0_design_test = torch.ones(m, 1)
# m2x_0 = 0 * 0 * torch.eye(m)

# However, Q and R might need to be specified here.