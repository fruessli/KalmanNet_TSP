#########################
# Define the model used #
#########################
# Ei, how f(x) and h(x) look like in the Simulation.

import math
import torch

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)

# May need to import some other things later.
from new_parameters import m, n
from KNet_data import F, H

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

def f(x):
    # f(x) = F * x
    return torch.matmul(F, x)

def h(x):
    # h(x) = H * x
    return torch.matmul(H,x).to(cuda0)