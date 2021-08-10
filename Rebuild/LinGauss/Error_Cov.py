######################################################
# Calculate the prior and posterior Error Covariance #
######################################################
# Goal: Posterior Error Covariance: P_plus
# Arguments: H, KG, R (R would not be available in applications)
# Prior Error Covariance: P_minus

# From the KF it is known that 
# P_plus = (I - K*H) * P_minus.
# P_minus = inv(H_T*H) * H_T * inv(inv(H*K) - I) * R*H * inv(H_T*H)
# which is equivalent to 
# K = P_minus*H_T * inv(R + H*P_minus*H_T).
# which is known from the KF.

import torch

# Not sure if needed.
if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

##############################
# Posterior Error Covariance #
##############################
def Calc_Error_Cov(H,R,K,m,n):
    H_T = torch.transpose(H, 0, 1)
    H_T_H = torch.matmul(H_T, H)

    # P_minus as described above.
    # torch.inverse() is an Alias for torch.linalg.inv()
    inv_HK = torch.linalg.inv( torch.matmul(H, K) )
    # torch.linalg.solve(A, B) solves A^-1 * B.
    # P_minus = inv(H_T*H) * H_T ...
    P_minus = torch.linalg.solve(H_T_H, H_T)
    # ... * inv(H*K - I) * R ...
    temp = torch.linalg.solve(inv_HK - torch.eye(n), R)
    P_minus = torch.matmul(P_minus, temp)
    # ... * H ...
    P_minus = torch.matmul(P_minus, H)
    # ... * inv(H_T*H)
    temp = torch.linalg.inv(H_T_H)
    P_minus = torch.matmul(P_minus, temp)

    # P_plus as described above.
    # P_plus = (I - K*H) * P_minus
    temp = torch.matmul(K, H)
    P_plus = torch.matmul(torch.eye(m) - temp, P_minus)
    return(P_plus)