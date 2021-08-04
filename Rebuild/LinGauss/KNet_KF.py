#################
# Kalman Filter #
#################
# Build a System Model for a KF.

import torch

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

# A KF tries to estimate the state x_t for given x_0 and y_t.
class KalmanFilter:
    # Removed semicolons since this is Python.

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    # Prediction
    # Predict the piors of x and y. Since this is a normal KF, 
    # the 2nd moments are also needed to calculate the KGain.
    # For EKF x_t|t-1 = f(x_t-1|t-1), y_t|t-1 = h(x_t|t-1).
    def Predict(self):
        # Predict the 1-st moment of x
        # x_t|t-1 = F * x_t-1|t-1
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 2-nd moment of x
        # Sigma_t|t-1 = F * Sigma_t-1|t-1 * F_T + Q
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        # y_t|t-1 = H * x_t|t-1
        self.m1y = torch.matmul(self.H, self.m1x_prior)

        # Predict the 2-nd moment of y
        # S_t|t-1 = H * Sigma_t|t-1 * H_T + R
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    # KG = Sigma_t|t-1 * H_T * inverse(S_t|t-1)
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    # Update step in KF algorithm
    def Correct(self):
        # Compute the 1-st posterior moment
        # x_t|t = x_t|t-1 + KG * dy
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        # Sigma_t|t = Sigma_t|t-1 - (KG * S_t|t-1 * KG_T)
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    # One step of the KF algorithm.
    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        # Return x_t|t and Sigma_t|t for the next step, 
        # ie, x_t-1|t-1 and Sigma_t-1|t-1
        return self.m1x_posterior,self.m2x_posterior

    # Initialize the KF with x_t=0 and Sigma_t=0.
    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    #####################
    # Generate Sequence #
    #####################
    # Generate a Sequence of x_t|t and Sigma_t|t for t in [1,T].
    # T: Sequence length.
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(cuda0)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(cuda0)

        # Initialize the Sequence.
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        # Do T steps of the KF algo.
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, t], 1);
            xt,sigmat = self.Update(yt);
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)