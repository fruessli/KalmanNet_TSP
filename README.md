# KalmanNet_TSP

## KNet_nn

F is the 2nd moment of x(t-1), since it is the derivative of f, ei, F(f(x(t-1))).\
H is the 2nd moment of x(t|t-1), since it is the derivative of h, ei, H(h(x(t|t-1))).

F_T is F.T\
m is the number of rows in F.

H_T is H.T\
n is the number of rows in H.

The KGain has dimmension mxn.

H1_KNet is the Number of neurons in the 1st hidden layer.\
H2_KNet is the Number of neurons in the 2nd hidden layer.

D_in are the input dimensions, ei, x(t-1), y(t).\
D_out are the output dimensions, ei, KGain.

KG_l1 is the linear layer.\
KG_relu1 is the ReLU activation fct.

GRU:\
Since the batch size is 1 it is simply stochastic training, one sample after another.\
The sequence length is also 1. Thus the GRU doesn't memorize anything. So why do we even need a RNN?\
nn.GRU(input_size, hidden_size, num_layers)

Input->Lin->ReLU->GRU->Lin->ReLU->Lin

KGain_step:\
L1_out: Output of 1st Lin Layer.\
La1_out: Output of 1st ReLU.

hn: hidden state tensor

Returns L3_out

## KNet_sysmdl

Builds the state space model.

b_matrix is a matrix of bernoulli drawn 1s and 0s, where a entry is 1 with prob p_outlier. On default p_outlier = 0.

What is T? Trajectory of what?

GenerateSequence:\
rsample: Takes random sample.

xt = xt-1 * F + eq\
yt = yt-1 * H + er + btdt

## KNet_KF

Normal KF

## Extended_KNet_nn

GRU has now nGRU layers (here 2).\
SystemDynamics can either be set to full or partial information.
H is called h, and F is now called f.

InitSystemDynamics:\
Some additional variable get init, like a counter i, or x_out.

@@@@@ I didn't draw the model of EKNet yet. @@@@@
@@@@@ At KGain_est @@@@@

How is squeeze used in here? y = torch.squeeze(f(x))

The EKNet can either have F1+F3 or F2+F4, KNet has only F2+F4.

## Plot.py

From the looks of it, only a class for plotting different graphs.

## How to run:
