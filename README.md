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

## How to run:
