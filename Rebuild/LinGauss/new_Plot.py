#######################################
# Contains Several Plotting Functions #
#######################################
# Contains Legends, Colors, Axis-Labels etc.
import torch
# Plotting library for Python and Numpy
import matplotlib as mpl
# The parameter chunksize essentially means 
# the number of rows to be read into a dataframe 
# at any single time in order to fit into 
# the local memory. So 1'000 rows get read at a time.
# This is the case at least in pandas.
mpl.rcParams['agg.path.chunksize'] = 1E4    
# matplotlib.pyplot is a collection of functions
# that make matplotlib work like MATLAB.
import matplotlib.pyplot as plt
# gridspec: A grid layout to place subplots within a figure.
import matplotlib.gridspec as gridspec
# Seaborn is a Python data visualization library
# based on matplotlib.
import seaborn as sns
import numpy as np

# A collection of functions and objects for 
# creating or placing inset axes.
# zoomed_inset_axes: Create an anchored inset 
#                    axes by scaling a parent axes.
# mark_inset: Draw a box to mark the location of an 
#             area represented by an inset axes.
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# find_peaks: This function takes a 1-D array and finds all 
#   local maxima by simple comparison of neighboring values.
from scipy.signal import find_peaks
# Axes3D: 3D axes object.
from mpl_toolkits.mplot3d import Axes3D

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    cuda0 = torch.device("cpu")
    print("Running on the CPU")

# Legend
Klegend = ["KNet - Train", "KNet - Validation", "KNet - Test", "Kalman Filter"]
# Color
KColor = ['-ro', 'k-', 'b-','g-']

########
# Plot #
########
class Plot:
    
    ##################
    # Init the Plots #
    ##################
    # Specify the Folder Name and Model Name
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    #########################################
    # Plot the loss as a function of Epochs #
    #########################################
    # Plot the loss of KNet in Train, CV and Test, and KF.
    # N_Epochs_plt: Number of Epochs - x_axis
    # MSE_KF_dB_avg: Avg loss of the KF in dB. In KF_test.py!
    # MSE_test_dB_avg: Avg loss of the NN in dB during Testing.
    # MSE_cv_dB_epoch: Avg loss of the NN in dB during Validation.
    # MSE_train_dB_epoch: Avg loss of the NN in dB during Training.
    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        # Width x Height in inches
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        # Plotting Training Loss
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

        # CV
        # Plotting Validation Loss
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

        # Test
        # Plotting Testing Loss
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        # # Plotting KF Loss
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)

    ##################
    # Plot Histogram #
    ##################
    # One axis is MSE loss, the other one is prob density.
    def NNPlot_Hist(self, MSE_KF_data_linear_arr, MSE_KN_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'

        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        # OUTDATED
        # WARNING! distplot will be removed in future versions of Seaborn!
        # Use displot or histplot instead. Update: It happened.
        # distplot: Combines plt's with kdeplot (and rugplot).
        # hist: Computes and plots a histogram.
        # log10: Takes the log to base 10 of each element.
        # hist: Whether to plot a (normed) histogram.
        # kde: Whether to plot a gaussian kernel density estimate.
        # kde_kws: Keyword arguments for kdeplot.
        # kdeplot: Show a univariate or bivariate distribution 
        # with a kernel density estimate.
        # sns.distplot(10 * torch.log10(MSE_KN_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = self.modelName)
        # # Would be for the designed KF
        # #sns.distplot(10 * torch.log10(MSE_KF_design_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter - design')
        # sns.distplot(10 * torch.log10(MSE_KF_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'r', label = 'Kalman Filter')

        # Error: Dataset has 0 variance; skipping density estimate.
        sns.displot(data = 10 * torch.log10(MSE_KN_linear_arr), kind = "kde", color='g', lw = 3, label = self.modelName)
        # KF alone causes no error.
        sns.displot(data = 10 * torch.log10(MSE_KF_data_linear_arr), kind = "kde", color='r', lw = 3, label = 'Kalman Filter')

        plt.title("Histogram [dB]",fontsize=32)
        plt.legend(fontsize=32)
        plt.savefig(fileName)

    # KFPlot and NNPlot_test are not used atm.