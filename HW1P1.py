''' HW1P1
    EECE5644
    Ethan Holand
    5/25/2022
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

# Given values:
N = 10000 #Number of samples generated

# Class priors:
priors = np.array([0.65, 0.35])
thresholds = np.cumsum(priors) # Cumulative sum of prior probabilities
thresholds = np.insert(thresholds, 0, 0) # Append a 0 at the start

# Mean and covariance of data pdfs:
mu = np.array([[-0.5, 0.5, 0.5],
               [1, 1, 1]]) # Gaussian distributions means
Sigma = np.array([[[1, -0.5, 0.3], 
                   [-0.5, 1, -0.5], 
                   [0.3, -0.5, 1]],
                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]]) # Gaussian distributions covariance matrices

n = mu.shape[1] # Determine dimensionality from mixture PDF parameters
C = len(priors)
u = np.random.rand(N) #string of N randomly generated variables, from 0 to 1

# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N) # KEEP TRACK OF THIS

# Plot for original data and their true labels
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
marker_shapes = 'd+.'
marker_colors = 'rbg' 

L = np.array(range(1, C+1))
for l in L:
    # Get randomly sampled indices for this component
    indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]

    # No. of samples in this component
    Nl = len(indices)  
    labels[indices] = l * np.ones(Nl)
    X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl)
    plt.plot(X[labels==l, 0], X[labels==l, 1], X[labels==l, 2], marker_shapes[l-1] + marker_colors[l-1], label="Class {}".format(l-1))

# Plot the original data and their true labels
plt.legend()
ax.set_xlabel(r"x-axis")
ax.set_ylabel(r"y-axis")
ax.set_zlabel(r"z-axis")
plt.title("Generated Original Data Samples")
plt.tight_layout()
plt.show()


## PART A: ERM CLASSIFICATION USING THE KNOWLEDGE OF TRUE DATA PDF:
