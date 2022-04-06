# Summary: cosmicvariance.py
# This file reads the data from the cosmicvariance.txt file and takes the H_0 values and determines
# the cosmic variance for the values. It also makes a histogram for the data. It is not very long or complicated.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Customisation of maplotlib plots
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

# Read text file in with data and create a dataframe for the whole thing
filename = "./cosmicvariance.txt"
df_cosvar = pd.read_csv(filename, sep="\t", header=None)
df_cosvar.columns = ["Trial", "H0", "H0_err_lower", "H0_err_upper", "logDeltaM", "logDeltaM_err_lower",
                     "logDeltaM_err_upper", "true_DeltaM", "true_logDeltaM"]

# Print the dataframe as well as the mean and actual H without h=0.73 incorporated and the standard deviation
print(df_cosvar)
print(df_cosvar["H0"].mean())
print(df_cosvar["H0"].mean() * 0.73)
print(df_cosvar["H0"].std())

# Convert all the H0 values to proper H0 without h=0.73
H0vals = (df_cosvar["H0"] * 0.73).to_numpy()

# Plot the histogram of the H0 values
plt.figure(figsize=(20, 15))
_ = plt.hist(H0vals, bins=10, facecolor="gray", rwidth=0.97)
plt.axvline(x=(df_cosvar["H0"].mean() * 0.73), lw=3, ls="--", color="black")
plt.xlabel(r"$H_0$ (km/s)/Mpc")
plt.ylabel("Frequency")
plt.show()
