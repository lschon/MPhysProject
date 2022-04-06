# Summary: plotter.py
# This file creates all the plots required throughout and is called in the main


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import models as m


# Creates scatter plot of the particles in the Universe within a cutoff range
def particle_scatter(df_near, df_max, cutoff):
    f, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(x="x_near", y="y_near", s=5, data=df_near, color="gray", label="Particles")
    ax.set_xlabel("x (Mpc/h)", fontsize=20)
    ax.set_ylabel("y (Mpc/h)", fontsize=20)
    # ax.set_title("Scatter plot for particles within " + str(
    #     cutoff) + " (Mpc/h) of the most massive halo from the Millenium simulation", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(float(df_max["x"]), float(df_max["y"]), marker="x", s=100, color="red",
                label="Most massive halo")  # position of max mass halo
    plt.legend()
    plt.show()


# KDE plot of the particles in the Universe within a cutoff range
def particle_kde(df_near, df_max, cutoff):
    f, ax = plt.subplots(figsize=(25, 20))
    sns.kdeplot(x="x_near", y="y_near", levels=5, data=df_near, shade=True,
                cbar=True)
    ax.set_xlabel("x (Mpc/h)", fontsize=20)
    ax.set_ylabel("y (Mpc/h)", fontsize=20)
    ax.set_title(
        "KDE plot for particles within " + str(
            cutoff) + " (Mpc/h) of the most massive halo from the Millenium simulation", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(float(df_max["x"]), float(df_max["y"]), marker="x", s=100)
    plt.show()


# Function to get the axes values to plot
def get_axes(df_mags):
    # Find the x_axis and Hubble Flow values for the Hubble Flow over-plotting
    x_axis = np.linspace(df_mags["dist_from_halo_Mpc"].min(), df_mags["dist_from_halo_Mpc"].max(), 1000)

    # Since H_0 = 100 h km/s/Mpc we multiply with h to get the Hubble Flow values
    y_HubbleFlow = 100 * 0.73 * x_axis
    return x_axis, y_HubbleFlow


# Make the residuals for the Hubble Flow
def make_residuals_HF(df_bins):
    # Finding the binned Hubble Flow values (i.e. so that there are 20 evenly spaced out values)
    HubbleFlow_binned = 100 * 0.73 * df_bins["bin_centers"].to_numpy()

    # Finding the residuals from a subtraction from observed values
    residuals_HF = HubbleFlow_binned - df_bins["mean_vels"].to_numpy()

    return residuals_HF


# Make the residuals for the Lam model and determine chi-squared of fit
def make_residuals_Lam(df_bins, DeltaM):
    Lam_binned = m.Lam(df_bins["bin_centers"], DeltaM)
    residuals_Lam = df_bins["mean_vels"] - Lam_binned
    chisq, _ = chisquare(f_obs=df_bins["mean_vels"][2:].to_numpy(), f_exp=np.asarray(Lam_binned[2:]))
    print("Lam model has chi-squared of: " + str(chisq))
    return residuals_Lam


# Plot the particle velocities and corresponding distances
def particle_vels(df_bins, df_mags, x_axis, y_HubbleFlow, residuals, df_models, label, best_fit_model):
    # Plot the radial velocities with Hubble flow versus magnitude of distance from halo
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 20),
                                   gridspec_kw={
                                       'height_ratios': [3, 1]})
    sns.scatterplot(x="dist_from_halo_Mpc", y="rad_vels_corr", s=2, data=df_mags, color="gray", label="Particles",
                    ax=ax1)
    ax1.set_xlabel("Magnitude of distance from most massive halo (Mpc)")
    ax1.set_ylabel("Radial velocity (km/s)")
    # ax1.set_title("Magnitude of distance from most massive halo versus particle radial velocities", fontsize=20)
    ax1.errorbar(df_bins["bin_centers"], df_bins["mean_vels"], yerr=df_bins["bin_errors"], color="black", capsize=5,
                 fmt="none", label=r"      $\sigma$ errors")
    ax1.set_ylim(-5000, 10000)
    ax1.legend()

    # Over-plot the Hubble flow
    sns.lineplot(x=x_axis, y=y_HubbleFlow, label="Hubble flow", color="black", linestyle="--", ax=ax1)

    # Over-plot binned values as scatter points
    sns.scatterplot(x="bin_centers", y="mean_vels", data=df_bins, color="black", label="Binned particles", marker="o",
                    ax=ax1)

    # Plot the residuals
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Magnitude of distance from most massive halo (Mpc)")

    # Plot Hubble Flow
    if label == "HF":
        ax2.set_ylabel("Hubble Flow residuals (km/s)")
        sns.scatterplot(data=df_bins, x="bin_centers", y=residuals, ax=ax2, label="Hubble flow residuals",
                        color="black")
        ax2.errorbar(df_bins["bin_centers"], residuals, yerr=df_bins["bin_errors"], color="black", capsize=5,
                     fmt="none",
                     label=r"$\sigma$ errors")
        ax2.legend()

    # Plot Lam Model
    elif label == "Lam":
        ax2.set_ylabel("Lam residuals (km/s)")
        sns.lineplot(data=df_models, x="x_axis", y="Lam", label="Lam", ax=ax1, color="blue")
        sns.scatterplot(x=df_bins["bin_centers"][2:], y=residuals[2:], ax=ax2, label="Lam",
                        color="blue")
        ax2.errorbar(df_bins["bin_centers"][2:], residuals[2:], yerr=df_bins["bin_errors"][2:], color="blue",
                     capsize=5, fmt="none", label=r"$\sigma$ errors")
        ax2.legend()

    # Plot MCMC fit
    elif label == "MCMC":

        residuals_MCMC = df_bins["mean_vels"][2:] - best_fit_model
        ax2.set_ylabel("Residuals (km/s)")
        sns.lineplot(x=df_bins["bin_centers"][2:], y=best_fit_model, label="MCMC", ax=ax1, color="red")
        sns.lineplot(data=df_models, x="x_axis", y="Lam", label="Lam model", ax=ax1, color="blue")
        sns.scatterplot(x=df_bins["bin_centers"][2:], y=residuals_MCMC, ax=ax2, label="MCMC residuals",
                        color="red")
        ax2.errorbar(df_bins["bin_centers"][2:], residuals_MCMC, yerr=df_bins["bin_errors"][2:], color="red",
                     capsize=5, fmt="none", label=r"$\sigma$ errors")
        sns.scatterplot(x=df_bins["bin_centers"][2:], y=residuals[2:], ax=ax2, label="Lam residuals",
                        color="blue")
        ax2.errorbar(df_bins["bin_centers"][2:], residuals[2:], yerr=df_bins["bin_errors"][2:], color="blue",
                     capsize=5, fmt="none", label=r"$\sigma$ errors")
        ax2.legend()
        chisq, _ = chisquare(f_obs=df_bins["mean_vels"][2:].to_numpy(), f_exp=best_fit_model)
        print("MCMC model has chi-squared of: " + str(chisq))

    plt.tight_layout()
    plt.show()
    # plt.savefig("test.pdf")
