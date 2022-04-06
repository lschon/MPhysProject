# Summary: main_kim.py
# This file uses the data from Kim et al. (2020) found at https://iopscience.iop.org/article/10.3847/1538-4357/abbd97/pdf
# This file is the "main" for this part of the project.
# The data is hard-coded in kim_data() and then the data is plotted and the MCMC is called.


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import models as m
import MCMC_kim
import MCMC_kim2


# Function with hard-coded values from the Kim paper
def kim_data():
    D_trgb = [11.87, 10.01, 9.92, 9.64, 9.28, 9.07, 9.07, 8.87, 7.80, 7.64, 6.78, 7.36, 6.54, 6.42, 6.39, 5.86,
              5.85, 5.47, 6.02, 5.03, 4.68, 4.75, 4.90, 4.46, 4.75, 4.93, 4.87, 4.69, 4.72, 4.75, 4.56, 3.94, 4.26]
    D_err_lower = [0.53, 0.91, 0.34, 0.75, 0.40, 0.34, 0.27, 0.30, 0.25, 0.42, 0.37, 0.20, 0.27, 0.32, 0.34, 0.22,
                   0.22, 0.20, 0.25, 0.32, 0.19, 0.15, 0.24, 0.13, 0.12, 0.26, 0.36, 0.30, 0.25, 0.23, 0.25, 0.11, 0.18]
    D_err_upper = [0.53, 1.00, 0.34, 1.31, 0.40, 0.34, 0.27, 0.30, 0.25, 0.42, 0.37, 0.20, 0.27, 0.32, 0.34, 0.22,
                   0.22, 0.20, 0.25, 0.32, 0.19, 0.15, 0.24, 0.13, 0.12, 0.26, 0.36, 0.35, 0.25, 0.23, 0.25, 0.11, 0.18]
    D_errors = [D_err_lower, D_err_upper]
    R_vc = [5.26, 6.56, 6.58, 6.90, 7.68, 7.69, 7.93, 8.30, 9.55, 9.79, 9.92, 9.93, 10.12, 10.25, 10.26, 10.79, 11.0,
            11.26, 11.41, 11.89, 11.90, 11.90, 11.97, 12.15, 12.18, 12.19, 12.23, 12.33, 12.37, 12.56, 12.56, 12.75,
            12.78]
    V_min = [-217, -37, -44, -205, 90, 318, 150, 287, 430, 500, 438, 462, 420, 532, 461, 469, 566, 591, 590, 664,
             633, 767, 694, 634, 705, 783, 692, 708, 734, 631, 665, 654, 765]
    V_max = [-461, -54, -46, -223, -7, 297, 57, 203, 353, 447, 421, 395, 402, 523, 448, 454, 546, 579, 523, 650, 629,
             772,
             688, 627, 696, 808, 674, 696, 730, 545, 621, 638, 765]
    # Note: 33 values in each

    # Create dataframe with all the values
    df_kim = pd.DataFrame(
        list(zip(D_trgb, D_err_lower, D_err_upper, R_vc, V_min, V_max)),
        columns=["D_trgb", "D_err_lower", "D_err_upper", "R_vc", "V_min", "V_max"])

    return df_kim, D_errors


# Call kim_data() function
df_kim, D_errors = kim_data()

# Create the arrays needed to plot and work with the data
x_axis = np.linspace(0, 14, 1000)
x_axis2 = np.linspace(0, 14, 33)
y_HubbleFlow = 100 * 0.73 * x_axis
virial_mass = 6.3 * 10 ** 14 * 0.73  # mass in M_sun / h
Lam_binned = m.Lam(x_axis, virial_mass)
Lam_binned2 = m.Lam(x_axis2, virial_mass)
Verr = np.array(D_errors) * 100  # from Hubble law v = H_0 * R


# Plots the minimum estimated velocities from Kim et al.
def kim_plotter_min(df_kim, MCMC, best_fit_model):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    sns.scatterplot(x=df_kim["R_vc"], y=df_kim["V_min"], s=15, color="blue", label="v_min",
                    ax=ax1)

    sns.lineplot(x=x_axis, y=y_HubbleFlow, label="Hubble flow", color="black", linestyle="--", ax=ax1)
    sns.lineplot(x=x_axis, y=Lam_binned, label="Lam model", ax=ax1, color="black")

    ax1.errorbar(x=df_kim["R_vc"], y=df_kim["v_min"], xerr=D_errors, yerr=Verr, color="blue", capsize=5,
                 fmt="none", label="v_min errors")
    if MCMC:
        sns.lineplot(x=df_kim["R_vc"], y=best_fit_model, label="MCMC model", ax=ax1, color="green", linestyle="--")

    ax1.axhline(0, color="gray", linestyle="-")
    ax1.set_xlabel("Distance (Mpc)")
    ax1.set_ylabel("Velocity (km/s)")
    ax1.set_ylim(-750, 1200)
    ax1.set_xlim(0, 14)
    ax1.legend()
    plt.tight_layout()
    plt.show()


# Plots the maximum estimated velocities from Kim et al.
def kim_plotter_max(df_kim, MCMC, best_fit_model):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
    sns.scatterplot(x=df_kim["R_vc"], y=df_kim["V_max"], s=15, color="red", label="v_max",
                    ax=ax1)

    sns.lineplot(x=x_axis, y=y_HubbleFlow, label="Hubble flow", color="black", linestyle="--", ax=ax1)
    sns.lineplot(x=x_axis, y=Lam_binned, label="Lam model", ax=ax1, color="black")
    ax1.errorbar(x=df_kim["R_vc"], y=df_kim["_max"], xerr=D_errors, yerr=Verr, color="red", capsize=5,
                 fmt="none", label="v_max errors")

    if MCMC:
        sns.lineplot(x=df_kim["R_vc"], y=best_fit_model, label="MCMC model", ax=ax1, color="green", linestyle="--")

    ax1.axhline(0, color="gray", linestyle="-")
    ax1.set_xlabel("Distance (Mpc)")
    ax1.set_ylabel("Velocity (km/s)")
    ax1.set_ylim(-750, 1200)
    ax1.set_xlim(0, 14)
    ax1.legend()
    plt.tight_layout()
    plt.show()


# Conducts the MCMC for the Kim et al. data
def do_MCMC_kim(mass, data, minimum):
    # Detect if it's the minimum or maximum estimated velocities
    if minimum:
        kim_plotter_min(df_kim, MCMC=False, best_fit_model=None)
    else:
        kim_plotter_max(df_kim, MCMC=False, best_fit_model=None)

    # Run the MCMC and present results
    print("Running 2 variable MCMC with H_0 and Omega_m.")
    sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, best_fit_model = MCMC_kim2.run_MCMC2(
        DeltaM=mass, df_kim=data, error=Verr, minimum=minimum)
    MCMC_kim2.parameter_plot(sampler, ndim)
    MCMC_kim2.autocorr_time(sampler)
    MCMC_kim2.present_results(sampler, H_0_true, logDeltaM_true)
    MCMC_kim2.corner_plot(sampler, H_0_true, logDeltaM_true)

    # Print the determined parameter values
    samples = sampler.flatchain
    mcmc1 = np.percentile(samples[:, 0], [16, 50, 84])
    q1 = np.diff(mcmc1)
    mcmc2 = np.percentile(samples[:, 1], [16, 50, 84])
    q2 = np.diff(mcmc2)
    print(mcmc1[1], q1[0], q1[1], mcmc2[1], q2[0], q2[1])

    # Plot the MCMC results
    if minimum:
        kim_plotter_min(df_kim, MCMC=True, best_fit_model=best_fit_model)
    else:
        kim_plotter_max(df_kim, MCMC=True, best_fit_model=best_fit_model)

    print("MCMC complete.")


# Conduct both minimum and maximum estimated velocity trials
do_MCMC_kim(mass=virial_mass, data=df_kim, minimum=True)
do_MCMC_kim(mass=virial_mass, data=df_kim, minimum=False)
