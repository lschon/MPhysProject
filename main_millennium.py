# Summary: main_millennium.py
# This file is the "main" for the Millennium Simulation data from Springel et al. (2005) at https://www.nature.com/articles/nature03597
# This reads in all the data and uses the other complementary files to construct them as dataframes and run the MCMC on them
# in order to determine the parameter values involved.

# Importing standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import seaborn as sns
import csv

# Importing Markov Chain Monte Carlo packages
import emcee
import corner
from tqdm import tqdm

# Importing other python scripts. Note each MCMC file uses different parameters
import dataframes as dfs
import plotter
import MCMC_3var as MCMC3
import MCMC_2var as MCMC2
import MCMC_2var2 as MCMC22
import MCMC_2var3 as MCMC23


# Retrieve all the data required and make them into dataframes (see dataframes.py)
def get_data(num):
    # Define filenames
    halos_filename = "./Millenium_Data/mbp_positions_063_M13_matched.txt"
    particles_filename = "./Millenium_Data/ParticleData_Point1Percent.txt"
    print("Loading in the required Dataframes:")

    # Get the halo dataframe
    df_halos = dfs.load_df("df_halos")
    if df_halos is None:
        df_halos = dfs.make_df_halos(halos_filename)

    # Get the one-line dataframe which is the halo of maximum mass & find the maximum mass and save variable
    df_max, max_mass = dfs.make_df_max(df_halos, num)
    max_mass_new = max_mass * 10 ** 10  # M is the halo mass in M_sun/h (timed by 10^10 because original data was divided by that)

    # Get the particles dataframe, containing all the particle data
    df_particles = dfs.load_df("df_particles")
    if df_particles is None:
        df_particles = dfs.make_df_particles(particles_filename)

    # Get the dataframe containing all the particles within
    cutoff = 50
    df_near = dfs.load_df(f"df_near{num}")
    if df_near is None:
        df_near = dfs.make_df_near(cutoff, df_particles, df_max)
        df_near.to_pickle(f'./Dataframes/df_near[{num}].pkl')

    # Get the dataframe involving all the velocity and distance values of the particles
    df_mags = dfs.make_df_mags(df_near)

    # Get the dataframe of the binned velocity values
    bins = 20
    df_bins = dfs.make_df_bins(df_mags, bins)

    # Get the x and y-axis for the standard plot
    x_axis, y_HubbleFlow = plotter.get_axes(df_mags)

    # Get the dataframe of the values of the model predictions
    df_models = dfs.make_df_models(DeltaM=max_mass_new, x_axis=x_axis)

    return df_halos, df_max, max_mass, max_mass_new, df_particles, df_near, df_mags, df_bins, x_axis, y_HubbleFlow, df_models, cutoff, bins


# Plot all the dark matter particles within the box
def plot_particles(df_near, df_max, cutoff):
    print("Plotting particles as scatter plot...")
    plotter.particle_scatter(df_near, df_max, cutoff)
    print("Plotting complete.")


# Plot the density distribution of the particles within the box
def plot_particles_KDE(df_near, df_max, cutoff):
    print("Plotting particles as KDE plot...")
    plotter.particle_kde(df_near, df_max, cutoff)
    print("Plotting complete.")


# Plot the particle velocities with residuals
def plot_HF(df_bins, df_mags, x_axis, y_HubbleFlow):
    print("Plotting the velocities of particles with Hubble Flow.")
    residuals_HF = plotter.make_residuals_HF(df_bins)
    plotter.particle_vels(df_bins, df_mags, x_axis, y_HubbleFlow, residuals_HF, df_models=None, label="HF",
                          best_fit_model=None)
    print("Plotting complete.")


# Plot the particle velocities with Lam model over-plotted
def plot_Lam(df_bins, df_mags, x_axis, y_HubbleFlow, df_models, DeltaM):
    print("Plotting the velocities of particles with Lam model.")
    residuals_Lam = plotter.make_residuals_Lam(df_bins, DeltaM)
    plotter.particle_vels(df_bins, df_mags, x_axis, y_HubbleFlow, residuals_Lam, df_models, label="Lam",
                          best_fit_model=None)
    print("Plotting complete.")


# Plot the particle velocities with Lam and MCMC fit
def plot_MCMC(df_bins, bins, df_mags, x_axis, y_HubbleFlow, df_models, best_fit_model, DeltaM):
    print("Plotting the velocities of particles with Hubble Flow.")
    residuals_Lam = plotter.make_residuals_Lam(df_bins, DeltaM)
    plotter.particle_vels(df_bins, df_mags, x_axis, y_HubbleFlow, residuals=residuals_Lam, df_models=df_models,
                          label="MCMC",
                          best_fit_model=best_fit_model)
    print("Plotting complete.")


# Conduct MCMC with 3 variables
def do_MCMC3(max_mass_new, df_bins):
    print("Running 3 variable MCMC with H_0, Omega_m, and logDeltaM.")
    sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, Omega_m_true, best_fit_model = MCMC3.run_MCMC3(
        DeltaM=max_mass_new, df_bins=df_bins)
    MCMC3.parameter_plot(sampler, ndim)
    MCMC3.autocorr_time(sampler)
    MCMC3.present_results(sampler, H_0_true, logDeltaM_true, Omega_m_true, ndim)
    MCMC3.corner_plot(sampler, H_0_true, logDeltaM_true, Omega_m_true)
    print("MCMC complete.")
    return best_fit_model


# Option to conduct MCMC with 2 variables, logDeltaM and Omega_m (not really used)
def do_MCMC2(max_mass_new, df_bins):
    print("Running 2 variable MCMC with logDeltaM and Omega_m")
    sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, Omega_m_true, best_fit_model = MCMC2.run_MCMC2(
        DeltaM=max_mass_new,
        df_bins=df_bins)
    MCMC2.parameter_plot(sampler, ndim)
    MCMC2.autocorr_time(sampler)
    MCMC2.present_results(sampler, logDeltaM_true, Omega_m_true, ndim)
    MCMC2.corner_plot(sampler, logDeltaM_true, Omega_m_true)
    print("MCMC complete.")
    return best_fit_model


# Option to conduct MCMC with 2 variables, H_0 and Omega_m (also not really used)
def do_MCMC22(max_mass_new, df_bins):
    print("Running 2 variable MCMC with H_0 and Omega_m")
    sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, Omega_m_true, best_fit_model = MCMC22.run_MCMC22(
        DeltaM=max_mass_new,
        df_bins=df_bins)
    MCMC22.parameter_plot(sampler, ndim)
    MCMC22.autocorr_time(sampler)
    MCMC22.present_results(sampler, H_0_true, Omega_m_true, ndim)
    MCMC22.corner_plot(sampler, H_0_true, Omega_m_true)
    print("MCMC complete.")
    return best_fit_model


# Option to conduct MCMC with 2 variables, H_0 and logDeltaM (this is the one that is used)
def do_MCMC23(max_mass_new, df_bins):
    print("Running 2 variable MCMC with H_0 and logDeltaM")
    sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, best_fit_model = MCMC23.run_MCMC23(
        DeltaM=max_mass_new,
        df_bins=df_bins)
    MCMC23.parameter_plot(sampler, ndim)
    MCMC23.autocorr_time(sampler)
    MCMC23.present_results(sampler, H_0_true, logDeltaM_true)
    MCMC23.corner_plot(sampler, H_0_true, logDeltaM_true)
    print("MCMC complete.")
    return best_fit_model, sampler


# Main function which combines everything. Used during the beginning of the development
def old_main():
    # Get all the data and plot all the desired plots
    df_halos, df_max, max_mass, max_mass_new, df_particles, df_near, \
    df_mags, df_bins, x_axis, y_HubbleFlow, df_models, cutoff, bins = get_data(num=0)
    plot_particles(df_near, df_max, cutoff)
    plot_particles_KDE(df_near, df_max, cutoff)
    plot_HF(df_bins, df_mags, x_axis, y_HubbleFlow)
    plot_Lam(df_bins, df_mags, x_axis, y_HubbleFlow, df_models, DeltaM=max_mass_new)

    # Options to do different MCMC fits but the H_0 and logDeltaM 2 parameter fit is the best one
    # best_fit_model = do_MCMC3(max_mass_new, df_bins)
    # best_fit_model = do_MCMC2(max_mass_new, df_bins)
    # best_fit_model = do_MCMC22(max_mass_new, df_bins)
    best_fit_model = do_MCMC23(max_mass_new, df_bins)
    plot_MCMC(df_bins, bins, df_mags, x_axis, y_HubbleFlow, df_models, best_fit_model, DeltaM=max_mass_new)


# Newer main function for a single dark matter halo (i.e. not repeated)
def single_main(num):
    print("Ongoing trial: " + str(num))

    # Get all the data and do the 2-parameter MCMC
    df_halos, df_max, max_mass, max_mass_new, df_particles, df_near, \
    df_mags, df_bins, x_axis, y_HubbleFlow, df_models, cutoff, bins = get_data(num=num)
    best_fit_model, sampler = do_MCMC23(max_mass_new, df_bins)
    plot_MCMC(df_bins, bins, df_mags, x_axis, y_HubbleFlow, df_models, best_fit_model, DeltaM=max_mass_new)
    logDeltaM_true = round(math.log10(max_mass_new), 10)

    # Get the MCMC parameter results
    samples = sampler.flatchain
    mcmc1 = np.percentile(samples[:, 0], [16, 50, 84])
    q1 = np.diff(mcmc1)
    mcmc2 = np.percentile(samples[:, 1], [16, 50, 84])
    q2 = np.diff(mcmc2)

    # Print the results of the MCMC
    print(num, mcmc1[1], q1[0], q1[1], mcmc2[1], q2[0], q2[1], max_mass_new, logDeltaM_true)

    print("Trial " + str(num) + " complete.")


# Similarly new main function which runs several trials in a row
def rep_main(start, stop):
    # Writes the results in a csv file. Note: the results appear only at the very end when the loop is complete
    # I'm not quite sure why that is, but that's the case. Also, change the filename because it'll re-write the file
    # and you'll lose the data otherwise if you do successive runs.
    with open('millennium_output.csv', 'w') as f1:
        # Open a csv file to write on
        writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )

        # Run trials in manually inputted range, where start and stop are the positions of
        # the sorted most massive halos list. e.g. start=0, stop=5 goes from the most massive halo to fifth most massive
        for i in range(start, stop):
            print("Ongoing trial: " + str(i))

            # Get all the data and do the 2-parameter MCMC
            df_halos, df_max, max_mass, max_mass_new, df_particles, df_near, \
            df_mags, df_bins, x_axis, y_HubbleFlow, df_models, cutoff, bins = get_data(num=i)
            best_fit_model, sampler = do_MCMC23(max_mass_new, df_bins)
            plot_MCMC(df_bins, bins, df_mags, x_axis, y_HubbleFlow, df_models, best_fit_model, DeltaM=max_mass_new)
            logDeltaM_true = round(math.log10(max_mass_new), 10)

            # Get the MCMC parameter results
            samples = sampler.flatchain
            mcmc1 = np.percentile(samples[:, 0], [16, 50, 84])
            q1 = np.diff(mcmc1)
            mcmc2 = np.percentile(samples[:, 1], [16, 50, 84])
            q2 = np.diff(mcmc2)

            # Print the results of the MCMC and write them into .csv file
            print(i, mcmc1[1], q1[0], q1[1], mcmc2[1], q2[0], q2[1], max_mass_new, logDeltaM_true)
            row = [i, mcmc1[1], q1[0], q1[1], mcmc2[1], q2[0], q2[1], max_mass_new, logDeltaM_true]
            writer.writerow(row)
            print("Trial " + str(i) + " complete.")


# Change this to choose your main and which halo you want as the reference point
if __name__ == "__main__":
    # old_main()
    single_main(num=0)
    # rep_main(start=0, stop=5)
