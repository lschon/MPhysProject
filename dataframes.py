# Summary: dataframes.py
# This file either reads in dataframes which are already constructed and stored in .pkl files (called pickled files)
# or creates the dataframes themselves. As such, this is mainly a data-handling file, required to have the data
# in an easy, workable form. It is called in the main to retrieve the dataframes, which are then used in the MCMC
# Note that df is short for dataframe. Dataframes are a very handy feature of pandas, in Python, so if unfamiliar
# please take a look online: https://www.geeksforgeeks.org/introduction-to-pandas-in-python/

import pandas as pd
from scipy import stats
import math
import models as m


# Retrieve df pickle files from folder if they're available
def load_df(df_name):
    file = "./Dataframes/{name}.pkl".format(name=df_name)
    try:
        df = pd.read_pickle(file)
        print("SUCCESS: {name} was loaded in".format(name=df_name))
        return df
    except Exception as ex:
        print("FAILURE: {name} was NOT loaded in, because of: {ex}".format(ex=ex, name=df_name))
        return None


# Make the dark matter halo dataframe from a given file
def make_df_halos(filename):
    # DO NOT USE .txt FILES IN THE FUTURE. CHANGE TO .CSVs AND USE pd.read_csv INSTEAD!
    # then you can skip the infer_nrwos bit, its required because read_fwf "guesses" the limits on the data
    # based on the first hundreds of values by default and restricts the rest which comes, otherwise.
    # Hence, it is set to 35000, the length of the file
    df_halos = pd.read_fwf(filename, infer_nrows=35000)
    df_halos.columns = ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5", "Column 6",
                        "x", "y", "z", "Column 10", "Column 11", "halo mass",
                        "Column 13", "r_200", "Column 15"]
    df_halos.to_pickle('./Dataframes/df_halos.pkl')
    print("The dark matter halo data has been successfully loaded.")
    return df_halos


# Make the one-lined dataframe containing the info of the halo with desired mass
def make_df_max(df_halos, num):
    # Take the halo mass column and sort by mass and choose which halo by setting num.
    # e.g. most massive halo has num=0
    # Note: it is called max_mass but it is not necessarily always the maximum mass, it is set by num
    sorted_halos = df_halos.sort_values(by='halo mass', ascending=False)
    max_mass = sorted_halos["halo mass"].to_numpy()[num]
    print("The reference halo mass is " + str(max_mass) + " M_sun / h / 10^10")

    # Make single-lined dataframe containing the attributes of the chosen halo. Option to pickle.
    df_max = df_halos.loc[df_halos["halo mass"] == max_mass]
    # df_max.to_pickle('./Dataframes/df_max.pkl')

    print("The information of the dark matter halo with max mass has been successfully obtained.")
    return df_max, max_mass


# Make the particles dataframe from a given file
def make_df_particles(filename):
    df_particles = pd.read_fwf(filename, infer_nrows=10000000)
    df_particles.columns = ["x", "y", "z", "vx", "vy", "vz"]
    df_particles.to_pickle('./Dataframes/df_particles.pkl')
    print("The particle data has been successfully loaded.")
    return df_particles


# Make the dataframe of particles within a cutoff range
def make_df_near(cutoff, df_particles, df_max):
    # Note: defining the cutoff range in Mpc/h
    # Define empty lists for the x, y, and z values of the particles within a certain range
    x_near = []
    y_near = []
    z_near = []
    vx_near = []
    vy_near = []
    vz_near = []
    xdiff_near = []
    ydiff_near = []
    zdiff_near = []

    # Scan df_particles for values and add to the position lists
    i = 0
    for ind in df_particles.index:

        # Find the difference in distance
        xdiff = df_particles["x"][ind] - df_max["x"]
        ydiff = df_particles["y"][ind] - df_max["y"]
        zdiff = df_particles["z"][ind] - df_max["z"]

        if i % 100000 == 0:
            print("Trial number: " + str(i))

        # Add values if close enough
        if (float(abs(xdiff)) < cutoff) and (float(abs(ydiff)) < cutoff) and (float(abs(zdiff)) < cutoff):
            x_near.append(df_particles["x"][ind])
            y_near.append(df_particles["y"][ind])
            z_near.append(df_particles["z"][ind])
            vx_near.append(df_particles["vx"][ind])
            vy_near.append(df_particles["vy"][ind])
            vz_near.append(df_particles["vz"][ind])
            xdiff_near.append(float(xdiff))
            ydiff_near.append(float(ydiff))
            zdiff_near.append(float(zdiff))
        i += 1

    # Construct a dataframe using these lists. This is less expensive than making one from the ground up.
    df_near = pd.DataFrame(
        list(zip(x_near, y_near, z_near, vx_near, vy_near, vz_near, xdiff_near, ydiff_near, zdiff_near)),
        columns=["x_near", "y_near", "z_near", "vx_near", "vy_near",
                 "vz_near", "xdiff_near", "ydiff_near", "zdiff_near"])
    print("Particles within {cutoff} Mpc/h have successfully been found, with {length} values.".format(cutoff=cutoff,
                                                                                                       length=len(
                                                                                                           df_near)))

    return df_near


# Define a function which calculates the magnitude given input values
def magnitude(x, y, z):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r


# Make the dataframe involving all the velocity and distance magnitude values of the particles
def make_df_mags(df_near, H_0=100, h=0.73):
    # Create empty lists for the velocities and distances
    velocities = []
    distances = []
    distances_from_halo = []
    distances_from_halo_Mpc = []

    # For each component of the df_near dataframe, find the overall distance and velocity and add to lists
    for ind in df_near.index:
        # Find the magnitudes of the distance, the distance from the halo, and the velocities of the particles
        dist = magnitude(float(df_near["x_near"][ind]), float(df_near["y_near"][ind]), float(df_near["z_near"][ind]))
        dist_from_halo = magnitude(float(df_near["xdiff_near"][ind]), float(df_near["ydiff_near"][ind]),
                                   float(df_near["zdiff_near"][ind]))
        vel = magnitude(float(df_near["vx_near"][ind]), float(df_near["vy_near"][ind]), float(df_near["vz_near"][ind]))

        # Append to the lists
        distances.append(dist)
        distances_from_halo.append(dist_from_halo)

        # Divide by h because you have the data in Mpc, but you want in Mpc/h so times data in Mpc by h/h and so h*data in Mpc/h
        # thus divide by h to cancel out
        distances_from_halo_Mpc.append(dist_from_halo / h)
        velocities.append(vel)

    # Create dataframe for these magnitudes from the lists. NOTE: vel is trans velocities not radial
    df_mags = pd.DataFrame(list(zip(distances, distances_from_halo, distances_from_halo_Mpc, velocities)),
                           columns=["dist", "dist_from_halo", "dist_from_halo_Mpc", "trans_vels"])

    # Also include the radial velocities
    # Without Hubble Flow, the calculated radial velocities are the following
    # units of Mpc/h for x/y/zdiff & dist_from_halo and km/s for vx/y/z so cancels to give km/s.
    df_mags["rad_vels"] = (((df_near.xdiff_near * df_near.vx_near) +
                            (df_near.ydiff_near * df_near.vy_near) +
                            (df_near.zdiff_near * df_near.vz_near)) /
                           df_mags.dist_from_halo)

    # With Hubble Flow, we just add H_0 * R
    # Units for H_0 are h*(km/s)/Mpc so  h km s^-1 Mpc^-1. Cancels with Mpc/h to give km/s on left term
    # Units are km/s overall for all velocities
    df_mags["rad_vels_corr"] = H_0 * df_mags["dist_from_halo"] + df_mags["rad_vels"]

    # Save as a .pkl
    df_mags.to_pickle("./Dataframes/df_mags.pkl")
    print("The magnitudes of the velocities of the particles has successfully been determined.")
    return df_mags


# Make the dataframe for the binned velocity values of the particles
def make_df_bins(df_mags, bins):
    # Find the relevant binned values
    mean_vels, bin_edges, binnumber = stats.binned_statistic(df_mags["dist_from_halo_Mpc"],
                                                             df_mags["rad_vels_corr"], statistic="mean", bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    # Add useful values to df_mags if it doesn't already have that column
    if not ("binnumber" in df_mags):
        df_mags["binnumber"] = binnumber

    # Create dataframe for the binned values with velocities (km/s) and bin centers (in Mpc)
    df_bins = pd.DataFrame(list(zip(mean_vels, bin_centers)),
                           columns=["mean_vels", "bin_centers"])

    # Finding the errors in each bin
    bin_errors = []
    for i in range(len(df_bins["bin_centers"])):
        bin_errors.append(df_mags[df_mags["binnumber"] == (i + 1)]["rad_vels_corr"].std())

    # Adding to bin dataframe
    df_bins["bin_errors"] = bin_errors

    # Save as a .pkl
    df_bins.to_pickle("./Dataframes/df_bins.pkl")
    print("The particle velocities have successfully been binned")
    return df_bins


# Make the dataframe of the values of the model predictions
def make_df_models(DeltaM, x_axis):
    # Determining all the theoretical model values and making into a dataframe
    df_models = pd.DataFrame(
        list(zip(x_axis, m.Penarrubia(x_axis, DeltaM), m.Aavik(x_axis, DeltaM), m.Lam(x_axis, DeltaM))),
        columns=["x_axis", "Penarrubia", "Aavik", "Lam"])
    # df_models.to_pickle("./Dataframes/df_models.pkl")
    print("The particle velocity model predictions have successfully been calculated.")
    return df_models
