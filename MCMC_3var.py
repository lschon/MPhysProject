# Summary: MCMC_3var.py
# This conducts the MCMC for H_0 and logDeltaM and Omega_m for the Millennium Simulation data.
# I'm not going to comment too much on this file because it's not really used.
# See MCMC_2var3.py for more notes.

import math
import emcee
import corner
import models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

# Allows use of LaTeX text in plots
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)


def run_MCMC3(DeltaM, df_bins):
    # Get the values for the constants used from models.py script
    H_0, h, km_in_Mpc, G, Omega_Lambda, Omega_m, t0, f, delta_c = models.get_constants()
    Omega_m = 0.3111

    # Save the "true" values of the parameters used
    def true_params():
        H_0_true = H_0
        DeltaM_true = DeltaM
        logDeltaM_true = round(math.log10(DeltaM_true), 10)
        Omega_m_true = Omega_m
        return H_0_true, logDeltaM_true, Omega_m_true

    H_0_true, logDeltaM_true, Omega_m_true = true_params()

    # Delete the parameters that are used in the fitting
    del H_0, Omega_m, DeltaM, f

    # Retrieve data
    def get_datasets():
        # Define our data sets as arrays
        dist = df_bins["bin_centers"][2:].to_numpy()
        V = df_bins["mean_vels"][2:].to_numpy()
        Verr = df_bins["bin_errors"][2:].to_numpy()
        return dist, V, Verr

    dist, V, Verr = get_datasets()

    # Define our model function (Lam's model)
    def model(theta, R=dist):
        # Defines our 3 variables from some inputted theta tuple with 3 values
        H_0, logDeltaM, Omega_m = theta
        # f = Omega_m ** 0.55
        # Lam's model
        model = H_0 * h * R * (1 - (Omega_m ** 0.55 * delta_c / 3) * (
                ((2 * G * 10 ** logDeltaM / h) / ((H_0 * h) ** 2 * Omega_m * R ** 3) + 1) ** (1 / delta_c) - 1))
        # model = H_0 * h * R * (1 - (f * delta_c / 3) * ( ( (2 * G * DeltaM / h) / ((H_0 * h)**2 * Omega_m * R**3) + 1)**(1 / delta_c) - 1) )
        return model

    # From document, lnlike:
    # returns a number corresponding to how good a fit your model is to your data for a given set of parameters,
    # weighted by the error in your data points (i.e. it is more important the fit be close to data points with
    # small error bars than points with large error bars)
    def lnlike(theta, x, y, yerr):
        LnLike = -0.5 * np.sum(((y - model(theta)) / yerr) ** 2)
        return LnLike

    #  Note: adjust this into the more advanced covariance type -- np.cov

    # lnprior defines the bounds for our variables
    def lnprior(theta):
        H_0, logDeltaM, Omega_m = theta
        if 50 < H_0 < 150 and 10 < logDeltaM < 20 and 0.2943 < Omega_m < 0.3279:
            # if 50 < H_0 < 150 and 10 < logDeltaM < 20 and 0.001 < Omega_m < 1:
            return 0.0
        else:
            return -np.inf

    # lnprob: Combines steps above to produce a probability of a result
    # If lnprior returns infinity, return infinity (unusable)
    # If lnprior returns 0.0, return lp + lnlike() but lp=0 so just returns lnlike()
    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if math.isinf(lp):
            return -np.inf
        else:
            return lp + lnlike(theta, x, y, yerr)  # recall if lp not -inf, its 0, so this just returns likelihood

    # Define the data as a tuple of arrays, the number of walkers, and the number of iterations
    data = (dist, V, Verr)
    nwalkers = 1000
    niter = 100000

    # Define the initial true values and the number of dimensions
    initial = np.array([H_0_true, logDeltaM_true, Omega_m_true])
    ndim = len(initial)

    print("Initial guesses for H_0, logDeltaM, and Omega_m (which are the 'true' values):")
    print(initial)

    # Small piece of code for the random adjustments to the p0 guess, where each value needs to be scaled because
    # of their vastly different orders of magnitude
    rand_adjustment = [np.random.randn(ndim) for i in range(nwalkers)]
    for i in range(len(rand_adjustment)):
        rand_adjustment[i][0] *= 5
        rand_adjustment[i][1] *= 1
        rand_adjustment[i][2] = 0.001 * abs(rand_adjustment[i][2])

    # Create the p0 guess by initializing an array with our true values and adding the random adjustments onto them
    p0_init = [initial for i in range(nwalkers)]
    p0 = np.zeros((len(p0_init), ndim))
    for i in range(len(p0_init)):
        p0[i] = p0_init[i] + rand_adjustment[i]
    x = 0

    # Main function which runs the MCMC

    def main(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("Running burn-in...")
        # p0, _, _ = sampler.run_mcmc(p0, 100, skip_initial_state_check=True, progress = False);
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=False);
        sampler.reset()

        print("Burn-in complete.")
        print()

        start = datetime.now()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        end = datetime.now()
        time_taken = end - start
        print("Time taken for MCMC: ", time_taken)
        return sampler, pos, prob, state

    sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

    def MCMC_plot():
        plt.ion()
        plt.figure(figsize=(20, 20))
        plt.scatter(dist, V, label="Binned particles", color="black")
        samples = sampler.flatchain
        for theta in samples[np.random.randint(len(samples), size=100)]:
            fig = plt.plot(dist, model(theta), color="r", alpha=0.1)
        plt.errorbar(dist, V, yerr=Verr, color="black", capsize=5, fmt="none")
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.xlabel("Magnitude of distance from most massive halo (Mpc)")
        plt.ylabel("Radial velocity (km/s)")
        # plt.title("MCMC results on radial velocity versus distance from most massive halo simulation data")
        plt.legend()
        # plt.ylim(-1000, 10000)

        plt.show()

    MCMC_plot()

    def MCMC_plot2():
        samples = sampler.flatchain
        theta_max = samples[np.argmax(sampler.flatlnprobability)]
        best_fit_model = model(theta_max)

        def sample_walkers(nsamples, flattened_chain):
            models = []
            draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
            thetas = flattened_chain[draw]
            for i in thetas:
                mod = model(i)
                models.append(mod)
            spread = np.std(models, axis=0)
            med_model = np.median(models, axis=0)
            return med_model, spread

        med_model, spread = sample_walkers(100, samples)

        plt.figure(figsize=(20, 20))
        plt.plot(dist, V, label="Original data")
        plt.plot(dist, best_fit_model, label="Highest likelihood model")
        plt.fill_between(dist, med_model - spread, med_model + spread, color='grey', alpha=0.5,
                         label=r'$1\sigma$ posterior spread')
        # plt.title("Comparison of MCMC highest likelihood model vs data")
        plt.xlabel("Magnitude of distance from most massive halo (Mpc)")
        plt.ylabel("Radial velocity (km/s)")
        plt.legend()
        plt.show()
        print("Theta max: " + str(theta_max))

        return best_fit_model

    best_fit_model = MCMC_plot2()

    return sampler, pos, prob, state, dist, V, ndim, H_0_true, logDeltaM_true, Omega_m_true, best_fit_model


# Display the parameter values over the walker steps
def parameter_plot(sampler, ndim):
    fig, axes = plt.subplots(3, figsize=(20, 14), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$H_0$", r"$\log \Delta M$", r"$\Omega_m$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    # axes[0].set_title("Parameter values for each walker at each step of MCMC chain")
    axes[-1].set_xlabel("Step number")
    plt.show()


# Display auto-correlation times for each parameter (?)
def autocorr_time(sampler):
    tau = sampler.get_autocorr_time()
    print("Auto-correlation time for H_0, logDeltaM, and Omega_m")
    print(tau)
    return tau


def present_results(sampler, H_0_true, logDeltaM_true, Omega_m_true, ndim):
    samples = sampler.flatchain
    results = samples[np.argmax(sampler.flatlnprobability)]
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    print("H_0 is found to be " + str(results[0]) + ", compared to H_0_true of " + str(H_0_true))
    print("Difference of: " + str(results[0] - H_0_true) + " (km  s^-1  Mpc^-1)")
    print()
    print("logDeltaM is found to be " + str(results[1]) + ", compared to logDeltaM_true of " + str(logDeltaM_true))
    print("Difference of: " + str(results[1] - logDeltaM_true) + " (M_sun / h)")
    print()
    print("Omega_m is found to be " + str(results[2]) + ", compared to Omega_m_true of " + str(Omega_m_true))
    print("Difference of: " + str(results[2] - Omega_m_true))

    # TODO: Add percentwise difference here ^^


def corner_plot(sampler, H_0_true, logDeltaM_true, Omega_m_true):
    # Flattening the samples means compressing all the walker's routes into one readable chain
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    # Note that confidence intervals are 1 sigma
    labels = [r"$H_0$", r"$\log \Delta M$", r"$\Omega_m$"]
    fig = corner.corner(
        flat_samples, labels=labels, truths=[H_0_true, logDeltaM_true, Omega_m_true], show_titles=True,
        quantiles=[0.16, 0.5, 0.84])
    fig.set_size_inches((15, 15))
    plt.show()
