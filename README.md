# MPhysProject
Title: MPhys Project - Understanding the Local Expansion Rate of the Universe

Author: Lucas Schön

Dates: September 2021 to April 2022

Institution: University of Edinburgh

Supervisors: Dr. Yan-Chuan Cai and Prof. Jorge Peñarrubia

Contact e-mail: lucas.mgs.schon@gmail.com

This MPhys project investigates the local expansion rate of the Universe using Millennium Simulation (2005) data and Virgo cluster data from Kim et al. (2020)

Their original papers are found here, https://www.nature.com/articles/nature03597, and here, https://iopscience.iop.org/article/10.3847/1538-4357/abbd97/pdf, respectively.

It applies an MCMC onto data from both, using a Spherical Collapse Model (SCM) solution from Lam et al. (2013), from https://arxiv.org/pdf/1305.5548.pdf, to try to determine values for the Hubble constant, which quantifies the expansion in the Universe.

The totality of the project is found in the file: Understanding the Local Expansion Rate of the Universe -  LS MPhys.pdf 

A summary of the files follows.

General files:

    dataframes.py:
        This reads in data from text files and constructs useful dataframes (using pandas).
        
    models.py:
        This defines all the constants and constructs all the different SCM values.
        
    plotter.py:
        This simply creates all the plots.
        
Millennium Simulation-specific files:

    main_millennium.py:
        This is the main file combining all the others for the Millennium Simulation.
        
    MCMC_2var.py, MCMC_2var2.py, MCMC_2var3.py, MCMC_3var.py: 
        Variations of the MCMC, each using different parameters. The MCMC_2var3.py file is the main one used, fitting H_0 and logDelta.
        
    cosmicvariance.py:
        This calculates the cosmic variance of H_0 using the data from cosmicvariance.txt.
        
    cosmicvariance.txt:
        This is a text file of the data from 40 trials using the most massive halos.

Virgo Cluster-specific files:

    main_kim.py:
        This is the main file combining all others for the Virgo Cluster data.
        
    MCMC_kim.py, MCMC_kim2:
        Variations of the MCMC specific to the Virgo Cluster, MCMC_kim.py uses 3 parameters and MCMC_kim2.py uses 2. The latter is the primary one.
        
