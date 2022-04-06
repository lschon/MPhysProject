# Summary: models.py
# This script defines all the constants used as well as creates the values for each Spherical Collapse Model

import math


def get_constants():
    H_0 = 100  # Define Hubble constant with implicit multiplication with h = 0.73
    h = 0.73
    km_in_Mpc = 3.086 * 10 ** 19  # number of kilometers in a Mpc
    G = 4.3 * 10 ** (-9)  # gravitational constant in Mpc M_sun^-1 (km / s)^2
    Omega_Lambda = 0.75  # Values used by Millenium simulation
    Omega_m = 0.25
    t0 = 4.34313792 * 10 ** 17  # Age of the universe. Using 13.772 billion years, converted into seconds
    f = Omega_m ** 0.55  # f is this by definition, related to growth rate of structure (how fast things are evolving)
    delta_c = 1.686  # given by Cai
    # R = function argument which is a list of distances to halo center in Mpc
    return H_0, h, km_in_Mpc, G, Omega_Lambda, Omega_m, t0, f, delta_c


H_0, h, km_in_Mpc, G, Omega_Lambda, Omega_m, t0, f, delta_c = get_constants()


# Define the Peñarrubia model
def Penarrubia(R, DeltaM):
    v_list = []
    for i in range(len(R)):
        v = (1.2 + 0.16 * Omega_Lambda) * (R[i] * km_in_Mpc / t0) - 1.1 * math.sqrt(G * (DeltaM / h) / R[i])
        v_list.append(v)
    return v_list


# Define the Aavik model
def Aavik(R, DeltaM):
    v_list = []
    for i in range(len(R)):
        v = (1.2 + 0.16 * Omega_Lambda) * (R[i] * km_in_Mpc / t0) - 1.1 * math.sqrt(
            G * (DeltaM / h) / R[i] + R[i] ** 2 * Omega_m * (H_0 * h) ** 2 / 2)
        v_list.append(v)
    return v_list


# Define the Lam model
def Lam(R, DeltaM):
    # EQUATION B2 in Paper https://arxiv.org/pdf/1305.5548.pdf
    v_list = []
    for i in range(len(R)):
        v = H_0 * h * R[i] * (1 - (Omega_m ** 0.55 * delta_c / 3) * (
                ((2 * G * DeltaM / h) / ((H_0 * h) ** 2 * Omega_m * R[i] ** 3) + 1) ** (1 / delta_c) - 1))
        v_list.append(v)
    return v_list
