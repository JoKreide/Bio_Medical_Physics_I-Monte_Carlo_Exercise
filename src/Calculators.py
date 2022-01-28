import numpy as np
import numpy.typing as npt

electron_mass = 0.511

def calc_positions(angles : npt.NDArray[np.float], distances : npt.NDArray[np.float]):
    return distances *  np.vstack((np.cos(angles), np.sin(angles))).T

def calc_compton_scatter_energy(scatter_angles, photon_energy):
    alpha = photon_energy / electron_mass
    return photon_energy / (1 + alpha * (1 - np.cos(scatter_angles)))