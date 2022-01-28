import numpy as np
import numpy.typing as npt
import numpy.random as random

from CDFLookup import CDFLookup
from CDFLookup2D import CDFLookup2D


def sample_depths(cross_sections: npt.NDArray[np.float]):
    """
    randomly samples interactions depths based on provided list of cross-sections.
    :param cross_sections: list of cross-sections, either just total cross-sections or 2D array with total cross-sections in last row
    :return: returns randomly sampled depths
    """
    css: npt.NDArray[np.float]
    match 1:
        case 1:
            print("works")
    match len(cross_sections.shape):
        case 2:
            css = cross_sections[-1]
        case 1:
            css = cross_sections
        case _:
            raise ValueError("cross_section must have 1 or 2 dimensions")

    rands = random.random_sample(len(css))
    return -css ** -1 * np.log(rands)


def sample_depth(cross_section: float):
    rand = random.rand()
    return -cross_section ** -1 * np.log(rand)


def sample_types(cross_sections: npt.NDArray[np.float]):
    totals = cross_sections[-1]
    rands = random.random_sample(len(totals)) * totals
    rayleigh = cross_sections[0]
    compton = rayleigh + cross_sections[1]
    return 2 * (rands >= compton) + (rands >= rayleigh) * (rands < compton)

def sample_compton_angle(lookup: CDFLookup2D, energies: npt.NDArray[np.float], number = None):
    if number is None:
        number = len(energies)
    rands = random.random_sample(number)
    return lookup.get_values(rands, energies)

def sample_rayleigh_angle(lookup: CDFLookup, number_of_angles):
    rands = random.random_sample(number_of_angles)
    return lookup.get_values(rands)