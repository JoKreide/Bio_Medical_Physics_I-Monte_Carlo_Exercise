import numpy as np
import Sampler
from CDFLookup2D import CDFLookup2D
from src.CrossSectionLookup import CrossSectionLookup as csl, CrossSectionLookup
import time
import Calculators
from CDFLookup import CDFLookup
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """water_csss = csl.from_csv("lookups/scattering crossections water.csv", has_keys = True)
    energies = np.random.random_sample(10**6) * (water_csss._end - water_csss._start) + water_csss._start
    dist = Calculators.calc_positions(np.array([0,np.pi,np.pi/2]),np.array([1]))"""

    number = 10**4

    rayleigh_angle_lookup = CDFLookup.from_csv('lookups/rayleigh angles.csv')
    sampled_rayleigh_angles = Sampler.sample_rayleigh_angle(rayleigh_angle_lookup, number)

    compton_angle_lookup = CDFLookup2D.from_csv('lookups/compton angles.csv')
    energies = np.array([0.099]*number)#np.random.random_sample(number) * (0.1 - 0.001) + 0.001
    sampled_compton_angles = Sampler.sample_compton_angle(compton_angle_lookup, energies, number = number)

    depths_lookup = CrossSectionLookup.from_csv('lookups/scattering crossections water.csv',has_keys = True)
    total_crosssections = depths_lookup.get_all_cross_sections(energies)[-1]

    dist = Sampler.sample_depths(total_crosssections)

    dist.sort()
    plt.figure("interaction depths")
    plt.plot(dist)

    sampled_rayleigh_angles.sort()
    plt.figure("rayleigh angles")
    plt.plot(sampled_rayleigh_angles)

    sampled_compton_angles.sort()
    plt.figure("compton angles")
    plt.plot(sampled_compton_angles)

    scatter_energies = Calculators.calc_compton_scatter_energy(sampled_compton_angles, 0.5)
    plt.figure("compton scatter energy cdf")
    plt.plot(scatter_energies)

    plt.figure("energy plot")
    plt.plot(sampled_compton_angles, scatter_energies)

    plt.show()




