import time

import numpy as np
import Photons
import Renderer
from CDFLookup2D import CDFLookup2D
from src.CrossSectionLookup import CrossSectionLookup
from CDFLookup import CDFLookup
import matplotlib.pyplot as plt


def calc_absorbed_dose(photon_number, photon_energy, cross_section_lookup, compton_look_up, rayleigh_look_up, xmin, xmax, ymin, ymax):
    df = Photons.initiate_pencil_beam(photon_energy, photon_number)
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, xmin, xmax, ymin, ymax)
    return np.sum(depositions['energy'].values) / photon_number

if __name__ == '__main__':
    cross_section_lookup = CrossSectionLookup.from_csv('lookups/scattering crossections water.csv', has_keys = True)
    compton_look_up = CDFLookup2D.from_csv('lookups/compton angles.csv')
    rayleigh_look_up = CDFLookup.from_csv('lookups/rayleigh angles.csv')

    photon_number = 10 ** 5 #number of photons per run
    iterations = 10 ** 2 #number of runs
    photon_energy = 0.05 #MeV

    start = time.time()
    doses = np.array([calc_absorbed_dose(photon_number, photon_energy, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10) for i in range(iterations)])
    print(f"done in {time.time() - start: .2f}s for {photon_number * iterations} photons")

    print(f"Photon Energy: {photon_energy*1000:.0f} keV")
    print(f"Total Simulated Energy:\t {photon_energy * photon_number * (1.603 * 10**-4) : .3f}nJ")
    print(f"Average Absorbed Energy: {photon_number * np.mean(doses) * (1.603 * 10**-4) : .3f}nJ"
          f"\t|\t"
          f"{np.mean(doses)/photon_energy * 100 : .2f} %")
    print(f"Run to Run Variance:\t {photon_number * np.std(doses) * (1.603 * 10**-4) : .3f}nJ"
          f"\t|\t"
          f"{np.std(doses) / photon_energy * 100 : .2f} %")

    df = Photons.initiate_pencil_beam(photon_energy, 1000)
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10)

    Renderer.draw_photon_paths((-2.5, 0), (5, 10), depositions, depositions['id'][:5], "Path of First 5 Interacting Photons", scale_points = False)
    plt.show()



