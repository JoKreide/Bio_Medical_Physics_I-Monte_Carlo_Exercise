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

    photon_number = 10 ** 8  # number of photons per run
    photon_energy = 0.01  # MeV

    start = time.time()
    depositions = Photons.calculate_depositions(
        photon_energy, photon_number, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10
        )
    time_taken = time.time() - start
    print(f"sampling done in {int(time_taken / 60)}min {time_taken % 60 : .2f}s")
    print(f"\t {time_taken * 10 ** 6 / photon_number : .3f}s / million photons")

    Renderer.draw_photon_paths((-2.5, 0), (5, 10), depositions, depositions['id'][:5], "Path of First 5 Interacting Photons", scale_points = False)

    start = time.time()
    Renderer.draw_deposition_image((-2.5, 0), (5, 10), depositions, res = 100, title = 'interaction loop', log = True)
    time_taken = time.time() - start
    print(f"rendering done in {int(time_taken / 60)}min {time_taken % 60 : .2f}s")
    print(f"\t {time_taken * 10 ** 6 / len(depositions) : .3f}s / million interactions")

    plt.show()
