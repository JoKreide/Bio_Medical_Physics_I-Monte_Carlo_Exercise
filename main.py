import time
import numpy as np
from src import Photons
from src import Renderer
from src.CDFLookup2D import CDFLookup2D
from src.CrossSectionLookup import CrossSectionLookup
from src.CDFLookup import CDFLookup
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def calc_absorbed_dose(photon_number, photon_energy, cross_section_lookup, compton_look_up, rayleigh_look_up, xmin, xmax, ymin, ymax):
    df = Photons.initiate_pencil_beam(photon_energy, photon_number)
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, xmin, xmax, ymin, ymax)
    return np.sum(depositions['energy'].values) / photon_number


if __name__ == '__main__':
    cross_section_lookup = CrossSectionLookup.from_csv('lookups/scattering crossections water.csv', has_keys = True)
    compton_look_up = CDFLookup2D.from_csv('lookups/compton angles.csv')
    rayleigh_look_up = CDFLookup.from_csv('lookups/rayleigh angles.csv')

    photon_number = 10 ** 7  # number of photons per run
    photon_energy = 0.1  # MeV
    loops = 10

    print(f"starting {loops} loops of {int(photon_number/loops)} photons each...")

    start = time.time()

    dep_doses = []
    all_depositions = []
    for i in tqdm(range(loops)):
        depositions = Photons.calculate_depositions(
            photon_energy, int(photon_number/loops), cross_section_lookup, compton_look_up, rayleigh_look_up,
            -2.5, 2.5, 0, 10, start_index = int(photon_number/loops)*i)
        if i == 0:
            all_depositions = depositions
        else:
            all_depositions = pd.concat([all_depositions, depositions])
        dep_doses.append(np.sum(depositions['energy']))

    time_taken = time.time() - start

    print(f"total deposition: {np.sum(dep_doses) / photon_number / photon_energy * 100:.3f}% of incoming energy")
    print(f"per run stab of deposition: {np.std(dep_doses) / photon_number / photon_energy * 100:.3f}% of incoming energy")

    print(f"sampling done in {int(time_taken / 60)}min {time_taken % 60 : .2f}s")
    print(f"\t {time_taken * 10 ** 6 / photon_number : .3f}s / million photons")

    Renderer.draw_photon_paths((-2.5, 0), (5, 10),
                               all_depositions, all_depositions['id'][:5],
                               "Path of First 5 Interacting Photons", scale_points = False)
    plt.draw()

    print("creating photon path image done")

    start = time.time()
    Renderer.draw_deposition_image((-2.5, 0), (5, 10), all_depositions, res = 100, title = 'interaction loop', log = True)
    time_taken = time.time() - start
    print(f"rendering done in {int(time_taken / 60)}min {time_taken % 60 : .2f}s")
    print(f"\t {time_taken * 10 ** 6 / len(all_depositions) : .3f}s / million interactions")

    plt.show()
