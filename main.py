import numpy as np
import Photons
import Renderer
from CDFLookup2D import CDFLookup2D
from src.CrossSectionLookup import CrossSectionLookup
import time
from CDFLookup import CDFLookup
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cross_section_lookup = CrossSectionLookup.from_csv('lookups/scattering crossections water.csv', has_keys = True)
    compton_look_up = CDFLookup2D.from_csv('lookups/compton angles.csv')
    rayleigh_look_up = CDFLookup.from_csv('lookups/rayleigh angles.csv')

    for i in range(5):
        photon_number = 10 ** 6
        df = Photons.initiate_pencil_beam(0.1, photon_number)
        depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10)
        absorbed_dose = np.sum(depositions['energy'].values) / photon_number
        print(absorbed_dose)



