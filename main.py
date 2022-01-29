import numpy as np
import Photons
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

    photon_number = 10 ** 6
    df = Photons.initiate_pencil_beam(0.1, photon_number)

    start = time.time()
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, -5, 5, 0, 25)
    end = time.time()

    print(f"\ndone in: {(end - start) * 10 ** 6 / len(depositions)} s / million events")
    print()
    print(f"average y-depth: {np.mean(depositions['y_pos'])}")
    print(f"max y-depth: {np.max(depositions['y_pos'])}")
    print(f"stab y-depth: {np.std(depositions['y_pos'])}")
    print(f"average x: {np.mean(depositions['x_pos'])}")
    print(f"max x: {np.max(np.abs(depositions['x_pos']))}")
    print(f"stab x: {np.std(depositions['x_pos'])}")

    print("\n---- FINAL DEPOSITIONS----\n")
    print(depositions)

    xs = depositions['x_pos']
    xs.values.sort()
    plt.figure("Transverse Distribution")
    plt.plot(xs)

    ys = depositions['y_pos']
    ys.values.sort()
    plt.figure("Longitudinal Distribution")
    plt.plot(ys)

    plt.show()



