import numpy as np
import Photons
from CDFLookup import CDFLookup
from CDFLookup2D import CDFLookup2D
from CrossSectionLookup import CrossSectionLookup
import matplotlib.pyplot as plt


def draw_photon_paths(box_position, box_dimensions, interaction_df, indices, title='', scale_points = False):
    fig = plt.figure(title)
    water_rect = plt.Rectangle(box_position, *box_dimensions, facecolor = 'cornflowerblue')
    plt.gca().add_patch(water_rect)

    for index in indices:
        interactions = interaction_df[interaction_df['id'] == index]
        positions = np.array([interactions['x_pos'], interactions['y_pos']]).T
        positions = np.vstack(([[0, 0]], positions))

        if scale_points:
            scaled_energy = interactions['energy'].values
            scaled_energy = scaled_energy.flatten()
            scaled_energy = np.log(scaled_energy)
            scaled_energy = scaled_energy / np.max(scaled_energy)
            scaled_energy = scaled_energy / 20 * min(box_dimensions)
            scaled_energy = np.insert(scaled_energy,0,0)
        else:
            scaled_energy = [1 / 40 * min(box_dimensions)] * (len(positions)-1)
        scaled_energy = np.insert(scaled_energy, 0, 0)

        for pos, energy in zip(positions, scaled_energy):
            circle = plt.Circle(pos, radius = energy, fc = 'r')
            plt.gca().add_patch(circle)

    for index in indices:
        interactions = interaction_df[interaction_df['id'] == index]
        positions = np.array([interactions['x_pos'], interactions['y_pos']]).T
        positions = np.vstack(([[0,0]],positions))
        line = plt.Polygon(positions, closed = None, fill = None, linewidth = 1.5)
        plt.gca().add_patch(line)

    plt.axis('scaled')
    return fig

if __name__ == '__main__':
    cross_section_lookup = CrossSectionLookup.from_csv('../lookups/scattering crossections water.csv', has_keys = True)
    compton_look_up = CDFLookup2D.from_csv('../lookups/compton angles.csv')
    rayleigh_look_up = CDFLookup.from_csv('../lookups/rayleigh angles.csv')

    photon_number = 10 ** 3
    df = Photons.initiate_pencil_beam(0.1, photon_number)
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10)
    draw_photon_paths((-2.5, 0), (5, 10), depositions, depositions['id'][:5])
    plt.show()