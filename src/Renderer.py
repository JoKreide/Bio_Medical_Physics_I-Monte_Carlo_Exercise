from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import Photons
from CDFLookup import CDFLookup
from CDFLookup2D import CDFLookup2D
from CrossSectionLookup import CrossSectionLookup
import matplotlib.pyplot as plt


def draw_photon_paths(box_position, box_dimensions, interaction_df, indices, title='', scale_points = False):
    plt.figure(title)
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
    plt.draw()

def calc_pixels(box_position, box_dimensions, interaction_df, res=10, log=True):
    pixels = np.zeros(np.ceil([box_dimensions[0]*res, box_dimensions[1]*res]).astype(int))

    x_pos = np.floor((interaction_df['x_pos'] - box_position[0]) * res)
    y_pos = np.floor((interaction_df['y_pos'] - box_position[1]) * res)

    pos = (np.vstack((x_pos, y_pos)).T).astype(int)

    for p, e in zip(pos, interaction_df['energy'].values):
        np.add.at(pixels,(p[0], p[1]),e)

    if log:
        pixels = pixels + 0.00001
        pixels = np.log(pixels)
        pixels = pixels - np.min(pixels)

    return pixels


def draw_deposition_image(box_position, box_dimensions, interaction_df, res=10, title='', log=True, chunks=25):
    chunk_size = int(len(interaction_df) / chunks)
    df_chunks = [interaction_df[i * chunk_size : min((i + 1) * chunk_size, len(interaction_df))] for i in range(chunks)]

    with ProcessPoolExecutor() as executor:
        results = executor.map(calc_pixels,repeat(box_position), repeat(box_dimensions), df_chunks, repeat(res), repeat(log))

    pixels = np.sum(results, axis = 0)
    pixels = pixels / np.max(pixels)

    plt.figure(title)
    plt.imshow(pixels, cmap = 'hot', interpolation = 'nearest')
    plt.draw()

def draw_deposition_image2(box_position, box_dimensions, interaction_df, res=10, title='', log=True):
    fig = plt.figure(title)
    pixels = np.zeros([int(box_dimensions[0]*res), int(box_dimensions[1]*res)])

    for x in range(int(box_dimensions[0]*res)):
        for y in range(int(box_dimensions[1]*res)):
            min_x = x / res + box_position[0]
            max_x = min_x + 1 / res
            min_y = y / res + box_position[1]
            max_y = min_y + 1 / res

            ints = interaction_df[
                (interaction_df['x_pos'] < max_x) &
                (interaction_df['x_pos'] > min_x) &
                (interaction_df['y_pos'] < max_y) &
                (interaction_df['y_pos'] > min_y)
            ]

            pixels[x, y] = np.sum(ints['energy'])

    pixels = pixels / np.max(pixels)
    plt.imshow(pixels, cmap = 'hot', interpolation = 'nearest')

if __name__ == '__main__':
    cross_section_lookup = CrossSectionLookup.from_csv('../lookups/scattering crossections water.csv', has_keys = True)
    compton_look_up = CDFLookup2D.from_csv('../lookups/compton angles.csv')
    rayleigh_look_up = CDFLookup.from_csv('../lookups/rayleigh angles.csv')

    photon_number = 10 ** 3
    df = Photons.initiate_pencil_beam(0.1, photon_number)
    depositions = Photons.calculate_depositions(df, cross_section_lookup, compton_look_up, rayleigh_look_up, -2.5, 2.5, 0, 10)
    draw_photon_paths((-2.5, 0), (5, 10), depositions, depositions['id'][:5])
    plt.show()