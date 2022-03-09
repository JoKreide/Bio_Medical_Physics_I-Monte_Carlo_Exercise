from itertools import repeat
import numpy as np
import pandas as pd
import Calculators
import Sampler
from CDFLookup import CDFLookup
from CDFLookup2D import CDFLookup2D
from CrossSectionLookup import CrossSectionLookup
from concurrent.futures import ProcessPoolExecutor


def initiate_pencil_beam(energy, photon_number, start_x_pos = 0, start_y_pos = 0, start_angle=0, start_index = 0):
    """
    Creates a pandas data frame with the columns energies, angles, x_pos and y_pos.
    :param energy: energy of the photons
    :param photon_number: number of photons
    :param start_angle: angle of pencil beam
    :param start_y_pos: y position of source
    :param start_x_pos: x position of source
    :return: frame containing photon data in rows
    """
    energies = np.array([energy] * photon_number, dtype = float)
    angles = np.array([start_angle] * photon_number, dtype = float)
    x_positions = np.array(start_x_pos * photon_number, dtype = float)
    y_positions = np.array(start_y_pos * photon_number, dtype = float)
    photon_id = np.arange(photon_number) + start_index
    d = {'id': photon_id, 'energies': energies, 'angles': angles, 'x_pos': x_positions, 'y_pos': y_positions}
    return pd.DataFrame(data=d)


def next_interactions(df,
                      cross_section_lookup: CrossSectionLookup, compton_look_up: CDFLookup2D, rayleigh_look_up: CDFLookup):
    cross_sections = cross_section_lookup.get_all_cross_sections(df['energies'].values)

    depths = Sampler.sample_depths(cross_sections[-1])
    types = Sampler.sample_types(cross_sections)

    angles = Sampler.sample_angle(df['energies'].values, types, compton_look_up, rayleigh_look_up)
    absorbed_energies = Sampler.sample_absorption(df['energies'].values, types, angles)

    x_change, y_change = Calculators.calc_positions(df['angles'], depths)
    x_pos = df['x_pos'] + x_change
    y_pos = df['y_pos'] + y_change

    data = {
        'id': df['id'].values, 'depths': depths, 'types': types, 'angles': angles, 'absorption': absorbed_energies,
        'x_pos': x_pos, 'y_pos': y_pos
    }

    return pd.DataFrame(data=data)

def update_photons(photons, interactions, x_min, x_max, y_min, y_max):
    x_change, y_change = Calculators.calc_positions(photons['angles'], interactions['depths'])
    photons['x_pos'] = photons['x_pos'] + x_change
    photons['y_pos'] = photons['y_pos'] + y_change

    in_bounds = (photons['x_pos'] <= x_max) & (photons['x_pos'] >= x_min) & (photons['y_pos'] <= y_max) & (photons['y_pos'] >= y_min)
    is_not_absorbed = interactions['types'] != 2

    photons['energies'] = np.mod(photons['energies'] - interactions['absorption'],2*np.pi)
    energy_in_bounds = photons['energies'] >= 0.001

    photons['angles'] = np.mod(photons['angles'] + interactions['angles'] + np.pi,2*np.pi) - np.pi

    photons = photons[in_bounds & is_not_absorbed & energy_in_bounds]
    return photons.reset_index(drop = True)

def update_depositions(interactions, x_min, x_max, y_min, y_max, depositions = None):
    new_df = pd.DataFrame({
        'id': interactions['id'].values,
        'x_pos': interactions['x_pos'].values, 'y_pos': interactions['y_pos'].values,
        'energy': interactions['absorption'].values
    })
    is_in_bounds = (new_df['x_pos'] < x_max) & (new_df['x_pos'] > x_min) & (new_df['y_pos'] < y_max) & (new_df['y_pos'] > y_min)
    new_df = new_df[is_in_bounds]
    if depositions is None:
        return new_df.reset_index(drop = True)
    else:
        return pd.concat([depositions, new_df], ignore_index = True).reset_index(drop = True)

def _calc_dep(photon_energy, photon_number, cross_section_lookup, compton_look_up, rayleigh_look_up,
              x_min, x_max, y_min, y_max, start_index):
    depositions = None
    photon_df = initiate_pencil_beam(photon_energy, photon_number, start_index=start_index)

    while len(photon_df) > 0:
        interactions = next_interactions(photon_df, cross_section_lookup, compton_look_up, rayleigh_look_up)
        photon_df = update_photons(photon_df, interactions, x_min, x_max, y_min, y_max)
        depositions = update_depositions(interactions, x_min, x_max, y_min, y_max, depositions = depositions)

    del photon_df
    return depositions

def calculate_depositions(photon_energy, photon_number,
                          cross_section_lookup, compton_look_up, rayleigh_look_up,
                          x_min, x_max, y_min, y_max, chunks = 10, start_index=0):

    with ProcessPoolExecutor() as executor:
        results = executor.map(_calc_dep, repeat(photon_energy), [int(photon_number/chunks)]*chunks,
                               repeat(cross_section_lookup), repeat(compton_look_up), repeat(rayleigh_look_up),
                               repeat(x_min), repeat(x_max), repeat(y_min), repeat(y_max),
                               np.arange(chunks)*int(photon_number/chunks) + start_index)

    return pd.concat(results)
