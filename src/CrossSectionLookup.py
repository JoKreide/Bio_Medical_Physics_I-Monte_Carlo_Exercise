import numpy as np
import numpy.typing as npt
from typing import List
import csv
import logging


class CrossSectionLookup:
    cross_sections: npt.NDArray[np.float] #contains all of the scattering crosssections and their derivatives in form [[crosss, derivative], [...]]
    indicies = {'r': 0, 'c': 1, 'p': 2,
                'ray': 0, 'com': 1, 'pho': 1,
                'rayleigh': 0, 'compton': 1, 'photoelectric': 2
    }
    cross_section_methods : list

    # private stuff
    _rayleigh_table: npt.NDArray[np.float]
    _rayleigh_interpol: npt.NDArray[np.float]
    _compton_table: npt.NDArray[np.float]
    _compton_interpol: npt.NDArray[np.float]
    _photoel_table: npt.NDArray[np.float]
    _photoel_interpol: npt.NDArray[np.float]
    _energy_start: np.float
    _energy_end: np.float
    _energy_step: np.float

    def __init__(self, rayleigh_table, rayleigh_interpol, compton_table, compton_interpol, photoel_table, photoel_interpol, energy_start, energy_end, energy_step):
        self._rayleigh_table = rayleigh_table
        self._rayleigh_interpol = rayleigh_interpol
        self._compton_table = compton_table
        self._compton_interpol = compton_interpol
        self._energy_start = energy_start
        self._energy_end = energy_end
        self._energy_step = energy_step
        self._photoel_table = photoel_table
        self._photoel_interpol = photoel_interpol

        self.cross_sections = np.concatenate([
            [np.concatenate([[self._rayleigh_table], [self._rayleigh_interpol]], axis = 0)],
            [np.concatenate([[self._compton_table], [self._compton_interpol]], axis = 0)],
            [np.concatenate([[self._photoel_table], [self._photoel_interpol]], axis = 0)]
        ], axis = 0)

        self.cross_section_methods = [self.get_rayleigh, self.get_compton, self.get_photoel]

    @classmethod
    def from_csv(cls, path: str, has_keys = False, delimiter = ','):
        """
        generates a cross section lookup from a csv. Columns are energies, rayleigh, compton
        :param path: path to the csv file
        :param keys: list of names for columns
        :param has_keys: whether the csv's first row is column names
        :param delimiter: delimiter used in the csv file
        :return: cross section lookup object
        """
        gen_rayleigh_table: npt.NDArray[np.float]
        gen_rayleigh_interpol: npt.NDArray[np.float]
        gen_compton_table: npt.NDArray[np.float]
        gen_compton_interpol: npt.NDArray[np.float]
        gen_photoel_table: npt.NDArray[np.float]
        gen_photoel_interpol: npt.NDArray[np.float]
        gen_start: np.float
        gen_end: np.float
        gen_step: np.float

        # get content of file as np array
        raw_data = np.genfromtxt(path, delimiter = delimiter).astype(np.float)

        # remove headers from data
        if has_keys:
            raw_data = raw_data[1:, :]

        # check step sizes
        energies = raw_data[:,0]
        gen_start = np.float(np.min(energies))
        gen_end = np.float(np.max(energies))
        assert gen_start == energies[0] and gen_end == energies[-1], f"energies do not appear in order.\n\t start: {energies[0]} \t min: {gen_start}\n\t end: {energies[-1]} \t max: {gen_end}"

        energy_window = gen_end - gen_start
        gen_step = energy_window / (len(energies)-1)
        errors = np.abs(energies - np.arange(1, len(energies)+1) * gen_step)
        assert np.max(errors) < gen_step / 2, "energies are not uniformly distributed"

        _, gen_rayleigh_table, gen_compton_table, gen_photoel_table = raw_data.transpose()
        gen_rayleigh_interpol = cls._generate_derivatives(gen_rayleigh_table)
        gen_compton_interpol = cls._generate_derivatives(gen_compton_table)
        gen_photoel_interpol = cls._generate_derivatives(gen_photoel_table)

        return cls(gen_rayleigh_table, gen_rayleigh_interpol, gen_compton_table, gen_compton_interpol, gen_photoel_table, gen_photoel_interpol, gen_start, gen_end, gen_step)

    @staticmethod
    def _generate_derivatives(array : npt.NDArray):
        left_shift = array[:-2]
        right_shift = array[2:]
        interpol = (right_shift - left_shift)
        interpol = np.insert(interpol, 0, array[1]-array[0])
        interpol = np.append(interpol, array[-1]-array[-2])
        return interpol

    def get_rayleigh(self, energy:float):
        assert self._energy_start < energy < self._energy_end, f"energy out of bounds {self._energy_start} to {self._energy_end}"
        index = (energy - self._energy_start)/self._energy_step
        intdex = int(index)
        return self._rayleigh_table[intdex] + (index-intdex)*self._rayleigh_interpol[intdex]

    def get_compton(self, energy:float):
        assert self._energy_start < energy < self._energy_end, f"energy out of bounds {self._energy_start} to {self._energy_end}"
        index = (energy - self._energy_start)/self._energy_step
        intdex = int(index)
        return self._compton_table[intdex] + (index-intdex)*self._compton_interpol[intdex]

    def get_photoel(self, energy:float):
        assert self._energy_start < energy < self._energy_end, f"energy out of bounds {self._energy_start} to {self._energy_end}"
        index = (energy - self._energy_start)/self._energy_step
        intdex = int(index)
        return self._compton_table[intdex] + (index-intdex)*self._compton_interpol[intdex]

    def get_cross_section(self,  interaction: str, energy: float):
        assert interaction in self.indicies.keys(), f"interaction {interaction} was not found in indicies"
        return self.cross_section_methods[self.indicies[interaction]](energy)

    def get_cross_sections(self, interaction: str, energies: npt.NDArray[np.float]):
        assert interaction in self.indicies.keys(), f"interaction {interaction} was not found in indicies"
        assert np.all((energies < self._energy_end) * (energies > self._energy_start)), "energies not all in bounds"
        indicies = (energies - self._energy_start)/self._energy_step
        intdicies = np.floor(indicies).astype(np.int)
        cross_lookup, interpol_lookup = self.cross_sections[self.indicies[interaction]]
        ramps = np.take(interpol_lookup,intdicies)
        bases = np.take(cross_lookup,intdicies)
        return bases + (indicies - intdicies) * ramps

    def get_all_cross_sections(self, energies: npt.NDArray[np.float]):
        assert np.all((energies <= self._energy_end) * (energies >= self._energy_start)), "energies not all in bounds"
        indicies = (energies - self._energy_start) / self._energy_step
        intdicies = np.floor(indicies).astype(np.int)
        values = np.take(self.cross_sections, intdicies, axis = 2)
        csss = values[:,0,:] + values[:,1,:] * (indicies-intdicies)
        tscs = np.sum(csss, axis = 0)
        return np.vstack((csss,tscs))

