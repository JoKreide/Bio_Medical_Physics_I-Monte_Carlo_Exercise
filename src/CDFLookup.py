import numpy as np
import numpy.typing as npt
from typing import List
import csv
import logging


class CDFLookup:
    # private stuff
    _value_table: npt.NDArray[np.float]
    _interpol_table: npt.NDArray[np.float]

    _start: np.float
    _end: np.float
    _step: np.float

    def __init__(self, value_table, interpol_table, energy_start, energy_end, energy_step):
        self._value_table = value_table
        self._interpol_table = interpol_table
        self._start = energy_start
        self._end = energy_end
        self._step = energy_step

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
        gen_value_table: npt.NDArray[np.float]
        gen_interpol: npt.NDArray[np.float]

        gen_start: np.float
        gen_end: np.float
        gen_step: np.float

        # get content of file as np array
        raw_data = np.genfromtxt(path, delimiter = delimiter).astype(float)

        # remove headers from data
        if has_keys:
            raw_data = raw_data[1:, :]

        # check step sizes
        xs = raw_data[:,0]
        gen_start = np.float(np.min(xs))
        gen_end = np.float(np.max(xs))
        assert gen_start == xs[0] and gen_end == xs[-1], f"xs do not appear in order.\n\t start: {xs[0]} \t min: {gen_start}\n\t end: {xs[-1]} \t max: {gen_end}"

        window = gen_end - gen_start
        gen_step = window / (len(xs)-1)
        errors = np.abs(xs - np.arange(gen_start,gen_end+gen_step,gen_step))

        assert np.max(errors) < gen_step / 2, "xs are not uniformly distributed"

        values = raw_data[:,1]
        interpol = cls._generate_derivatives(values)

        return cls(values, interpol, gen_start, gen_end, gen_step)

    @staticmethod
    def _generate_derivatives(array : npt.NDArray):
        left_shift = array[:-2]
        right_shift = array[2:]
        interpol = (right_shift - left_shift)
        interpol = np.insert(interpol, 0, array[1]-array[0])
        interpol = np.append(interpol, array[-1]-array[-2])
        return interpol

    def get_values(self, xs) -> npt.NDArray[np.float]:
        index = (xs - self._start)/self._step
        intdex = np.floor(index).astype(np.int)
        bases = np.take(self._value_table,intdex)
        ramps = np.take(self._interpol_table,intdex)
        return bases + ramps * (index - intdex)

