import numpy as np
import numpy.typing as npt
from typing import List
import csv
import logging


class CDFLookup2D:
    # private stuff
    _data: npt.NDArray[np.float]
    _param_interpol_table: npt.NDArray[np.float]
    _p_interpol_table: npt.NDArray[np.float]

    _param_start: np.float
    _param_end: np.float
    _param_step: np.float

    _p_start: np.float
    _p_end: np.float
    _p_step: np.float

    def __init__(self, value_table, param_diffs, p_diffs, param_start, param_end, param_step, p_start, p_end, p_step):
        self._data = value_table
        self._param_interpol_table = param_diffs
        self._p_interpol_table = p_diffs
        self._param_start = param_start
        self._param_end = param_end
        self._param_step = param_step
        self._p_start = p_start
        self._p_end = p_end
        self._p_step = p_step

    @classmethod
    def from_csv(cls, path: str, delimiter = ','):
        """

                :param path:
                :param has_keys:
                :param delimiter:
                :return:
                """
        raw_table = np.genfromtxt(path, delimiter = delimiter).astype(float)
        params = raw_table[0, 1:]
        ps = raw_table[1:, 0]
        data = raw_table[1:, 1:]

        param_start = params[0]
        param_end = params[-1]
        param_step = np.median(np.diff(params))

        p_start = ps[0]
        p_end = ps[-1]
        p_step = np.median(np.diff(ps))

        param_diffs = (data[:,2:] - data[:,:-2])/2
        p_diffs = (data[2:,:] - data[:-2,:])/2


        first_diff = np.array([data[:,1]-data[:,0]]).T
        last_diff = np.array([data[:,-1]-data[:,-2]]).T
        param_diffs = np.concatenate((first_diff,param_diffs,last_diff), axis = 1)

        first_diff = np.array([data[1,:] - data[0,:]])
        last_diff = np.array([data[-1,:] - data[-2,:]])

        p_diffs = np.concatenate((first_diff, p_diffs, last_diff), axis = 0)

        return cls(data, param_diffs, p_diffs, param_start, param_end, param_step, p_start, p_end, p_step)



    def get_values(self, ps, params) -> npt.NDArray[np.float]:
        p_indicies = (ps - self._p_start) / self._p_step
        p_inticies = np.floor(p_indicies).astype(int)
        param_indicies = (params - self._param_start) / self._param_step
        param_inticies = np.floor(param_indicies).astype(int)

        intdex = param_inticies + p_inticies * self._data.shape[1]

        bases = np.take(self._data, intdex)
        p_der = np.take(self._p_interpol_table.T, intdex)
        param_der = np.take(self._param_interpol_table.T, intdex)

        return bases + p_der * (p_indicies-p_inticies) + param_der * (param_indicies-param_inticies)

