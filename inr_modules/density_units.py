"""Shared display units for ionospheric electron-density figures."""

import numpy as np


DENSITY_SCALE_M3 = 1.0e11
DENSITY_UNIT_LABEL = r'$10^{11}$ m$^{-3}$'


def density_to_display(values):
    """Convert electron density from m^-3 to units of 10^11 m^-3."""
    return np.asarray(values, dtype=np.float64) / DENSITY_SCALE_M3


def log10_density_to_display(values):
    """Convert log10 electron density to units of 10^11 m^-3."""
    return np.power(10.0, np.asarray(values, dtype=np.float64) - 11.0)
