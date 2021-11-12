import numpy as np
from numpy import log10, seterr, exp

seterr(all='ignore')

# Conversion functions


def db2lin(value):
    lin_value = 10 ** (value / 10)
    return lin_value


def lin2db(value):
    db_value = 10*np.log10(value)
    return db_value


def lin2dbm(value):
    dbm_value = 10*np.log10(value/0.001)
    return dbm_value


def alfa2lin(alfa_db):
    alfa_lin = alfa_db/1e3/(20*np.log10(np.e))
    return alfa_lin


def lin2dBm(array):
    return lin2db(array) + 30


def dBm2lin(array):
    return db2lin(array) * 1e-3

