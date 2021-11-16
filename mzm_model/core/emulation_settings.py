"emulation_settings.py"

import scipy.constants as con
import numpy as np
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin

'Physical constants'

h = con.h   # Plank constant [J*s]
q = 1.602e-19   # electron charge [C]
light_speed = con.c   # speed of light in vacuum [m/s]

'Ideal light source params'

wavelength = 1550   # [nm]
frequency = light_speed/wavelength    # THz
eta_s = 0.03   # quantum efficiency [%, photons/electrons]
input_current = 350-3  # [mA] [taken in range between 200 mA and 500 mA]
phi_laser = 0

'Splitter and Combiner params'

# gammas values at 1/sqrt(2) is due to maximize Griffin model tf
gamma_1 = 1/np.sqrt(2)  # input field split ratio (equal to gamma_1**2 for power splitter ratio)
gamma_2 = 1/np.sqrt(2)    # output field split ratio of the combiner (consider a symmetrical split ratio, so 0.5*2=1)
k_splitter = 1  # arbitrary electric field conversion value

'Waveguides params'

v_A = 8   # [V]
v_B = 0    # [V]
# define static v_pi only for classic LiNbO3 MZM, because v_pi in InP MZM varies with v_cm = v_bias
v_pi = 0.5    # [V]

'''MZM Params'''
v_cm = (v_A+v_B)/2  # common mode voltage
v_diff = (v_A-v_B)/2    # differential voltage

v_in_limit = v_A  # for v_in range limit
v_in_step = 0.01    # v_in step

'''Non Ideal MZM Additional Params'''

# For ER values consider 40, 32 and 27 dB values (40 is similar to infinite)
er = db2lin(40)
# er = db2lin(1000)   # extinction ratio, in linear units, but usually written in dB (using lower values some issues,
# but recovered by MF presence)

insertion_loss = 1
v_off = 0     # [V]

'''Griffin MZM Params'''
v_bias = - np.pi / v_pi
v_pi_griffin = - np.pi / v_bias
b = 0.0    # Phase Non-linearity parameter, between 0.0 (linear) and 0.15 (non-linear)
c = 20    # Transmission Absorption parameter, between 20 (same LiNbO3 behavior)and 4.3

phase_offset = 0

# ER OF I AND Q MZMS
er_i = db2lin(40)
er_q = db2lin(40)
# BIAS OFFSET OF I AND Q GRIFFIN MZMS
bias_offset_i = 0.0
bias_offset_q = 0.0
# Noise params
noise_flag = True
std = 0.01     # std is the standard deviation of normal distribution, for AWGN (default value is 0.01)
sim_duration = 10e-9     # simulation duration
SNRdB_InP = 40 # for AWGN channel of InP
SNRdB_LiNb = 40 # for AWGN channel of LiNb

# symbol generator params
'PRBS'
poldeg = 8  # polynomial degree for PRBS generator, between 5 and 28
prbs_counter = 50   # number of the first generating polynomials, between 1 and 176
zero_pad = True     # flag to add the last sequence of zeros to have 2^n sequences
modulation_format = '16qam'
modulation_format = modulation_format.upper()
Rs = 32e9  # baud-rate, in Gbaud
Ts = 1/Rs  # symbol period
num_signals = 2**poldeg     # number of transmitted signals

'Raised cosine Params'
sps = 20   # define samples per symbol (N.B. CHOOSE ONLY EVEN NUMBERS TO GET CORRECT RESULTS)
N_taps = sps*num_signals    # length of the filter in samples
beta = 0.09     # roll-off for raised cosine (ideal is 0, actually use 0.09)
samp_f = sps*Rs     # sampling frequency

channel_bw = (1 + beta)*Rs  # [Hz] Frequency 'amplitude' of the primary lobe
delta_f = channel_bw    # channel spacing

# sym_len = r_s**(-1)    # number of samples to represent a signal (bit duration, in ps)

if modulation_format == 'QPSK':
    # L = number of levels in each dimension
    n_bit = 2
    L = n_bit
    norm_factor = (np.sqrt(2)/2)
    v_drive = 2*v_pi
    evm_factor = 1
elif modulation_format == '16QAM':
    n_bit = 4
    L = n_bit
    norm_factor = 0.776
    v_drive = v_pi
    evm_factor = 9/5

norm_rx = sps
n_bit_nrz = n_bit/2     # number of bits for sqrt(M)-PAM modulation
r_b = Rs*n_bit     # bit-rate, in Gbitps

M = 2**n_bit
op_freq = 32e9     #GHz
spectral_efficiency = n_bit/(1 + beta)

"Driving voltage signal params"
v_tx_param = 0.5*v_pi/(np.sqrt(M) - 1)
# v_tx_param_griffin = 0.5*v_pi_griffin/(np.sqrt(M) - 1)
