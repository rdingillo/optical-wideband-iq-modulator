"iq_mzm.py"

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal as sig
import scipy.io
import pandas as pd
import random
import control
import time
import collections
from scipy.interpolate import interp1d

import commpy.filters as flt
import copy
import seaborn as sns; sns.set_theme()
from colorama import Fore
from pathlib import Path
from scipy.signal import butter, bessel, freqz, lfilter
from collections import defaultdict
from mzm_model.core.elements import IdealLightSource, Splitter, Combiner, MZMSingleElectrode, GriffinMZM
from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, v_A, v_B, v_pi, v_cm, \
    v_diff, v_in_limit, v_in_step, er, insertion_loss, phase_offset, v_off, b, c, wavelength, gamma_1, gamma_2, \
    modulation_format, n_bit, num_signals, std, op_freq, M, poldeg, prbs_counter, zero_pad, sps, beta, Ts, Rs, \
    samp_f, N_taps, v_tx_param, noise_flag, norm_factor, norm_rx, v_drive, v_bias, v_pi_griffin, er_i, er_q, \
    bias_offset_i, bias_offset_q
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin
from mzm_model.core.science_utils import rand_key, evm_rms_estimation, snr_estimation, ber_estimation
from mzm_model.core.prbs_generator import prbs_generator
from mzm_model.core.utils import plot_constellation, eo_tf_iq_draw, savitzky_golay
from scipy import signal
start_time = time.time()

matplotlib.use('Qt5Agg')
root = Path(__file__).parent.parent
input_folder = root/'resources'
folder_results = root/'mzm_model'/'results'

baudrates = [32, 68, 96, 128]
types = ('Pockel like', 'b var', 'c var', 'ER var', 'offset var', 'real case')

"Pockel's effect results"
evm_pockel_qpsk = np.array([0.0568, 0.1247, 0.217, 0.3239])
ber_pockel_qpsk = np.array([1.004e-136, 8.69e-30, 7.22e-11, 1.26e-5])
evm_pockel_16qam = np.array([0.0719, 0.1702, 0.2915, 0.411])
ber_pockel_16qam = np.array([1.157e-18, 1.515e-4, 2.25e-2, 9.27e-2])

snr_pockel_qpsk = -2*lin2db(evm_pockel_qpsk)
snr_pockel_16qam = -2*lin2db(evm_pockel_16qam)

snr_penalty_16qam = snr_pockel_qpsk - snr_pockel_16qam

ber_pockel_qpsk = np.log10(ber_pockel_qpsk)
ber_pockel_16qam = np.log10(ber_pockel_16qam)

"b variable results"
evm_bvar_qpsk = np.array([0.1072, 0.1546, 0.2284, 0.3264])
ber_bvar_qpsk = np.array([9.81e-40, 6.10e-20, 5.98e-10, 1.47e-5])
evm_bvar_16qam = np.array([0.18, 0.2416, 0.3377, 0.4433])
ber_bvar_16qam = np.array([3.69e-4, 6.64e-3, 4.58e-2, 1.15e-1])

snr_bvar_qpsk = -2*lin2db(evm_bvar_qpsk)
snr_bvar_16qam = -2*lin2db(evm_bvar_16qam)


ber_bvar_qpsk = np.log10(ber_bvar_qpsk)
ber_bvar_16qam = np.log10(ber_bvar_16qam)

"c var results"
evm_cvar_qpsk = np.array([0.0568, 0.1247, 0.217, 0.3239])
ber_cvar_qpsk = np.array([9.4e-136, 8.69e-30, 7.22e-11, 1.26e-5])
evm_cvar_16qam = np.array([0.0721, 0.1712, 0.2935, 0.412])
ber_cvar_16qam = np.array([1.27e-18, 1.515e-4, 2.25e-2, 9.27e-2])

snr_cvar_qpsk = -2*lin2db(evm_cvar_qpsk)
snr_cvar_16qam = -2*lin2db(evm_cvar_16qam)
ber_cvar_qpsk = np.log10(ber_cvar_qpsk)
ber_cvar_16qam = np.log10(ber_cvar_16qam)

"ER var results"
evm_er_var_qpsk = np.array([0.0568, 0.1247, 0.217, 0.3239])
ber_er_var_qpsk = np.array([7.34e-136, 7.82e-29, 7.22e-11, 1.78e-5])
evm_er_var_16qam = np.array([0.0721, 0.1702, 0.2915, 0.421])
ber_er_var_16qam = np.array([1.39e-18, 1.515e-4, 2.25e-2, 9.84e-1])

snr_er_var_qpsk = -2*lin2db(evm_er_var_qpsk)
snr_er_var_16qam = -2*lin2db(evm_er_var_16qam)
ber_er_var_qpsk = np.log10(ber_er_var_qpsk)
ber_er_var_16qam = np.log10(ber_er_var_16qam)

"Bias offset variable results"
evm_offset_var_qpsk = np.array([0.0752, 0.1346, 0.223, 0.3327])
ber_offset_var_qpsk = np.array([7.53e-79, 7.84e-26, 2.29e-10, 1.92e-5])
evm_offset_var_16qam = np.array([0.0877, 0.1709, 0.2921, 0.418])
ber_offset_var_16qam = np.array([4.15e-13, 1.62e-4, 2.4e-2, 9.57e-2])

snr_offset_var_qpsk = -2*lin2db(evm_offset_var_qpsk)
snr_offset_var_16qam = -2*lin2db(evm_offset_var_16qam)
ber_offset_var_qpsk = np.log10(ber_offset_var_qpsk)
ber_offset_var_16qam = np.log10(ber_offset_var_16qam)

"Realistic case"
evm_realistic_qpsk = np.array([0.0792, 0.1335, 0.2189, 0.3232])
ber_realistic_qpsk = np.array([2.64e-71, 3.16e-26, 1.05e-10, 1.21e-5])
evm_realistic_16qam = np.array([0.0874, 0.1778, 0.2963, 0.4145])
ber_realistic_16qam = np.array([3.53e-13, 2.81e-4, 2.46e-2, 9.53e-2])

snr_realistic_qpsk = -2*lin2db(evm_realistic_qpsk)
snr_realistic_16qam = -2*lin2db(evm_realistic_16qam)
ber_realistic_qpsk = np.log10(ber_realistic_qpsk)
ber_realistic_16qam = np.log10(ber_realistic_16qam)

# create a new vector to interpolate all
vector = np.linspace(32, 128, num=4, endpoint=True)
f1 = interp1d(vector, snr_realistic_qpsk, kind='cubic')
vector_new = np.linspace(32, 128, num=1001, endpoint=True)

ber_max_value1 = -3
ber_max_value2 = -2

pre_fec_ber1 = [ber_max_value1, ber_max_value1, ber_max_value1, ber_max_value1]
pre_fec_ber2 = [ber_max_value2, ber_max_value2, ber_max_value2, ber_max_value2]

plt.figure()

plt.plot(baudrates, snr_pockel_qpsk, '-', label='Pockel, QPSK', linewidth=1.5, color='b')
plt.plot(baudrates, snr_pockel_16qam, '--', label='Pockel, 16QAM', linewidth=1.5, color='b')
plt.plot(baudrates, snr_bvar_qpsk, '-', label='b var, QPSK', linewidth=1.5, color='g')
plt.plot(baudrates, snr_bvar_16qam, '--', label='b var, 16QAM', linewidth=1.5, color='g')
# plt.plot(baudrates, snr_cvar_qpsk, '-', label='c var, QPSK', linewidth=1.5, color='m')
# plt.plot(baudrates, snr_cvar_16qam, '--',label='c var, 16QAM', linewidth=1.5, color='m')
# plt.plot(baudrates, snr_er_var_qpsk, '-',label='ER var, QPSK', linewidth=1.5, color='k')
# plt.plot(baudrates, snr_er_var_16qam, '--', label='ER var, 16QAM', linewidth=1.5, color='k')
# plt.plot(baudrates, snr_offset_var_qpsk, '-',label='Bias offset var, QPSK', linewidth=1.5, color='r')
# plt.plot(baudrates, snr_offset_var_16qam, '--',label='Bias offset var, 16QAM', linewidth=1.5, color='r')
plt.plot(baudrates, snr_realistic_qpsk, '-',label='Realistic case var, QPSK', linewidth=1.5, color='r')
plt.plot(baudrates, snr_realistic_16qam, '--',label='Realistic case var, 16QAM', linewidth=1.5, color='r')
plt.title('SNR vs Baud-rate')
plt.xlabel('Baud-rate [Gbaud]')
plt.ylabel('SNR [dB]')
plt.legend()


plt.figure()

plt.plot(baudrates, ber_pockel_qpsk, '-', label='Pockel, QPSK', linewidth=1.5, color='b')
plt.plot(baudrates, ber_pockel_16qam, '--', label='Pockel, 16QAM', linewidth=1.5, color='b')
plt.plot(baudrates, ber_bvar_qpsk, '-', label='b var, QPSK', linewidth=1.5, color='g')
plt.plot(baudrates, ber_bvar_16qam, '--', label='b var, 16QAM', linewidth=1.5, color='g')
# plt.plot(baudrates, ber_cvar_qpsk, '-', label='c var, QPSK', linewidth=1.5, color='m')
# plt.plot(baudrates, ber_cvar_16qam, '--',label='c var, 16QAM', linewidth=1.5, color='m')
# plt.plot(baudrates, ber_er_var_qpsk, '-',label='ER var, QPSK', linewidth=1.5, color='k')
# plt.plot(baudrates, ber_er_var_16qam, '--', label='ER var, 16QAM', linewidth=1.5, color='k')
# plt.plot(baudrates, ber_offset_var_qpsk, '-',label='Bias offset var, QPSK', linewidth=1.5, color='r')
# plt.plot(baudrates, ber_offset_var_16qam, '--',label='Bias offset var, 16QAM', linewidth=1.5, color='r')
plt.plot(baudrates, ber_realistic_qpsk, '-',label='Realistic case var, QPSK', linewidth=1.5, color='orange')
plt.plot(baudrates, ber_realistic_16qam, '--',label='Realistic case var, 16QAM', linewidth=1.5, color='orange')
plt.plot(baudrates, pre_fec_ber1, '-', linewidth=1.5, color='r', label='BER$_{MAX,1}$ = -3')
plt.plot(baudrates, pre_fec_ber2, '-', linewidth=1.5, color='b', label='BER$_{MAX,2}$ = -2')

plt.legend()
plt.title('pre-FEC BER vs Baud-rate')
plt.xlabel('Baud-rate [Gbaud]')
plt.ylabel('log(pre-FEC BER)')

evm_qpsk_list = [evm_pockel_qpsk, evm_bvar_qpsk, evm_cvar_qpsk, evm_er_var_qpsk, ]

# plt.figure()
# plt.plot(baudrates, evm_offset_var_qpsk)
# plt.plot(baudrates, evm_offset_var_16qam)



# retrieve the number of PRBS symbols
num_PRBS = int(np.sqrt(M))
# generate a PRBS for each sqrt(M)-PAM
prbs = prbs_generator(num_PRBS, poldeg, prbs_counter, zero_pad)
# build PRBS dataframe
prbs_df=pd.DataFrame(prbs[0])
prbs_df['full'] = prbs_df[prbs_df.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1)
prbs_df = pd.DataFrame(prbs_df['full'])
prbs_no_duplicates = prbs_df.drop_duplicates()
index_dict = {}

for i in prbs_no_duplicates['full'].tolist():
    index_dict[i] = list(np.where(prbs_df['full']==i)[0])
# index_df = pd.DataFrame(index_dict)
# Retrieve PRBS for each I and Q signals
i_seq = []
q_seq = []
if modulation_format == 'QPSK':
    i_seq = prbs[0][:, 0]
    q_seq = prbs[0][:, 1]
elif modulation_format == '16QAM':
    i_seq1 = prbs[0][:, 0]
    i_seq2 = prbs[0][:, 1]
    i_seq = [str(i_seq1[i]) + str(i_seq2[i]) for i in range(len(i_seq1))]
    q_seq1 = prbs[0][:, 2]
    q_seq2 = prbs[0][:, 3]
    q_seq = [str(q_seq1[i]) + str(q_seq2[i]) for i in range(len(q_seq1))]
# backup digital sequences for testing
# i_seq = np.ones(len(i_seq)).astype(int)
# q_seq = np.ones(len(i_seq)).astype(int)

# retrieve optical input source
source = IdealLightSource(input_current)

# evaluate optical input power
input_power = source.calculate_optical_input_power(source.current)
source.input_power = input_power
# evaluate power in dBm
input_power_dbm = lin2dbm(input_power)
input_field = source.out_field

# define TX amplitude parameter in time (for the field)
k_tx = (np.sqrt(input_power)/(np.sqrt(2)))*(np.pi/(2*v_pi))

# evaluate electric fields in splitter
splitter = Splitter(source.input_power, source.out_field)
splitter_fields = splitter.calculate_arms_input_fields()
# save p and q field for the 2 arms (I, Q)
p_field = splitter_fields[0]   # [mV/m]
q_field = splitter_fields[1]   # [mV/m]
splitter.A_in_field = p_field
splitter.B_in_field = q_field

# initialize two lists, one for i_field and one for q_field
i_field_list = []
q_field_list = []
out_iq_list = []
vp_list = []
vq_list = []
vp_norm_list = []
vq_norm_list = []
# add some random AWG noise to signal
# 0 is the mean of the normal distribution I am choosing for
# std is the standard deviation of normal distribution
# num_signals is the number of signals where I want to add AWGN
if noise_flag == True:
    noise_i = np.random.normal(0, std, 2*num_signals*sps-1)
    noise_q = np.random.normal(0, std, 2*num_signals*sps-1)
else:
    noise_i = 0
    noise_q = 0

# initialize counter for scrolling noise lists
count = 0
count_i = 0
count_q = 0

if modulation_format == 'QPSK':
    if noise_flag == True:
        for i in i_seq:
            if i == 0:
                v_p_norm = -1 + noise_i[count_i]
                v_p = (-1 + noise_i[count_i])*v_pi
            elif i == 1:
                v_p_norm = 1 + noise_i[count_i]
                v_p = (1 + noise_i[count_i])*v_pi
            vp_list.append(v_p)
            vp_norm_list.append(v_p_norm)
            count_i += 1

        for i in q_seq:
            if i == 0:
                v_q_norm = -1 + noise_q[count_q]
                v_q = (-1 + noise_q[count_q]) * v_pi
            elif i == 1:
                v_q_norm = 1 + noise_q[count_q]
                v_q = (1 + noise_q[count_q]) * v_pi
            vq_list.append(v_q)
            vq_norm_list.append(v_q_norm)
            count_q += 1

    else:
        for i in i_seq:
            if i == 0:
                v_p_norm = -1
                v_p = (-1) * v_pi
            elif i == 1:
                v_p_norm = 1
                v_p = (1) * v_pi
            vp_list.append(v_p)
            vp_norm_list.append(v_p_norm)
            count_i += 1

        for i in q_seq:
            if i == 0:
                v_q_norm = -1
                v_q = (-1) * v_pi
            elif i == 1:
                v_q_norm = 1
                v_q = (1) * v_pi

            vq_list.append(v_q)
            vq_norm_list.append(v_q_norm)
            count_q += 1

        # evaluate, for each vp vq couple, the out field of  modulator (check results)
        out_iq_mod = 0.5*input_field*((np.sin((np.pi/2)*(v_p/v_pi)) - ((1/np.sqrt(er))*np.cos((np.pi/2)*(v_p/v_pi)))*1j)
                        + ((np.sin((np.pi/2)*(v_q/v_pi)) - ((1/np.sqrt(er))*np.cos((np.pi/2)*(v_q/v_pi)))*1j))*1j)

        out_iq_list.append(out_iq_mod)

elif modulation_format == '16QAM':
    if noise_flag == True:
        for i in i_seq:
            if i == '00':
                v_p_norm = -1 + noise_i[count_i]
                v_p = (-1 + noise_i[count_i])*v_pi
            elif i == '01':
                v_p_norm = -1/3 + noise_i[count_i]
                v_p = ((-1/(3*np.sqrt(2.377))) + noise_i[count_i])*v_pi
            elif i == '10':
                v_p_norm = 1 + noise_i[count_i]
                v_p = (1 + noise_i[count_i])*v_pi
            elif i == '11':
                v_p_norm = 1/3 + noise_i[count_i]
                v_p = ((1/(3*np.sqrt(2.377))) + noise_i[count_i])*v_pi
            vp_norm_list.append(v_p_norm)
            vp_list.append(v_p)
            count_i += 1

        for q in q_seq:
            if q == '00':
                v_q_norm = -1 + noise_q[count_q]
                v_q = (-1 + noise_q[count_q]) * v_pi
            elif q == '01':
                v_q_norm = -1 / 3 + noise_q[count_q]
                v_q = ((-1 / (3 * np.sqrt(2.377))) + noise_q[count_q]) * v_pi
            elif q == '10':
                v_q_norm = 1 + noise_q[count_q]
                v_q = (1 + noise_q[count_q]) * v_pi
            elif q == '11':
                v_q_norm = 1 / 3 + noise_q[count_q]
                v_q = ((1 / (3 * np.sqrt(2.377))) + noise_q[count_q]) * v_pi

            vq_norm_list.append(v_q_norm)
            vq_list.append(v_q)
            count_q += 1

    else:
        for i in i_seq:
            if i == '00':
                v_p_norm = -1
                v_p = (-1) * v_pi
            elif i == '01':
                v_p_norm = -1 / 3
                v_p = ((-1 / (3 * np.sqrt(2.377)))) * v_pi
            elif i == '10':
                v_p_norm = 1
                v_p = (1) * v_pi
            elif i == '11':
                v_p_norm = 1 / 3
                v_p = ((1 / (3 * np.sqrt(2.377)))) * v_pi
            vp_norm_list.append(v_p_norm)
            vp_list.append(v_p)
            count_i += 1

        for q in q_seq:
            if q == '00':
                v_q_norm = -1
                v_q = (-1) * v_pi
            elif q == '01':
                v_q_norm = -1 / 3
                v_q = ((-1 / (3 * np.sqrt(2.377)))) * v_pi
            elif q == '10':
                v_q_norm = 1
                v_q = (1) * v_pi
            elif q == '11':
                v_q_norm = 1 / 3
                v_q = ((1 / (3 * np.sqrt(2.377)))) * v_pi

            vq_norm_list.append(v_q_norm)
            vq_list.append(v_q)
            count_q += 1

        # evaluate, for each vp vq couple, the out field of  modulator (check results)
        out_iq_mod = 0.5*input_field*((np.sin((np.pi/2)*(v_p/v_pi)) - ((1/np.sqrt(er))*np.cos((np.pi/2)*(v_p/v_pi)))*1j)
                        + ((np.sin((np.pi/2)*(v_q/v_pi)) - ((1/np.sqrt(er))*np.cos((np.pi/2)*(v_q/v_pi)))*1j))*1j)
        out_iq_list.append(out_iq_mod)

# define ideal values of constellation points in constellation diagram
if noise_flag == True:
    ideal_norm_constellation = np.array(vp_norm_list - noise_i[:len(vp_norm_list)]) + np.array(vq_norm_list - noise_q[:len(vq_norm_list)])*1j
else:
    ideal_norm_constellation = np.array(vp_norm_list) + np.array(vq_norm_list)*1j

# plot_constellation(ideal_norm_constellation, 'Ideal Constellation Diagram')
# define Oversampled signals for I and Q sequences, adding zeros
i_sig = np.array([])
q_sig = np.array([])

if modulation_format == 'QPSK':
    # I bits
    for bit in i_seq:
        pulse = np.zeros(sps)
        pulse[0] = bit*2 - 1    # set the first value to either a 1 or -1
        i_sig = np.concatenate((i_sig, pulse))  # add the N samples to the signal

    # Q bits
    for bit in q_seq:
        pulse = np.zeros(sps)
        pulse[0] = bit*2 - 1    # set the first value to either a 1 or -1
        q_sig = np.concatenate((q_sig, pulse))  # add the N samples to the signal

elif modulation_format == '16QAM':
    # I bits
    for bit in i_seq:
        pulse = np.zeros(sps)
        # set the first value to 1, 0.3, -0.3 or -1
        if bit == '11':
            pulse[0] = 1/3
        elif bit == '10':
            pulse[0] = 1
        elif bit == '01':
            pulse[0] = -1/3
        elif bit == '00':
            pulse[0] = -1
        i_sig = np.concatenate((i_sig, pulse))  # add the N samples to the signal

    # Q bits
    for bit in q_seq:
        pulse = np.zeros(sps)
        # set the first value to 1, 0.3, -0.3 or -1
        if bit == '11':
            pulse[0] = 1/3
        elif bit == '10':
            pulse[0] = 1
        elif bit == '01':
            pulse[0] = -1/3
        elif bit == '00':
            pulse[0] = -1
        q_sig = np.concatenate((q_sig, pulse))  # add the N samples to the signal

# define time axis for the 'square wave'
t1 = np.linspace(0, num_signals*Ts*1e9, len(i_sig))
fig, axs = plt.subplots(2, 1, figsize=(9, 10))
fig.suptitle('I and Q samples')
axs[0].set_title('I samples')
axs[0].set(xlabel='Time [ns] ', ylabel='Bit values')
axs[0].plot(t1, i_sig, label='I')
axs[1].set_title('Q samples')
axs[1].set(xlabel='Time [ns] ', ylabel='Bit values')
axs[1].plot(t1, q_sig, label='Q')
plt.grid(True)

input_power_wave = input_power_dbm*np.sin(2*np.pi*frequency*t1)
power_wave_freq = (np.fft.fft(input_power_wave))
magn_pow = np.abs(power_wave_freq)
# plt.figure()
# plt.plot(magn_pow)
# plt.show(block=False)

# Create our root-raised-cosine (RRC) filter
# rrcos params:
# N = N_taps = length of the filter in samples = samples per symbol * number of transmitted signals
# beta = roll-off
# Ts = symbol period (in seconds)
# samp_f = sampling frequency = Rs*sps
rrcos_filter = flt.rrcosfilter(N_taps, beta, Ts, samp_f)

# Perform FFT of RRCOS filter and retrieve the frequencies associated
rrcos_fft = np.fft.fft(rrcos_filter[1])
rrcos_fft_mag = 20*np.log10(np.abs(rrcos_fft))
rrcos_freq = np.fft.fftfreq(n=rrcos_filter[1].size, d=1/samp_f)

# Everytime perform an FFT, perform the shift in order to make first FFT of negative freqs
# and then of the positive ones, contrary to default approach of numpy FFT
rrcos_fft = np.fft.fftshift(rrcos_fft)
rrcos_freq = np.fft.fftshift(rrcos_freq)

# plt.figure()
# plt.plot(rrcos_filter[0], rrcos_filter[1])
# plt.grid(True)
# plt.title('RRC Filter Pulse Time response')
# plt.xlabel('Time')
# plt.ylabel('h(t)')
#
plt.figure()
plt.plot(rrcos_freq, np.abs(rrcos_fft)/sps)
plt.grid(True)
plt.title('RRC Filter Pulse Frequency response')
plt.show(block=False)
plt.xlabel('Frequency [GHz]')
plt.ylabel('H(f)')
# here we need to create the custom LPF generated by misurated values and apply it at the input of the modulator
lpf = pd.read_csv(input_folder/'FreqResponse_belmonte.csv', usecols= ['Freq [GHz]','dB20(S21diff) norm2'], index_col=False)
lpf['Freq [GHz]'] = lpf['Freq [GHz]']*1e9
# lpf.plot(kind = 'line',
#         x = 'Freq [GHz]',
#         y = 'dB20(S21diff) norm2',
#         color = 'green')
# plt.title('Custom LPF')
# plt.show(block=False)

freqs_filter = lpf['Freq [GHz]']
pow_filter = lpf['dB20(S21diff) norm2']
# Add negative frequencies to filter to perform multiplication with signal FFT later
freqs_no_zero = freqs_filter[1::]
pow_no_zero = pow_filter[1::]
negative_freqs = freqs_no_zero[::-1]*(-1)
negative_pows = pow_no_zero[::-1]
tot_freqs = negative_freqs.append(freqs_filter)
tot_pows = negative_pows.append(pow_filter)

# lpf_dict = {'Freqs': tot_freqs, 'Power [dB20]': tot_pows}
# lpf_pow_df = pd.DataFrame(lpf_dict)
# lpf_pow_df.plot(kind = 'line',
#         x = 'Freqs',
#         y = 'Power [dB20]',
#         color = 'green')
# plt.title('Custom LPF (Power)')
# plt.show(block=False)

# Take the amplitude values to perform FFT
magn_filter = control.db2mag(tot_pows)
# magn_db = 10*np.log10(magn_filter)
lpf_dict_magn = {'Frequency': tot_freqs, 'Amplitude': magn_filter}

lpf_mag_df = pd.DataFrame(lpf_dict_magn)
# lpf_mag_df.plot(kind = 'line',
#         x = 'Frequency',
#         y = 'Amplitude',
#         color = 'green')
# plt.title('Custom LPF (Amplitude)')
# plt.ylabel('Amplitude (normalized)')
# plt.show(block=False)

# retrieve the frequency values of RRCOS associated to LPF frequencies to obtain a new set of
# frequencies useful to interpolate data of the custom LPF
boolean_array = np.logical_and(rrcos_freq >= min(np.array(tot_freqs)), rrcos_freq <= max(np.array(tot_freqs)))
new_array = rrcos_freq[np.where(boolean_array)]
# take the values for interpolation
x = rrcos_fft[np.where(boolean_array)]
y = new_array
f = scipy.interpolate.griddata(np.array(tot_freqs), np.array(lpf_mag_df['Amplitude']), new_array)
# create the lists related to empty values of frequencies, and give them the minimum value. then insert in df
lost_freqs = np.array([x for x in rrcos_freq if x not in new_array])
min_amp_array = np.array([min(f) for x in range(len(lost_freqs))])
lpf_dict_magn_interp = {'Freqs': y, 'Amplitude': f}

lpf_mag_df_interp = pd.DataFrame(lpf_dict_magn_interp)

fill_values_df_dict = {'Freqs': lost_freqs, 'Amplitude': min_amp_array}

fill_values_df = pd.DataFrame(fill_values_df_dict)
# lpf_mag_df_interp.plot(kind = 'line',
#         x = 'Freqs',
#         y = 'Amplitude',
#         color = 'green')
#
# plt.title('Custom LPF (Amplitude) after interpolation')
# plt.show(block=False)

# merge the two dfs to have the same dimension of the RRCOS filter
lpf_filled_df = pd.merge_ordered(fill_values_df, lpf_mag_df_interp)
# lpf_filled_df.plot(kind = 'line',
#         x = 'Freqs',
#         y = 'Amplitude',
#         color = 'green')
#
# plt.title('Custom LPF (Amplitude) on the whole set of freqs')
# plt.show(block=False)
# Perform FFT of signals and retrieve the frequencies
# FFT performed only on the wanted samples
if noise_flag == True:
    i_samples = [i_sig[i*sps] + noise_i[i] for i in range(num_signals)]
    q_samples = [q_sig[i*sps] + noise_q[i] for i in range(num_signals)]
else:
    i_samples = [i_sig[i*sps] for i in range(num_signals)]
    q_samples = [q_sig[i*sps] for i in range(num_signals)]


i_sig_fft = np.fft.fft(i_sig)
q_sig_fft = np.fft.fft(q_sig)
i_sig_fft = np.fft.fftshift(i_sig_fft)
q_sig_fft = np.fft.fftshift(q_sig_fft)
i_sig_fft_mag = 20*np.log10(np.abs(i_sig_fft))
q_sig_fft_mag = 20*np.log10(np.abs(q_sig_fft))

sig_freq = np.fft.fftfreq(n=i_sig.size, d=1/samp_f)
sig_freq = np.fft.fftshift(sig_freq)

# fig, axs = plt.subplots(2, 1, figsize=(9, 10))
# fig.suptitle('I and Q Signals FFT')
# axs[0].set_title('I Signal (FFT)')
# axs[0].plot(sig_freq, i_sig_fft_mag, label='I')

# axs[1].set_title('Q Signal (FFT)')
# axs[1].plot(sig_freq, q_sig_fft_mag, label='Q')
# plt.grid(True)
# plt.show(block=False)

# Filter our signals, in order to apply the pulse shaping
# These are the shaped signals we can use
# add some noise generated randomly before

# in time domain
i_shaped_t = scipy.signal.fftconvolve(i_sig, rrcos_filter[1]) + noise_i
q_shaped_t = scipy.signal.fftconvolve(q_sig, rrcos_filter[1]) + noise_q

# in frequency domain
i_shaped = np.multiply(i_sig_fft, rrcos_fft)
q_shaped = np.multiply(q_sig_fft, rrcos_fft)
i_shaped_mag = 20*np.log10(np.abs(i_shaped))
q_shaped_mag = 20*np.log10(np.abs(q_shaped))

# plot I and Q signals after rrcos filtering in frequency

# fig, axs = plt.subplots(2, 1, figsize=(9, 10))
# fig.suptitle('I and Q PSD after RRCOS filtering (in frequency)')
# axs[0].set_title('I pulse shaped')
# axs[0].plot(rrcos_freq, i_shaped_mag, label='I')
# axs[0].set_xlabel('Frequency')
# axs[0].set_ylabel('PSD')
#
# axs[1].set_title('Q pulse shaped')
# axs[1].plot(rrcos_freq, q_shaped_mag, label='Q')
#
# fig, axs = plt.subplots(2, 1, figsize=(9, 10))
# fig.suptitle('I and Q after RRCOS filtering (in time)')
# axs[0].set_title('I pulse shaped')
# axs[0].plot(i_shaped_t, label='I')
#
# axs[1].set_title('Q pulse shaped')
# axs[1].plot(q_shaped_t, label='Q')

# for i in range(num_signals):
#     axs[0].plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(i_shaped_t), max(i_shaped_t)])
#     axs[1].plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(q_shaped_t), max(q_shaped_t)])

# plt.grid(True)
# plt.show(block=False)

# here we need to create the custom LPF generated by measured values and apply it at the input of the modulator
# Apply LPF, but before that convert the dB20 values to magnitude values
bandcut_i = np.multiply(i_shaped, lpf_filled_df['Amplitude'])
bandcut_q = np.multiply(q_shaped, lpf_filled_df['Amplitude'])

bandcut_i_mag = 20*np.log10(np.abs(bandcut_i))
bandcut_q_mag = 20*np.log10(np.abs(bandcut_q))

# plot I and Q signals after LPF filtering in frequency

# fig, axs = plt.subplots(2, 1, figsize=(9, 10))
# fig.suptitle('I and Q PSD after LPF filtering (in frequency)')
# axs[0].set_title('I pulse shaped')
# axs[0].plot(rrcos_freq, bandcut_i_mag, label='I')
# axs[0].set_xlabel('Frequency')
# axs[0].set_ylabel('PSD')
# axs[1].set_title('Q pulse shaped')
# axs[1].plot(rrcos_freq, bandcut_q_mag, label='Q')
# axs[1].set_xlabel('Frequency')
# axs[1].set_ylabel('PSD')

# plot of the wanted signal in time
# here we add AWGN noise to see what happens to the signal when a noise is present
# if we add it before the IFFT, no effect is visible
if noise_flag == True:
    i_shaped_time = np.fft.ifft(bandcut_i) + noise_i[0:len(bandcut_i)]
    q_shaped_time = np.fft.ifft(bandcut_q) + noise_q[0:len(bandcut_q)]
else:
    i_shaped_time = np.fft.ifft(bandcut_i)
    q_shaped_time = np.fft.ifft(bandcut_q)

# fig, axs = plt.subplots(2, 1, figsize=(9, 10))
# fig.suptitle('I and Q Pulse shaped (in time)')
# axs[0].set_title('I pulse shaped')
# axs[0].set(xlabel='Time [ns] ', ylabel='Voltage (V)')
# axs[0].plot(t1, i_shaped_time, label='I')
#
# axs[1].set_title('Q pulse shaped')
# axs[1].set(xlabel='Time [ns] ', ylabel='Voltage (V)')
# axs[1].plot(t1, q_shaped_time, label='Q')
#
# plt.grid(True)

# plt.figure()
# plt.plot(t1, i_shaped_mag)
# plt.grid(True)

# for i in range(num_signals):
#     axs[0].plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(i_shaped_t), max(i_shaped_t)])
#     axs[1].plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(q_shaped_t), max(q_shaped_t)])

# plt.grid(True)
# plt.show(block=False)

# define actual input signals going to RF electrodes of the MZM, taking only the correct samples at every Ts
# p_sign = [i_shaped_t[i*sps+N_taps//2+1] * v_tx_param for i in range(num_signals)]
# q_sign = [q_shaped_t[i*sps+N_taps//2+1] * v_tx_param for i in range(num_signals)]

# multiply I and Q pulses for the transmission amplitude parameter for voltages
p_sign = i_shaped_time * v_tx_param
q_sign = q_shaped_time * v_tx_param

# try using signals from convolution in time
# p_sign = i_shaped_t * v_tx_param
# q_sign = q_shaped_t * v_tx_param

# define Single Electrode Classic MZM for both I and Q arms
i_MZM_list = [MZMSingleElectrode(p_field, p_signal, v_pi) for p_signal in p_sign]
i_field_list = [i_MZM.out_mzm_field_evaluation() for i_MZM in i_MZM_list]
q_MZM_list = [MZMSingleElectrode(q_field, q_signal, v_pi) for q_signal in q_sign]
# for Q field evaluation, apply Phase Modulator effect rotating it, multiplying it for -1j
q_field_list = [q_MZM.out_mzm_field_evaluation()*(-1)*(1j) for q_MZM in q_MZM_list]

# combiner output field

combiner_list = [Combiner(i_field_list[i], q_field_list[i]) for i in range(len(i_field_list))]
out_fields_combiner_list = [combiner.combiner_out_field(combiner.in_A, combiner.in_B) for combiner in combiner_list]
out_fields_samples = [out_fields_combiner_list for i in range(num_signals)]

# plt.figure()
# plt.plot(out_fields_combiner_list)
# # for i in range(num_signals):
# #     plt.plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(out_fields_combiner_list), max(out_fields_combiner_list)])
# plt.grid(True)
# plt.title('Out_field (Classical MZM)')

# plot constellations
constellation_samples_classic = [out_fields_combiner_list[i*sps] for i in range(num_signals)]

# plot constellations, normalized wrt a constant value 0.75, v_pi,
# the constant transmission parameter and insertion loss
classic_tx_const = (constellation_samples_classic/
                   (np.sqrt(input_power)))/(v_tx_param/v_pi)*(1/np.sqrt(insertion_loss))/norm_factor

# plot_constellation(classic_tx_const, 'LiNbO_3 MZM Constellation at TX side, before RX MF')

# create a list of Griffin MZMs using this input voltage interval
# take into account the non-ideal params as b, c
# for this evaluation, consider common mode voltage at v_bias
v_bias = -np.pi/v_pi

# inserting *np.arcsin(np.sqrt(1/(1+er_i))) we get the dependence on ER
i_Griffin_list = [GriffinMZM(bias_offset_i + v_bias,
                             (v_p_tem)/2 + 1.5*v_pi + bias_offset_i/2,
                             -1.5*v_pi + (-v_p_tem)/2 - bias_offset_i/2, gamma_1, gamma_2,
                             phase_offset, b, c, er_i) for v_p_tem in p_sign]
i_Griffin_field_list = [i_Griffin.griffin_eo_tf_field()*p_field for i_Griffin in i_Griffin_list]
q_Griffin_list = [GriffinMZM(bias_offset_q + v_bias,
                             1.5*v_pi + (v_q_tem)/2 + bias_offset_q/2,
                             -1.5*v_pi + (-v_q_tem)/2 - bias_offset_q/2, gamma_1, gamma_2,
                             phase_offset, b, c, er_q) for v_q_tem in q_sign]
# for Q field evaluation, apply Phase Modulator effect rotating it, multiplying it for -1j
q_Griffin_field_list = [q_Griffin.griffin_eo_tf_field()*q_field*(-1)*(1j) for q_Griffin
                        in q_Griffin_list]
i_griffin_er_list = np.array([lin2db(mzm.griffin_il_er()[1]) for mzm in i_Griffin_list])
q_griffin_er_list = np.array([lin2db(mzm.griffin_il_er()[1]) for mzm in q_Griffin_list])

# combiner output field Griffin

combiner_Griffin_list = [Combiner(i_Griffin_field_list[i], q_Griffin_field_list[i]) for i
                         in range(len(i_Griffin_field_list))]
out_fields_combiner_Griffin_list = [combiner.combiner_out_field(combiner.in_A, combiner.in_B) for combiner
                                    in combiner_Griffin_list]
# plt.figure()
#
# plt.plot(out_fields_combiner_Griffin_list)
# # for i in range(num_signals):
# #     plt.plot([i*sps+N_taps//2+1, i*sps+N_taps//2+1], [min(out_fields_combiner_Griffin_list),
# #     max(out_fields_combiner_Griffin_list)])
# plt.grid(True)
# plt.title('Out_field (Griffin)')

# plot constellations, normalized wrt a constant value 0.75, v_pi and
# the constant transmission parameter
out_fields_samples_Griffin = [out_fields_combiner_Griffin_list for i in range(num_signals)]
constellation_samples_Griffin = [out_fields_combiner_Griffin_list[i*sps] for i in range(num_signals)]

griffin_tx_const = constellation_samples_Griffin/\
                   (np.sqrt(input_power))/(v_tx_param/v_pi)/norm_factor
# plot_constellation(griffin_tx_const, 'InP MZM Constellation at TX side, before RX MF')

# Perform FFT on the mixed signal output
# here don't perform the fftshift because alreadout_fields_combiner_Griffin_listy done previously on fft functions

# fft_power_out = (np.fft.fft(out_fields_samples_Griffin))**2
fft_classical_out = (np.fft.fft(out_fields_samples))
fft_classical_mag = 20*np.log10(np.abs(fft_classical_out))

# fft_classical_mag = savitzky_golay(fft_classical_mag[0], 51, 3)

# plt.figure()
# plt.plot(rrcos_freq, savitzky_golay(fft_classical_mag[0], 51, 3))
# plt.grid(True)
# plt.xlabel('Frequency')
# plt.ylabel('PSD')
# plt.title('Out_field spectrum (LiNbO_3 MZM)')

# fft_power_out = (np.fft.fft(out_fields_samples_Griffin))**2
fft_griffin_out = (np.fft.fft(out_fields_samples_Griffin))

fft_freqs = np.fft.fftfreq(len(out_fields_samples_Griffin))
fft_griffin_mag = 20*np.log10(np.abs(fft_griffin_out))

# fft_griffin_mag = savitzky_golay(fft_griffin_mag[0], 51, 3)
# fft_mag_clean = np.fft.fftshift(fft_mag[0])

# plt.figure()
# plt.plot(rrcos_freq, savitzky_golay(fft_griffin_mag[0], 51, 3))
# plt.xlabel('Frequency')
# plt.ylabel('PSD')
# plt.grid(True)
# plt.title('Out_field spectrum (InP MZM)')

# Filter our signals, in order to simulate the coherent receiver
# Apply again the RRCOS filter on output shape and see the results

# Generate FFT of fields out of combiner, and take only magnitude
fft_classical_rx = (np.fft.fft(out_fields_combiner_list))
fft_classical_rx_mag = 20*np.log10(np.abs(fft_classical_rx))

fft_griffin_rx = (np.fft.fft(out_fields_combiner_Griffin_list))
fft_griffin_rx_mag = 20*np.log10(np.abs(fft_griffin_rx))

# apply RRCOS filter in frequency by multiplication
griffin_adapt_filter = np.multiply(fft_griffin_rx, rrcos_fft)
classical_adapt_filter = np.multiply(fft_classical_rx, rrcos_fft)

# apply IFFT to obtain the waveforms in time domain
griffin_rx_time = np.fft.ifft(griffin_adapt_filter)
classical_rx_time = np.fft.ifft(classical_adapt_filter)

# Plot constellations at RX side
constellation_rx_classic = [classical_rx_time[i*sps] for i in range(num_signals)]
classic_rx_const = constellation_rx_classic/(np.sqrt(input_power))/(v_tx_param/v_pi)*(1/np.sqrt(insertion_loss))\
                   /norm_factor/norm_rx
# plot_constellation(classic_rx_const, 'Standard MZM Constellation at RX side')
# In Griffin, apply a phase correction filter (CPE) based on the phase non-linearity correction
# using as parameters b and v_pi, considering that for a correct functionality of Griffin
# v_cm = 2*v_pi and (DC values) v_A = -v_B = 1.5 v_pi
# cpe_griffin = np.exp(-((1 + 2*b)*v_bias*(1.5*v_pi) - b*(1.5*v_pi)**2)*1j)
cpe_griffin = np.exp((b*(1.5*v_pi)**2)*1j)
# cpe_griffin = 1
constellation_rx_Griffin = [griffin_rx_time[i*sps]*cpe_griffin for i in range(num_signals)]
# *np.exp(np.mean([griffin_rx_time[i].imag for i in range(len(griffin_rx_time))])*(1j))
griffin_rx_const = constellation_rx_Griffin/(np.sqrt(input_power))/(v_tx_param/v_pi)/norm_factor/norm_rx
# plot_constellation(griffin_rx_const, 'InP MZM Constellation at RX side')

# shift transmitted elements to take back the correct timing synchronization after fft shift
shift_list = collections.deque(classic_tx_const)
shift_list.rotate(int(np.ceil(len(classic_tx_const)/2)))
shifted_list_classic = np.array(list(shift_list))

shift_list2 = collections.deque(griffin_tx_const)
shift_list2.rotate(int(np.ceil(len(griffin_tx_const)/2)))
shifted_list_griffin = np.array(list(shift_list2))
# evm_trial = evm_rms_estimation(prbs_no_duplicates, index_dict, ideal_norm_constellation, ideal_norm_constellation)
evm_classic_tx = evm_rms_estimation(prbs_no_duplicates, index_dict, ideal_norm_constellation, shifted_list_classic)*100
evm_classic_rx = evm_rms_estimation(prbs_no_duplicates, index_dict, ideal_norm_constellation, classic_rx_const)*100
evm_griffin_tx = evm_rms_estimation(prbs_no_duplicates, index_dict, ideal_norm_constellation, shifted_list_griffin)*100
evm_griffin_rx = evm_rms_estimation(prbs_no_duplicates, index_dict, ideal_norm_constellation, griffin_rx_const)*100

# print('EVM CLASSIC TX:', evm_classic_tx, '%')
print('EVM CLASSIC RX:', evm_classic_rx, '%')
# print('EVM GRIFFIN TX:', evm_griffin_tx, '%')
print('EVM GRIFFIN RX:', evm_griffin_rx, '%')

ber_classic_tx = ber_estimation(evm_classic_tx/100)
ber_classic_rx = ber_estimation(evm_classic_rx/100)
ber_griffin_tx = ber_estimation(evm_griffin_tx/100)
ber_griffin_rx = ber_estimation(evm_griffin_rx/100)

snr_classic_rx = -2*lin2db(evm_classic_rx/100)
snr_griffin_rx = -2*lin2db(evm_griffin_rx/100)

# print('BER CLASSIC TX:', ber_classic_tx)
print('BER CLASSIC RX:', ber_classic_rx)
print('SNR CLASSIC RX:', snr_classic_rx)
print('BER GRIFFIN RX:', ber_griffin_rx)
print('SNR GRIFFIN RX:', snr_griffin_rx)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Modulation space and PSD for '+modulation_format+' at TX side')
axs[0, 0].set_title('LiNbO_3 MZM')
x = [ele.real for ele in classic_tx_const]
y = [ele.imag for ele in classic_tx_const]

axs[0, 0].scatter(x, y)
axs[0, 0].set_ylabel('Im{E_out}')
axs[0, 0].set_xlabel('Re{E_out}')

axs[0, 1].set_title('InP MZM')
x = [ele.real for ele in griffin_tx_const]
y = [ele.imag for ele in griffin_tx_const]

axs[0, 1].scatter(x, y)
axs[0, 1].set_ylabel('Im{E_out}')
axs[0, 1].set_xlabel('Re{E_out}')

axs[1, 0].plot(rrcos_freq, savitzky_golay(fft_classical_mag[0], 51, 3))
axs[1, 0].grid(True)
axs[1, 0].set_xlabel('Frequency')
axs[1, 0].set_ylabel('PSD')
axs[1, 0].set_title('LiNbO_3 MZM')

axs[1, 1].plot(rrcos_freq, savitzky_golay(fft_griffin_mag[0], 51, 3))
axs[1, 1].grid(True)
axs[1, 1].set_xlabel('Frequency')
axs[1, 1].set_ylabel('PSD')
axs[1, 1].set_title('InP MZM')

plt.tight_layout()
plt.interactive(False)


fig, axs = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle('Modulation space for '+modulation_format+' at RX side')
axs[0].set_title('LiNbO_3 MZM')
x = [ele.real for ele in classic_rx_const]
y = [ele.imag for ele in classic_rx_const]

axs[0].scatter(x, y)
axs[0].set_ylabel('Im{E_out}')
axs[0].set_xlabel('Re{E_out}')

axs[1].set_title('InP MZM')
x = [ele.real for ele in griffin_rx_const]
y = [ele.imag for ele in griffin_rx_const]

axs[1].scatter(x, y)
axs[1].set_ylabel('Im{E_out}')
axs[1].set_xlabel('Re{E_out}')

plt.tight_layout()
plt.interactive(False)



plt.show(block=True)

