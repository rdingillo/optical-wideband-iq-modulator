import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import seaborn as sns; sns.set_theme()
from colorama import Fore
from pathlib import Path

from mzm_model.core.elements import IdealLightSource, Splitter, Combiner, MZMSingleElectrode, GriffinMZM
from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, v_A, v_B, v_pi, v_cm, \
    v_diff, v_in_limit, v_in_step, er, insertion_loss, phase_offset, v_off, b, c, wavelength, gamma_1, gamma_2, \
    modulation_format, n_bit, num_signals, std, op_freq, M, r_b, r_s, sim_duration, num_samples, num_real_samples, Ts, \
    poldeg, prbs_counter, zero_pad
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin
from mzm_model.core.science_utils import rand_key
from mzm_model.core.utils import plot_constellation, eo_tf_iq_draw
from scipy import signal
from mzm_model.core.prbs_generator import prbs_generator

matplotlib.use('Qt5Agg')
start_time = time.time()

# print()

root = Path(__file__).parent.parent
input_folder = root/'resources'
folder_results = root/'mzm_model'/'results'

# retrieve optical input source
source = IdealLightSource(input_current)

# evaluate optical input power
input_power = source.calculate_optical_input_power(source.current)
source.input_power = input_power
# evaluate power in dBm
input_power_dbm = lin2dbm(input_power)
input_field = source.out_field

# define TX amplitude parameter in time
k_tx = (np.sqrt(input_power)/(np.sqrt(2)))*(np.pi/(2*v_pi))

# retrieve the number of PRBS
num_PRBS = int(np.sqrt(M))

# define driver code according to modulation format
driver_code = prbs_generator(num_PRBS, poldeg, prbs_counter, zero_pad)
i_seq = np.array(driver_code[0][:, 0])
q_seq = np.array(driver_code[0][:, 0])
print(driver_code)

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
noise_i = np.random.normal(0, std, num_signals)
noise_q = np.random.normal(0, std, num_signals)
# initialize counter for scrolling noise lists
count = 0
count_i = 0
count_q = 0
# define a constant parameter for signal, given the cardinality M
v_param = 0.5*v_pi*(np.sqrt(M)-1)
if modulation_format == 'QPSK':
    for i in i_seq:
        if i == 0:
            v_p_norm = -1
            v_p = (-1 + noise_i[count_i]) * v_pi
        elif i == 1:
            v_p_norm = 1
            v_p = (1 + noise_i[count_i]) * v_pi
        vp_list.append(v_p)
        vp_norm_list.append(v_p_norm)
        count_i += 1

    for i in q_seq:
        if i == 0:
            v_q_norm = -1
            v_q = (-1 + noise_i[count_q]) * v_pi
        elif i == 1:
            v_q_norm = 1
            v_q = (1 + noise_i[count_q]) * v_pi
        vq_list.append(v_q)
        vq_norm_list.append(v_q_norm)
        count_q += 1
        v_p = v_param*v_p_norm
        v_q = v_param*v_q_norm
        vp_list.append(v_p)
        vq_list.append(v_q)
        vp_norm_list.append(v_p_norm)
        vq_norm_list.append(v_q_norm)

        count += 1

elif modulation_format == '16QAM':
    for i in driver_code:
        if i[:2] == '00':
            v_p_norm = -1
            v_p = (-1 + noise_i[count])*v_pi
        elif i[:2] == '01':
            v_p_norm = -1/3
            v_p = ((-1/(3*np.sqrt(2.377))) + noise_i[count])*v_pi
        elif i[:2] == '10':
            v_p_norm = 1
            v_p = (1 + noise_i[count])*v_pi
        elif i[:2] == '11':
            v_p_norm = 1/3
            v_p = ((1/(3*np.sqrt(2.377)))+ noise_i[count])*v_pi
        if i[2:4] == '00':
            v_q_norm = -1
            v_q = (-1 + noise_q[count])*v_pi
        elif i[2:4] == '01':
            v_q_norm = -1/3
            v_q = ((-1/(3*np.sqrt(2.377))) + noise_q[count])*v_pi
        elif i[2:4] == '10':
            v_q_norm = 1
            v_q = (1 + noise_q[count])*v_pi
        elif i[2:4] == '11':
            v_q_norm = 1/3
            v_q = ((1/(3*np.sqrt(2.377)))+noise_q[count])*v_pi
        vp_norm_list.append(v_p_norm)
        vq_norm_list.append(v_q_norm)
        vp_list.append(v_p)
        vq_list.append(v_q)

        count += 1

print()

"""In this piece of code, emulation of a theoretical behavior, without considering DSP and DAC"""

# define the 2 signals for P and Q vectors
sig1 = np.zeros(int(np.ceil(sim_duration*num_real_samples/Ts)))
sig2 = np.zeros(int(np.ceil(sim_duration*num_real_samples/Ts)))

# find the indexes where the bits are equal to 1
id_p = np.where(i_seq == 1)
for i in id_p[0]:
    temp = int(i*num_real_samples)
    # then assign the actual value of the field retrieving it from vp_list
    sig1[temp:temp+num_real_samples] = vp_list[i]


# repeat the same operations on the Q signal
id_q = np.where(q_seq == 1)
for i in id_q[0]:
    temp = int(i*num_real_samples)
    sig2[temp:temp+num_real_samples] = vq_list[i]

# set after how long you want to take the samples
n_samp = int(np.ceil(num_real_samples/num_samples))
# set time axis
t = np.linspace(0, sim_duration, int(np.ceil(sim_duration*num_real_samples/Ts)))
# plot the squarewaves of the input digital signals
fig, axs = plt.subplots(2, 1, figsize=(9, 10))
fig.suptitle('P and Q signals')
# plot P signals
axs[0].set_title('P signals')
axs[0].set(xlabel='Time [s]', ylabel='Amplitude')
plt.grid()
axs[0].plot(t, sig1)
axs[0].scatter(t[n_samp-3::n_samp], sig1[n_samp-3::n_samp])

# plot Q signals
axs[1].set_title('Q signals')
axs[1].set(xlabel='Time [s]', ylabel='Amplitude')
plt.grid()
axs[1].plot(t, sig2)
axs[1].scatter(t[n_samp-3::n_samp], sig2[n_samp-3::n_samp])

# take only the samples
t_samp = t[n_samp-3::n_samp]
p_sig_samp = sig1[n_samp-3::n_samp]*np.sinc(t_samp)
q_sig_samp = sig2[n_samp-3::n_samp]*np.sinc(t_samp)
plt.figure(2)
plt.plot(t_samp, p_sig_samp)
plt.grid()

# apply the sinc to all samples

tx_field_time_list = [(k_tx/np.sqrt(2))*(vp_list[i] + vq_list[i]*1j) for i in range(len(driver_code))]

# plot the sqrt(M)-PAM constellations and, consequently, the effective constellation in use
plt.figure(figsize=(7, 7))
plt.title('Modulation space for '+modulation_format)
x = [ele.real for ele in tx_field_time_list]
y = [ele.imag for ele in tx_field_time_list]
plt.scatter(x, y)
plt.ylabel('Im{E_out}')
plt.xlabel('Re{E_out}')
plt.tight_layout()
plt.interactive(False)

plt.show(block=True)



