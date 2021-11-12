import scipy.constants as con
import scipy.special
import math
import numpy as np
import random
import copy

from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, gamma_1, gamma_2, k_splitter, \
    b, c, wavelength, phi_laser, er, insertion_loss, v_off, v_cm, M, evm_factor, L
from mzm_model.core.math_utils import lin2db

'''Ideal Light Source methods'''


def calculate_alfa_s(self):
    alfa_s = ((h*frequency)/q)*eta_s    # [V]
    return alfa_s


def calculate_optical_input_power(self, input_source):
    alfa = self.calculate_alfa_s()
    optical_input_power = alfa*input_source     # [W]
    self.input_power = optical_input_power
    self.out_field = np.sqrt(optical_input_power)*np.exp(phi_laser*1j)
    return optical_input_power


'''Splitter methods'''


def calculate_arms_input_fields(self):
    A_in_power = self.input_power*(gamma_1**2)
    B_in_power = self.input_power*(1-(gamma_1**2))
    a_amplitude = (np.sqrt(A_in_power))/(np.sqrt(k_splitter))
    b_amplitude = (np.sqrt(B_in_power))/(np.sqrt(k_splitter))
    A_in_field = self.input_field*gamma_1
    B_in_field = (self.input_field*gamma_1)*1j
    return A_in_field, B_in_field


'''Waveguides methods'''


def delta_phi_evaluation(self, v_arm, v_pi):
    delta_phi = np.pi*(v_arm/v_pi)
    return delta_phi


def delta_phi_evaluation_k(self, v_arm, k_arm):
    delta_phi = v_arm*k_arm
    return delta_phi


def out_field_evaluation(self, arm_field, v_arm, v_pi):
    delta_phi = self.delta_phi_evaluation(v_arm, v_pi)
    out_field = arm_field*np.exp(delta_phi*1j)
    return out_field


'''Combiner methods'''


def combiner_out_field(self, in_A, in_B):
    comb_out_field = gamma_2*(in_A+in_B*1j)
    self.out_field = comb_out_field
    return comb_out_field


'''Ideal MZM methods'''


def field_ratio_evaluation(self):
    v_cm = self.v_cm
    v_diff = self.v_diff
    v_pi = self.v_pi
    field_ratio = np.exp(((np.pi * v_cm) / (2 * v_pi)) * 1j) * np.sin((np.pi / 2) * (v_diff / v_pi))
    return field_ratio


# define electro optic transfer function
def eo_tf_evaluation(self):
    v_cm = self.v_cm
    v_diff = self.v_diff
    v_pi = self.v_pi
    eo_tf = (np.sin((np.pi / 2) * (v_diff / v_pi)))**2
    return eo_tf


'''Non-Ideal MZM methods'''


def field_ratio_evaluation_non_ideal(self):
    v_cm = self.v_cm
    v_diff = self.v_diff
    v_pi = self.v_pi
    er = self.er
    v_off = self.v_off
    insertion_loss = self.insertion_loss
    phase_offset = self.phase_offset
    field_ratio = np.sqrt(insertion_loss) * np.exp(((np.pi * v_cm) / (2 * v_pi)) * 1j) * \
                  (np.sin((np.pi / 2) * ((v_diff - v_off)/v_pi)) - (1/np.sqrt(er))*1j*np.cos((np.pi / 2) * \
                    ((v_diff - v_off)/v_pi)))
    return field_ratio


# define electro-optic transfer function
def eo_tf_evaluation_non_ideal(self):
    v_cm = self.v_cm
    v_diff = self.v_diff
    v_pi = self.v_pi
    er = self.er
    v_off = self.v_off
    insertion_loss = self.insertion_loss
    phase_offset = self.phase_offset
    eo_tf = insertion_loss * ((np.sin((np.pi / 2) * ((v_diff - v_off) / v_pi)))**2 + (1 / er) * (np.cos((np.pi / 2) \
            * ((v_diff - v_off) / v_pi)))**2)
    return eo_tf


'''Single Electrode MZM methods'''


def out_mzm_field_evaluation(self):
    v_pi = self.v_pi
    in_field = self.in_field
    v_in = self.v_in
    arg = (np.pi/2)*((v_in - v_off)/v_pi)
    # out_field = in_field*(np.sin(arg) - np.cos(arg)*(1/np.sqrt(er))*1j)
    out_field = in_field*np.sqrt(insertion_loss)*(np.sin(arg) - ((1/np.sqrt(er))*np.cos(arg))*1j)
    self.out_field = out_field
    return out_field


'''Griffin Model methods'''


# define V-dependent intensity tf
def griffin_intensity_tf(self, v_arm):
    c = self.c
    tf = (1+np.exp((v_arm - c)/0.8))**-1.25
    return tf


def griffin_phase(self, v_arm):
    b = self.b
    v_pi = self.v_pi
    v_bias = self.v_bias
    phase = ((2*b*v_bias*v_pi - np.pi)/v_pi)*v_arm - b*v_arm**2
    return phase


def griffin_eo_tf(self):
    gamma_1 = self.gamma_1
    gamma_2 = self.gamma_2
    phase_offset = self.phase_offset
    v_l = self.v_A
    v_r = self.v_B
    transmission_vl = self.griffin_intensity_tf(v_l)
    transmission_vr = self.griffin_intensity_tf(v_r)
    phase_vl = self.griffin_phase(v_l)
    phase_vr = self.griffin_phase(v_r)
    self.transmission_vl = transmission_vl
    self.transmission_vr = transmission_vr
    self.phase_vl = phase_vl
    self.phase_vr = phase_vr
    ext_r = self.er

    eo_tf_field = np.sqrt(1-1/ext_r)*gamma_1*gamma_2*np.sqrt(transmission_vl)*np.exp(phase_vl*1j) + \
                  np.sqrt(1-1/ext_r)*np.sqrt(np.abs(1 - gamma_1**2)*np.abs(1 - gamma_2**2))*np.sqrt(transmission_vr)*\
                  np.exp((phase_vr + phase_offset)*1j)
    eo_field_conj = np.conjugate(eo_tf_field)
    eo_tf_power = eo_tf_field*eo_field_conj
    return eo_tf_power


def griffin_eo_tf_field(self):
    gamma_1 = self.gamma_1
    gamma_2 = self.gamma_2
    phase_offset = self.phase_offset
    v_l = self.v_A
    v_r = self.v_B
    ext_r = self.er
    transmission_vl = self.griffin_intensity_tf(v_l)
    transmission_vr = self.griffin_intensity_tf(v_r)
    phase_vl = self.griffin_phase(v_l)
    phase_vr = self.griffin_phase(v_r)
    self.transmission_vl = transmission_vl
    self.transmission_vr = transmission_vr
    self.phase_vl = phase_vl
    self.phase_vr = phase_vr

    eo_tf_field = np.sqrt(1-1/ext_r) * (gamma_1 * gamma_2 * np.sqrt(transmission_vl) * np.exp(phase_vl * 1j)) + \
                  np.sqrt(1-1/ext_r) * np.sqrt(np.abs(1 - gamma_1 ** 2) * np.abs(1 - gamma_2 ** 2)) * \
                  np.sqrt(transmission_vr) *np.exp((phase_vr + phase_offset) * 1j)
    return eo_tf_field


def griffin_il_er(self):
    v_l = self.v_A
    v_r = self.v_B
    t_vl = self.griffin_intensity_tf(v_l)
    t_vr = self.griffin_intensity_tf(v_r)
    il = np.sqrt(2)*t_vl
    er = (t_vr + t_vl + 2*np.sqrt(t_vl*t_vr))/(t_vr + t_vl - 2*np.sqrt(t_vl*t_vr))
    er_lin = lin2db(er)
    il_on_er = il/er
    return il, er, il_on_er


'''Modulation format methods'''


# Function to create random binary string
def rand_key(p):
    # variable to store the string
    key1 = ''
    # loop to find the string of the desired length:
    for i in range(p):
        # randint function to generate 0,1 randomly and converting result into str
        temp = str(random.randint(0, 1))
        # concatenate random 0,1 to the final result
        key1 += temp

    return key1


'''EVM AND BER ESTIMATION'''


# inputs are the two lists of constellation points
def evm_rms_estimation(prbs_in, indexes, ideal_const, actual_const):
    prbs_no_duplicates = prbs_in['full'].tolist()

    evm_partial = []
    for i in prbs_no_duplicates:
        Pi = ideal_const[indexes[i]]    # ideal constellation points
        Ps = actual_const[indexes[i]]   # actual constellation points
        T = len(actual_const[indexes[i]])   # number of transmitted symbols

        # define normalization factors
        sum_p_measure = np.abs(np.sum(Ps.real**2 + Ps.imag**2))
        norm_measure = np.abs(np.sqrt(T/sum_p_measure))
        sum_p_ideal = np.abs(np.sum(Pi.real**2 + Pi.imag**2))
        norm_ideal = np.abs(np.sqrt(T/sum_p_ideal))
        norm_i_meas = Ps.real * norm_measure
        norm_i_ideal = Pi.real * norm_ideal
        norm_q_meas = Ps.imag * norm_measure
        norm_q_ideal = Pi.imag * norm_ideal
        num = np.sqrt(np.average((np.abs(norm_i_meas - norm_i_ideal)**2 + np.abs(norm_q_meas - norm_q_ideal)**2)))
        den = np.sqrt(np.average((np.abs(norm_i_ideal)**2 + np.abs(norm_q_ideal)**2)))
        # num = np.sqrt(np.mean((abs(Ps.real - Pi.real) ** 2 + abs(Ps.imag - Pi.imag) ** 2)))
        # den = np.mean(abs(Pi) ** 2)*evm_factor
        evm = (num/den)
        # numerator = (np.average((np.abs(actual_const[indexes[i]] - ideal_const[indexes[i]]))))
        # # FOR DENOMINATOR: FORMULA FROM PAPER, AVERAGE OF ALL SYMBOLS IN CONSTELLATION MULTIPLIED BY NORM FACTOR
        # denominator = (np.average((np.abs(ideal_const[indexes[i]]))))
        evm_partial.append(evm)
    evm_fin = np.mean((np.array(evm_partial)))
    return evm_fin


# SNR estimation based on EVM (in input)
def snr_estimation(evm):
    snr = 1/evm**2
    return snr


# BER estimation based on EVM (in input)
def ber_estimation(evm):
    # ber = (1 - (1/np.sqrt(M)))/(0.5*np.log2(M))*\
    #     scipy.special.erfc(np.sqrt(1.5/((M-1)*evm**2)))
    ber = (2*(1 - 1/L)/np.log2(L))*\
          math.erfc(np.sqrt((((3*np.log2(L))/(L**(2)- 1)))*2/(evm**2 * np.log2(M))))
    return ber