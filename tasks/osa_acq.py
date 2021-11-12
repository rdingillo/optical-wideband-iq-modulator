import numpy as np
from numpy import squeeze, sum, argmax, append, sign, array, mean, max, inf
from scipy.io import loadmat
from pandas import DataFrame
from os import listdir, makedirs
from pathlib import Path
from os.path import exists
from matplotlib.pyplot import plot, xlabel, ylabel, legend,\
    figure, show, tight_layout, xticks, yticks, xlim, ylim, savefig, errorbar, fill_between
from seaborn import set as set_sns
from seaborn import distplot
from mzm_model.core.math_utils import lin2dBm, dBm2lin

set_sns(palette="deep", font_scale=1.1, color_codes=True, rc={"figure.figsize": [8, 5]})


def gradient(array):
    return array[1:] - array[:-1]


def get_or_create_folder(path):
    if not exists(path):
        makedirs(path)
    return path

root = Path(__file__).parent.parent
RESOURCES_PATH = root/'resources'
RESULTS_PATH = root/'mzm_model'/'results'
#roadm_bst_dir = RESOURCES_PATH/'EDFA_ROADM_BST'
#roadm_pre_dir = RESOURCES_PATH/'EDFA_ROADM_PRE'


def easy_pp(frequency, power, tot_power, integration_window=40e9, aspect_ratio=40, step=6, plot_flag=False):

    if plot_flag:
        plot(frequency*1e-12, power)
    # Cut most of the noise
    noise_threshold = max(power) - aspect_ratio
    where_not_noise = (power > noise_threshold)
    power = power[where_not_noise]
    frequency = frequency[where_not_noise]

    # Find stationary points
    ds = sign(gradient(power))
    dds = ds[1:] * ds[:-1]
    dds = append(0, append(dds, 0))
    stationary = (dds < 0) + (dds == 0)
    # Select stationary points within the spectrum band case id-th
    indices = array(range(frequency.size))
    left_boundary = indices[stationary][append(gradient(power[stationary]), 0) > step][0]
    right_boundary = indices[stationary][append(0, gradient(power[stationary])) < -step][-1]
    # plot(frequency*1e-12, power)
    # plot(frequency[left_boundary]*1e-12, power[left_boundary], '>')
    # plot(frequency[right_boundary]*1e-12, power[right_boundary], '<')
    # show()

    inner_stationary = append(left_boundary,
                              append(indices[stationary * (frequency[left_boundary] < frequency) * (
                                      frequency < frequency[right_boundary])],
                                     right_boundary))
    if plot_flag:
        plot(frequency*1e-12, power)
        plot(frequency[inner_stationary]*1e-12, power[inner_stationary], 'o')

    # Find channel ascent/descent
    left_boundary = inner_stationary[
        append(power[inner_stationary[1:]] - power[inner_stationary[:-1]] > step / 2, False)]
    right_boundary = \
        inner_stationary[
            append(False, power[inner_stationary[1:]] - power[inner_stationary[:-1]] < -step / 2)]
    boundary = zip(left_boundary, right_boundary)

    if plot_flag:
        plot(frequency[left_boundary]*1e-12, power[left_boundary], '>b')
        plot(frequency[right_boundary]*1e-12, power[right_boundary], '<r')

    eval_power = array([])
    eval_freq = array([])
    for b in boundary:
        sub_index = indices[(b[0] <= indices) * (indices <= b[-1])]
        sub_power = power[sub_index]
        sub_freq = frequency[sub_index]
        where = argmax(sub_power)
        eval_power = append(eval_power, sub_power[where])
        eval_freq = append(eval_freq, sub_freq[where])

    boundary = zip(left_boundary, right_boundary)
    eval_ase = array([])
    for b in boundary:
        eval_ase = append(eval_ase, (power[b[0]] + power[b[1]])/2)

    if plot_flag:
        plot(eval_freq*1e-12, eval_power, 'go')
        plot(eval_freq*1e-12, eval_ase, 'bo')
        show()

    shift = tot_power - lin2dBm(sum(dBm2lin(eval_power)))
    eval_power += shift
    eval_ase += shift

    eval_ase_lin = dBm2lin(eval_ase)/integration_window  # W/Hz

    return eval_freq, eval_power, eval_ase_lin


model = 'EDFA_ROADM_BST'

edfa_char_folder = RESOURCES_PATH/'characterizations'/model
results_folder = get_or_create_folder(RESULTS_PATH/model)

files = listdir(edfa_char_folder)

Bn = 32e9
dataset = DataFrame()
case_id = 1
tot = 1
skipped = 0
for file in files:
    content = loadmat(edfa_char_folder/file)
    gain_target = squeeze(content['Gain_target'])
    tilt_target = squeeze(content['Tilt_target'])
    osa_freq = squeeze(content['spectrum_freq'])
    osa_p_rx = squeeze(content['spectrum_RX_power'])
    osa_p_tx = squeeze(content['spectrum_TX_power'])
    p_in = squeeze(content['TOT_Power_IN'])
    p_out = squeeze(content['TOT_Power_OUT'])

    f_max = 196
    f_min = 191.5
    mask = (f_min < osa_freq) * (osa_freq < f_max) == 1
    osa_freq = osa_freq[mask]
    osa_p_tx = osa_p_tx[mask]
    osa_p_rx = osa_p_rx[:, mask]

    df = DataFrame()
    eval_freq_in, eval_power_in, eval_ase_in_lin = easy_pp(osa_freq, osa_p_tx, p_in)
    eval_ase_in = lin2dBm(eval_ase_in_lin * Bn)
    f_label = [f'freq_{i}' for i in range(1, eval_freq_in.size + 1)]

    for i, g_tar in enumerate(gain_target):
        tot += 1

        # Read OSA
        eval_freq_out, eval_out_power, eval_ase_out_lin = easy_pp(osa_freq, osa_p_rx[i, :], p_out[i])
        eval_ase_out = lin2dBm(eval_ase_out_lin * Bn)

        # Gain
        gain = p_out[i] - p_in
        if len(eval_out_power) != len(eval_power_in):
            # eval_power_in = np.hstack((eval_power_in, eval_power_in[-1]))
            eval_out_power = np.resize(eval_out_power, eval_power_in.size)
        gain_profile = eval_out_power - eval_power_in
        gain_ripple = gain_profile - gain

        # ASE
        if len(eval_ase_out) != len(gain_profile):
            eval_ase_out = np.resize(eval_ase_out, gain_profile.size)
        ase_profile = dBm2lin(eval_ase_out - gain_profile)  # - dBm2lin(eval_ase_in)
        ase_profile = ase_profile * (ase_profile > 0)
        ase = lin2dBm(mean(ase_profile))
        ase_profile = lin2dBm(ase_profile)
        ase_ripple = ase_profile - ase

        if (-inf in ase_profile) + (inf in ase_profile):
            skipped += 1
            continue

        ddf = DataFrame()
        ddf['ID'] = [case_id]
        case_id += 1

        # Scalar field
        ddf['gain_target'] = [g_tar]
        ddf['gain_real'] = [gain]
        ddf['pin_real'] = [p_in]
        ddf['pout_real'] = [p_out[i]]
        ddf['pout_target'] = [p_in + g_tar]
        ddf['ase_real'] = ase

        # Vector field
        gain_profile_label = [f'gain_profile_{i}' for i in range(1, eval_freq_in.size + 1)]
        gain_ripple_label = [f'gain_ripple_{i}' for i in range(1, eval_freq_in.size + 1)]
        ase_profile_label = [f'ase_profile{i}' for i in range(1, eval_freq_in.size + 1)]
        ase_ripple_label = [f'ase_ripple{i}' for i in range(1, eval_freq_in.size + 1)]

        ddf[f_label] = eval_freq_in
        ddf[gain_profile_label] = gain_profile
        ddf[ase_profile_label] = ase_profile
        ddf[gain_ripple_label] = gain_ripple
        ddf[ase_ripple_label] = ase_ripple

        df = df.append(ddf)
    show()

dataset = dataset.append(df)

# Gain fixed with saturation
pin_real = dataset.pin_real.values
gain_real = dataset.gain_real.values
gain_target = dataset.gain_target.values
pout_real = dataset.pout_real.values
pout_max = max(pout_real)

gain_pred = [gp[0] if (gp[0] + gp[1] <= pout_max) else pout_max - gp[1] for gp in zip(gain_target, pin_real)]

figure()
plot(gain_real, 'go')
plot(gain_pred, 'r.')

figure()
distplot(gain_real - gain_pred)
show()

dataset['gain_pred'] = gain_pred

dataset.to_csv(results_folder/'characterizations.csv')

print(f'Done. Total: {tot}. Processed: {case_id}. Skipped: {skipped}')


# import scipy.io
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import control
# import collections
#
# import copy
# import seaborn as sns; sns.set_theme()
# from colorama import Fore
# from pathlib import Path
#
#
# matplotlib.use('Qt5Agg')
# root = Path(__file__).parent.parent
# input_folder = root/'resources'
# folder_results = root/'mzm_model'/'results'
# roadm_bst_dir = input_folder/'EDFA_ROADM_BST'
# roadm_pre_dir = input_folder/'EDFA_ROADM_PRE'
#
# bst_00 = scipy.io.loadmat(roadm_bst_dir/'EDFA_BST_ROADM_1_C-band_EDFA_Pin_0.0.mat')
# bst_25 = scipy.io.loadmat(roadm_bst_dir/'EDFA_BST_ROADM_1_C-band_EDFA_Pin_-2.5.mat')
# bst_50 = scipy.io.loadmat(roadm_bst_dir/'EDFA_BST_ROADM_1_C-band_EDFA_Pin_-5.0.mat')
# bst_75 = scipy.io.loadmat(roadm_bst_dir/'EDFA_BST_ROADM_1_C-band_EDFA_Pin_-7.5.mat')
# bst_100 = scipy.io.loadmat(roadm_bst_dir/'EDFA_BST_ROADM_1_C-band_EDFA_Pin_-10.0.mat')
#
# pre_00 = scipy.io.loadmat(roadm_pre_dir/'EDFA_PRE_ROADM_1_C-band_EDFA_Pin_0.0.mat')
# pre_25 = scipy.io.loadmat(roadm_pre_dir/'EDFA_PRE_ROADM_1_C-band_EDFA_Pin_-2.5.mat')
# pre_50 = scipy.io.loadmat(roadm_pre_dir/'EDFA_PRE_ROADM_1_C-band_EDFA_Pin_-5.0.mat')
# pre_75 = scipy.io.loadmat(roadm_pre_dir/'EDFA_PRE_ROADM_1_C-band_EDFA_Pin_-7.5.mat')
# pre_100 = scipy.io.loadmat(roadm_pre_dir/'EDFA_PRE_ROADM_1_C-band_EDFA_Pin_-10.0.mat')

