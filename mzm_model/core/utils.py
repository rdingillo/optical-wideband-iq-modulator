"utils.py"

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()

from scipy.stats import gaussian_kde
from mzm_model.core.emulation_settings import v_in_limit, v_pi, er, insertion_loss, phase_offset, v_off, gamma_1, gamma_2, b, c, v_cm, modulation_format
from mzm_model.core.math_utils import lin2dbm, lin2db, db2lin


# EO-TF draw method
def eo_tf_draw(self, v_in_range, eo_tf_list, non_ideal_eo_tf_list, griffin_eo_tf_list, il_list, er_list, il_on_er_list, v_pi_gr):
    fig, axs = plt.subplots(3, 1, figsize=(9, 10))
    fig.suptitle('EO Transfer Function of MZM')
    # plot definition of Ideal MZM
    axs[0].set_title('Ideal MZM')
    axs[0].set(xlabel='V_in', ylabel='P_out/P_in')
    axs[0].plot(v_in_range, eo_tf_list.real, label='Re')
    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    # plt.legend(fontsize=10)
    textstr = r'V_pi: ' + ' $%.2f$ V' % v_pi

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[0].text(1.03, 0.98, textstr, transform=axs[0].transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

    # plot definition of non-ideal MZM
    # define IL/ER to plot it
    plt.ylim(-0.4, 1.1 * insertion_loss)
    er_plot = np.ones(len(non_ideal_eo_tf_list)) * (insertion_loss / er)
    v_on = v_pi + v_off
    axs[1].set_title('Non-Ideal MZM')
    axs[1].set_ylim(-0.1, 1.1*insertion_loss)
    axs[1].set(xlabel='V_in', ylabel='P_out/P_in')
    axs[1].plot(v_in_range, non_ideal_eo_tf_list.real, label='TF')
    axs[1].plot(v_in_range, er_plot, label='1/ER')
    # axs[1].vlines(v_on, -0.1, 1.1*insertion_loss, label='V_on', colors='green')
    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    axs[1].legend(fontsize=10)
    textstr = '\n'.join((
        r'V_pi: ' + ' $%.2f$ V' % (v_pi,),
        r'ER: ' + ' $%.2f$ dB' % (lin2db(er),),
        r'IL: ' + ' $%.2f$' % (insertion_loss,),
        r'IL/ER: ' + ' $%.4f$' % (insertion_loss/er,),
        r'Phase Offset: ' + ' $%.2f$' % (phase_offset,),
        r'V_off: ' + ' $%.2f$ V' % (v_off,),
        r'V_on: ' + ' $%.2f$ V' % (v_on,)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[1].text(1.03, 0.98, textstr, transform=axs[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)


    # plot definition of Griffin MZM
    # define IL/ER to plot it
    plt.ylim(-0.1, 1.1)
    er_plot = np.ones(len(griffin_eo_tf_list)) * (1 - 1 / er)
    axs[2].set_title('InP MZM')
    axs[2].set(xlabel='V_L - V_R', ylabel='P_out/P_in')
    axs[2].plot(v_in_range, griffin_eo_tf_list, label='TF')
    axs[2].plot(v_in_range, er_plot, label='1/ER')
    # if v_cm < v_pi:
    #     v_off_g = 2*v_cm
    # elif v_cm == v_pi:
    #     v_off_g = 0
    # else:
    #     count = np.floor(v_cm/v_pi)
    #     v_off_g = 2*v_cm - 2*count*v_pi
    # if phase_offset == 0 and b == 0 and c == 20:
    #     axs[2].vlines(v_off_g, -0.1, 1.1, label='V_off', colors='red')
    #     v_on_g = v_pi + v_off_g
    #     axs[2].vlines(v_on_g, -0.1, 1.1, label='V_on', colors='green')

    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    axs[2].legend(fontsize=10)
    if phase_offset == 0 and b == 0 and  c == 20:
        textstr = '\n'.join((
            r'V_pi: ' + ' $%.2f$ V' % (v_pi_gr,),
            r'b: ' + ' $%.2f$ ' % (b,),
            r'c: ' + ' $%.2f$' % (c,),
            r'ER: ' + ' $%.2f$ dB' % (lin2db(er),),
            r'Phase Offset: ' + ' $%.2f$' % (phase_offset,)))
            # r'V_off: ' + ' $%.2f$ V' % (v_off_g,),
            # r'V_on: ' + ' $%.2f$ V' % (v_on_g,)))
    else:
        textstr = '\n'.join((
            r'V_pi: ' + ' $%.2f$ V' % (v_pi_gr,),
            r'b: ' + ' $%.2f$ ' % (b,),
            r'c: ' + ' $%.2f$' % (c,),
            r'ER: ' + ' $%.2f$ dB' % (lin2db(er),),
            r'Phase Offset: ' + ' $%.2f$' % (phase_offset,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[2].text(1.03, 0.98, textstr, transform=axs[2].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.interactive(False)


# phase and intensity draw
def phase_transmission_draw(self, v_in_range, phase_list, transmission_list):
    fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    fig.suptitle('Phase and Transmission Transfer Function of InP MZM')
    # plot definition of Phase
    axs[0].set_title('Phase')
    axs[0].set(xlabel='V_in', ylabel='Phase (rad)')
    axs[0].plot(v_in_range, phase_list, label='Phase')
    # axs[0].set_ylim(-np.pi - 0.1, np.pi + 0.1)
    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    # plt.legend(fontsize=10)
    textstr = r'V_pi: ' + ' $%.2f$ V' % v_pi

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[0].text(1.03, 0.98, textstr, transform=axs[0].transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

    # plot definition of Transmission
    axs[1].set_title('Transmission')
    axs[1].set(xlabel='V_in', ylabel='Transmission')
    axs[1].plot(v_in_range, transmission_list, label='Transmission')
    axs[1].set_ylim(0 - 0.1, 1 + 0.1)

    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    # plt.legend(fontsize=10)
    textstr = r'V_pi: ' + ' $%.2f$ V' % v_pi

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[1].text(1.03, 0.98, textstr, transform=axs[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.interactive(False)


# phase and intensity draw
def phase_parametric_draw(self, v_in_range, b_list, param_phase_list):
    plt.figure(figsize=(7, 7))
    plt.title('Phase Transfer Function of InP MZM varying b')
    # plot definition of Phase
    plt.xlabel('V_in')
    plt.ylabel('Phase (rad)')
    counter = 0
    for phase_list in param_phase_list:
        plt.plot(v_in_range, phase_list, label='b = '+str(round(b_list[counter], 2)))
        plt.legend(fontsize=10)
        counter += 1

    plt.tight_layout()
    plt.interactive(False)


# phase and intensity draw
def transmission_parametric_draw(self, v_in_range, c_list, param_transmission_list):
    plt.figure(figsize=(7, 7))
    plt.title('Transmission Transfer Function of InP MZM varying c')
    # plot definition of Phase
    plt.xlabel('V_in')
    plt.ylabel('Transmission')
    counter = 0
    for transmission_list in param_transmission_list:
        plt.plot(v_in_range, transmission_list, label='c = '+str(c_list[counter]))
        plt.legend(fontsize=10)
        counter += 1

    plt.tight_layout()
    plt.interactive(False)


# plot constellations
def plot_constellation(field_list, type):
    plt.figure(figsize=(7, 7))
    plt.title('Modulation space for '+modulation_format+' ('+type+')')
    x = [ele.real for ele in field_list]
    y = [ele.imag for ele in field_list]

    plt.scatter(x, y)
    plt.ylabel('Im{E_out}')
    plt.xlabel('Re{E_out}')
    plt.tight_layout()
    plt.interactive(False)


# EO-TF draw method
def eo_tf_iq_draw(v_in_range, eo_tf_list, griffin_eo_tf_list):
    fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    fig.suptitle('EO Transfer Function of MZM')

    # plot definition of non-ideal MZM
    # define IL/ER to plot it
    er_plot = np.ones(len(eo_tf_list)) * (insertion_loss / er)
    v_on = v_pi + v_off
    axs[0].set_title('Classic LiNbO3 MZM')
    axs[0].set(xlabel='V_in', ylabel='P_out/P_in')
    axs[0].plot(v_in_range, eo_tf_list.real, label='TF')
    axs[0].plot(v_in_range, er_plot, label='IL/ER')
    axs[0].vlines(v_on, -0.1, 1.1*insertion_loss, label='V_on', colors='green')
    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    axs[1].legend(fontsize=10)
    textstr = '\n'.join((
        r'V_pi: ' + ' $%.2f$ V' % (v_pi,),
        r'ER: ' + ' $%.2f$ dB' % (lin2db(er),),
        r'IL: ' + ' $%.2f$' % (insertion_loss,),
        r'IL/ER: ' + ' $%.4f$' % (insertion_loss/er,),
        r'Phase Offset: ' + ' $%.2f$' % (phase_offset,),
        r'V_off: ' + ' $%.2f$ V' % (v_off,),
        r'V_on: ' + ' $%.2f$ V' % (v_on,)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[1].text(1.03, 0.98, textstr, transform=axs[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    plt.ylim(-0.1, 1.1*insertion_loss)

    # plot definition of Griffin MZM
    # define IL/ER to plot it
    axs[1].set_title('InP MZM')
    axs[1].set(xlabel='V_L - V_R', ylabel='P_out/P_in')
    axs[1].plot(v_in_range, griffin_eo_tf_list, label='TF')
    if v_cm < v_pi:
        v_off_g = 2*v_cm
    elif v_cm == v_pi:
        v_off_g = 0
    else:
        count = np.floor(v_cm/v_pi)
        v_off_g = 2*v_cm - 2*count*v_pi
    if phase_offset == 0 and b == 0 and c == 20:
        axs[2].vlines(v_off_g, -0.1, 1.1, label='V_off', colors='red')
        v_on_g = v_pi + v_off_g
        axs[2].vlines(v_on_g, -0.1, 1.1, label='V_on', colors='green')

    # plt.plot(v_in_range, eo_tf_list.imag, label='Im')
    axs[1].legend(fontsize=10)
    if phase_offset == 0 and b == 0 and  c == 20:
        textstr = '\n'.join((
            r'V_pi: ' + ' $%.2f$ V' % (v_pi,),
            r'b: ' + ' $%.2f$ ' % (b,),
            r'c: ' + ' $%.2f$' % (c,),
            r'Phase Offset: ' + ' $%.2f$' % (phase_offset,),
            r'V_off: ' + ' $%.2f$ V' % (v_off_g,),
            r'V_on: ' + ' $%.2f$ V' % (v_on_g,)))
    else:
        textstr = '\n'.join((
            r'V_pi: ' + ' $%.2f$ V' % (v_pi,),
            r'b: ' + ' $%.2f$ ' % (b,),
            r'c: ' + ' $%.2f$' % (c,),
            r'Phase Offset: ' + ' $%.2f$' % (phase_offset,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[1].text(1.03, 0.98, textstr, transform=axs[2].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    plt.ylim(-0.1, 1.1*(np.sqrt(er)+1)/(np.sqrt(er)-1))
    plt.tight_layout()
    plt.interactive(False)
    plt.show(block=True)


# smooth the figure by using Savitzky-Golay filter. It uses least squares to regress a small window of data
# onto a polynomial, then uses the polynomial to estimate the point in the center of the window.
# Finally the window is shifted forward by one data point and the process repeats.
# This continues until every point has been optimally adjusted relative to its neighbors.
# It works great even with noisy samples from non-periodic and non-linear sources.

# use window size 51, polynomial order 3

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
