import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import copy
import seaborn as sns; sns.set_theme()
from colorama import Fore
from pathlib import Path
from scipy import signal
from mzm_model.core.elements import IdealLightSource, Splitter, Waveguide, Combiner, IdealMZM, NonIdealMZM, GriffinMZM
from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, v_A, v_B, v_pi, v_cm, \
    v_diff, v_in_limit, v_in_step, er, insertion_loss, phase_offset, v_off, v_bias, b, c, wavelength, gamma_1, gamma_2
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin

start_time = time.time()
matplotlib.use('Qt5Agg')

root = Path(__file__).parent.parent
input_folder = root/'resources'
folder_results = root/'mzm_model'/'results'

# retrieve optical input source
source = IdealLightSource(input_current)

# evaluate optical input power
input_power = source.calculate_optical_input_power(source.current)
source.input_power = input_power
input_power_dbm = lin2dbm(input_power)

# evaluate electric fields in splitter
splitter = Splitter(source.input_power, source.out_field)
splitter_fields = splitter.calculate_arms_input_fields()
A_field = splitter_fields[0]   # [mV/m]
B_field = splitter_fields[1]   # [mV/m]
splitter.A_in_field = A_field
splitter.B_in_field = B_field

# evaluate electric fields in waveguides
arm_a = Waveguide(A_field)
arm_b = Waveguide(B_field)

out_arm_A = arm_a.out_field_evaluation(arm_a.in_field, v_A, v_pi)
out_arm_B = arm_b.out_field_evaluation(arm_b.in_field, v_B, v_pi)

arm_a.out_field = out_arm_A
arm_b.out_field = out_arm_B

# combiner output field

combiner = Combiner(arm_a.out_field, arm_b.out_field)
out_field_combiner = combiner.combiner_out_field(combiner.in_A, combiner.in_B)
combiner.out_field = out_field_combiner
# to check that combiner field is the same wrt eo_tf, try to take the abs**2 of combiner_out_field/sqrt()input_power)
power_tf_check = (np.abs(combiner.out_field)/np.sqrt(input_power))**2
# evaluate electro_optic transfer function of ideal MZM (single-point)
ideal_mzm = IdealMZM(v_cm, v_diff, v_pi)
eo_tf = ideal_mzm.eo_tf_evaluation()

# evaluate electro_optic tf on a range of input voltages
v_in_range = np.arange(v_B, v_A, v_in_step)
# create a list of ideal MZMs using this input voltage interval
# for this evaluation, consider common mode voltage at 0
ideal_mzm_list = [IdealMZM(v_cm, v_in, v_pi) for v_in in v_in_range]
eo_tf_list = np.array([mzm.eo_tf_evaluation() for mzm in ideal_mzm_list])

# create a list of non-ideal MZMs using this input voltage interval
# take into account the non-ideal params as ER, IL and Voff
# for this evaluation, consider common mode voltage at 0
non_ideal_mzm = NonIdealMZM(v_cm, v_diff, v_pi, er, insertion_loss, phase_offset, v_off)
non_ideal_mzm_list = [NonIdealMZM(v_cm, v_in, v_pi, er, insertion_loss, phase_offset, v_off) for v_in in v_in_range]
non_ideal_eo_tf_list = np.array([mzm.eo_tf_evaluation_non_ideal() for mzm in non_ideal_mzm_list])

# create a list of Griffin MZMs using this input voltage interval
# take into account the non-ideal params as b, c
# for this evaluation, consider common mode voltage at 0
griffin_mzm = GriffinMZM(v_bias, v_A, v_B, gamma_1, gamma_2, phase_offset, b, c, er)
# evaluate Griffin V_bias to have the same V_pi of the classic MZM
'Set a constant v_bias to get the same v_pi of the classic mzm, otherwise leave the variable one'
v_bias = np.pi / v_pi
# griffin_mzm_list = [GriffinMZM(v_bias, v_in/2 + v_pi/2, (2*v_bias - v_in/2 - v_pi/2), gamma_1, gamma_2, phase_offset, b, c) for v_in in v_in_range]
griffin_mzm_list = [GriffinMZM(v_bias, v_in/2, 0, gamma_1, gamma_2, phase_offset, b, c, er) for v_in in v_in_range]
griffin_eo_tf_list = np.array([mzm.griffin_eo_tf() for mzm in griffin_mzm_list])
griffin_phase_vl_list = np.array([mzm.phase_vl for mzm in griffin_mzm_list])/np.pi
griffin_phase_vr_list = np.array([mzm.phase_vr for mzm in griffin_mzm_list])/np.pi
griffin_transmission_vl_list = np.array([mzm.transmission_vl for mzm in griffin_mzm_list])
griffin_transmission_vr_list = np.array([mzm.transmission_vr for mzm in griffin_mzm_list])
griffin_il_list = np.array([mzm.griffin_il_er()[0] for mzm in griffin_mzm_list])
griffin_er_list = np.array([mzm.griffin_il_er()[1] for mzm in griffin_mzm_list])
griffin_il_on_er_list = np.array([mzm.griffin_il_er()[2] for mzm in griffin_mzm_list])

b_list = np.arange(0.00, 0.21, 0.05)
c_list = [20, 7.3, 6.3, 5.3, 4.3]


griffin_b_list = [[GriffinMZM(v_bias, v_in, 2*v_bias - v_in, gamma_1, gamma_2, phase_offset, b_par, 20, er) for v_in in v_in_range] for b_par in b_list]
# griffin_phase_vin = np.array([[[mzm.griffin_phase(v_in) for mzm in griffin_b_list[i]] for v_in in v_in_range]for i in range(len(griffin_b_list))])
griffin_c_list = [[GriffinMZM(v_bias, v_in, 0, gamma_1, gamma_2, phase_offset, 0.00, c_par, er) for v_in in v_in_range] for c_par in c_list]

griffin_eo_tf_bvar = np.array([[mzm.griffin_eo_tf() for mzm in griffin_b_list[i]] for i in range(len(griffin_b_list))])
griffin_eo_tf_cvar = np.array([[mzm.griffin_eo_tf() for mzm in griffin_c_list[i]] for i in range(len(griffin_c_list))])



griffin_phase_vl_bvar = np.array([[mzm.phase_vl for mzm in griffin_b_list[i]] for i in range(len(griffin_b_list))])/np.pi
griffin_transmission_vl_cvar = np.array([[mzm.transmission_vl for mzm in griffin_c_list[i]] for i in range(len(griffin_c_list))])

v_pi_gr = griffin_mzm_list[0].v_pi
# plot EO TFs
non_ideal_mzm.eo_tf_draw(v_in_range, eo_tf_list, non_ideal_eo_tf_list, griffin_eo_tf_list, griffin_il_list, griffin_er_list, griffin_il_on_er_list, v_pi_gr)
griffin_mzm.phase_transmission_draw(v_in_range, griffin_phase_vl_list, griffin_transmission_vl_list)
griffin_mzm.phase_parametric_draw(v_in_range, b_list, griffin_phase_vl_bvar)
griffin_mzm.transmission_parametric_draw(v_in_range, c_list, griffin_transmission_vl_cvar)

plt.show(block=True)


print()
