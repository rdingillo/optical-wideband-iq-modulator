"elements.py"


import math
import numpy as np

from mzm_model.core.emulation_settings import h, frequency, q, eta_s, input_current, v_pi


class IdealLightSource(object):
    def __init__(self, current):
        self._current = current
        self._input_power = 0
        self._out_field = 0

    @property
    def current(self):
        return self._current

    @property
    def input_power(self):
        return self._input_power

    @input_power.setter
    def input_power(self, input_power):
        self._input_power = input_power

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field
        
    from mzm_model.core.science_utils import calculate_alfa_s, calculate_optical_input_power


class Splitter(object):
    def __init__(self, input_power, input_field):
        self._input_power = input_power
        self._input_field = input_field
        self._A_in_field = 0
        self._B_in_field = 0

    @property
    def input_power(self):
        return self._input_power

    @property
    def input_field(self):
        return self._input_field

    @property
    def A_in_field(self):
        return self._A_in_field

    @A_in_field.setter
    def A_in_field(self, A_in_field):
        self._A_in_field = A_in_field

    @property
    def B_in_field(self):
        return self._B_in_field

    @B_in_field.setter
    def B_in_field(self, B_in_field):
        self._B_in_field = B_in_field

    from mzm_model.core.science_utils import calculate_arms_input_fields


class Waveguide(object):
    def __init__(self, in_field):
        self._in_field = in_field
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    from mzm_model.core.science_utils import delta_phi_evaluation, delta_phi_evaluation_k, out_field_evaluation


class Combiner(object):
    def __init__(self, in_A, in_B):
        self._in_A = in_A
        self._in_B = in_B
        self._out_field = 0

    @property
    def in_A(self):
        return self._in_A

    @property
    def in_B(self):
        return self._in_B

    @property
    def out_field(self):
        return self._out_field
    
    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    from mzm_model.core.science_utils import combiner_out_field


class IdealMZM(object):
    def __init__(self, v_cm, v_diff, v_pi):
        self._v_cm = v_cm
        self._v_diff = v_diff
        self._v_pi = v_pi

    @property
    def v_cm(self):
        return self._v_cm

    @property
    def v_diff(self):
        return self._v_diff

    @property
    def v_pi(self):
        return self._v_pi

    from mzm_model.core.science_utils import field_ratio_evaluation, eo_tf_evaluation
    from mzm_model.core.utils import eo_tf_draw


class MZMSingleElectrode(object):
    def __init__(self, in_field, v_in, v_pi):
        self._in_field = in_field
        self._v_in = v_in
        self._v_pi = v_pi
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def v_in(self):
        return self._v_in

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field
    from mzm_model.core.science_utils import out_mzm_field_evaluation
    from mzm_model.core.utils import eo_tf_draw


class NonIdealMZM(object):
    def __init__(self, v_cm, v_diff, v_pi, er, insertion_loss, phase_offset, v_off):
        self._v_cm = v_cm
        self._v_diff = v_diff
        self._v_pi = v_pi
        self._er = er
        self._insertion_loss = insertion_loss
        self._phase_offset = phase_offset
        self._v_off = v_off

    @property
    def v_cm(self):
        return self._v_cm

    @property
    def v_diff(self):
        return self._v_diff

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def er(self):
        return self._er

    @property
    def insertion_loss(self):
        return self._insertion_loss

    @property
    def phase_offset(self):
        return self._phase_offset

    @property
    def v_off(self):
        return self._v_off

    from mzm_model.core.science_utils import field_ratio_evaluation_non_ideal, eo_tf_evaluation_non_ideal
    from mzm_model.core.utils import eo_tf_draw


class GriffinMZM(object):
    def __init__(self, v_bias, v_A, v_B, gamma_1, gamma_2, phase_offset, b, c, er):
        # bias voltage v_cm
        self._v_bias = v_bias
        # driving voltages
        self._v_A = v_A
        self._v_B = v_B
        # define v_pi as it depends on bias voltage v_cm
        self._v_pi = -np.pi/v_bias
        self._phase_offset = phase_offset
        self._gamma_1 = gamma_1
        self._gamma_2 = gamma_2
        self._b = b
        self._c = c
        self._phase_vl = 0
        self._phase_vr = 0
        self._transmission_vl = 0
        self._transmission_vr = 0
        self._er = er

    @property
    def v_bias(self):
        return self._v_bias

    @property
    def v_A(self):
        return self._v_A

    @property
    def v_B(self):
        return self._v_B

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def phase_offset(self):
        return self._phase_offset

    @property
    def gamma_1(self):
        return self._gamma_1

    @property
    def gamma_2(self):
        return self._gamma_2

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def phase_vl(self):
        return self._phase_vl

    @property
    def er(self):
        return self._er

    @phase_vl.setter
    def phase_vl(self, phase_vl):
        self._phase_vl = phase_vl

    @property
    def phase_vr(self):
        return self._phase_vr

    @phase_vr.setter
    def phase_vr(self, phase_vr):
        self._phase_vr = phase_vr

    @property
    def transmission_vl(self):
        return self._transmission_vl

    @transmission_vl.setter
    def transmission_vl(self, transmission_vl):
        self._transmission_vl = transmission_vl

    @property
    def transmission_vr(self):
        return self._transmission_vr

    @transmission_vr.setter
    def transmission_vr(self, transmission_vr):
        self._transmission_vr = transmission_vr

    from mzm_model.core.science_utils import griffin_intensity_tf, griffin_phase, griffin_eo_tf, griffin_il_er, griffin_eo_tf_field
    from mzm_model.core.utils import eo_tf_draw, phase_transmission_draw, phase_parametric_draw, transmission_parametric_draw
