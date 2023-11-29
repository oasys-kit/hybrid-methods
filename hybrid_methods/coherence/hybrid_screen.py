#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2023. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
import numpy
import copy
from abc import abstractmethod

from wofry.propagator.propagator import PropagationManager, PropagationParameters, PropagationElements
from wofry.propagator.wavefront import Wavefront
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofryimpl.propagator.propagators1D.fresnel import Fresnel1D
from wofryimpl.propagator.propagators2D.fresnel import Fresnel2D
from wofryimpl.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofryimpl.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D as Screen1D, WOScreen as Screen2D
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates

wofry_propagation_manager = PropagationManager.Instance()
wofry_propagation_manager.add_propagator(Fraunhofer1D())
wofry_propagation_manager.add_propagator(Fresnel1D())
wofry_propagation_manager.add_propagator(Fraunhofer2D())
wofry_propagation_manager.add_propagator(Fresnel2D())
wofry_propagation_manager.set_initialized()

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix
from srxraylib.util.inverse_method_sampler import Sampler2D, Sampler1D

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------

class HybridDiffractionPlane:
    SAGITTAL   = 1
    TANGENTIAL = 2
    BOTH_2D    = 3
    BOTH_2X1D  = 4

class HybridCalculationType:
    SIMPLE_APERTURE                = 1
    MIRROR_OR_GRATING_SIZE         = 2
    MIRROR_SIZE_AND_ERROR_PROFILE  = 3
    GRATING_SIZE_AND_ERROR_PROFILE = 4
    CRL_SIZE                       = 5
    CRL_SIZE_AND_ERROR_PROFILE     = 6

class HybridPropagationType:
    FAR_FIELD  = 1
    NEAR_FIELD = 2
    BOTH       = 3

class HybridLengthUnits:
    METERS = 0
    CENTIMETERS = 1
    MILLIMETERS = 2

# -------------------------------------------------------------
# WAVE OPTICS PROVIDER
# -------------------------------------------------------------

class HybridWaveOpticsProvider():
    @abstractmethod
    def initialize_wavefront_from_range(self,
                                        dimension : int = 1,
                                        x_min=0.0,
                                        x_max=0.0,
                                        y_min=0.0,
                                        y_max=0.0,
                                        number_of_points=(100, 100),
                                        wavelength=1e-10): raise NotImplementedError
    @abstractmethod
    def do_propagation(self, propagation_parameters, propagation_type : int = HybridPropagationType.FAR_FIELD): raise NotImplementedError

# WOFRY IS DEFAULT (provider not specified)
class _DefaultWaveOpticsProvider(HybridWaveOpticsProvider):
    def __init__(self):
        self.__propagation_manager = PropagationManager().Instance()

    def initialize_wavefront_from_range(self,
                                        dimension : int = 1,
                                        x_min=0.0,
                                        x_max=0.0,
                                        y_min=0.0,
                                        y_max=0.0,
                                        number_of_points=(100, 100),
                                        wavelength=1e-10):
        assert (dimension in [1, 2])

        if dimension == 1:
            assert (type(number_of_points)==int)
            return GenericWavefront1D.initialize_wavefront_from_range(x_min=x_min,
                                                                      x_max=x_max,
                                                                      number_of_points=number_of_points,
                                                                      wavelength=wavelength)
        elif dimension == 2:
            assert (type(number_of_points)==list)
            return GenericWavefront2D.initialize_wavefront_from_range(x_min=x_min,
                                                                      x_max=x_max,
                                                                      y_min=y_min,
                                                                      y_max=y_max,
                                                                      number_of_points=number_of_points,
                                                                      wavelength=wavelength)
    def do_propagation(self,
                       dimension : int == 1,
                       wavefront : Wavefront = None,
                       propagation_distance : float = 0.0,
                       propagation_type : int = HybridPropagationType.FAR_FIELD):
        assert (dimension in [1, 2])
        assert (propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.NEAR_FIELD])

        propagation_elements = PropagationElements()
        propagation_elements.add_beamline_element(BeamlineElement(optical_element=Screen1D() if dimension==1 else Screen2D(),
                                                                  coordinates=ElementCoordinates(p=propagation_distance)))

        if propagation_type == HybridPropagationType.FAR_FIELD:
            if dimension ==  1:  handler_name = Fraunhofer1D.HANDLER_NAME
            elif dimension == 2: handler_name = Fraunhofer2D.HANDLER_NAME
        elif propagation_type == HybridPropagationType.NEAR_FIELD:
            if dimension ==  1:  handler_name = Fresnel1D.HANDLER_NAME
            elif dimension == 2: handler_name = Fresnel2D.HANDLER_NAME

        return self.__propagation_manager.do_propagation(propagation_parameters=PropagationParameters(wavefront=wavefront,
                                                                                                      propagation_elements=propagation_elements),
                                                         handler_name=handler_name)

# -------------------------------------------------------------
# RAY-TRACING WRAPPERS
# -------------------------------------------------------------

class HybridBeamWrapper():
    def __init__(self, beam, lenght_units):
        assert (beam is not None)
        assert (lenght_units in [HybridLengthUnits.METERS, HybridLengthUnits.CENTIMETERS, HybridLengthUnits.MILLIMETERS])

        self._beam          = beam
        self._length_units = lenght_units

    @property
    def wrapped_beam(self): return self._beam
    @property
    def length_units(self) -> int: return self._length_units
    @property
    def length_units_to_m(self) -> float:
        if   self._length_units == HybridLengthUnits.METERS:      return 1.0
        elif self._length_units == HybridLengthUnits.CENTIMETERS: return 0.01
        elif self._length_units == HybridLengthUnits.MILLIMETERS: return 0.001

    @abstractmethod
    def duplicate(self): raise NotImplementedError

class HybridOEWrapper():
    def __init__(self, optical_element, name):
        self._optical_element = optical_element
        self._name            = name

    @property
    def wrapped_optical_element(self): return self._optical_element
    @property
    def name(self): return self._name

    @abstractmethod
    def check_congruence(self, calculation_type : int): raise NotImplementedError
    @abstractmethod
    def duplicate(self): raise NotImplementedError


class HybridNotNecessaryWarning(Exception):
    def __init__(self, *args, **kwargs):
        super(HybridNotNecessaryWarning, self).__init__(*args, **kwargs)

# -------------------------------------------------------------
# HYBRID I/O OBJECTS
# -------------------------------------------------------------

class HybridListener():
    @abstractmethod
    def status_message(self, message : str): raise NotImplementedError
    @abstractmethod
    def set_progress_value(self, percentage : int): raise NotImplementedError
    @abstractmethod
    def warning_message(self, message : str): raise NotImplementedError
    @abstractmethod
    def error_message(self, message : str): raise NotImplementedError


class HybridInputParameters():
    def __init__(self,
                 listener : HybridListener,
                 beam     : HybridBeamWrapper,
                 optical_element: HybridOEWrapper,
                 diffraction_plane : int = HybridDiffractionPlane.TANGENTIAL,
                 propagation_type : int = HybridPropagationType.FAR_FIELD,
                 focal_length : float = -1,
                 propagation_distance : float = -1,
                 n_bins_x : int = 200,
                 n_bins_z : int = 200,
                 n_peaks : int = 20,
                 fft_n_pts : int = 1e6,
                 analyze_geometry : bool = True,
                 random_seed : int = 0,
                 **kwargs):
        self.__listener                 = listener
        self.__beam                     = beam
        self.__original_beam            = beam.duplicate()
        self.__optical_element          = optical_element
        self.__original_optical_element = optical_element.duplicate()
        self.__diffraction_plane        = diffraction_plane
        self.__focal_length             = focal_length
        self.__propagation_distance     = propagation_distance
        self.__propagation_type         = propagation_type
        self.__n_bins_x                 = n_bins_x
        self.__n_bins_z                 = n_bins_z
        self.__n_peaks                  = n_peaks
        self.__fft_n_pts                = fft_n_pts
        self.__analyze_geometry         = analyze_geometry
        self.__random_seed              = random_seed
        self.__additional_parameters    = kwargs

    @property
    def listener(self) -> HybridListener: return self.__listener
    @property
    def beam(self) -> HybridBeamWrapper: return self.__beam
    @property
    def original_beam(self) -> HybridBeamWrapper: return self.__original_beam
    @property
    def optical_element(self) -> HybridOEWrapper: return self.__optical_element
    @property
    def original_optical_element(self) -> HybridOEWrapper: return self.__original_optical_element
    @property
    def diffraction_plane(self) -> int: return self.__diffraction_plane
    @property
    def focal_length(self) -> float: return self.__focal_length
    @property
    def propagation_distance(self) -> float: return self.__propagation_distance
    @property
    def propagation_type(self) -> int: return self.__propagation_type
    @property
    def n_bins_x(self) -> int: return self.__n_bins_x
    @property
    def n_bins_z(self) -> int: return self.__n_bins_z
    @property
    def n_peaks(self) -> int: return self.__n_peaks
    @property
    def fft_n_pts(self) -> int: return self.__fft_n_pts
    @property
    def analyze_geometry(self) -> bool: return self.__analyze_geometry
    @property
    def random_seed(self) -> int: return self.__random_seed

    def get(self, name): return self.__additional_parameters.get(name, None)

class HybridGeometryAnalysis:
    BEAM_NOT_CUT_TANGENTIALLY = 1
    BEAM_NOT_CUT_SAGITTALLY   = 2

    def __init__(self): self.__analysis = []

    def add_analysis_result(self, result : int): self.__analysis.append(result)
    def get_analysis_result(self): return copy.deepcopy(self.__analysis)
    def has_result(self, result : int): return result in self.__analysis

class HybridCalculationResult():
    def __init__(self,
                 far_field_beam : HybridBeamWrapper = None,
                 near_field_beam : HybridBeamWrapper = None,
                 geometry_analysis : HybridGeometryAnalysis = None):
        self.__far_field_beam = far_field_beam
        self.__near_field_beam = near_field_beam
        self.__geometry_analysis = geometry_analysis

    @property
    def far_field_beam(self): return self.__far_field_beam
    @far_field_beam.setter
    def far_field_beam(self, value): self.__far_field_beam = value

    @property
    def near_field_beam(self): return self.__near_field_beam
    @near_field_beam.setter
    def near_field_beam(self, value): self.__near_field_beam = value

    @property
    def geometry_analysis(self): return self.__geometry_analysis
    @geometry_analysis.setter
    def geometry_analysis(self, value): self.geometry_analysis = value

'''
    ghy_mirrorfile = "mirror.dat"


    crl_error_profiles = None
    crl_material = None
    crl_delta = None
    crl_scaling_factor = 1.0
'''
'''
class HybridCalculationParameters(object):
    beam_not_cut_in_z = False
    beam_not_cut_in_x = False

    shadow_oe_end = None

    original_beam_history = None

    image_plane_beam = None
    image_plane_beam_lost = None
    ff_beam = None
    nf_beam = None

    screen_plane_beam = None

    # Screen
    wenergy     = None
    wwavelength = None
    xp_screen   = None
    yp_screen   = None
    zp_screen   = None
    ref_screen  = None
    xx_screen = None
    ghy_x_min = 0.0
    ghy_x_max = 0.0
    zz_screen = None
    ghy_z_min = 0.0
    ghy_z_max = 0.0
    dx_ray = None
    dz_ray = None
    gwavelength = 0.0
    gknum = 0.0

    # Mirror
    xx_mirr = None
    zz_mirr = None
    angle_inc = None
    angle_ref = None

    # Mirror Surface
    w_mirr_1D_values = None
    w_mirr_2D_values = None

    # Mirror Fitted Functions
    wangle_x = None
    wangle_z = None
    wangle_ref_x = None
    wangle_ref_z = None
    wl_x     = None
    wl_z     = None

    xx_focal_ray = None
    zz_focal_ray = None

    w_mirror_lx = None
    w_mirror_lz = None
    w_mirror_l = None

    wIray_x = None
    wIray_z = None
    wIray_2d = None

    do_ff_x = True
    do_ff_z = True

    # Propagation output
    dif_xp = None
    dif_zp = None
    dif_x = None
    dif_z = None
    dif_xpzp = None
    dif_xz = None

    # Conversion Output
    dx_conv = None
    dz_conv = None
    xx_image_ff = None
    zz_image_ff = None
    xx_image_nf = None
    zz_image_nf = None

    crl_delta = None
'''

# -------------------------------------------------------------
# HYBRID SCREEN OBJECT
# -------------------------------------------------------------

class AbstractHybridScreen():
    #inner classes for calculations only

    class GeometricalParameters:
        ticket_tangential = None
        ticket_sagittal = None
        max_tangential = numpy.Inf
        min_tangential = -numpy.Inf
        max_sagittal = numpy.Inf
        min_sagittal = -numpy.Inf
        is_infinite = False

    class HybridCalculationParameters:
        pass

    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        self._wave_optics_provider = wave_optics_provider

    @classmethod
    @abstractmethod
    def get_specific_calculation_type(cls): raise NotImplementedError

    def run_hybrid_method(self, input_parameters : HybridInputParameters):
        try:
            geometry_analysis = self._check_input_congruence(input_parameters)

            if input_parameters.analyze_geometry: self._check_geometry_analysis(input_parameters, geometry_analysis)

            hybrid_result = HybridCalculationResult(geometry_analysis=geometry_analysis)

            input_parameters.listener.status_message("Starting HYBRID calculation")
            input_parameters.listener.set_progress_bar(0)

            calculation_parameters = self._extract_calculation_parameters(input_parameters)

            input_parameters.listener.status_message("Analysis of Input Beam and OE completed")
            input_parameters.listener.set_progress_bar(10)


        except HybridNotNecessaryWarning as w:
            input_parameters.listener.warning_message(message=str(w))

            hybrid_result = HybridCalculationResult(far_field_beam=input_parameters.original_beam,
                                                    near_field_beam=None,
                                                    geometry_analysis=geometry_analysis)

        return hybrid_result

    def _check_input_congruence(self, input_parameters : HybridInputParameters) -> HybridGeometryAnalysis:
        self._check_oe_congruence(input_parameters.optical_element)
        self._check_oe_displacements(input_parameters)

        return self._do_geometry_analysis(input_parameters)

    def _check_oe_congruence(self, optical_element : HybridOEWrapper):
        optical_element.check_congruence(self.get_specific_calculation_type())

    @abstractmethod
    def _check_oe_displacements(self, input_parameters : HybridInputParameters): raise NotImplementedError

    def _do_geometry_analysis(self, input_parameters : HybridInputParameters):
        geometry_analysis = HybridGeometryAnalysis()

        if self._no_lost_rays_from_oe(input_parameters):
            geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY)
            geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY)
        else:
            geometrical_parameters = self._calculate_geometrical_parameters(input_parameters)

            if geometrical_parameters.is_infinite:
                geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY)
                geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY)
            else: # ANALYSIS OF THE HISTOGRAMS
                def get_intensity_cut(ticket, _max, _min):
                    intensity       = ticket['histogram']
                    coordinates     = ticket['bin_center']
                    cursor_up       = numpy.where(coordinates < _min)
                    cursor_down     = numpy.where(coordinates > _max)
                    total_intensity = numpy.sum(intensity)

                    return (numpy.sum(intensity[cursor_up]) + numpy.sum(intensity[cursor_down])) / total_intensity

                intensity_sagittal_cut   = get_intensity_cut(geometrical_parameters.ticket_sagittal,
                                                             geometrical_parameters.min_sagittal,
                                                             geometrical_parameters.max_sagittal)
                intensity_tangential_cut = get_intensity_cut(geometrical_parameters.ticket_tangential,
                                                             geometrical_parameters.min_tangential,
                                                             geometrical_parameters.max_tangential)

                if intensity_sagittal_cut < 0.05:   geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY)
                if intensity_tangential_cut < 0.05: geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY)

        return geometry_analysis

    def _check_geometry_analysis(self, input_parameters : HybridInputParameters, geometry_analysis : HybridGeometryAnalysis):
        if self._is_geometry_analysis_enabled():
            if (input_parameters.diffraction_plane == HybridDiffractionPlane.SAGITTAL and
                geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY)) \
                    or \
                (input_parameters.diffraction_plane == HybridDiffractionPlane.TANGENTIAL and \
                 geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY)) \
                    or \
                ((input_parameters.diffraction_plane in [HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]) and \
                 (geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY) and
                  geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY))) :
                raise HybridNotNecessaryWarning("O.E. contains almost the whole beam, diffraction effects are not expected:\nCalculation aborted, beam remains unaltered")

            if input_parameters.diffraction_plane in [HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
                if geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY):
                    input_parameters.ghy_diff_plane = HybridDiffractionPlane.TANGENTIAL
                    input_parameters.listener.warning_message("O.E. does not cut the beam in the Sagittal plane:\nCalculation is done in Tangential plane only")
                elif geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY):
                    input_parameters.ghy_diff_plane = HybridDiffractionPlane.SAGITTAL
                    input_parameters.listener.warning_message("O.E. does not cut the beam in the Tangential plane:\nCalculation is done in Sagittal plane only")
    @classmethod
    def _is_geometry_analysis_enabled(cls): return True
    @abstractmethod
    def _no_lost_rays_from_oe(self, input_parameters : HybridInputParameters): raise NotImplementedError
    @abstractmethod
    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters): raise NotImplementedError

    @abstractmethod
    def _extract_calculation_parameters(self, input_parameters: HybridInputParameters): raise NotImplementedError


class AbstractSimpleApertureHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractSimpleApertureHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.SIMPLE_APERTURE


class AbstractMirrorOrGratingSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractMirrorOrGratingSizeHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.MIRROR_OR_GRATING_SIZE

class AbstractMirrorSizeAndErrorHybridScreen(AbstractMirrorOrGratingSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractMirrorSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.MIRROR_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

class AbstractGratingSizeAndErrorHybridScreen(AbstractMirrorOrGratingSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractGratingSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.GRATING_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

class AbstractCRLSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractCRLSizeHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.CRL_SIZE

class AbstractCRLSizeAndErrorHybridScreen(AbstractCRLSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractCRLSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.CRL_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False


# -------------------------------------------------------------
# HYBRID SCREEN FACTORY METHOD
# -------------------------------------------------------------

from typing import Type
from srxraylib.util.threading import Singleton, synchronized_method

@Singleton
class HybridScreenManager(object):

    def __init__(self):
        self.__chains_hashmap         = {}

    @synchronized_method
    def add_hybrid_screen_class(self, hybrid_implementation, hybrid_screen_class: Type[AbstractHybridScreen]):
        if not hybrid_implementation in self.__chains_hashmap.keys(): self.__chains_hashmap[hybrid_implementation] = {}

        hybrid_chain_of_responsibility = self.__chains_hashmap.get(hybrid_implementation)
        key                            = str(hybrid_screen_class.get_specific_calculation_type())

        if key in hybrid_chain_of_responsibility.keys(): raise ValueError("HybridScreenManager " + hybrid_screen_class.__name__ + " already in the Chain")
        else: hybrid_chain_of_responsibility[key] = hybrid_screen_class


    @synchronized_method
    def create_hybrid_screen_manager(self,
                                     hybrid_implementation,
                                     calculation_type : int = HybridCalculationType.SIMPLE_APERTURE,
                                     wave_optics_provider : HybridWaveOpticsProvider = None, **kwargs) -> AbstractHybridScreen:
        hybrid_screen_class = self.__chains_hashmap.get(hybrid_implementation, {}).get(str(calculation_type), None)

        if hybrid_screen_class is None: raise Exception("HybridScreenManager not found for calculation type: "+ str(calculation_type))
        else: return hybrid_screen_class(wave_optics_provider, **kwargs)
