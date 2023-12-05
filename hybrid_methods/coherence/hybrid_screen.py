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
import xraylib

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
    SAGITTAL   = 0
    TANGENTIAL = 1
    BOTH_2D    = 2
    BOTH_2X1D  = 3

class HybridCalculationType:
    SIMPLE_APERTURE                = 0
    MIRROR_OR_GRATING_SIZE         = 1
    MIRROR_SIZE_AND_ERROR_PROFILE  = 2
    GRATING_SIZE_AND_ERROR_PROFILE = 3
    CRL_SIZE                       = 4
    CRL_SIZE_AND_ERROR_PROFILE     = 5

class HybridPropagationType:
    FAR_FIELD  = 0
    NEAR_FIELD = 1
    BOTH       = 2

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
                 far_field_image_distance : float = -1,
                 near_field_image_distance : float = -1,
                 n_bins_x : int = 200,
                 n_bins_z : int = 200,
                 n_peaks : int = 20,
                 fft_n_pts : int = 1e6,
                 analyze_geometry : bool = True,
                 random_seed : int = 0,
                 **kwargs):
        self.__listener                  = listener
        self.__beam                      = beam
        self.__original_beam             = beam.duplicate()
        self.__optical_element           = optical_element
        self.__original_optical_element  = optical_element.duplicate()
        self.__diffraction_plane         = diffraction_plane
        self.__propagation_type          = propagation_type
        self.__far_field_image_distance  = far_field_image_distance
        self.__near_field_image_distance = near_field_image_distance
        self.__n_bins_x                  = n_bins_x
        self.__n_bins_z                  = n_bins_z
        self.__n_peaks                   = n_peaks
        self.__fft_n_pts                 = fft_n_pts
        self.__analyze_geometry          = analyze_geometry
        self.__random_seed               = random_seed
        self.__additional_parameters     = kwargs

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
    def propagation_type(self) -> int: return self.__propagation_type
    @property
    def far_field_image_distance(self) -> float: return self.__far_field_image_distance
    @property
    def near_field_image_distance(self) -> float: return self.__near_field_image_distance
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

    # INPUT PARAMETERS TO BE CHANGED BY CALCULATION
    @far_field_image_distance.setter
    def far_field_image_distance(self, value : float): self.__far_field_image_distance = value
    @near_field_image_distance.setter
    def near_field_image_distance(self, value : float): self.__near_field_image_distance = value
    @n_bins_x.setter
    def n_bins_x(self, value : int): self.__n_bins_x = value
    @n_bins_z.setter
    def n_bins_z(self, value : int): self.__n_bins_z = value
    @n_peaks.setter
    def n_peaks(self, value : int): self.__n_peaks = value
    @fft_n_pts.setter
    def fft_n_pts(self, value : int): self.__fft_n_pts = value

    def get(self, name): return self.__additional_parameters.get(name, None)

class HybridGeometryAnalysis:
    BEAM_NOT_CUT_TANGENTIALLY = 1
    BEAM_NOT_CUT_SAGITTALLY   = 2

    def __init__(self): self.__analysis = []

    def add_analysis_result(self, result : int): self.__analysis.append(result)
    def get_analysis_result(self): return copy.deepcopy(self.__analysis)
    def has_result(self, result : int): return result in self.__analysis

    def __str__(self):
        text = "Geometry Analysis:"
        if len(self.__analysis) == 0: text += " beam is cut in both directions"
        else:
            if self.BEAM_NOT_CUT_SAGITTALLY in self.__analysis: text += " beam not cut sagittally"
            if self.BEAM_NOT_CUT_TANGENTIALLY in self.__analysis: text += " beam not cut tangentially"
        return text

class HybridCalculationResult():
    def __init__(self,
                 far_field_beam : HybridBeamWrapper = None,
                 near_field_beam : HybridBeamWrapper = None,
                 divergence_sagittal : ScaledArray = None,
                 divergence_tangential : ScaledArray = None,
                 divergence_2D : ScaledMatrix = None,
                 position_sagittal: ScaledArray = None,
                 position_tangential: ScaledArray = None,
                 position_2D: ScaledMatrix = None,
                 geometry_analysis : HybridGeometryAnalysis = None):
        self.__far_field_beam = far_field_beam
        self.__near_field_beam = near_field_beam
        self.__divergence_sagittal = divergence_sagittal
        self.__divergence_tangential = divergence_tangential
        self.__divergence_2D = divergence_2D
        self.__position_sagittal = position_sagittal
        self.__position_tangential = position_tangential
        self.__position_2D = position_2D
        self.__geometry_analysis = geometry_analysis

    @property
    def far_field_beam(self) -> HybridBeamWrapper: return self.__far_field_beam
    @far_field_beam.setter
    def far_field_beam(self, value : HybridBeamWrapper): self.__far_field_beam = value

    @property
    def near_field_beam(self) -> HybridBeamWrapper: return self.__near_field_beam
    @near_field_beam.setter
    def near_field_beam(self, value : HybridBeamWrapper): self.__near_field_beam = value

    @property
    def divergence_sagittal(self) -> ScaledArray: return self.__divergence_sagittal
    @divergence_sagittal.setter
    def divergence_sagittal(self, value : ScaledArray): self.__divergence_sagittal = value

    @property
    def divergence_tangential(self) -> ScaledArray: return self.__divergence_tangential
    @divergence_tangential.setter
    def divergence_tangential(self, value : ScaledArray): self.__divergence_tangential = value

    @property
    def divergence_2D(self) -> ScaledMatrix: return self.__divergence_2D
    @divergence_2D.setter
    def divergence_2D(self, value : ScaledMatrix): self.__divergence_2D = value

    @property
    def position_sagittal(self) -> ScaledArray: return self.__position_sagittal
    @position_sagittal.setter
    def position_sagittal(self, value : ScaledArray): self.__position_sagittal = value

    @property
    def position_tangential(self) -> ScaledArray: return self.__position_tangential
    @position_tangential.setter
    def position_tangential(self, value : ScaledArray): self.__position_tangential = value

    @property
    def position_2D(self) -> ScaledMatrix: return self.__position_2D
    @position_2D.setter
    def position_2D(self, value : ScaledMatrix): self.__position_2D = value

    @property
    def geometry_analysis(self) -> HybridGeometryAnalysis: return self.__geometry_analysis
    @geometry_analysis.setter
    def geometry_analysis(self, value : HybridGeometryAnalysis): self.geometry_analysis = value

# -------------------------------------------------------------
# HYBRID SCREEN OBJECT
# -------------------------------------------------------------

class AbstractHybridScreen():
    #**************************************
    #inner classes: for internal calculations only
    #**************************************

    class GeometricalParameters:
        def __init__(self,
                     ticket_tangential: dict=None,
                     ticket_sagittal: dict = None,
                     max_tangential: float = numpy.Inf,
                     min_tangential: float = -numpy.Inf,
                     max_sagittal: float = numpy.Inf,
                     min_sagittal: float = -numpy.Inf,
                     is_infinite: bool = False):
            self.__ticket_tangential = ticket_tangential
            self.__ticket_sagittal   = ticket_sagittal  
            self.__max_tangential    = max_tangential   
            self.__min_tangential    = min_tangential   
            self.__max_sagittal      = max_sagittal     
            self.__min_sagittal      = min_sagittal     
            self.__is_infinite       = is_infinite      
        
        @property
        def ticket_tangential(self) -> dict: return self.__ticket_tangential
        @ticket_tangential.setter
        def ticket_tangential(self, value: dict): self.__ticket_tangential = value
        
        @property
        def ticket_sagittal(self) -> dict: return self.__ticket_sagittal
        @ticket_sagittal.setter
        def ticket_sagittal(self, value: dict): self.__ticket_sagittal = value

        @property
        def max_tangential(self) -> float: return self.__max_tangential
        @max_tangential.setter
        def max_tangential(self, value: float): self.__max_tangential = value

        @property
        def min_tangential(self) -> float: return self.__min_tangential
        @min_tangential.setter
        def min_tangential(self, value: float): self.__min_tangential = value

        @property
        def max_sagittal(self) -> float: return self.__max_sagittal
        @max_sagittal.setter
        def max_sagittal(self, value: float): self.__max_sagittal = value
        
        @property
        def min_sagittal(self) -> float: return self.__min_sagittal
        @min_sagittal.setter
        def min_sagittal(self, value: float): self.__min_sagittal = value

        @property
        def is_infinite(self) -> bool: return self.__is_infinite
        @min_sagittal.setter
        def is_infinite(self, value: bool): self.__is_infinite = value

        
    class CalculationParameters: # Keep generic to allow any possible variation with the chosen raytracing tool
        def __init__(self,
                     energy: float=None,
                     wavelength: float=None,
                     xx_screen: numpy.ndarray=None,
                     zz_screen: numpy.ndarray=None,
                     xp_screen: numpy.ndarray=None,
                     yp_screen: numpy.ndarray=None,
                     zp_screen: numpy.ndarray=None,
                     x_min: float=None,
                     x_max: float=None,
                     z_min: float=None,
                     z_max: float=None,
                     dx_rays: numpy.ndarray=None,
                     dz_rays: numpy.ndarray=None,
                     dif_x: ScaledArray=None,
                     dif_z: ScaledArray=None,
                     dif_xp: ScaledArray=None,
                     dif_zp: ScaledArray=None,
                     dif_xpzp: ScaledMatrix=None,
                     dx_convolution: numpy.ndarray=None,
                     dz_convolution: numpy.ndarray=None,
                     xx_propagated: numpy.ndarray=None,
                     zz_propagated: numpy.ndarray=None,
                     xx_image_ff: numpy.ndarray=None,
                     zz_image_ff: numpy.ndarray=None,
                     xx_image_nf: numpy.ndarray=None,
                     zz_image_nf: numpy.ndarray=None,
                     ff_beam: HybridBeamWrapper=None,
                     nf_beam: HybridBeamWrapper=None,
                     ): 
            self.__energy         = energy
            self.__wavelength     = wavelength
            self.__xp_screen      = xp_screen
            self.__yp_screen      = yp_screen
            self.__zp_screen      = zp_screen
            self.__xx_screen      = xx_screen
            self.__zz_screen      = zz_screen
            self.__x_min          = x_min
            self.__x_max          = x_max
            self.__z_min          = z_min
            self.__z_max          = z_max
            self.__dx_rays        = dx_rays
            self.__dz_rays        = dz_rays
            self.__dif_x          = dif_x
            self.__dif_z          = dif_z
            self.__dif_xp         = dif_xp
            self.__dif_zp         = dif_zp
            self.__dif_xpzp       = dif_xpzp
            self.__dx_convolution = dx_convolution
            self.__dz_convolution = dz_convolution
            self.__xx_propagated       = xx_propagated
            self.__zz_propagated       = zz_propagated
            self.__xx_image_ff    = xx_image_ff
            self.__zz_image_ff    = zz_image_ff
            self.__xx_image_nf    = xx_image_nf
            self.__zz_image_nf    = zz_image_nf
            self.__ff_beam        = ff_beam
            self.__nf_beam        = nf_beam

            self.__calculation_parameters = {}

        @property
        def energy(self) -> float: return self.__energy
        @energy.setter
        def energy(self, value: float): self.__energy = value

        @property
        def wavelength(self) -> float: return self.__wavelength
        @energy.setter
        def wavelength(self, value: float): self.__wavelength = value

        @property
        def xx_screen(self) -> numpy.ndarray: return self.__xx_screen
        @energy.setter
        def xx_screen(self, value: numpy.ndarray): self.__xx_screen = value

        @property
        def zz_screen(self) -> numpy.ndarray: return self.__zz_screen
        @energy.setter
        def zz_screen(self, value: numpy.ndarray): self.__zz_screen = value

        @property
        def xp_screen(self) -> numpy.ndarray: return self.__xp_screen
        @energy.setter
        def xp_screen(self, value: numpy.ndarray): self.__xp_screen = value

        @property
        def yp_screen(self) -> numpy.ndarray: return self.__yp_screen
        @energy.setter
        def yp_screen(self, value: numpy.ndarray): self.__yp_screen = value

        @property
        def zp_screen(self) -> numpy.ndarray: return self.__zp_screen
        @energy.setter
        def zp_screen(self, value: numpy.ndarray): self.__zp_screen = value

        @property
        def x_min(self) -> float: return self.__x_min
        @energy.setter
        def x_min(self, value: float): self.__x_min = value

        @property
        def x_max(self) -> float: return self.__x_max
        @energy.setter
        def x_max(self, value: float): self.__x_max = value

        @property
        def z_min(self) -> float: return self.__z_min
        @energy.setter
        def z_min(self, value: float): self.__z_min = value

        @property
        def z_max(self) -> float: return self.__z_max
        @energy.setter
        def z_max(self, value: float): self.__z_max = value

        @property
        def dx_rays(self) -> numpy.ndarray: return self.__dx_rays
        @energy.setter
        def dx_rays(self, value: numpy.ndarray): self.__dx_rays = value

        @property
        def dz_rays(self) -> numpy.ndarray: return self.__dz_rays
        @energy.setter
        def dz_rays(self, value: numpy.ndarray): self.__dz_rays = value

        @property
        def dif_x(self) -> ScaledArray: return self.__dif_x
        @energy.setter
        def dif_x(self, value: ScaledArray): self.__dif_x = value

        @property
        def dif_z(self) -> ScaledArray: return self.__dif_z
        @energy.setter
        def dif_z(self, value: ScaledArray): self.__dif_z = value

        @property
        def dif_xp(self) -> ScaledArray: return self.__dif_xp
        @energy.setter
        def dif_xp(self, value: ScaledArray): self.__dif_xp = value

        @property
        def dif_zp(self) -> ScaledArray: return self.__dif_zp
        @energy.setter
        def dif_zp(self, value: ScaledArray): self.__dif_zp = value

        @property
        def dif_xpzp(self) -> ScaledMatrix: return self.__dif_xpzp
        @energy.setter
        def dif_xpzp(self, value: ScaledMatrix): self.__dif_xpzp = value

        @property
        def dx_convolution(self) -> numpy.ndarray: return self.__dx_convolution
        @energy.setter
        def dx_convolution(self, value: numpy.ndarray): self.__dx_convolution = value

        @property
        def dz_convolution(self) -> numpy.ndarray: return self.__dz_convolution
        @energy.setter
        def dz_convolution(self, value: numpy.ndarray): self.__dz_convolution = value

        @property
        def xx_propagated(self) -> numpy.ndarray: return self.__xx_propagated
        @energy.setter
        def xx_propagated(self, value: numpy.ndarray): self.__xx_propagated = value

        @property
        def zz_propagated(self) -> numpy.ndarray: return self.__zz_propagated
        @energy.setter
        def zz_propagated(self, value: numpy.ndarray): self.__zz_propagated = value

        @property
        def xx_image_ff(self) -> numpy.ndarray: return self.__xx_image_ff
        @energy.setter
        def xx_image_ff(self, value: numpy.ndarray): self.__xx_image_ff = value

        @property
        def xx_image_nf(self) -> numpy.ndarray: return self.__xx_image_nf
        @energy.setter
        def xx_image_nf(self, value: numpy.ndarray): self.__xx_image_nf = value

        @property
        def zz_image_ff(self) -> numpy.ndarray: return self.__zz_image_ff
        @energy.setter
        def zz_image_ff(self, value: numpy.ndarray): self.__zz_image_ff = value

        @property
        def zz_image_nf(self) -> numpy.ndarray: return self.__zz_image_nf
        @energy.setter
        def zz_image_nf(self, value: numpy.ndarray): self.__zz_image_nf = value

        @property
        def ff_beam(self) -> HybridBeamWrapper: return self.__ff_beam
        @energy.setter
        def ff_beam(self, value: HybridBeamWrapper): self.__ff_beam = value

        @property
        def nf_beam(self) -> HybridBeamWrapper: return self.__nf_beam
        @energy.setter
        def nf_beam(self, value: HybridBeamWrapper): self.__nf_beam = value

        def get(self, parameter_name): return self.__calculation_parameters.get(parameter_name, None)
        def set(self, parameter_name, parameter_value): self.__calculation_parameters[parameter_name] = parameter_value
        def has(self, parameter_name): return parameter_name in self.__calculation_parameters.keys()

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

            calculation_parameters = self._manage_initial_screen_projection(input_parameters)

            input_parameters.listener.status_message("Analysis of Input Beam and OE completed")
            input_parameters.listener.set_progress_bar(10)

            self._initialize_hybrid_calculation(input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Initialization if Hybrid calculation completed")
            input_parameters.listener.set_progress_bar(20)

            input_parameters.listener.status_message("Start Wavefront Propagation")

            self._perform_wavefront_propagation(input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Start Ray Resampling")
            input_parameters.listener.set_progress_bar(80)

            self._convolve_wavefront_with_rays(input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Creating Output Beam")

            self._generate_output_result(input_parameters, calculation_parameters, hybrid_result)
            
            return hybrid_result

        except HybridNotNecessaryWarning as w:
            input_parameters.listener.warning_message(message=str(w))

            hybrid_result = HybridCalculationResult(far_field_beam=input_parameters.original_beam,
                                                    near_field_beam=None,
                                                    geometry_analysis=geometry_analysis)

        return hybrid_result

    # -----------------------------------------------
    # INPUT ANALYSIS: CONGRUENCE AND GEOMETRY

    def _check_input_congruence(self, input_parameters : HybridInputParameters) -> HybridGeometryAnalysis:
        self._check_oe_congruence(input_parameters.optical_element)
        self._check_oe_displacements(input_parameters)

        return self._do_geometry_analysis(input_parameters)

    def _check_oe_congruence(self, optical_element : HybridOEWrapper):
        optical_element.check_congruence(self.get_specific_calculation_type())

    @abstractmethod
    def _check_oe_displacements(self, input_parameters : HybridInputParameters): raise NotImplementedError

    def _do_geometry_analysis(self, input_parameters : HybridInputParameters) -> HybridGeometryAnalysis:
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
                def get_intensity_cut(ticket, _min, _max):
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
    def _is_geometry_analysis_enabled(cls) -> bool: return True

    @abstractmethod
    def _no_lost_rays_from_oe(self, input_parameters : HybridInputParameters) -> bool: raise NotImplementedError

    @abstractmethod
    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters) -> GeometricalParameters: raise NotImplementedError

    # -----------------------------------------------
    # CALCULATION OF ALL DATA ON THE HYBRID SCREEN

    def _manage_initial_screen_projection(self, input_parameters: HybridInputParameters) -> CalculationParameters:
        calculation_parametes = self._manage_common_initial_screen_projection_data(input_parameters)

        self._manage_specific_initial_screen_projection_data(input_parameters, calculation_parametes)

    @abstractmethod
    def _manage_common_initial_screen_projection_data(self, input_parameters: HybridInputParameters) -> CalculationParameters: raise NotImplementedError

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: CalculationParameters): pass



    # -----------------------------------------------
    # CREATION OF ALL DATA NECESSARY TO WAVEFRONT PROPAGATION

    @abstractmethod
    def _initialize_hybrid_calculation(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters):
        ray_tracing_focal_plane, ray_tracing_image_plane = self._get_ray_tracing_planes(input_parameters, calculation_parameters)
        
        # --------------------------------------------------
        # Propagation distances 
        #
        if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
            if input_parameters.far_field_image_distance < 0.0:
                input_parameters.far_field_image_distance = ray_tracing_image_plane
                input_parameters.widget.status_message("FF image distance not set (<-1), set as T_IMAGE" + str(ray_tracing_image_plane))
            else:
                if (input_parameters.far_field_image_distance == ray_tracing_image_plane):
                    input_parameters.widget.status_message("Defined FF image distance is different from T_IMAGE, used the defined distance = " + str(input_parameters.far_field_image_distance))
                else:
                    input_parameters.widget.status_message("FF image distance = " + str(input_parameters.far_field_image_distance))

        if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
            if input_parameters.near_field_image_distance < 0.0: 
                input_parameters.near_field_image_distance = ray_tracing_focal_plane
                input_parameters.listener.status_message("NF image distance not set: set as SIMAG" + str(ray_tracing_focal_plane))
            else:
                if input_parameters.near_field_image_distance != ray_tracing_focal_plane:
                    input_parameters.listener.status_message("Defined NF image distance is different from SIMAG, used the defined focal length = " + str(input_parameters.near_field_image_distance))
                else:
                    input_parameters.listener.status_message("NF image distance = " + str(input_parameters.near_field_image_distance))

            # --------------------------------------------------
            # propagated NF spatial distributions
            #
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
                calculation_parameters.xx_propagated = copy.deepcopy(calculation_parameters.xx_screen) + \
                                                       input_parameters.near_field_image_distance * numpy.tan(calculation_parameters.dx_ray)

            if input_parameters.diffraction_plane == [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
                calculation_parameters.zz_propagated = copy.deepcopy(calculation_parameters.zz_screen) + \
                                                       input_parameters.near_field_image_distance * numpy.tan(calculation_parameters.dz_ray)

        # --------------------------------------------------
        # Intensity profiles (histogram): I_ray(z) curve
        #
        histogram_s, bins_s, histogram_t, bins_t, histogram_2D = self._get_screen_plane_histograms(input_parameters, calculation_parameters)

        if input_parameters.ghy_diff_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in X
            if (input_parameters.n_bins_x < 0): input_parameters.n_bins_x = 200

            input_parameters.n_bins_x = min(input_parameters.n_bins_x, round(len(calculation_parameters.xx_screen) / 20))  # xshi change from 100 to 20
            input_parameters.n_bins_x = max(input_parameters.n_bins_x, 10)

            calculation_parameters.wIray_x = ScaledArray.initialize_from_range(histogram_s, bins_s[0], bins_s[-1])
        elif input_parameters.ghy_diff_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in Z
            if (input_parameters.n_bins_z < 0): input_parameters.n_bins_z = 200

            input_parameters.nbins_z = min(input_parameters.nbins_z, round(len(calculation_parameters.zz_screen) / 20))  # xshi change from 100 to 20
            input_parameters.nbins_z = max(input_parameters.nbins_z, 10)

            calculation_parameters.wIray_z = ScaledArray.initialize_from_range(histogram_t, bins_t[0], bins_t[-1])
        elif input_parameters.ghy_diff_plane == 3:  # 2D
            if (input_parameters.ghy_nbins_x < 0): input_parameters.ghy_nbins_x = 50
            if (input_parameters.ghy_nbins_z < 0): input_parameters.ghy_nbins_z = 50

            input_parameters.ghy_nbins_x = min(input_parameters.nbins_x, round(numpy.sqrt(len(calculation_parameters.xx_screen) / 10)))
            input_parameters.ghy_nbins_z = min(input_parameters.nbins_z, round(numpy.sqrt(len(calculation_parameters.zz_screen) / 10)))
            input_parameters.ghy_nbins_x = max(input_parameters.nbins_x, 10)
            input_parameters.ghy_nbins_z = max(input_parameters.nbins_z, 10)

            calculation_parameters.wIray_x  = ScaledArray.initialize_from_range(histogram_s, bins_s[0], bins_s[-1])
            calculation_parameters.wIray_z  = ScaledArray.initialize_from_range(histogram_t, bins_t[0], bins_t[-1])
            calculation_parameters.wIray_2d = ScaledMatrix.initialize_from_range(histogram_2D, bins_s[0], bins_s[-1], bins_t[0], bins_t[-1])

    @abstractmethod
    def _get_ray_tracing_planes(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): raise NotImplementedError

    # -----------------------------------------------
    # WAVEFRONT PROPAGATION

    @abstractmethod
    def _perform_wavefront_propagation(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): raise NotImplementedError

    # -----------------------------------------------
    # CONVOLUTION WAVEOPTICS + RAYS

    def _convolve_wavefront_with_rays(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters):
        if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d calculation in x direction
            if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                s1d     = Sampler1D(calculation_parameters.dif_xp.get_values(), calculation_parameters.dif_xp.get_abscissas())
                pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 1))
                dx_conv = numpy.arctan(pos_dif) + calculation_parameters.dx_rays  # add the ray divergence kicks

                calculation_parameters.dx_convolution = dx_conv
                calculation_parameters.xx_image_ff    = calculation_parameters.xx_screen + input_parameters.far_field_image_distance * numpy.tan(dx_conv)  # ray tracing to the image plane
            elif input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                s1d     = Sampler1D(calculation_parameters.dif_x.get_values(), calculation_parameters.dif_x.get_abscissas())
                pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xx_propagated), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 2))

                calculation_parameters.xx_image_nf = pos_dif + calculation_parameters.xx_propagated
        elif input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d calculation in z direction
            if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                s1d     = Sampler1D(calculation_parameters.dif_zp.get_values(), calculation_parameters.dif_zp.get_abscissas())
                pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 1))
                dz_conv =  numpy.arctan(pos_dif) + calculation_parameters.dz_rays  # add the ray divergence kicks

                calculation_parameters.dz_convolution = dz_conv
                calculation_parameters.zz_image_ff    = calculation_parameters.zz_screen + input_parameters.far_field_image_distance * numpy.tan(dz_conv)  # ray tracing to the image plane
            elif input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                s1d     = Sampler1D(calculation_parameters.dif_z.get_values(), calculation_parameters.dif_z.get_abscissas())
                pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.zz_propagated), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 2))

                calculation_parameters.zz_image_nf = pos_dif + calculation_parameters.zz_propagated
        elif input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:  # 2D
            s2d = Sampler2D(calculation_parameters.dif_xpzp.z_values,
                            calculation_parameters.dif_xpzp.x_coord,
                            calculation_parameters.dif_xpzp.y_coord)
            pos_dif_x, pos_dif_z = s2d.get_n_sampled_points(len(calculation_parameters.zp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 5))
            dx_conv              = numpy.arctan(pos_dif_x) + calculation_parameters.dx_ray  # add the ray divergence kicks
            dz_conv              = numpy.arctan(pos_dif_z) + calculation_parameters.dz_ray  # add the ray divergence kicks

            calculation_parameters.dx_conv     = dx_conv
            calculation_parameters.dz_conv     = dz_conv
            calculation_parameters.xx_image_ff = calculation_parameters.xx_screen + input_parameters.far_field_image_distance * numpy.tan(dx_conv)  # ray tracing to the image plane
            calculation_parameters.zz_image_ff = calculation_parameters.zz_screen + input_parameters.far_field_image_distance * numpy.tan(dz_conv)  # ray tracing to the image plane

    # -----------------------------------------------
    # OUTPUT BEAM GENERATION

    def _generate_output_result(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, hybrid_result : HybridCalculationResult):
        self._apply_convolution_to_rays(input_parameters, calculation_parameters)

        hybrid_result.position_sagittal     = calculation_parameters.dif_x
        hybrid_result.position_tangential   = calculation_parameters.dif_z
        hybrid_result.divergence_sagittal   = calculation_parameters.dif_xp
        hybrid_result.divergence_tangential = calculation_parameters.dif_zp
        hybrid_result.divergence_2D         = calculation_parameters.dif_xpzp
        hybrid_result.far_field_beam        = calculation_parameters.ff_beam
        hybrid_result.near_field_beam       = calculation_parameters.nf_beam

    @abstractmethod
    def _apply_convolution_to_rays(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): raise NotImplementedError


# -------------------------------------------------------------
# SUBCLASSES OF HYBRID SCREEN OBJECT - BY CALCULATION TYPE
# -------------------------------------------------------------

class MissingRequiredCalculationParameter(Exception):
    def __init__(self, parameter_names):
        self.__parameter_names = parameter_names

        error_message = "missing required calculation parameter(s): "
        for name in parameter_names: error_message += name + ","

        super(MissingRequiredCalculationParameter, self).__init__(error_message[:-1])

    @property
    def parameter_names(self) -> list: return self.__parameter_names


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

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        xx_mirr, yy_mirr                    = self._get_footprint_spatial_coordinates(input_parameters, calculation_parameters)
        incidence_angles, reflection_angles = self._get_ray_angles(input_parameters, calculation_parameters)

        calculation_parameters.set("incidence_angles",  incidence_angles)
        calculation_parameters.set("reflection_angles", reflection_angles)

        xx_screen = calculation_parameters.xx_screen
        zz_screen = calculation_parameters.zz_screen

        # generate theta(z) and l(z) curve over a continuous grid
        if numpy.amax(xx_screen) == numpy.amin(xx_screen):
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
                raise Exception("Inconsistent calculation: Diffraction plane is set on SAGITTAL, but the beam has no extension in that direction")
        else:
            calculation_parameters.set("incidence_angle_function_x", numpy.poly1d(numpy.polyfit(xx_screen, incidence_angles, self.NPOLY_ANGLE)))
            calculation_parameters.set("footprint_function_x",       numpy.poly1d(numpy.polyfit(xx_screen, xx_mirr,   self.NPOLY_L)))

        if numpy.amax(zz_screen) == numpy.amin(zz_screen):
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
                raise Exception("Inconsistent calculation: Diffraction plane is set on TANGENTIAL, but the beam has no extension in that direction")
        else:
            calculation_parameters.set("incidence_angle_function_z", numpy.poly1d(numpy.polyfit(zz_screen, incidence_angles, self.NPOLY_ANGLE)))
            calculation_parameters.set("footprint_function_z",       numpy.poly1d(numpy.polyfit(zz_screen, yy_mirr,   self.NPOLY_L)))

    @abstractmethod
    def _get_footprint_spatial_coordinates(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters): raise NotImplementedError
    @abstractmethod
    def _get_ray_angles(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters): raise NotImplementedError

class _AbstractMirrorOrGratingSizeAndErrorHybridScreen(AbstractMirrorOrGratingSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(_AbstractMirrorOrGratingSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        calculation_parameters = super(_AbstractMirrorOrGratingSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

        error_profile = self._get_error_profile(input_parameters, calculation_parameters)

        w_mirror_lx = None
        w_mirror_lz = None

        if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D, HybridDiffractionPlane.BOTH_2D]:  # X
            offset_y_index = self._get_tangential_displacement_index(input_parameters, calculation_parameters)

            w_mirror_lx = ScaledArray.initialize_from_steps(error_profile.z_values[:, int(len(error_profile.y_coord) / 2 - offset_y_index)],
                                                            error_profile.x_coord[0],
                                                            error_profile.x_coord[1] - error_profile.x_coord[0])

        if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D, HybridDiffractionPlane.BOTH_2D]:  # Z
            offset_x_index = self._get_sagittal_displacement_index(input_parameters, calculation_parameters)

            w_mirror_lz = ScaledArray.initialize_from_steps(error_profile.z_values[int(len(calculation_parameters.w_mirr_2D_values.x_coord) / 2 - offset_x_index), :],
                                                            error_profile.y_coord[0],
                                                            error_profile.y_coord[1] - error_profile.y_coord[0])

        calculation_parameters.set("error_profile_projection_s", w_mirror_lx)
        calculation_parameters.set("error_profile_projection_t", w_mirror_lz)
        calculation_parameters.set("error_profile",              error_profile)

    @abstractmethod
    def _get_error_profile(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters): raise NotImplementedError
    @abstractmethod
    def _get_tangential_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters): raise NotImplementedError
    @abstractmethod
    def _get_sagittal_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters): raise NotImplementedError


class AbstractMirrorSizeAndErrorHybridScreen(_AbstractMirrorOrGratingSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractMirrorSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.MIRROR_SIZE_AND_ERROR_PROFILE

class AbstractGratingSizeAndErrorHybridScreen(_AbstractMirrorOrGratingSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractGratingSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.GRATING_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        calculation_parameters = super(AbstractGratingSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

        reflection_angles = calculation_parameters.get("reflection_angles")

        calculation_parameters.set("reflection_angle_function_x", numpy.poly1d(numpy.polyfit(calculation_parameters.xx_screen, reflection_angles, self.NPOLY_ANGLE)))
        calculation_parameters.set("reflection_angle_function_z", numpy.poly1d(numpy.polyfit(calculation_parameters.zz_screen, reflection_angles, self.NPOLY_ANGLE)))

class AbstractCRLSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractCRLSizeHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.CRL_SIZE

    def _initialize_hybrid_calculation(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        super(AbstractCRLSizeHybridScreen, self)._initialize_hybrid_calculation(input_parameters, calculation_parameters)

        crl_delta = input_parameters.get("crl_delta")

        if crl_delta is None: calculation_parameters.set("crl_delta", self.get_delta(input_parameters, calculation_parameters))
        else:                 calculation_parameters.set("crl_delta", crl_delta)

    @staticmethod
    def get_delta(input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        material = input_parameters.get("crl_material")
        density  = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(material))
        energy   = calculation_parameters.energy/1000 # in KeV

        return 1 - xraylib.Refractive_Index_Re(material, energy, density)


class AbstractCRLSizeAndErrorHybridScreen(AbstractCRLSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider):
        super(AbstractCRLSizeAndErrorHybridScreen, self).__init__(wave_optics_provider)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.CRL_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        calculation_parameters = super(AbstractCRLSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

        calculation_parameters.set("error_profiles", self._get_error_profiles(input_parameters, calculation_parameters))

    @abstractmethod
    def _get_error_profiles(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters): raise NotImplementedError

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
