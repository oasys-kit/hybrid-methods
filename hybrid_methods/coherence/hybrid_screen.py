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
from typing import Union, Type, Tuple
import numpy
import copy
from abc import abstractmethod
import xraylib
from scipy.interpolate import RectBivariateSpline

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
try:
    wofry_propagation_manager.add_propagator(Fresnel1D())
    wofry_propagation_manager.add_propagator(Fresnel2D())
    wofry_propagation_manager.set_initialized()
except ValueError as e:
    print(e)

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
    KB_SIZE                        = 6
    KB_SIZE_AND_ERROR_PROFILE      = 7

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
    def do_propagation(self, 
                       dimension : int = 1,
                       wavefront : Wavefront = None,
                       propagation_distance : float = 0.0,
                       propagation_type : int = HybridPropagationType.FAR_FIELD): raise NotImplementedError

# WOFRY IS DEFAULT (provider not specified)
class _DefaultWaveOpticsProvider(HybridWaveOpticsProvider):
    def __init__(self):
        self.__propagation_manager = PropagationManager.Instance()

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
            assert (type(number_of_points) == int)
            return GenericWavefront1D.initialize_wavefront_from_range(x_min=x_min,
                                                                      x_max=x_max,
                                                                      number_of_points=number_of_points,
                                                                      wavelength=wavelength)
        elif dimension == 2:
            assert (type(number_of_points) == tuple)
            return GenericWavefront2D.initialize_wavefront_from_range(x_min=x_min,
                                                                      x_max=x_max,
                                                                      y_min=y_min,
                                                                      y_max=y_max,
                                                                      number_of_points=number_of_points,
                                                                      wavelength=wavelength)
    def do_propagation(self,
                       dimension : int = 1,
                       wavefront : Wavefront = None,
                       propagation_distance : float = 0.0,
                       propagation_type : int = HybridPropagationType.FAR_FIELD):
        assert (dimension in [1, 2])
        assert (propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.NEAR_FIELD])

        propagation_elements = PropagationElements()
        propagation_elements.add_beamline_element(BeamlineElement(optical_element=Screen1D() if dimension==1 else Screen2D(),
                                                                  coordinates=ElementCoordinates(p=propagation_distance)))

        # N.B.: Hybrid uses Fresnel Propagator only, but it could be different in the future
        if propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.NEAR_FIELD]:
            if dimension ==  1:  handler_name = Fresnel1D.HANDLER_NAME
            elif dimension == 2: handler_name = Fresnel2D.HANDLER_NAME

        return self.__propagation_manager.do_propagation(propagation_parameters=PropagationParameters(wavefront=wavefront,
                                                                                                      propagation_elements=propagation_elements,
                                                                                                      shift_half_pixel=True),
                                                         handler_name=handler_name)

# -------------------------------------------------------------
# RAY-TRACING WRAPPERS
# -------------------------------------------------------------

class HybridBeamWrapper():
    def __init__(self, beam, lenght_units, **kwargs):
        assert (beam is not None)
        assert (lenght_units in [HybridLengthUnits.METERS, HybridLengthUnits.CENTIMETERS, HybridLengthUnits.MILLIMETERS])
        if isinstance(beam, list):  self._beam = [b.duplicate(**kwargs) for b in beam]
        else:                       self._beam = beam.duplicate(**kwargs)
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
    def duplicate(self, **kwargs): raise NotImplementedError

class HybridOEWrapper():
    def __init__(self, optical_element, name, **kwargs):
        if isinstance(optical_element, list):  self._optical_element = [oe.duplicate(**kwargs) for oe in optical_element]
        else:                                  self._optical_element = optical_element.duplicate(**kwargs)
        self._name = name

    @property
    def wrapped_optical_element(self): return self._optical_element
    @property
    def name(self): return self._name

    @abstractmethod
    def check_congruence(self, calculation_type : int): raise NotImplementedError
    @abstractmethod
    def duplicate(self, **kwargs): raise NotImplementedError


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

class StdIOHybridListener(HybridListener):
    def status_message(self, message : str): print(f"MESSAGE: {message}")
    def set_progress_value(self, percentage : int): print(f"PROGRESS: {str(percentage)}%")
    def warning_message(self, message : str): print(f"WARNING: {message}")
    def error_message(self, message : str): print(f"ERROR: {message}")

class HybridInputParameters():
    def __init__(self,
                 listener : HybridListener,
                 beam     : HybridBeamWrapper,
                 optical_element: HybridOEWrapper,
                 diffraction_plane : int = HybridDiffractionPlane.TANGENTIAL,
                 propagation_type : int = HybridPropagationType.FAR_FIELD,
                 n_bins_x : int = 200,
                 n_bins_z : int = 200,
                 n_peaks : int = 20,
                 fft_n_pts : int = 1e6,
                 analyze_geometry : bool = True,
                 random_seed : int = 0,
                 **kwargs):
        self.__listener                        = listener
        self.__beam                            = beam
        self.__original_beam                   = beam.duplicate()
        self.__optical_element                 = optical_element
        self.__original_optical_element        = optical_element.duplicate()
        self.__diffraction_plane               = diffraction_plane
        self.__propagation_type                = propagation_type
        self.__n_bins_x                        = n_bins_x
        self.__n_bins_z                        = n_bins_z
        self.__n_peaks                         = n_peaks
        self.__fft_n_pts                       = fft_n_pts
        self.__analyze_geometry                = analyze_geometry
        self.__random_seed                     = random_seed
        self.__additional_parameters           = kwargs

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
    @n_bins_x.setter
    def n_bins_x(self, value : int): self.__n_bins_x = value
    @n_bins_z.setter
    def n_bins_z(self, value : int): self.__n_bins_z = value
    @n_peaks.setter
    def n_peaks(self, value : int): self.__n_peaks = value
    @fft_n_pts.setter
    def fft_n_pts(self, value : int): self.__fft_n_pts = value

    def get(self, name): return self.__additional_parameters.get(name, None)

    @property
    def additional_parameters(self) -> dict: return self.__additional_parameters

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
        @is_infinite.setter
        def is_infinite(self, value: bool): self.__is_infinite = value

    class CalculationParameters: # Keep generic to allow any possible variation with the chosen raytracing tool
        def __init__(self,
                     energy: float=None,
                     wavelength: float=None,
                     image_plane_distance: float=None,
                     xx_screen: numpy.ndarray=None,
                     zz_screen: numpy.ndarray=None,
                     xp_screen: numpy.ndarray=None,
                     yp_screen: numpy.ndarray=None,
                     zp_screen: numpy.ndarray=None,
                     x_min: float=None,
                     x_max: float=None,
                     z_min: float=None,
                     z_max: float=None,
                     wIray_x: ScaledArray=None,
                     wIray_z: ScaledArray=None,
                     wIray_2D: ScaledMatrix=None,
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
                     xx_focal: numpy.ndarray=None,
                     zz_focal: numpy.ndarray=None,
                     xx_image_ff: numpy.ndarray=None,
                     zz_image_ff: numpy.ndarray=None,
                     xx_image_nf: numpy.ndarray=None,
                     zz_image_nf: numpy.ndarray=None,
                     ff_beam: HybridBeamWrapper=None,
                     nf_beam: HybridBeamWrapper=None,
                     ): 
            self.__energy               = energy
            self.__wavelength           = wavelength
            self.__image_plane_distance = image_plane_distance
            self.__xp_screen            = xp_screen
            self.__yp_screen            = yp_screen
            self.__zp_screen            = zp_screen
            self.__xx_screen            = xx_screen
            self.__zz_screen            = zz_screen
            self.__x_min                = x_min
            self.__x_max                = x_max
            self.__z_min                = z_min
            self.__z_max                = z_max
            self.__wIray_x              = wIray_x
            self.__wIray_z              = wIray_z
            self.__wIray_2D             = wIray_2D
            self.__dx_rays              = dx_rays
            self.__dz_rays              = dz_rays
            self.__dif_x                = dif_x
            self.__dif_z                = dif_z
            self.__dif_xp               = dif_xp
            self.__dif_zp               = dif_zp
            self.__dif_xpzp             = dif_xpzp
            self.__dx_convolution       = dx_convolution
            self.__dz_convolution       = dz_convolution
            self.__xx_propagated        = xx_propagated
            self.__zz_propagated        = zz_propagated
            self.__xx_focal             = xx_focal
            self.__zz_focal             = zz_focal
            self.__xx_image_ff          = xx_image_ff
            self.__zz_image_ff          = zz_image_ff
            self.__xx_image_nf          = xx_image_nf
            self.__zz_image_nf          = zz_image_nf
            self.__ff_beam              = ff_beam
            self.__nf_beam              = nf_beam

            self.__calculation_parameters = {}

        @property
        def energy(self) -> float: return self.__energy
        @energy.setter
        def energy(self, value: float): self.__energy = value

        @property
        def wavelength(self) -> float: return self.__wavelength
        @wavelength.setter
        def wavelength(self, value: float): self.__wavelength = value

        @property
        def image_plane_distance(self) -> float: return self.__image_plane_distance
        @image_plane_distance.setter
        def image_plane_distance(self, value: float): self.__image_plane_distance = value

        @property
        def xx_screen(self) -> numpy.ndarray: return self.__xx_screen
        @xx_screen.setter
        def xx_screen(self, value: numpy.ndarray): self.__xx_screen = value

        @property
        def zz_screen(self) -> numpy.ndarray: return self.__zz_screen
        @zz_screen.setter
        def zz_screen(self, value: numpy.ndarray): self.__zz_screen = value

        @property
        def xp_screen(self) -> numpy.ndarray: return self.__xp_screen
        @xp_screen.setter
        def xp_screen(self, value: numpy.ndarray): self.__xp_screen = value

        @property
        def yp_screen(self) -> numpy.ndarray: return self.__yp_screen
        @yp_screen.setter
        def yp_screen(self, value: numpy.ndarray): self.__yp_screen = value

        @property
        def zp_screen(self) -> numpy.ndarray: return self.__zp_screen
        @zp_screen.setter
        def zp_screen(self, value: numpy.ndarray): self.__zp_screen = value

        @property
        def x_min(self) -> float: return self.__x_min
        @x_min.setter
        def x_min(self, value: float): self.__x_min = value

        @property
        def x_max(self) -> float: return self.__x_max
        @x_max.setter
        def x_max(self, value: float): self.__x_max = value

        @property
        def z_min(self) -> float: return self.__z_min
        @z_min.setter
        def z_min(self, value: float): self.__z_min = value

        @property
        def z_max(self) -> float: return self.__z_max
        @z_max.setter
        def z_max(self, value: float): self.__z_max = value

        @property
        def wIray_x(self) -> ScaledArray: return self.__wIray_x
        @wIray_x.setter
        def wIray_x(self, value: ScaledArray): self.__wIray_x = value

        @property
        def wIray_z(self) -> ScaledArray: return self.__wIray_z
        @wIray_z.setter
        def wIray_z(self, value: ScaledArray): self.__wIray_z = value

        @property
        def wIray_2D(self) -> ScaledMatrix: return self.__wIray_2D
        @wIray_2D.setter
        def wIray_2D(self, value: ScaledMatrix): self.__wIray_2D = value

        @property
        def dx_rays(self) -> numpy.ndarray: return self.__dx_rays
        @dx_rays.setter
        def dx_rays(self, value: numpy.ndarray): self.__dx_rays = value

        @property
        def dz_rays(self) -> numpy.ndarray: return self.__dz_rays
        @dz_rays.setter
        def dz_rays(self, value: numpy.ndarray): self.__dz_rays = value

        @property
        def dif_x(self) -> ScaledArray: return self.__dif_x
        @dif_x.setter
        def dif_x(self, value: ScaledArray): self.__dif_x = value

        @property
        def dif_z(self) -> ScaledArray: return self.__dif_z
        @dif_z.setter
        def dif_z(self, value: ScaledArray): self.__dif_z = value

        @property
        def dif_xp(self) -> ScaledArray: return self.__dif_xp
        @dif_xp.setter
        def dif_xp(self, value: ScaledArray): self.__dif_xp = value

        @property
        def dif_zp(self) -> ScaledArray: return self.__dif_zp
        @dif_zp.setter
        def dif_zp(self, value: ScaledArray): self.__dif_zp = value

        @property
        def dif_xpzp(self) -> ScaledMatrix: return self.__dif_xpzp
        @dif_xpzp.setter
        def dif_xpzp(self, value: ScaledMatrix): self.__dif_xpzp = value

        @property
        def dx_convolution(self) -> numpy.ndarray: return self.__dx_convolution
        @dx_convolution.setter
        def dx_convolution(self, value: numpy.ndarray): self.__dx_convolution = value

        @property
        def dz_convolution(self) -> numpy.ndarray: return self.__dz_convolution
        @dz_convolution.setter
        def dz_convolution(self, value: numpy.ndarray): self.__dz_convolution = value

        @property
        def xx_propagated(self) -> numpy.ndarray: return self.__xx_propagated
        @xx_propagated.setter
        def xx_propagated(self, value: numpy.ndarray): self.__xx_propagated = value

        @property
        def zz_propagated(self) -> numpy.ndarray: return self.__zz_propagated
        @zz_propagated.setter
        def zz_propagated(self, value: numpy.ndarray): self.__zz_propagated = value

        @property
        def xx_focal(self) -> numpy.ndarray: return self.__xx_focal
        @xx_focal.setter
        def xx_focal(self, value: numpy.ndarray): self.__xx_focal = value

        @property
        def zz_focal(self) -> numpy.ndarray: return self.__zz_focal
        @zz_focal.setter
        def zz_focal(self, value: numpy.ndarray): self.__zz_focal = value

        @property
        def xx_image_ff(self) -> numpy.ndarray: return self.__xx_image_ff
        @xx_image_ff.setter
        def xx_image_ff(self, value: numpy.ndarray): self.__xx_image_ff = value

        @property
        def xx_image_nf(self) -> numpy.ndarray: return self.__xx_image_nf
        @xx_image_nf.setter
        def xx_image_nf(self, value: numpy.ndarray): self.__xx_image_nf = value

        @property
        def zz_image_ff(self) -> numpy.ndarray: return self.__zz_image_ff
        @zz_image_ff.setter
        def zz_image_ff(self, value: numpy.ndarray): self.__zz_image_ff = value

        @property
        def zz_image_nf(self) -> numpy.ndarray: return self.__zz_image_nf
        @zz_image_nf.setter
        def zz_image_nf(self, value: numpy.ndarray): self.__zz_image_nf = value

        @property
        def ff_beam(self) -> HybridBeamWrapper: return self.__ff_beam
        @ff_beam.setter
        def ff_beam(self, value: HybridBeamWrapper): self.__ff_beam = value

        @property
        def nf_beam(self) -> HybridBeamWrapper: return self.__nf_beam
        @nf_beam.setter
        def nf_beam(self, value: HybridBeamWrapper): self.__nf_beam = value

        def get(self, parameter_name): return self.__calculation_parameters.get(parameter_name, None)
        def set(self, parameter_name, parameter_value): self.__calculation_parameters[parameter_name] = parameter_value
        def has(self, parameter_name): return parameter_name in self.__calculation_parameters.keys()

    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        if wave_optics_provider is None: wave_optics_provider = _DefaultWaveOpticsProvider()

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
            input_parameters.listener.set_progress_value(0)

            calculation_parameters = self.CalculationParameters()

            self._set_image_distance_from_optical_element(input_parameters, calculation_parameters)

            self._manage_initial_screen_projections(input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Analysis of Input Beam and OE completed")
            input_parameters.listener.set_progress_value(10)

            self._initialize_hybrid_calculation(input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Initialization if Hybrid calculation completed")
            input_parameters.listener.set_progress_value(20)

            input_parameters.listener.status_message("Start Wavefront Propagation")

            self._perform_wavefront_propagation(input_parameters, calculation_parameters, geometry_analysis)

            input_parameters.listener.status_message("Start Ray Resampling")
            input_parameters.listener.set_progress_value(80)

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
                    coordinates     = ticket['bins']
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
                    input_parameters.diffraction_plane = HybridDiffractionPlane.TANGENTIAL
                    input_parameters.listener.warning_message("O.E. does not cut the beam in the Sagittal plane:\nCalculation is done in Tangential plane only")
                elif geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY):
                    input_parameters.diffraction_plane = HybridDiffractionPlane.SAGITTAL
                    input_parameters.listener.warning_message("O.E. does not cut the beam in the Tangential plane:\nCalculation is done in Sagittal plane only")

    @classmethod
    def _is_geometry_analysis_enabled(cls) -> bool: return True

    @abstractmethod
    def _no_lost_rays_from_oe(self, input_parameters : HybridInputParameters) -> bool: raise NotImplementedError

    @abstractmethod
    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters) -> GeometricalParameters: raise NotImplementedError

    # -----------------------------------------------
    # CALCULATION OF ALL DATA ON THE HYBRID SCREEN

    def _manage_initial_screen_projections(self, input_parameters: HybridInputParameters, calculation_parameters: CalculationParameters):
        self._manage_common_initial_screen_projection_data(input_parameters, calculation_parameters)
        self._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

    @abstractmethod
    def _set_image_distance_from_optical_element(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): raise NotImplementedError

    @abstractmethod
    def _manage_common_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: CalculationParameters): raise NotImplementedError

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: CalculationParameters): pass

    # -----------------------------------------------
    # CREATION OF ALL DATA NECESSARY TO WAVEFRONT PROPAGATION

    def _initialize_hybrid_calculation(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters):
        if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
            calculation_parameters.xx_propagated = copy.deepcopy(calculation_parameters.xx_screen) + \
                                                   calculation_parameters.image_plane_distance * numpy.tan(calculation_parameters.dx_rays)
            if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                calculation_parameters.xx_focal = copy.deepcopy(calculation_parameters.xx_screen) + \
                                                  self._get_focal_length_from_optical_element(input_parameters, calculation_parameters) * numpy.tan(calculation_parameters.dx_rays)

        if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2D, HybridDiffractionPlane.BOTH_2X1D]:
            calculation_parameters.zz_propagated = copy.deepcopy(calculation_parameters.zz_screen) + \
                                                   calculation_parameters.image_plane_distance * numpy.tan(calculation_parameters.dz_rays)
            if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                calculation_parameters.zz_focal = copy.deepcopy(calculation_parameters.zz_screen) + \
                                                       self._get_focal_length_from_optical_element(input_parameters, calculation_parameters) * numpy.tan(calculation_parameters.dz_rays)

        # --------------------------------------------------
        # Intensity profiles (histogram): I_ray(z) curve
        #

        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D: # 2D
            if (input_parameters.n_bins_x < 0): input_parameters.n_bins_x = 50
            if (input_parameters.n_bins_z < 0): input_parameters.n_bins_z = 50
            input_parameters.n_bins_x = min(input_parameters.n_bins_x, round(numpy.sqrt(len(calculation_parameters.xx_screen) / 10)))
            input_parameters.n_bins_z = min(input_parameters.n_bins_z, round(numpy.sqrt(len(calculation_parameters.zz_screen) / 10)))
            input_parameters.n_bins_x = max(input_parameters.n_bins_x, 10)
            input_parameters.n_bins_z = max(input_parameters.n_bins_z, 10)
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in X
                if (input_parameters.n_bins_x < 0): input_parameters.n_bins_x = 200
                input_parameters.n_bins_x = min(input_parameters.n_bins_x, round(len(calculation_parameters.xx_screen) / 20))  # xshi change from 100 to 20
                input_parameters.n_bins_x = max(input_parameters.n_bins_x, 10)
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in Z
                if (input_parameters.n_bins_z < 0): input_parameters.n_bins_z = 200
                input_parameters.n_bins_z = min(input_parameters.n_bins_z, round(len(calculation_parameters.zz_screen) / 20))  # xshi change from 100 to 20
                input_parameters.n_bins_z = max(input_parameters.n_bins_z, 10)

        histogram_s, bins_s, histogram_t, bins_t, histogram_2D = self._get_screen_plane_histograms(input_parameters, calculation_parameters)

        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:  # 2D
            calculation_parameters.wIray_x  = ScaledArray.initialize_from_range(histogram_s, bins_s[0], bins_s[-1])
            calculation_parameters.wIray_z  = ScaledArray.initialize_from_range(histogram_t, bins_t[0], bins_t[-1])
            calculation_parameters.wIray_2D = ScaledMatrix.initialize_from_range(histogram_2D, bins_s[0], bins_s[-1], bins_t[0], bins_t[-1])
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in X
                calculation_parameters.wIray_x = ScaledArray.initialize_from_range(histogram_s, bins_s[0], bins_s[-1])

            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in Z
                calculation_parameters.wIray_z = ScaledArray.initialize_from_range(histogram_t, bins_t[0], bins_t[-1])

    # -----------------------------------------------
    # WAVEFRONT PROPAGATION

    @abstractmethod
    def _perform_wavefront_propagation(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis):
        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:
            self._propagate_wavefront_2D(input_parameters, calculation_parameters, geometry_analysis)
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:
                self._propagate_wavefront_sagittally(input_parameters, calculation_parameters, geometry_analysis)

            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:
                self._propagate_wavefront_tangentially(input_parameters, calculation_parameters, geometry_analysis)

    def _propagate_wavefront_sagittally(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis):
        sagittal_phase_shift, scale_factor  = self._initialize_sagittal_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        calculation_parameters.set("sagittal_phase_shift",  sagittal_phase_shift)
        calculation_parameters.set("sagittal_scale_factor", scale_factor)

        # -------------------------------------------------
        # FAR FIELD PROPAGATION
        if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
            focallength_ff = self._calculate_focal_length_ff_1D(calculation_parameters.x_min,
                                                                calculation_parameters.x_max,
                                                                input_parameters.n_peaks,
                                                                calculation_parameters.wavelength)
            focallength_ff = self._adjust_sagittal_focal_length_ff(focallength_ff, input_parameters, calculation_parameters)

            fft_size = int(scale_factor * self._calculate_fft_size(calculation_parameters.x_min,
                                                                   calculation_parameters.x_max,
                                                                   calculation_parameters.wavelength,
                                                                   focallength_ff,
                                                                   input_parameters.fft_n_pts))

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(27)
            else:                                                               input_parameters.listener.set_progress_value(30)

            input_parameters.listener.status_message("Sagittal FF: creating plane wave, fft_size = " + str(fft_size))

            wavefront = self._wave_optics_provider.initialize_wavefront_from_range(dimension=1,
                                                                                   wavelength=calculation_parameters.wavelength,
                                                                                   number_of_points=fft_size,
                                                                                   x_min=scale_factor * calculation_parameters.x_min,
                                                                                   x_max=scale_factor * calculation_parameters.x_max)

            if scale_factor == 1.0:
                try:
                    wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_x.interpolate_values(wavefront.get_abscissas())))
                except IndexError:
                    raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

            self._add_ideal_lens_phase_shift_1D(wavefront, focallength_ff)
            self._add_specific_sagittal_phase_shift(wavefront, input_parameters, calculation_parameters)

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(35)
            else:                                                               input_parameters.listener.set_progress_value(50)
            input_parameters.listener.status_message("Sagittal FF: begin propagation (distance = " + str(focallength_ff) + ")")

            propagated_wavefront = self._wave_optics_provider.do_propagation(dimension=1,
                                                                             wavefront=wavefront,
                                                                             propagation_distance=focallength_ff,
                                                                             propagation_type=HybridPropagationType.FAR_FIELD)

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(50)
            else:                                                               input_parameters.listener.set_progress_value(70)
            input_parameters.listener.status_message("Sagittal FF - dif_xp: begin calculation")

            image_size = min(abs(calculation_parameters.x_max), abs(calculation_parameters.x_min)) * 2
            image_size = min(image_size,
                            input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * focallength_ff / abs(calculation_parameters.x_max - calculation_parameters.x_min))
            
            image_size = self._adjust_sagittal_image_size_ff(image_size, focallength_ff, input_parameters, calculation_parameters)
            
            image_n_pts = int(round(image_size / propagated_wavefront.delta() / 2) * 2 + 1)

            dif_xp = ScaledArray.initialize_from_range(numpy.ones(propagated_wavefront.size()),
                                                       -(image_n_pts - 1) / 2 * propagated_wavefront.delta(),
                                                       (image_n_pts - 1) / 2 * propagated_wavefront.delta())

            dif_xp.np_array = numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_xp.scale)) ** 2

            dif_xp.set_scale_from_range(-(image_n_pts - 1) / 2 * propagated_wavefront.delta() / focallength_ff,
                                        (image_n_pts - 1) / 2 * propagated_wavefront.delta() / focallength_ff)

            calculation_parameters.dif_xp = dif_xp

            if input_parameters.propagation_type == HybridPropagationType.FAR_FIELD: input_parameters.listener.set_progress_value(80)

        # -------------------------------------------------
        # NEAR FIELD PROPAGATION
        if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
            focallength_nf = self._get_sagittal_near_field_effective_focal_length(input_parameters, calculation_parameters)

            fft_size = int(scale_factor * self._calculate_fft_size(calculation_parameters.x_min,
                                                                   calculation_parameters.x_max,
                                                                   calculation_parameters.wavelength,
                                                                   calculation_parameters.image_plane_distance,
                                                                   input_parameters.fft_n_pts))

            input_parameters.listener.status_message("Sagittal NF: creating plane wave, fft_size = " + str(fft_size))
            input_parameters.listener.set_progress_value(55)

            wavefront = self._wave_optics_provider.initialize_wavefront_from_range(dimension=1,
                                                                                   wavelength=calculation_parameters.wavelength,
                                                                                   number_of_points=fft_size,
                                                                                   x_min=scale_factor * calculation_parameters.x_min,
                                                                                   x_max=scale_factor * calculation_parameters.x_max)

            if scale_factor == 1.0:
                try:
                    wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_x.interpolate_values(wavefront.get_abscissas())))
                except IndexError:
                    raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

            self._add_ideal_lens_phase_shift_1D(wavefront, focallength_nf)
            self._add_specific_sagittal_phase_shift(wavefront, input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Sagittal NF: begin propagation (distance = " + str(focallength_nf) + ")")
            input_parameters.listener.set_progress_value(60)

            propagation_distance = calculation_parameters.image_plane_distance

            propagated_wavefront = self._wave_optics_provider.do_propagation(dimension=1,
                                                                             wavefront=wavefront,
                                                                             propagation_distance=propagation_distance,
                                                                             propagation_type=HybridPropagationType.NEAR_FIELD)

            image_size = (input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * numpy.abs(propagation_distance) / abs(calculation_parameters.x_max - calculation_parameters.x_min))
            image_size = self._adjust_sagittal_image_size_nf(image_size, propagation_distance, input_parameters, calculation_parameters)

            image_n_pts = int(round(image_size / propagated_wavefront.delta() / 2) * 2 + 1)

            input_parameters.listener.set_progress_value(75)
            input_parameters.listener.status_message("Tangential NF - dif_x: begin calculation")

            dif_x = ScaledArray.initialize_from_range(numpy.ones(image_n_pts),
                                                      -(image_n_pts - 1) / 2 * propagated_wavefront.delta(),
                                                      (image_n_pts - 1) / 2 * propagated_wavefront.delta())

            dif_x.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_x.scale))**2

            calculation_parameters.dif_x = dif_x

            input_parameters.listener.set_progress_value(80)

    def _propagate_wavefront_tangentially(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis):
        tangential_phase_shift, scale_factor  = self._initialize_tangential_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        calculation_parameters.set("tangential_phase_shift",  tangential_phase_shift)
        calculation_parameters.set("tangential_scale_factor", scale_factor)

        # -------------------------------------------------
        # FAR FIELD PROPAGATION
        if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
            focallength_ff = self._calculate_focal_length_ff_1D(calculation_parameters.z_min,
                                                                calculation_parameters.z_max,
                                                                input_parameters.n_peaks,
                                                                calculation_parameters.wavelength)
            focallength_ff = self._adjust_tangential_focal_length_ff(focallength_ff, input_parameters, calculation_parameters)
            
            fft_size = int(scale_factor * self._calculate_fft_size(calculation_parameters.z_min,
                                                                   calculation_parameters.z_max,
                                                                   calculation_parameters.wavelength,
                                                                   focallength_ff,
                                                                   input_parameters.fft_n_pts))

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(27)
            else:                                                               input_parameters.listener.set_progress_value(30)
            input_parameters.listener.status_message("FF: creating plane wave, fft_size = " + str(fft_size))

            wavefront = self._wave_optics_provider.initialize_wavefront_from_range(dimension=1,
                                                                                   wavelength=calculation_parameters.wavelength,
                                                                                   number_of_points=fft_size,
                                                                                   x_min=scale_factor * calculation_parameters.z_min,
                                                                                   x_max=scale_factor * calculation_parameters.z_max)

            if scale_factor == 1.0:
                try:
                    wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_z.interpolate_values(wavefront.get_abscissas())))
                except IndexError:
                    raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")
            
            self._add_ideal_lens_phase_shift_1D(wavefront, focallength_ff)
            self._add_specific_tangential_phase_shift(wavefront, input_parameters, calculation_parameters)

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(35)
            else:                                                               input_parameters.listener.set_progress_value(50)
            input_parameters.listener.status_message("Tangential FF: begin propagation (distance = " + str(focallength_ff) + ")")

            propagated_wavefront = self._wave_optics_provider.do_propagation(dimension=1,
                                                                             wavefront=wavefront,
                                                                             propagation_distance=focallength_ff,
                                                                             propagation_type=HybridPropagationType.FAR_FIELD)

            if input_parameters.propagation_type == HybridPropagationType.BOTH: input_parameters.listener.set_progress_value(50)
            else:                                                               input_parameters.listener.set_progress_value(70)
            input_parameters.listener.status_message("Tangential FF - dif_zp: begin calculation")

            image_size = min(abs(calculation_parameters.z_max), abs(calculation_parameters.z_min)) * 2
            image_size = min(image_size,
                             input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * focallength_ff / abs(calculation_parameters.z_max - calculation_parameters.z_min))

            image_size = self._adjust_tangential_image_size_ff(image_size, focallength_ff, input_parameters, calculation_parameters)

            image_n_pts = int(round(image_size / propagated_wavefront.delta() / 2) * 2 + 1)

            dif_zp = ScaledArray.initialize_from_range(numpy.ones(propagated_wavefront.size()),
                                                       -(image_n_pts - 1) / 2 * propagated_wavefront.delta(),
                                                       (image_n_pts - 1) / 2 * propagated_wavefront.delta())

            dif_zp.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_zp.scale))**2

            dif_zp.set_scale_from_range(-(image_n_pts - 1) / 2 * propagated_wavefront.delta() / focallength_ff,
                                        (image_n_pts - 1) / 2 * propagated_wavefront.delta() / focallength_ff)

            calculation_parameters.dif_zp = dif_zp

        # -------------------------------------------------
        # NEAR FIELD PROPAGATION
        if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
            focallength_nf = self._get_tangential_near_field_effective_focal_length(input_parameters, calculation_parameters)

            fft_size = int(scale_factor * self._calculate_fft_size(calculation_parameters.z_min,
                                                                   calculation_parameters.z_max,
                                                                   calculation_parameters.wavelength,
                                                                   calculation_parameters.image_plane_distance,
                                                                   input_parameters.fft_n_pts))

            input_parameters.listener.status_message("Tangential NF: creating plane wave, fft_size = " + str(fft_size))
            input_parameters.listener.set_progress_value(55)

            wavefront = self._wave_optics_provider.initialize_wavefront_from_range(dimension=1,
                                                                                   wavelength=calculation_parameters.wavelength,
                                                                                   number_of_points=fft_size,
                                                                                   x_min=scale_factor * calculation_parameters.z_min,
                                                                                   x_max=scale_factor * calculation_parameters.z_max)

            if scale_factor == 1.0:
                try:
                    wavefront.set_plane_wave_from_complex_amplitude(numpy.sqrt(calculation_parameters.wIray_z.interpolate_values(wavefront.get_abscissas())))
                except IndexError:
                    raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

            self._add_ideal_lens_phase_shift_1D(wavefront, focallength_nf)
            self._add_specific_tangential_phase_shift(wavefront, input_parameters, calculation_parameters)

            input_parameters.listener.status_message("Tangential NF: begin propagation (distance = " + str(calculation_parameters.image_plane_distance) + ")")
            input_parameters.listener.set_progress_value(60)

            propagated_wavefront = self._wave_optics_provider.do_propagation(dimension=1,
                                                                             wavefront=wavefront,
                                                                             propagation_distance=calculation_parameters.image_plane_distance,
                                                                             propagation_type=HybridPropagationType.NEAR_FIELD)

            image_size = (input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * numpy.abs(focallength_nf) / abs(calculation_parameters.z_max - calculation_parameters.z_min))
            image_size = self._adjust_tangential_image_size_nf(image_size, focallength_nf, input_parameters, calculation_parameters)

            image_n_pts = int(round(image_size / propagated_wavefront.delta() / 2) * 2 + 1)

            input_parameters.listener.set_progress_value(75)
            input_parameters.listener.status_message("Tangential NF - dif_z: begin calculation")

            dif_z = ScaledArray.initialize_from_range(numpy.ones(image_n_pts),
                                                      -(image_n_pts - 1) / 2 * propagated_wavefront.delta(),
                                                      (image_n_pts - 1) / 2 * propagated_wavefront.delta())

            dif_z.np_array *= numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitudes(dif_z.scale))**2

            calculation_parameters.dif_z = dif_z

            input_parameters.listener.set_progress_value(80)

    def _propagate_wavefront_2D(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis):
        phase_shift  = self._initialize_2D_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        calculation_parameters.set("2D_phase_shift",  phase_shift)

        focallength_ff = self._calculate_focal_length_ff_2D(calculation_parameters.x_min,
                                                            calculation_parameters.x_max,
                                                            calculation_parameters.z_min,
                                                            calculation_parameters.z_max,
                                                            input_parameters.n_peaks,
                                                            calculation_parameters.wavelength)

        focallength_ff = self._adjust_2D_focal_length_ff(focallength_ff, input_parameters, calculation_parameters)

        fftsize_x = int(self._calculate_fft_size(calculation_parameters.x_min,
                                                 calculation_parameters.x_max,
                                                 calculation_parameters.wavelength,
                                                 focallength_ff,
                                                 input_parameters.fft_n_pts,
                                                 factor=20))
        fftsize_z = int(self._calculate_fft_size(calculation_parameters.z_min,
                                                 calculation_parameters.z_max,
                                                 calculation_parameters.wavelength,
                                                 focallength_ff,
                                                 input_parameters.fft_n_pts,
                                                 factor=20))

        input_parameters.listener.set_progress_value(30)
        input_parameters.listener.status_message("FF: creating plane wave, fftsize_x = " +  str(fftsize_x) + ", fftsize_z = " +  str(fftsize_z))

        wavefront = self._wave_optics_provider.initialize_wavefront_from_range(dimension=2,
                                                                               wavelength=calculation_parameters.wavelength,
                                                                               number_of_points=(fftsize_x, fftsize_z),
                                                                               x_min=calculation_parameters.x_min,
                                                                               x_max=calculation_parameters.x_max,
                                                                               y_min=calculation_parameters.z_min,
                                                                               y_max=calculation_parameters.z_max)

        try:
            x_coord           = wavefront.get_coordinate_x()
            z_coord           = wavefront.get_coordinate_y()
            complex_amplitude = wavefront.get_complex_amplitude()

            for i in range(0, x_coord.shape[0]):
                for j in range(0, z_coord.shape[0]):
                    interpolated = calculation_parameters.wIray_2D.interpolate_value(x_coord[i], z_coord[j])
                    complex_amplitude[i, j] = numpy.sqrt(interpolated if interpolated > 0.0 else 0.0)

        except IndexError:
            raise Exception("Unexpected Error during interpolation: try reduce Number of bins for I(Tangential) histogram")

        self._add_ideal_lens_phase_shift_2D(wavefront, focallength_ff)
        self._add_specific_2D_phase_shift(wavefront, input_parameters, calculation_parameters)

        input_parameters.listener.set_progress_value(50)
        input_parameters.listener.status_message("2D FF: begin propagation (distance = " + str(focallength_ff) + ")")

        propagated_wavefront = self._wave_optics_provider.do_propagation(dimension=2,
                                                                         wavefront=wavefront,
                                                                         propagation_distance=focallength_ff,
                                                                         propagation_type=HybridPropagationType.FAR_FIELD)

        input_parameters.listener.set_progress_value(70)
        input_parameters.listener.status_message("2D FF - dif_xpzp: begin calculation")

        image_size_x = min(abs(calculation_parameters.x_max), abs(calculation_parameters.x_min)) * 2
        image_size_x = min(image_size_x,
                         input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * focallength_ff / abs(calculation_parameters.x_max - calculation_parameters.x_min))

        image_size_x = self._adjust_sagittal_image_size_ff(image_size_x, focallength_ff, input_parameters, calculation_parameters)

        delta_x       = propagated_wavefront.delta()[0]
        image_n_pts_x = int(round(image_size_x / delta_x / 2) * 2 + 1)

        image_size_z = min(abs(calculation_parameters.z_max), abs(calculation_parameters.z_min)) * 2
        image_size_z = min(image_size_z,
                         input_parameters.n_peaks * 2 * 0.88 * calculation_parameters.wavelength * focallength_ff / abs(calculation_parameters.z_max - calculation_parameters.z_min))

        image_size_z = self._adjust_tangential_image_size_ff(image_size_z, focallength_ff, input_parameters, calculation_parameters)

        delta_z       = propagated_wavefront.delta()[1]
        image_n_pts_z = int(round(image_size_z / delta_z / 2) * 2 + 1)

        dif_xpzp = ScaledMatrix.initialize_from_range(numpy.ones((image_n_pts_x, image_n_pts_z)),
                                                      min_scale_value_x = -(image_n_pts_x - 1) / 2 * delta_x,
                                                      max_scale_value_x =(image_n_pts_x - 1) / 2 * delta_x,
                                                      min_scale_value_y = -(image_n_pts_z - 1) / 2 * delta_z,
                                                      max_scale_value_y =(image_n_pts_z - 1) / 2 * delta_z)

        for i in range(0, dif_xpzp.shape()[0]):
            for j in range(0, dif_xpzp.shape()[1]):
                dif_xpzp.set_z_value(i, j, numpy.absolute(propagated_wavefront.get_interpolated_complex_amplitude(dif_xpzp.x_coord[i], dif_xpzp.y_coord[j]))**2)

        dif_xpzp.set_scale_from_range(0,
                                      -(image_n_pts_x - 1) / 2 * delta_x / focallength_ff,
                                      (image_n_pts_x - 1) / 2 * delta_x / focallength_ff)

        dif_xpzp.set_scale_from_range(1,
                                      -(image_n_pts_z - 1) / 2 * delta_z / focallength_ff,
                                      (image_n_pts_z - 1) / 2 * delta_z / focallength_ff)

        calculation_parameters.dif_xpzp = dif_xpzp

        input_parameters.listener.set_progress_value(80)

    def _get_sagittal_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float:
        return self._get_focal_length_from_optical_element(input_parameters, calculation_parameters)

    def _get_tangential_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float:
        return self._get_focal_length_from_optical_element(input_parameters, calculation_parameters)

    def _get_2D_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float:
        return self._get_focal_length_from_optical_element(input_parameters, calculation_parameters)

    @abstractmethod
    def _get_focal_length_from_optical_element(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: raise NotImplementedError

    @staticmethod
    def _add_ideal_lens_phase_shift_1D(wavefront: GenericWavefront1D, focal_length: float):
        wavefront.add_phase_shift((-1.0) * wavefront.get_wavenumber() * (wavefront.get_abscissas() ** 2 / focal_length) / 2)

    @staticmethod
    def _add_ideal_lens_phase_shift_2D(wavefront: GenericWavefront2D, focal_length: float):
        wavefront.add_phase_shifts(-1.0 * wavefront.get_wavenumber() * ((wavefront.get_mesh_x() ** 2 + wavefront.get_mesh_y() ** 2) / focal_length) / 2)


    def _initialize_sagittal_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float]:   return None, 1.0
    def _adjust_sagittal_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return focallength_ff
    def _add_specific_sagittal_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): pass
    def _adjust_sagittal_image_size_ff(self, image_size: float, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return image_size
    def _adjust_sagittal_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return image_size

    def _initialize_tangential_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float]: return None, 1.0
    def _adjust_tangential_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return focallength_ff
    def _add_specific_tangential_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): pass
    def _adjust_tangential_image_size_ff(self, image_size: float, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return image_size
    def _adjust_tangential_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return image_size

    def _initialize_2D_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Union[ScaledMatrix, None]: return None
    def _adjust_2D_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters) -> float: return focallength_ff
    def _add_specific_2D_phase_shift(self, wavefront: GenericWavefront2D, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters): pass

    # -----------------------------------------------
    # WAVEFRONT PROPAGATION UTILITY
    @staticmethod
    def _get_rms_slope_error_from_height(wmirror_l):
        array_first_derivative = numpy.gradient(wmirror_l.np_array, wmirror_l.delta())

        return AbstractHybridScreen._get_rms_error(ScaledArray(array_first_derivative, wmirror_l.scale))

    @staticmethod
    def _get_rms_error(data):
        wfftcol = numpy.absolute(numpy.fft.fft(data.np_array))

        waPSD = (2 * data.delta() * wfftcol[0:int(len(wfftcol) / 2)] ** 2) / data.size()  # uniformed with IGOR, FFT is not simmetric around 0
        waPSD[0] /= 2
        waPSD[len(waPSD) - 1] /= 2

        fft_scale = numpy.fft.fftfreq(data.size()) / data.delta()

        waRMS = numpy.trapz(waPSD, fft_scale[0:int(len(wfftcol) / 2)])  # uniformed with IGOR: Same kind of integration, with automatic range assignement

        return numpy.sqrt(waRMS)

    @staticmethod
    def _calculate_focal_length_ff_1D(min_value, max_value, n_peaks, wavelength):
        return (max_value - min_value) ** 2 / n_peaks / 2 / 0.88 / wavelength  # xshi suggested, but need to first fix the problem of getting the fake solution of mirror aperture by SHADOW.

    @staticmethod
    def _calculate_focal_length_ff_2D(min_x_value, max_x_value, min_z_value, max_z_value, n_peaks, wavelength):
        return (min((max_z_value - min_z_value), (max_x_value - min_x_value))) ** 2 / n_peaks / 2 / 0.88 / wavelength

    @staticmethod
    def _calculate_fft_size(min_value, max_value, wavelength, propagation_distance, fft_npts, factor=100):
        return int(min(factor * (max_value - min_value) ** 2 / wavelength / propagation_distance / 0.88, fft_npts))

    # -----------------------------------------------
    # CONVOLUTION WAVEOPTICS + RAYS

    def _convolve_wavefront_with_rays(self, input_parameters: HybridInputParameters, calculation_parameters : CalculationParameters):
        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:  # 2D
            s2d = Sampler2D(calculation_parameters.dif_xpzp.z_values,
                            calculation_parameters.dif_xpzp.x_coord,
                            calculation_parameters.dif_xpzp.y_coord)
            pos_dif_x, pos_dif_z = s2d.get_n_sampled_points(len(calculation_parameters.zp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 5))
            dx_conv              = numpy.arctan(pos_dif_x) + calculation_parameters.dx_rays  # add the ray divergence kicks
            dz_conv              = numpy.arctan(pos_dif_z) + calculation_parameters.dz_rays  # add the ray divergence kicks

            calculation_parameters.dx_convolution = dx_conv
            calculation_parameters.dz_convolution = dz_conv
            calculation_parameters.xx_image_ff    = calculation_parameters.xx_screen + calculation_parameters.image_plane_distance * numpy.tan(dx_conv)  # ray tracing to the image plane
            calculation_parameters.zz_image_ff    = calculation_parameters.zz_screen + calculation_parameters.image_plane_distance * numpy.tan(dz_conv)  # ray tracing to the image plane
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d calculation in x direction
                if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                    s1d     = Sampler1D(calculation_parameters.dif_xp.get_values(), calculation_parameters.dif_xp.get_abscissas())
                    pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 1))
                    dx_conv = numpy.arctan(pos_dif) + calculation_parameters.dx_rays  # add the ray divergence kicks

                    calculation_parameters.dx_convolution = dx_conv
                    calculation_parameters.xx_image_ff    = calculation_parameters.xx_screen + calculation_parameters.image_plane_distance * numpy.tan(dx_conv)  # ray tracing to the image plane

                if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                    s1d     = Sampler1D(calculation_parameters.dif_x.get_values(), calculation_parameters.dif_x.get_abscissas())
                    pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xx_focal), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 2))

                    calculation_parameters.xx_image_nf = pos_dif + calculation_parameters.xx_focal

            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d calculation in z direction
                if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                    s1d     = Sampler1D(calculation_parameters.dif_zp.get_values(), calculation_parameters.dif_zp.get_abscissas())
                    pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.xp_screen), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 1))
                    dz_conv =  numpy.arctan(pos_dif) + calculation_parameters.dz_rays  # add the ray divergence kicks

                    calculation_parameters.dz_convolution = dz_conv
                    calculation_parameters.zz_image_ff    = calculation_parameters.zz_screen + calculation_parameters.image_plane_distance * numpy.tan(dz_conv)  # ray tracing to the image plane

                if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                    s1d     = Sampler1D(calculation_parameters.dif_z.get_values(), calculation_parameters.dif_z.get_abscissas())
                    pos_dif = s1d.get_n_sampled_points(len(calculation_parameters.zz_focal), seed=None if input_parameters.random_seed is None else (input_parameters.random_seed + 2))

                    calculation_parameters.zz_image_nf = pos_dif + calculation_parameters.zz_focal

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

class AbstractSimpleApertureHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(AbstractSimpleApertureHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.SIMPLE_APERTURE

    def _get_sagittal_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        return (calculation_parameters.x_max-calculation_parameters.x_min)**2/calculation_parameters.wavelength/input_parameters.n_peaks

    def _get_tangential_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        return (calculation_parameters.z_max-calculation_parameters.z_min)**2/calculation_parameters.wavelength/input_parameters.n_peaks

    def _get_2D_near_field_effective_focal_length(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        return (max(numpy.abs(calculation_parameters.x_max-calculation_parameters.x_min),
                    numpy.abs(calculation_parameters.z_max-calculation_parameters.z_min)))**2/calculation_parameters.wavelength/input_parameters.n_peaks

class AbstractMirrorOrGratingSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(AbstractMirrorOrGratingSizeHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.MIRROR_OR_GRATING_SIZE

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        xx_mirr, yy_mirr                    = self._get_footprint_spatial_coordinates(input_parameters, calculation_parameters)
        incidence_angles, reflection_angles = self._get_rays_angles(input_parameters, calculation_parameters) # in radians

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
    def _get_footprint_spatial_coordinates(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[numpy.ndarray, numpy.ndarray]: raise NotImplementedError
    @abstractmethod
    def _get_rays_angles(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> Tuple[numpy.ndarray, numpy.ndarray]: raise NotImplementedError

    def _initialize_sagittal_phase_shift(self,
                                         input_parameters: HybridInputParameters,
                                         calculation_parameters :  AbstractHybridScreen.CalculationParameters,
                                         geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float, float]:
        sagittal_phase_shift, scale_factor = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_sagittal_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        has_roll_displacement, rotation_angle = self._has_roll_displacement(input_parameters, calculation_parameters)

        if has_roll_displacement:
            sag_min, sag_max, _, _ = self._get_optical_element_spatial_limits(input_parameters, calculation_parameters)

            sagittal_phase_shift = ScaledArray.initialize_from_range(numpy.zeros(3), sag_min, sag_max)
            sagittal_phase_shift.set_values(sagittal_phase_shift.get_values() + sagittal_phase_shift.get_abscissas()*numpy.sin(-rotation_angle))

        return sagittal_phase_shift, scale_factor

    def _initialize_tangential_phase_shift(self,
                                           input_parameters: HybridInputParameters,
                                           calculation_parameters :  AbstractHybridScreen.CalculationParameters,
                                           geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float]:
        tangential_phase_shift, scale_factor = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_tangential_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        has_pitch_displacement, rotation_angle = self._has_pitch_displacement(input_parameters, calculation_parameters)

        if has_pitch_displacement:
            _, _, tan_min, tan_max = self._get_optical_element_spatial_limits(input_parameters, calculation_parameters)

            tangential_phase_shift = ScaledArray.initialize_from_range(numpy.zeros(3), tan_min, tan_max)
            tangential_phase_shift.set_values(tangential_phase_shift.get_values() + tangential_phase_shift.get_abscissas() * numpy.sin(-rotation_angle))

        return tangential_phase_shift, scale_factor

    def _initialize_2D_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Union[ScaledMatrix, None]:
        phase_shift = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_2D_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        has_pitch_displacement, rotation_angle = self._has_pitch_displacement(input_parameters, calculation_parameters)

        if has_pitch_displacement:
            sag_min, sag_max, tan_min, tan_max = self._get_optical_element_spatial_limits(input_parameters, calculation_parameters)
            phase_shift = ScaledMatrix.initialize_from_range(numpy.zeros((3, 3)),
                                                             sag_min, sag_max,
                                                             tan_min, tan_max)

            for x_index in range(phase_shift.size_x()):
                phase_shift.z_values[x_index, :] += phase_shift.get_y_values()*numpy.sin(numpy.radians(-rotation_angle))

        return phase_shift


    def _adjust_sagittal_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        sagittal_phase_shift = calculation_parameters.get("sagittal_phase_shift")
        # TODO: PATCH to be found with a formula
        return focallength_ff if sagittal_phase_shift is None else min(focallength_ff, calculation_parameters.image_plane_distance * 4)


    def _adjust_tangential_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        tangential_phase_shift = calculation_parameters.get("tangential_phase_shift")

        return focallength_ff if tangential_phase_shift is None else min(focallength_ff, calculation_parameters.image_plane_distance * 4)

    def _adjust_2D_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        phase_shift = calculation_parameters.get("2D_phase_shift")

        return focallength_ff if phase_shift is None else min(focallength_ff, calculation_parameters.image_plane_distance * 4)

    def _add_specific_sagittal_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        sagittal_phase_shift = calculation_parameters.get("sagittal_phase_shift")

        if not sagittal_phase_shift is None:
            wavefront.add_phase_shifts(self._get_reflector_phase_shift(wavefront.get_abscissas(),
                                                                       calculation_parameters.wavelength,
                                                                       calculation_parameters.get("incidence_angle_function_x"),
                                                                       calculation_parameters.get("footprint_function_x"),
                                                                       sagittal_phase_shift))

    def _add_specific_tangential_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        tangential_phase_shift = calculation_parameters.get("tangential_phase_shift")

        if not tangential_phase_shift is None:
            wavefront.add_phase_shifts(self._get_reflector_phase_shift(wavefront.get_abscissas(),
                                                                       calculation_parameters.wavelength,
                                                                       calculation_parameters.get("incidence_angle_function_z"),
                                                                       calculation_parameters.get("footprint_function_z"),
                                                                       tangential_phase_shift))

    def _add_specific_2D_phase_shift(self, wavefront: GenericWavefront2D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        total_phase_shift          = calculation_parameters.get("2D_phase_shift")
        incidence_angle_function_z = calculation_parameters.get("incidence_angle_function_z"),
        footprint_function_z       = calculation_parameters.get("footprint_function_z"),

        if not total_phase_shift is None:
            phase_shifts = numpy.zeros(wavefront.size())

            for index in range(0, phase_shifts.shape[0]):
                total_phase_shift_z = ScaledArray.initialize_from_steps(total_phase_shift.z_values[index, :],
                                                                        total_phase_shift.y_coord[0],
                                                                        total_phase_shift.y_coord[1] - total_phase_shift.y_coord[0])

                phase_shifts[index, :] = self._get_reflector_phase_shift(wavefront.get_coordinate_y(),
                                                                         calculation_parameters.wavelength,
                                                                         incidence_angle_function_z,
                                                                         footprint_function_z,
                                                                         total_phase_shift_z)
            wavefront.add_phase_shifts(phase_shifts)

    def _adjust_sagittal_image_size_ff(self, image_size: float, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        return self.__adjust_sagittal_image_size(image_size, focallength_ff, input_parameters, calculation_parameters)

    def _adjust_tangential_image_size_ff(self, image_size: float, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> float:
        return self.__adjust_tangential_image_size(image_size, focallength_ff, input_parameters, calculation_parameters)

    def _adjust_sagittal_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        return self.__adjust_sagittal_image_size(image_size, focallength_nf, input_parameters, calculation_parameters)

    def _adjust_tangential_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> float:
        return self.__adjust_tangential_image_size(image_size, focallength_nf, input_parameters, calculation_parameters)

    def __adjust_sagittal_image_size(self, image_size: float, focallength: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        has_roll_displacement, rotation_angle = self._has_roll_displacement(input_parameters, calculation_parameters)

        if has_roll_displacement:
            _, sagittal_offset = self._has_sagittal_offset(input_parameters, calculation_parameters)

            image_size = max(image_size, 8 * (focallength * numpy.tan(numpy.radians(numpy.abs(rotation_angle))) + numpy.abs(sagittal_offset)))
        
        return image_size

    def __adjust_tangential_image_size(self, image_size: float, focallength: float, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> float:
        has_pitch_displacement, rotation_angle = self._has_pitch_displacement(input_parameters, calculation_parameters)

        if has_pitch_displacement:
            _, normal_offset = self._has_normal_offset(input_parameters, calculation_parameters)

            image_size = max(image_size, 8 * (focallength * numpy.tan(numpy.radians(numpy.abs(rotation_angle))) + numpy.abs(normal_offset)))

        return image_size

    @staticmethod
    def _get_reflector_phase_shift(abscissas,
                                   wavelength,
                                   incidence_angle_function,
                                   footprint_function,
                                   current_phase_shift):
        return (-1.0) * 4 * numpy.pi / wavelength * numpy.sin(incidence_angle_function(abscissas)) * current_phase_shift.interpolate_values(footprint_function(abscissas))

    @abstractmethod
    def _has_pitch_displacement(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]: raise NotImplementedError
    @abstractmethod
    def _has_roll_displacement(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]: raise NotImplementedError
    @abstractmethod
    def _has_sagittal_offset(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]: raise NotImplementedError
    @abstractmethod
    def _has_normal_offset(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]:raise NotImplementedError

    @abstractmethod
    def _get_optical_element_angles(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[float, float]: raise NotImplementedError
    @abstractmethod
    def _get_optical_element_spatial_limits(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[float, float, float, float]: raise NotImplementedError

class _AbstractMirrorOrGratingSizeAndErrorHybridScreen(AbstractMirrorOrGratingSizeHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(_AbstractMirrorOrGratingSizeAndErrorHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        super(_AbstractMirrorOrGratingSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

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

            w_mirror_lz = ScaledArray.initialize_from_steps(error_profile.z_values[int(len(error_profile.x_coord) / 2 - offset_x_index), :],
                                                            error_profile.y_coord[0],
                                                            error_profile.y_coord[1] - error_profile.y_coord[0])

        calculation_parameters.set("sagittal_error_profile_projection",   w_mirror_lx)
        calculation_parameters.set("tangential_error_profile_projection", w_mirror_lz)
        calculation_parameters.set("error_profile",                       error_profile)

    @abstractmethod
    def _get_error_profile(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> ScaledMatrix: raise NotImplementedError
    @abstractmethod
    def _get_tangential_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters): raise NotImplementedError
    @abstractmethod
    def _get_sagittal_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters): raise NotImplementedError

    def _initialize_sagittal_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float, float]:
        _, scale_factor = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_sagittal_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        sagittal_phase_shift = calculation_parameters.get("sagittal_error_profile_projection")

        has_roll_displacement, rotation_angle = self._has_roll_displacement(input_parameters, calculation_parameters)
        if has_roll_displacement:
            sagittal_phase_shift.set_values(sagittal_phase_shift.get_values() + sagittal_phase_shift.get_abscissas() * numpy.sin(-rotation_angle))

        rms_slope = AbstractHybridScreen._get_rms_slope_error_from_height(sagittal_phase_shift)

        input_parameters.listener.status_message("Using RMS slope error = " + str(rms_slope*1e6) + "\u03BCrad")

        central_incidence_angle, central_reflection_angle = self._get_optical_element_angles(input_parameters, calculation_parameters)

        if HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY in geometry_analysis.get_analysis_result():
            dp_image = numpy.std(calculation_parameters.xx_propagated) / calculation_parameters.image_plane_distance
            dp_se = 2 * rms_slope * numpy.sin(central_incidence_angle)
            dp_error = calculation_parameters.wavelength / 2 / (calculation_parameters.x_max - calculation_parameters.x_min)

            scale_factor = max(1, 5 * min(dp_error / dp_image, dp_error / dp_se))

        calculation_parameters.set("sagittal_rms_slope", rms_slope)
        calculation_parameters.set("central_incidence_angle", central_incidence_angle)
        calculation_parameters.set("central_reflection_angle", central_reflection_angle)

        return sagittal_phase_shift, scale_factor

    def _initialize_tangential_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Tuple[Union[ScaledArray, None], float]:
        _, scale_factor = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_sagittal_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        tangential_phase_shift = calculation_parameters.get("tangential_error_profile_projection")

        has_pitch_displacement, rotation_angle = self._has_pitch_displacement(input_parameters, calculation_parameters)
        if has_pitch_displacement:
            tangential_phase_shift.set_values(tangential_phase_shift.get_values() + tangential_phase_shift.get_abscissas() * numpy.sin(-rotation_angle))

        rms_slope = AbstractHybridScreen._get_rms_slope_error_from_height(tangential_phase_shift)

        input_parameters.listener.status_message("Using RMS slope error = " + str(rms_slope*1e6) + "\u03BCrad")

        if HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY in geometry_analysis.get_analysis_result():
            dp_image = numpy.std(calculation_parameters.zz_propagated) / calculation_parameters.image_plane_distance
            dp_se = 2 * rms_slope
            dp_error = calculation_parameters.wavelength / 2 / (calculation_parameters.z_max - calculation_parameters.z_min)

            scale_factor = max(1, 5 * min(dp_error / dp_image, dp_error / dp_se))

        calculation_parameters.set("tangential_rms_slope", rms_slope)

        return tangential_phase_shift, scale_factor

    def _initialize_2D_phase_shift(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters, geometry_analysis : HybridGeometryAnalysis) -> Union[ScaledMatrix, None]:
        _ = super(AbstractMirrorOrGratingSizeHybridScreen, self)._initialize_2D_phase_shift(input_parameters, calculation_parameters, geometry_analysis)

        phase_shift = calculation_parameters.get("error_profile")

        has_pitch_displacement, rotation_angle = self._has_pitch_displacement(input_parameters, calculation_parameters)
        if has_pitch_displacement:
            phase_shift.set_values(phase_shift.get_values() + phase_shift.get_abscissas() * numpy.sin(-rotation_angle))

        rms_slope = AbstractHybridScreen._get_rms_slope_error_from_height(ScaledArray(np_array=phase_shift.z_values[int(phase_shift.size_x()/2), :],
                                                                                      scale=phase_shift.get_y_values()))

        input_parameters.listener.status_message("Using RMS slope error = " + str(rms_slope*1e6) + "\u03BCrad")

        calculation_parameters.set("tangential_rms_slope", rms_slope)

        return phase_shift

    def _adjust_tangential_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope                = calculation_parameters.get("tangential_rms_slope")

        image_size = max(image_size, 16 * rms_slope * numpy.abs(focallength_nf))

        return super(_AbstractMirrorOrGratingSizeAndErrorHybridScreen, self)._adjust_tangential_image_size_nf(image_size, focallength_nf, input_parameters, calculation_parameters)

class AbstractMirrorSizeAndErrorHybridScreen(_AbstractMirrorOrGratingSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(AbstractMirrorSizeAndErrorHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.MIRROR_SIZE_AND_ERROR_PROFILE

    def _adjust_sagittal_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope               = calculation_parameters.get("sagittal_rms_slope")
        central_incidence_angle = calculation_parameters.get("central_incidence_angle")

        if not (rms_slope == 0.0 or central_incidence_angle == 0.0):
            focallength_ff = min(focallength_ff, (calculation_parameters.x_max - calculation_parameters.x_min) / 16 / rms_slope / numpy.sin(central_incidence_angle))  # xshi changed

        return focallength_ff

    def _adjust_tangential_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope = calculation_parameters.get("tangential_rms_slope")

        if rms_slope != 0.0:
            focallength_ff = min(focallength_ff, (calculation_parameters.z_max-calculation_parameters.z_min) / 16 / rms_slope )  # xshi changed

        return focallength_ff

    def _adjust_2D_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        return self._adjust_tangential_focal_length_ff(focallength_ff, input_parameters, calculation_parameters)

    def _adjust_sagittal_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope               = calculation_parameters.get("sagittal_rms_slope")
        central_incidence_angle = calculation_parameters.get("central_incidence_angle")

        image_size = max(image_size,
                         16 * rms_slope * numpy.abs(focallength_nf) * numpy.sin(central_incidence_angle))

        return super(AbstractMirrorSizeAndErrorHybridScreen, self)._adjust_sagittal_image_size_nf(image_size, focallength_nf, input_parameters, calculation_parameters)
    
class AbstractGratingSizeAndErrorHybridScreen(_AbstractMirrorOrGratingSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(AbstractGratingSizeAndErrorHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.GRATING_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        super(AbstractGratingSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

        reflection_angles = calculation_parameters.get("reflection_angles")

        calculation_parameters.set("reflection_angle_function_x", numpy.poly1d(numpy.polyfit(calculation_parameters.xx_screen, reflection_angles, self.NPOLY_ANGLE)))
        calculation_parameters.set("reflection_angle_function_z", numpy.poly1d(numpy.polyfit(calculation_parameters.zz_screen, reflection_angles, self.NPOLY_ANGLE)))

    def _adjust_sagittal_focal_length_ff(self, focallength_ff: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope                = calculation_parameters.get("sagittal_rms_slope")
        central_incidence_angle  = calculation_parameters.get("central_incidence_angle")
        central_reflection_angle = calculation_parameters.get("central_reflection_angle")

        if not (rms_slope == 0.0 or central_incidence_angle == 0.0):
            focallength_ff =  min(focallength_ff, (calculation_parameters.x_max - calculation_parameters.x_min) / 8 / rms_slope / (numpy.sin(central_incidence_angle) + numpy.sin(central_reflection_angle)))

        return focallength_ff

    def _add_specific_sagittal_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        sagittal_phase_shift = calculation_parameters.get("sagittal_phase_shift")

        if not sagittal_phase_shift is None:
            wavefront.add_phase_shifts(self._get_grating_phase_shift(wavefront.get_abscissas(),
                                                                     calculation_parameters.wavelength,
                                                                     calculation_parameters.get("incidence_angle_function_x"),
                                                                     calculation_parameters.get("reflection_angle_function_x"),
                                                                     calculation_parameters.get("footprint_function_x"),
                                                                     sagittal_phase_shift))


    def _add_specific_tangential_phase_shift(self, wavefront: GenericWavefront1D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        tangential_phase_shift = calculation_parameters.get("tangential_phase_shift")

        if not tangential_phase_shift is None:
            wavefront.add_phase_shifts(self._get_grating_phase_shift(wavefront.get_abscissas(),
                                                                     calculation_parameters.wavelength,
                                                                     calculation_parameters.get("incidence_angle_function_z"),
                                                                     calculation_parameters.get("reflection_angle_function_z"),
                                                                     calculation_parameters.get("footprint_function_z"),
                                                                     tangential_phase_shift))

    def _add_specific_2D_phase_shift(self, wavefront: GenericWavefront2D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        total_phase_shift           = calculation_parameters.get("2D_phase_shift")
        incidence_angle_function_z  = calculation_parameters.get("incidence_angle_function_z"),
        reflection_angle_function_z = calculation_parameters.get("reflection_angle_function_z"),
        footprint_function_z        = calculation_parameters.get("footprint_function_z"),

        phase_shifts = numpy.zeros(wavefront.size())

        for index in range(0, phase_shifts.shape[0]):
            total_phase_shift_z = ScaledArray.initialize_from_steps(total_phase_shift.z_values[index, :],
                                                                    total_phase_shift.y_coord[0],
                                                                    total_phase_shift.y_coord[1] - total_phase_shift.y_coord[0])

            phase_shifts[index, :] = self._get_grating_phase_shift(wavefront.get_coordinate_y(),
                                                                   calculation_parameters.wavelength,
                                                                   incidence_angle_function_z,
                                                                   reflection_angle_function_z,
                                                                   footprint_function_z,
                                                                   total_phase_shift_z)
        wavefront.add_phase_shifts(phase_shifts)

    def _adjust_sagittal_image_size_nf(self, image_size: float, focallength_nf: float, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        rms_slope                = calculation_parameters.get("sagittal_rms_slope")
        central_incidence_angle  = calculation_parameters.get("central_incidence_angle")
        central_reflection_angle = calculation_parameters.get("central_reflection_angle")

        image_size = max(image_size,
                         8 * rms_slope * numpy.abs(focallength_nf) * (numpy.sin(central_incidence_angle) + numpy.sin(central_reflection_angle)))

        return super(AbstractGratingSizeAndErrorHybridScreen, self)._adjust_sagittal_image_size_nf(image_size, focallength_nf, input_parameters, calculation_parameters)


    @staticmethod
    def _get_grating_phase_shift(abscissas,
                                 wavelength,
                                 incidence_angle_function,
                                 reflection_angle_function,
                                 footprint_function,
                                 current_phase_shift):
        return (-1.0) * 2 * numpy.pi / wavelength * (numpy.sin(incidence_angle_function(abscissas)) + numpy.sin(reflection_angle_function(abscissas))) * current_phase_shift.interpolate_values(footprint_function(abscissas))

class AbstractCRLSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
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
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        super(AbstractCRLSizeAndErrorHybridScreen, self).__init__(wave_optics_provider, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.CRL_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _is_geometry_analysis_enabled(cls): return False

    def _manage_specific_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        super(AbstractCRLSizeAndErrorHybridScreen, self)._manage_specific_initial_screen_projection_data(input_parameters, calculation_parameters)

        calculation_parameters.set("error_profiles", self._get_error_profiles(input_parameters, calculation_parameters))

    def _add_specific_2D_phase_shift(self, wavefront: GenericWavefront2D, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        error_profiles = calculation_parameters.get("error_profiles")

        for error_profile in error_profiles:
            phase_shift = self._get_crl_phase_shift(error_profile,
                                                    input_parameters,
                                                    calculation_parameters,
                                                    [wavefront.get_coordinate_x(), wavefront.get_coordinate_y()])

            wavefront.add_phase_shift(phase_shift)

    @abstractmethod
    def _get_error_profiles(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters): raise NotImplementedError

    @staticmethod
    def _get_crl_phase_shift(thickness_error_profile: ScaledMatrix, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters, coordinates: list):
        coord_x         = thickness_error_profile.x_coord
        coord_y         = thickness_error_profile.y_coord
        thickness_error = thickness_error_profile.z_values

        interpolator = RectBivariateSpline(coord_x, coord_y, thickness_error, bbox=[None, None, None, None], kx=1, ky=1, s=0)

        wavefront_coord_x = coordinates[0]
        wavefront_coord_y = coordinates[1]

        thickness_error = interpolator(wavefront_coord_x, wavefront_coord_y)
        thickness_error[numpy.where(numpy.isnan(thickness_error))] = 0.0
        thickness_error *= input_parameters.get("crl_scaling_factor")

        return -2 * numpy.pi * calculation_parameters.get("crl_delta") * thickness_error / calculation_parameters.wavelength

class AbstractKBMirrorSizeHybridScreen(AbstractHybridScreen):
    def __init__(self, wave_optics_provider, implementation, **kwargs):
        super(AbstractKBMirrorSizeHybridScreen, self).__init__(wave_optics_provider=wave_optics_provider, **kwargs)
        self._implementation = implementation

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.KB_SIZE
    @classmethod
    def _get_internal_calculation_type(self): return HybridCalculationType.MIRROR_OR_GRATING_SIZE

    def run_hybrid_method(self, input_parameters : HybridInputParameters):
        kb_mirror_1_hybrid_screen = HybridScreenManager.Instance().create_hybrid_screen_manager(self._implementation, self._get_internal_calculation_type())
        kb_mirror_2_hybrid_screen = HybridScreenManager.Instance().create_hybrid_screen_manager(self._implementation, self._get_internal_calculation_type())

        kb_mirror_1            = input_parameters.optical_element.wrapped_optical_element[0].duplicate()
        kb_mirror_1_input_beam = input_parameters.beam.wrapped_beam[0]
        kb_mirror_2            = input_parameters.optical_element.wrapped_optical_element[1]
        kb_mirror_2_input_beam = input_parameters.beam.wrapped_beam[1]

        self._modify_image_plane_distance_on_kb_1(kb_mirror_1, kb_mirror_2)

        input_parameters_1 = HybridInputParameters(listener=input_parameters.listener,
                                                   beam=self._get_hybrid_beam_instance(input_parameters, kb_mirror_1_input_beam),
                                                   optical_element=self._get_hybrid_oe_instance(input_parameters, kb_mirror_1),
                                                   diffraction_plane=HybridDiffractionPlane.TANGENTIAL,
                                                   propagation_type=input_parameters.propagation_type,
                                                   n_bins_x=input_parameters.n_bins_x,
                                                   n_bins_z=input_parameters.n_bins_z,
                                                   n_peaks=input_parameters.n_peaks,
                                                   fft_n_pts=input_parameters.fft_n_pts,
                                                   analyze_geometry=input_parameters.analyze_geometry,
                                                   random_seed=input_parameters.random_seed,  # TODO: add field
                                                   **input_parameters.additional_parameters)

        input_parameters_2 = HybridInputParameters(listener=input_parameters.listener,
                                                   beam=self._get_hybrid_beam_instance(input_parameters, kb_mirror_2_input_beam),
                                                   optical_element=self._get_hybrid_oe_instance(input_parameters, kb_mirror_2),
                                                   diffraction_plane=HybridDiffractionPlane.TANGENTIAL,
                                                   propagation_type=input_parameters.propagation_type,
                                                   n_bins_x=input_parameters.n_bins_x,
                                                   n_bins_z=input_parameters.n_bins_z,
                                                   n_peaks=input_parameters.n_peaks,
                                                   fft_n_pts=input_parameters.fft_n_pts,
                                                   analyze_geometry=input_parameters.analyze_geometry,
                                                   random_seed=input_parameters.random_seed,  # TODO: add field
                                                   **input_parameters.additional_parameters)

        kb_mirror_1_result = kb_mirror_1_hybrid_screen.run_hybrid_method(input_parameters_1)
        kb_mirror_2_result = kb_mirror_2_hybrid_screen.run_hybrid_method(input_parameters_2)

        return self._merge_results(kb_mirror_1_result, kb_mirror_2_result)

    def _get_hybrid_beam_instance(self, input_parameters: HybridInputParameters, kb_mirror_input_beam):
        return input_parameters.beam.__class__(beam=kb_mirror_input_beam)

    def _get_hybrid_oe_instance(self, input_parameters: HybridInputParameters, kb_mirror):
        return input_parameters.optical_element.__class__(optical_element=kb_mirror, name=None)

    def _merge_results(self, kb_mirror_1_result: HybridCalculationResult, kb_mirror_2_result: HybridCalculationResult):
        geometry_analysis = HybridGeometryAnalysis()
        if kb_mirror_1_result.geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY):
            geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_SAGITTALLY)
        if kb_mirror_2_result.geometry_analysis.has_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY):
            geometry_analysis.add_analysis_result(HybridGeometryAnalysis.BEAM_NOT_CUT_TANGENTIALLY)

        return HybridCalculationResult(far_field_beam=self._merge_beams(kb_mirror_1_result.far_field_beam, kb_mirror_2_result.far_field_beam),
                                       near_field_beam=self._merge_beams(kb_mirror_1_result.near_field_beam, kb_mirror_2_result.near_field_beam),
                                       divergence_sagittal=kb_mirror_1_result.divergence_tangential,
                                       divergence_tangential=kb_mirror_2_result.divergence_tangential,
                                       position_sagittal=kb_mirror_1_result.position_tangential,
                                       position_tangential=kb_mirror_2_result.position_tangential,
                                       geometry_analysis=geometry_analysis)

    @abstractmethod
    def _modify_image_plane_distance_on_kb_1(self, kb_mirror_1: BeamlineElement, kb_mirror_2: BeamlineElement): raise NotImplementedError
    @abstractmethod
    def _merge_beams(self, beam_1: HybridBeamWrapper, beam_2: HybridBeamWrapper): raise NotImplementedError


class AbstractKBMirrorSizeAndErrorHybridScreen(AbstractKBMirrorSizeHybridScreen):
    def __init__(self, wave_optics_provider, implementation, **kwargs):
        super(AbstractKBMirrorSizeAndErrorHybridScreen, self).__init__(wave_optics_provider=wave_optics_provider, implementation=implementation, **kwargs)

    @classmethod
    def get_specific_calculation_type(cls): return HybridCalculationType.KB_SIZE_AND_ERROR_PROFILE
    @classmethod
    def _get_internal_calculation_type(self): return HybridCalculationType.MIRROR_SIZE_AND_ERROR_PROFILE


# -------------------------------------------------------------
# HYBRID SCREEN FACTORY METHOD
# -------------------------------------------------------------

from srxraylib.util.threading import Singleton, synchronized_method

@Singleton
class HybridScreenManager(object):

    def __init__(self):
        self.__chains_hashmap = {}

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
        else: return hybrid_screen_class(wave_optics_provider, implementation=hybrid_implementation, **kwargs)
