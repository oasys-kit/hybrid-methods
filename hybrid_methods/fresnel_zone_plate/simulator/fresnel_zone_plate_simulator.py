# #########################################################################
# Copyright (c) 2020, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2020. UChicago Argonne, LLC. This software was produced       #
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
# #########################################################################
# %% Fresnel zone plate simulation code
# %--------------------------------------------------------------------------
# % by Joan Vila-Comamala from original IDL version of Ana Diaz (February, 2009)
# % June, 2010
# %
# % code modified by Michael Wojcik Oct, 2013
# %
# % It simulates wavefront after FZP and propagates to the focal plane
# % plots wavefield all through the propagation from the FZP to the focus.
# %
# % 2D Extension Code --> Wavefront propagation in made using the Hankel
# % transform for a circularly symmetric function a long the radial
# % coordinate. Hankel transform routine is Hankel_Transform_MGS.
# %
# % Keep your eyes open, this code has not been throughly debugged or tested!
# %
# %
# %% ------------------------------------------------------------------------

import numpy
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.special import jn_zeros
import scipy.constants as codata

from hybrid_methods.fresnel_zone_plate.simulator.hankel_transform import hankel_transform
from hybrid_methods.fresnel_zone_plate.simulator.refractive_index import get_delta_beta

m2ev = codata.c * codata.h / codata.e


# % ------------------------------------------
#
# cs_diameter = beamstop diameter [m]
# osa_position = distance FZP-OSA [m]
# osa_diameter = OSA diameter [m]
#
# zone_plate_type = 0  # equal to 0 --> Ordinary FZP
#                      # equal to 1 --> Zone-Doubled FZP
#                      # equal to 2 --> Zone-Filled FZP
#                      # equal to 3 --> Two-Level FZP
#                      # equal to 4 --> Three-Level FZP           - not implemented
#                      # equal to 5 --> ALD Multideposition FZP 1 - not implemented
#                      # equal to 6 --> ALD Multideposition FZP 2 - not implemented
#                      # equal to 7 --> Zone-Edge Slanted FZP     - not implemented
#
# width_coating  = Width of the coated material for a zone-filled FZP
#
# height1_factor = multiply height < 1 for Two-level profile
# height2_factor = multiply height < 1 for Two-level profile
#
# with_range =  False --> plot to focal length
#               True  --> plot in a given range
#
# range_i  = initial position of the range
# range_f  = final position of the range
# n_z      = number of positions
#
# n_slices           = number of slices
# with_complex_amplitude = False
#
# % ------------------------------------------
class FZPSimulatorOptions():
    def __init__(self,
                 with_central_stop=True,
                 cs_diameter=10e-6,
                 with_order_sorting_aperture=False,
                 osa_position=0.01,
                 osa_diameter=30e-6,
                 zone_plate_type=0,
                 width_coating=20e-9,
                 height1_factor=(1 / 3),
                 height2_factor=(2 / 3),
                 with_range=False,
                 range_i=-2e-6,
                 range_f=2e-6,
                 n_z=3,
                 with_multi_slicing=False,
                 n_slices=100,
                 with_complex_amplitude=False,
                 store_partial_results=True):
        self.with_central_stop = with_central_stop
        self.cs_diameter = cs_diameter
        self.with_order_sorting_aperture = with_order_sorting_aperture
        self.osa_position = osa_position
        self.osa_diameter = osa_diameter
        self.zone_plate_type = zone_plate_type
        self.width_coating = width_coating
        self.height1_factor = height1_factor
        self.height2_factor = height2_factor
        self.with_range = with_range
        self.range_i = range_i
        self.range_f = range_f
        self.n_z = n_z
        self.with_multi_slicing = with_multi_slicing
        self.n_slices = n_slices
        self.with_complex_amplitude = with_complex_amplitude
        self.store_partial_results = store_partial_results

# % ------------------------------------------
#
# height =  zone thickness or height [m]
# diameter = FZP diameter [m]
# b_min = outermost zone width [m] / outermost period for ZD [m]
#
# % ------------------------------------------
class FZPAttributes():
    def __init__(self,
                 height=20000e-9,
                 diameter=50e-6,
                 b_min=50e-9,
                 zone_plate_material='Au',
                 template_material='SiO2'
                 ):
        self.height = height
        self.diameter = diameter
        self.b_min = b_min
        self.zone_plate_material = zone_plate_material
        self.template_material = template_material

class FZPCalculationInputParameters():
    def __init__(self,
                 source_distance=0.0,
                 image_distance=None,
                 n_points=5000,
                 multipool=True,
                 profile_last_index=-1,
                 increase_resolution=False,
                 increase_points=-1):
        self.source_distance=source_distance
        self.image_distance=image_distance
        self.n_points=n_points
        self.multipool=multipool
        self.profile_last_index=profile_last_index
        self.increase_resolution=increase_resolution
        self.increase_points=increase_points

class FZPCalculationResult():
    def __init__(self,
                 radius: numpy.ndarray,
                 intensity: numpy.ndarray,
                 complex_amplitude: numpy.ndarray,
                 efficiency: numpy.ndarray):
        self.__intensity         = intensity
        self.__intensity_profile = self.__intensity[-1, :]
        self.__radius            = radius
        self.__complex_amplitude = complex_amplitude
        self.__efficiency        = efficiency
        
        self.__xp       = None
        self.__zp       = None
        self.__dif_xpzp = None

    @property
    def intensity(self): return self.__intensity
    @property
    def complex_amplitude(self): return self.__complex_amplitude
    @property
    def efficiency(self): return self.__efficiency
    @property
    def intensity_profile(self): return self.__intensity_profile
    @intensity_profile.setter
    def intensity_profile(self, value): self.__intensity_profile = value
    @property
    def radius(self): return self.__radius
    @radius.setter
    def radius(self, value): self.__radius = value

    @property
    def xp(self): return self.__xp
    @xp.setter
    def xp(self, value): self.__xp = value
    @property
    def zp(self): return self.__zp
    @zp.setter
    def zp(self, value): self.__zp = value
    @property
    def dif_xpzp(self): return self.__dif_xpzp
    @dif_xpzp.setter
    def dif_xpzp(self, value): self.__dif_xpzp = value


class FresnelZonePlateSimulator(object):
    def __init__(self, options: FZPSimulatorOptions, attributes: FZPAttributes):
        self.__options    = options
        self.__attributes = attributes

        # to be populated by initialize method
        self.__source_distance = None
        self.__image_distance  = None
        self.__n_points        = None
        self.__multipool       = None
        self.__wavelength      = None
        self.__k               = None
        self.__focal_distance  = None
        self.__n_zones         = None
        self.__max_radius      = None
        self.__step            = None
        self.__n_zeros         = None
        self.__delta_FZP       = None
        self.__beta_FZP        = None
        self.__delta_template  = None
        self.__beta_template   = None
        self.__n_slices        = None
        self.__n_z             = None

        self.__energy_in_KeV  = None

    @property
    def options(self): return self.__options
    @property
    def attributes(self): return self.__attributes

    @property
    def energy_in_KeV(self): return self.__energy_in_KeV
    @property
    def n_zones(self): return self.__n_zones

    @property
    def source_distance(self): return self.__source_distance
    @property
    def image_distance(self): return self.__image_distance
    @property
    def zp_focal_distance(self): return self.__focal_distance
    @property
    def zp_image_distance(self): return (1 / ((1 / self.__focal_distance) - (1 / self.__source_distance)))

    def initialize(self, energy_in_KeV, input_parameters: FZPCalculationInputParameters):
        at = self.__attributes
        op = self.__options

        if energy_in_KeV <= 0.0: raise ValueError("Energy must be > 0")
        if input_parameters.n_points <= 0: raise ValueError("Number of integration points must be > 0")

        if at.height <= 0.0: raise ValueError("ZP Height must be > 0")
        if at.b_min <= 0.0: raise ValueError("ZP outermost zone width must be > 0")
        if at.diameter <= 0.0: raise ValueError("ZP Diameter must be > 0")

        if op.zone_plate_type in [1, 2]:
            if op.width_coating <= 0.0: raise ValueError("Coating Width must be > 0")
        if op.zone_plate_type == 3:
            if op.height1_factor <= 0.0: raise ValueError("Height 1 Factor must be > 0")
            if op.height2_factor <= 0.0: raise ValueError("Height 2 Factor must be > 0")

        self.__energy_in_KeV = energy_in_KeV
        self.__source_distance = input_parameters.source_distance
        self.__image_distance  = input_parameters.image_distance
        self.__n_points  = input_parameters.n_points
        self.__multipool = input_parameters.multipool

        self.__wavelength = m2ev / (1e3*self.__energy_in_KeV)  # wavelength [m]
        self.__k          = 2 * numpy.pi / self.__wavelength  # wavevector [m-1]

        self.__focal_distance = at.diameter * at.b_min / self.__wavelength  # focal distance [m]
        self.__n_zones = int(numpy.floor(1.0 / 4.0 * (at.diameter / at.b_min)))

        self.__max_radius = at.diameter
        self.__step = self.__max_radius / self.__n_points

        self.__n_zeros = int(numpy.floor(1.25 * at.diameter / 2 / self.__max_radius * self.__n_points))  # Parameter to speed up the Hankel Transform
        # when the function has zeros for N > Nzero

        self.__delta_FZP, self.__beta_FZP = get_delta_beta(self.__energy_in_KeV, at.zone_plate_material)
        self.__delta_template, self.__beta_template = get_delta_beta(self.__energy_in_KeV, at.template_material)

        if op.with_multi_slicing:
            if op.n_slices <= 1: raise ValueError("Number of slices position must be > 1")
            self.__n_slices = op.n_slices
        else:
            self.__n_slices = 1

        if op.with_range:
            if op.range_f <= op.range_i: raise ValueError("Range final position is smaller than initial position")
            if op.n_z < 2: raise ValueError("Number of position must be >= 2")

            self.__n_z = op.n_z
        else:
            self.__n_z = 1

        if op.with_order_sorting_aperture:
            if op.osa_position <= 0: raise ValueError("OSA position must be > 0")
            if op.osa_position >= self.__focal_distance: raise ValueError("OSA position beyond focal distance")
            if op.osa_diameter <= 0: raise ValueError("OSA diameter must be > 0")

    def simulate(self) -> FZPCalculationResult:
        at = self.__attributes
        op = self.__options

        profile, membrane_transmission = self.__build_zone_plate_profile()

        # Loading the position of the zeros of the 1st order Bessel function, as much position as N+1.
        c = jn_zeros(0, self.__n_points + 1)

        # Definition of the position where the calculated input and transformed
        # functions are evaluated. We define also the maximum frequency in the
        # angular domain.
        q_max = c[self.__n_points] / (2 * numpy.pi * self.__max_radius) # Maximum frequency
        r = c[:self.__n_points] * self.__max_radius / c[self.__n_points]  # Radius vector
        q = c[:self.__n_points] / (2 * numpy.pi * self.__max_radius)    # Frequency vector

        # Recalculation of the position where the initial profile is defined.
        profile_h = self.__get_profile_h(profile, r)

        if op.store_partial_results:
            map_int     = numpy.zeros((self.__n_slices + self.__n_z, self.__n_points))
            map_complex = numpy.full((self.__n_slices + self.__n_z, self.__n_points), 0j)
        else:
            map_int     = numpy.zeros((self.__n_z, self.__n_points))
            map_complex = numpy.full((self.__n_z, self.__n_points), 0j)

        # Calculation of the first angular spectrum
        # --------------------------------------------------------------------------

        print("Initialization, (or Slice #: ", 1, ")" )

        field0 = profile_h * membrane_transmission
        if op.store_partial_results:
            map_int[0, :]     = numpy.multiply(numpy.abs(field0), numpy.abs(field0))
            map_complex[0, :] = field0[0: self.__n_points]
        four0 = hankel_transform(field0, self.__max_radius, c, multipool=self.__multipool)
        field0 = profile_h

        if op.with_multi_slicing: four0 = self.__propagate_multislicing(map_int, map_complex, field0, four0, q_max, q, c)

        if op.with_range: self.__propagate_on_range(map_int, map_complex, four0, q_max, q, c)
        else:             self.__propagate_to_focus(map_int, map_complex, four0, q_max, q, c)

        efficiency = self.__calculate_efficiency(-1, map_int, profile_h, r, int(numpy.floor(10 * at.b_min / self.__step)))

        return FZPCalculationResult(radius=None,
                                    intensity=map_int,
                                    complex_amplitude=map_complex,
                                    efficiency=efficiency)

    def set_divergence_distribution(self, calculation_result: FZPCalculationResult, last_index=-1, increase_resolution=False, increase_points=-1):
        intensity_profile = calculation_result.intensity_profile[:last_index]
        radius            = self.__get_profile_radius(last_index)

        X, Y, data_2D = self.__create_2D_profile_from_1D(radius, intensity_profile)

        dif_xpzp = data_2D
        xp       = X[0, :] / self.__focal_distance
        zp       = Y[:, 0] / self.__focal_distance

        if increase_resolution:
            assert increase_points > 0
            interpolator = interp1d(radius, intensity_profile, bounds_error=False, fill_value=0.0, kind='quadratic')
            radius = numpy.linspace(radius[0], radius[-1], increase_points)

            intensity_profile = interpolator(radius)

            interpolator = RectBivariateSpline(xp, zp, dif_xpzp, bbox=[None, None, None, None], kx=2, ky=2, s=0)
            xp = numpy.linspace(xp[0], xp[-1], increase_points)
            zp = numpy.linspace(zp[0], zp[-1], increase_points)

            dif_xpzp = interpolator(xp, zp)

        calculation_result.radius            = radius
        calculation_result.intensity_profile = intensity_profile
        calculation_result.xp                = xp
        calculation_result.zp                = zp
        calculation_result.dif_xpzp          = dif_xpzp

    ###################################################
    #
    def __get_profile_radius(self, last_index=-1):
        return numpy.arange(0, self.__max_radius, self.__step)[:last_index]

    ###################################################
    #
    @classmethod
    def __create_2D_profile_from_1D(cls, radius, profile_1D):
        interpol_index = interp1d(radius, profile_1D, bounds_error=False, fill_value=0.0)

        xv = numpy.arange(-radius[-1], radius[-1], radius[1] - radius[0])  # adjust your matrix values here
        X, Y = numpy.meshgrid(xv, xv)
        profilegrid = numpy.zeros(X.shape, float)
        for i, x in enumerate(X[0, :]):
            for k, y in enumerate(Y[:, 0]):
                current_radius = numpy.sqrt(x ** 2 + y ** 2)
                profilegrid[i, k] = interpol_index(current_radius)

        return X, Y, profilegrid

    ###################################################
    #
    def __build_zone_plate_profile(self):
        at = self.__attributes
        op = self.__options

        radia = numpy.sqrt(numpy.arange(0, self.__n_zones + 1) * self.__wavelength * self.__focal_distance + ((numpy.arange(0, self.__n_zones + 1) * self.__wavelength) ** 2) / 4)
        profile = numpy.full(self.__n_points, 1 + 0j)
        profile[int(numpy.floor(radia[self.__n_zones] / self.__step)):self.__n_points] = 0

        # Ordinary FZP
        if op.zone_plate_type == 0:
            for i in range(1, self.__n_zones, 2):
                position_i = int(numpy.floor(radia[i] / self.__step))
                position_f = int(numpy.floor(radia[i + 1] / self.__step))  # N.B. the index is excluded
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

            membrane_transmission = 1

        # Zone-doubled FZP
        if op.zone_plate_type == 1:
            for i in range(1, self.__n_zones, 2):
                position_i = int(numpy.floor((radia[i] + at.b_min / 4) / self.__step))
                position_f = int(numpy.floor((radia[i + 1] - at.b_min / 4) / self.__step))
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_template / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_template / self.__wavelength * at.height))

                position_i = int(numpy.floor((radia[i] - op.width_coating / 2) / self.__step))
                position_f = int(numpy.floor((radia[i] + op.width_coating / 2) / self.__step))
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

                position_i = int(numpy.floor((radia[i + 1] - op.width_coating / 2) / self.__step))
                position_f = int(numpy.floor((radia[i + 1] + op.width_coating / 2) / self.__step))
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

            # including absorption of coating material 
            membrane_transmission = numpy.exp(-1j * (-1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * op.width_coating / 2))

        # Zone-filled FZP
        if op.zone_plate_type == 2:
            for i in range(1, self.__n_zones, 2):

                position_i = int(numpy.floor(radia[i] / self.__step))
                position_f = int(numpy.floor(radia[i + 1] / self.__step))

                width = numpy.abs(int(numpy.floor((radia[i + 1] - radia[i]) / self.__step)))
                op.width_coating_step = numpy.abs(int(numpy.floor(op.width_coating / self.__step / 2)))

                if op.width_coating < width:
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_template / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_template / self.__wavelength * at.height))

                    position_i = int(numpy.floor((radia[i] - op.width_coating) / self.__step))
                    position_f = int(numpy.floor(radia[i] / self.__step))
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

                    position_i = int(numpy.floor(radia[i + 1] / self.__step))
                    position_f = int(numpy.floor((radia[i + 1] + op.width_coating) / self.__step))
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))
                else:
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_template / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_template / self.__wavelength * at.height))

                    position_i = int(numpy.floor((radia[i] - op.width_coating) / self.__step))
                    position_f = int(numpy.floor(radia[i] / self.__step))
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

                    position_i = int(numpy.floor(radia[i + 1] / self.__step))
                    position_f = int(numpy.floor((radia[i + 1] - op.width_coating) / self.__step))
                    profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * at.height - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * at.height))

            # including absorption of coating material 
            membrane_transmission = numpy.exp(-1j * (-1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * op.width_coating))

        # Two-Level FZP - stop here refactoring
        if op.zone_plate_type == 3:
            height1 = op.height1_factor * at.height
            height2 = op.height2_factor * at.height

            for i in range(1, self.__n_zones, 2):
                position_i = int(numpy.floor((2 * radia[i - 1] / 3 + radia[i + 1] / 3) / self.__step))
                position_f = int(numpy.floor((radia[i - 1] / 3 + 2 * radia[i + 1] / 3) / self.__step))
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * height1 - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * height1))

                position_i = int(numpy.floor((radia[i - 1] / 3 + 2 * radia[i + 1] / 3) / self.__step))
                position_f = int(numpy.floor((radia[i + 1]) / self.__step))
                profile[position_i:position_f] = numpy.exp(-1j * (-2 * numpy.pi * self.__delta_FZP / self.__wavelength * height2 - 1j * 2 * numpy.pi * self.__beta_FZP / self.__wavelength * height2))

            membrane_transmission = 1

        # Inserting the CS
        # --------------------------------------------------------------------------

        if op.with_central_stop:
            cs_pix = numpy.floor(op.cs_diameter / self.__step)
            profile[0: int(numpy.floor(cs_pix / 2))] = 0

        return profile, membrane_transmission

    ###################################################
    #
    def __get_profile_h(self, profile, r):
        # Recalculation of the position where the initial profile is defined.
        # Originally the profile is defined in position r0, that are linear for all
        # the values of position. Now we need to define the function in a new
        # coordinates that are by r. The next loop is interpolating the values of
        # the profile from the coordinates in r0 to the coordinates in r.
        # The output intensity profiles will be defined in r coordinates.
        r0 = numpy.arange(0, self.__max_radius, self.__step)
        profile_h = numpy.full(self.__n_points, 0j)
        for i in range(0, self.__n_points - 1):
            profile_h[i] = profile[i] + (profile[i + 1] - profile[i]) / (r0[i + 1] - r0[i]) * (r[i] - r0[i])
        profile_h[self.__n_points - 1] = profile[self.__n_points - 1]

        return profile_h

    ###################################################
    #
    def __propagate_multislicing(self, map_int, map_complex, field0, four0, q_max, q, c):
        step_slice            = self.__attributes.height
        store_partial_results = self.__options.store_partial_results

        for n in range(self.__n_slices - 1):
            print("Propagation to slice #: ", n+2)

            proj = numpy.exp(-1j * step_slice * ((2 * numpy.pi * q) ** 2) / (2 * self.__k))

            fun = numpy.multiply(proj, four0)
            field = hankel_transform(fun, q_max, c, multipool=self.__multipool)
            fun = numpy.multiply(field0, field)

            if store_partial_results:
                map_int[1 + n, :] = numpy.multiply(numpy.abs(fun), numpy.abs(fun))
                map_complex[1 + n, :] = fun

            four0 = hankel_transform(fun, self.__max_radius, c, n_zeros=self.__n_zeros, multipool=self.__multipool)

        return four0

    ###################################################
    #
    def __propagate_to_focus(self, map_int, map_complex, four0, q_max, q, c):
        op = self.__options

        if not op.with_order_sorting_aperture:
            self.__propagate_to_distance(map_int, map_complex, self.__focal_distance, four0, q_max, q, c)
        else:
            # Propagation to OSA position and OSA insertion
            # --------------------------------------------------------------------------
            four_OSA = self.__propagate_to_OSA(four0, q_max, q, c)

            # Propagation at the focus position
            # --------------------------------------------------------------------------
            self.__propagate_to_distance(map_int, map_complex, self.__focal_distance - op.osa_position, four_OSA, q_max, q, c)

    ###################################################
    #
    def __propagate_on_range(self, map_int, map_complex, four0, q_max, q, c):
        op = self.__options

        stepz = (op.range_f - op.range_i) / (op.n_z - 1)
        z = (op.range_i + numpy.arange(op.n_z) * stepz)

        if not op.with_order_sorting_aperture:
            for o in range(op.n_z):
                self.__propagate_to_distance(map_int, map_complex, z[o], four0, q_max, q, c, map_index=o)
        else:
            if op.osa_position < op.range_i:
                # Propagation to OSA position and OSA insertion
                # --------------------------------------------------------------------------
                four_OSA = self.__propagate_to_OSA(four0, q_max, q, c)

                # Continue the propagation from OSA on the range
                #--------------------------------------------------------------------------
                for o in range(op.n_z):
                    self.__propagate_to_distance(map_int, map_complex, z[o] - op.osa_position, four_OSA, q_max, q, c, map_index=o)
            else:
                z_before = z[numpy.where(z <= op.osa_position)]
                last_before = len(z_before)

                # Propagation from initial position to last position before OSA
                # ------------------------------------------------------------------
                for o in range(last_before):
                    self.__propagate_to_distance(map_int, map_complex, z_before[o], four0, q_max, q, c, map_index=o)

                # Propagation to OSA position and OSA insertion
                # --------------------------------------------------------------------------
                four_OSA = self.__propagate_to_OSA(four0, q_max, q, c)

                # Continue the propagation from first position after OSA to final position
                #--------------------------------------------------------------------------
                for o in range(last_before, op.n_z):
                    self.__propagate_to_distance(map_int, map_complex, z[o] - op.osa_position, four_OSA, q_max, q, c, map_index=o)

    ###################################################
    #
    def __propagate_to_distance(self, map_int, map_complex, z, four0, q_max, q, c, map_index=0):
        store_partial_results = self.__options.store_partial_results

        print("Propagation to distance: ", z, " m")

        proj = numpy.exp(-1j * z * ((2 * numpy.pi * q) ** 2) / (2 * self.__k))
        fun = numpy.multiply(proj, four0)
        four11 = hankel_transform(fun, q_max, c, multipool=self.__multipool)

        if store_partial_results:
            map_int[map_index + self.__n_slices, :] = numpy.multiply(numpy.abs(four11), numpy.abs(four11))
            map_complex[map_index + self.__n_slices, :] = four11
        else:
            map_int[map_index  :] = numpy.multiply(numpy.abs(four11), numpy.abs(four11))
            map_complex[map_index, :] = four11

    ###################################################
    #
    def __propagate_to_OSA(self, four0, q_max, q, c):
        op = self.__options

        print("Propagation to OSA: ", op.osa_position, " m")
        # Propagation at the OSA position
        # --------------------------------------------------------------------------
        proj_OSA = numpy.exp(-1j * op.osa_position * ((2 * numpy.pi * q) ** 2) / (2 * self.__k))
        fun = numpy.multiply(proj_OSA, four0)
        field_OSA = hankel_transform(fun, q_max, c, multipool=self.__multipool)

        # Inserting OSA
        # --------------------------------------------------------------------------
        OSA_pix = int(numpy.floor(op.osa_diameter / self.__step) - 1)
        field_OSA[int(OSA_pix / 2) + 1:self.__n_points] = 0
        four_OSA = hankel_transform(field_OSA, self.__max_radius, c, multipool=self.__multipool)

        return four_OSA

    ###################################################
    #
    def __calculate_efficiency(self, map_index, map_out, profile_h, r, n_integration_points=0) -> float:
        shape = map_out.shape

        map_d = numpy.full((2 * shape[1], shape[0]), None)
        map_d[:self.__n_points, :] = numpy.flipud(map_out[:, :self.__n_points].T)
        map_d[self.__n_points:2 * self.__n_points, :] = map_out[:, :self.__n_points].T

        if n_integration_points <= 0: n_integration_points = self.__n_points

        I_dens_0 = numpy.zeros(self.__n_points)
        I_dens_0[numpy.where(profile_h != 0.j)] = 1.0
        I_0 = numpy.trapz(numpy.multiply(r, I_dens_0), r)
        I = numpy.trapz(numpy.multiply(r[:n_integration_points + 1], map_d[self.__n_points - 1:(self.__n_points + n_integration_points), map_index]), r[:n_integration_points + 1])

        return numpy.divide(I, I_0)
