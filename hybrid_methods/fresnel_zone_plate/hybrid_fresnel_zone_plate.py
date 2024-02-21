#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2024. UChicago Argonne, LLC. This software was produced       #
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
from abc import abstractmethod
from typing import Any, Tuple

from hybrid_methods.fresnel_zone_plate.fresnel_zone_plate_simulator import FZPCalculationInputParameters, FZPAttributes, FZPSimulatorOptions, FresnelZonePlateSimulator, FZPCalculationResult

class HybridFresnelZonePlate():
    def __init__(self,
                 options: FZPSimulatorOptions,
                 attributes: FZPAttributes):
        self._simulator = FresnelZonePlateSimulator(options, attributes)

    def run_fzp_hybrid_method(self, input_parameters: FZPCalculationInputParameters, **kwargs) -> Tuple[Any, FZPCalculationResult]:
        zone_plate_beam, energy_in_KeV = self._get_zone_plate_aperture_beam(self._simulator.attributes, **kwargs)

        self._simulator.initialize(energy_in_KeV, input_parameters)

        calculation_result = self._simulator.simulate()

        self._simulator.set_divergence_distribution(calculation_result=calculation_result,
                                                    last_index=input_parameters.profile_last_index,
                                                    increase_resolution=input_parameters.increase_resolution,
                                                    increase_points=input_parameters.increase_points)

        output_beam  = self._get_ideal_lens_beam(zone_plate_beam, **kwargs)

        self._apply_convolution_to_rays(output_beam, calculation_result, **kwargs)

        return output_beam, calculation_result

    def get_energy_in_KeV(self):  return self._simulator.energy_in_KeV
    def get_n_zones(self):        return self._simulator.n_zones
    def get_zp_focal_distance(self): return self._simulator.zp_focal_distance
    def get_zp_image_distance(self): return self._simulator.zp_image_distance

    @abstractmethod
    def _get_zone_plate_aperture_beam(self, attributes: FZPAttributes, **kwargs) -> Tuple[Any, float]: raise NotImplementedError
    @abstractmethod
    def _get_ideal_lens_beam(self, zone_plate_beam: Any, **kwargs) -> Any: raise NotImplementedError
    @abstractmethod
    def _apply_convolution_to_rays(self, output_beam: Any, calculation_result: FZPCalculationResult, **kwargs): raise NotImplementedError

