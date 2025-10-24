#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
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
from typing import List, Any
import sys
import numpy
import time
import random
import copy

from scipy.signal import convolve2d

from srxraylib.util.histograms import get_fwhm, get_sigma
from srxraylib.util.inverse_method_sampler import Sampler2D
from srxraylib.util.random_distributions import Distribution2D, Grid2D, distribution_from_grid
from srxraylib.util.custom_distribution import CustomDistribution

import scipy.constants as codata

m2ev = codata.c * codata.h / codata.e

class Distribution:
    POSITION = 0
    DIVERGENCE = 1

from hybrid_methods.util.srw import SRW_INSTALLED

if SRW_INSTALLED:
    from hybrid_methods.util.srw import (
        array as srw_array, pi, srwl, srwl_uti_save_intens_ascii,
        SRWLMagFldC, SRWLMagFldH, SRWLMagFldU, SRWLStokes, SRWLWfr, SRWLPartBeam, SRWLOptD, SRWLOptC
    )

class HybridUndulatorListener:
    def signal_event(self, event: str, data: Any): raise NotImplementedError()
    def receive_messages(self, messages: List[str], data: Any): raise NotImplementedError()


class DefaultHybridUndulatorListener(HybridUndulatorListener):
    def __init__(self): pass
    
    def signal_event(self, event: str, data: Any):
        print("Received event: {}".format(event))
        print("Received data: {}".format(data))

    def receive_messages(self, messages: List[str], data: Any):
        print("Received messages: {}".format(messages))
        print("Received data: {}".format(data))

class HybridUndulatorInputParameters:
    def __init__(self,
                 number_of_rays                                              = 5000,
                 seed                                                        = 6775431,
                 coherent_beam                                               = 0,
                 phase_diff                                                  = 0.0,
                 polarization_degree                                         = 1.0,
                 use_harmonic                                                = 0,
                 harmonic_number                                             = 1,
                 energy                                                      = 10000.0,
                 energy_to                                                   = 10100.0,
                 energy_points                                               = 10,
                 number_of_periods                                           = 184,
                 undulator_period                                            = 0.025,
                 Kv                                                          = 0.857,
                 Kh                                                          = 0,
                 Bh                                                          = 0.0,
                 Bv                                                          = 1.5,
                 magnetic_field_from                                         = 0,
                 initial_phase_vertical                                      = 0.0,
                 initial_phase_horizontal                                    = 0.0,
                 symmetry_vs_longitudinal_position_vertical                  = 1,
                 symmetry_vs_longitudinal_position_horizontal                = 0,
                 horizontal_central_position                                 = 0.0,
                 vertical_central_position                                   = 0.0,
                 longitudinal_central_position                               = 0.0,
                 electron_energy_in_GeV                                      = 6.0,
                 electron_energy_spread                                      = 1.35e-3,
                 ring_current                                                = 0.2,
                 electron_beam_size_h                                        = 1.45e-05,
                 electron_beam_size_v                                        = 2.8e-06,
                 electron_beam_divergence_h                                  = 2.9e-06,
                 electron_beam_divergence_v                                  = 1.5e-06,
                 type_of_initialization                                      = 0,
                 use_stokes                                                  = 1,
                 auto_expand                                                 = 0,
                 auto_expand_rays                                            = 0,
                 source_dimension_wf_h_slit_gap                              = 0.0015,
                 source_dimension_wf_v_slit_gap                              = 0.0015,
                 source_dimension_wf_h_slit_c                                = 0.0,
                 source_dimension_wf_v_slit_c                                =  0.0,
                 source_dimension_wf_h_slit_points                           = 301,
                 source_dimension_wf_v_slit_points                           = 301,
                 source_dimension_wf_distance                                = 28.0,
                 horizontal_range_modification_factor_at_resizing            = 0.5,
                 horizontal_resolution_modification_factor_at_resizing       = 5.0,
                 vertical_range_modification_factor_at_resizing              = 0.5,
                 vertical_resolution_modification_factor_at_resizing         = 5.0,
                 waist_position_calculation                                  = 0,
                 waist_position                                              = 0.0,
                 waist_position_auto                                         = 0,
                 waist_position_auto_h                                       = 0.0,
                 waist_position_auto_v                                       = 0.0,
                 waist_back_propagation_parameters                           = 1,
                 waist_horizontal_range_modification_factor_at_resizing      = 0.5,
                 waist_horizontal_resolution_modification_factor_at_resizing = 5.0,
                 waist_vertical_range_modification_factor_at_resizing        = 0.5,
                 waist_vertical_resolution_modification_factor_at_resizing   = 5.0,
                 which_waist                                                 = 2,
                 number_of_waist_fit_points                                  = 10,
                 degree_of_waist_fit                                         = 3,
                 use_sigma_or_fwhm                                           = 0,
                 waist_position_user_defined                                 = 0.0,
                 kind_of_sampler                                             = 1,
                 save_srw_result                                             = 0,
                 source_dimension_srw_file                                   = "intensity_source_dimension.dat",
                 angular_distribution_srw_file                               = "intensity_angular_distribution.dat",
                 x_positions_file                                            = "x_positions.txt",
                 z_positions_file                                            = "z_positions.txt",
                 x_positions_factor                                          = 0.01,
                 z_positions_factor                                          = 0.01,
                 x_divergences_file                                          = "x_divergences.txt",
                 z_divergences_file                                          = "z_divergences.txt",
                 x_divergences_factor                                        = 1.0,
                 z_divergences_factor                                        = 1.0,
                 combine_strategy                                            = 0,
                 distribution_source                                         = 0,
                 energy_step                                                 = None,
                 power_step                                                  = None,
                 compute_power                                               = False,
                 integrated_flux                                             = None,
                 power_density                                               = None,
                 ):
        self.number_of_rays                                              = number_of_rays
        self.seed                                                        = seed
        self.coherent_beam                                               = coherent_beam
        self.phase_diff                                                  = phase_diff
        self.polarization_degree                                         = polarization_degree
        self.use_harmonic                                                = use_harmonic
        self.harmonic_number                                             = harmonic_number
        self.energy                                                      = energy
        self.energy_to                                                   = energy_to
        self.energy_points                                               = energy_points
        self.number_of_periods                                           = number_of_periods
        self.undulator_period                                            = undulator_period
        self.Kv                                                          = Kv
        self.Kh                                                          = Kh
        self.Bh                                                          = Bh
        self.Bv                                                          = Bv
        self.magnetic_field_from                                         = magnetic_field_from
        self.initial_phase_vertical                                      = initial_phase_vertical
        self.initial_phase_horizontal                                    = initial_phase_horizontal
        self.symmetry_vs_longitudinal_position_vertical                  = symmetry_vs_longitudinal_position_vertical
        self.symmetry_vs_longitudinal_position_horizontal                = symmetry_vs_longitudinal_position_horizontal
        self.horizontal_central_position                                 = horizontal_central_position
        self.vertical_central_position                                   = vertical_central_position
        self.longitudinal_central_position                               = longitudinal_central_position
        self.electron_energy_in_GeV                                      = electron_energy_in_GeV
        self.electron_energy_spread                                      = electron_energy_spread
        self.ring_current                                                = ring_current
        self.electron_beam_size_h                                        = electron_beam_size_h
        self.electron_beam_size_v                                        = electron_beam_size_v
        self.electron_beam_divergence_h                                  = electron_beam_divergence_h
        self.electron_beam_divergence_v                                  = electron_beam_divergence_v
        self.type_of_initialization                                      = type_of_initialization
        self.use_stokes                                                  = use_stokes
        self.auto_expand                                                 = auto_expand
        self.auto_expand_rays                                            = auto_expand_rays
        self.source_dimension_wf_h_slit_gap                              = source_dimension_wf_h_slit_gap
        self.source_dimension_wf_v_slit_gap                              = source_dimension_wf_v_slit_gap
        self.source_dimension_wf_h_slit_c                                = source_dimension_wf_h_slit_c
        self.source_dimension_wf_v_slit_c                                = source_dimension_wf_v_slit_c
        self.source_dimension_wf_h_slit_points                           = source_dimension_wf_h_slit_points
        self.source_dimension_wf_v_slit_points                           = source_dimension_wf_v_slit_points
        self.source_dimension_wf_distance                                = source_dimension_wf_distance
        self.horizontal_range_modification_factor_at_resizing            = horizontal_range_modification_factor_at_resizing
        self.horizontal_resolution_modification_factor_at_resizing       = horizontal_resolution_modification_factor_at_resizing
        self.vertical_range_modification_factor_at_resizing              = vertical_range_modification_factor_at_resizing
        self.vertical_resolution_modification_factor_at_resizing         = vertical_resolution_modification_factor_at_resizing
        self.waist_position_calculation                                  = waist_position_calculation
        self.waist_position                                              = waist_position
        self.waist_position_auto                                         = waist_position_auto
        self.waist_position_auto_h                                       = waist_position_auto_h
        self.waist_position_auto_v                                       = waist_position_auto_v
        self.waist_back_propagation_parameters                           = waist_back_propagation_parameters
        self.waist_horizontal_range_modification_factor_at_resizing      = waist_horizontal_range_modification_factor_at_resizing
        self.waist_horizontal_resolution_modification_factor_at_resizing = waist_horizontal_resolution_modification_factor_at_resizing
        self.waist_vertical_range_modification_factor_at_resizing        = waist_vertical_range_modification_factor_at_resizing
        self.waist_vertical_resolution_modification_factor_at_resizing   = waist_vertical_resolution_modification_factor_at_resizing
        self.which_waist                                                 = which_waist
        self.number_of_waist_fit_points                                  = number_of_waist_fit_points
        self.degree_of_waist_fit                                         = degree_of_waist_fit
        self.use_sigma_or_fwhm                                           = use_sigma_or_fwhm
        self.waist_position_user_defined                                 = waist_position_user_defined
        self.kind_of_sampler                                             = kind_of_sampler
        self.save_srw_result                                             = save_srw_result
        self.source_dimension_srw_file                                   = source_dimension_srw_file
        self.angular_distribution_srw_file                               = angular_distribution_srw_file
        self.x_positions_file                                            = x_positions_file
        self.z_positions_file                                            = z_positions_file
        self.x_positions_factor                                          = x_positions_factor
        self.z_positions_factor                                          = z_positions_factor
        self.x_divergences_file                                          = x_divergences_file
        self.z_divergences_file                                          = z_divergences_file
        self.x_divergences_factor                                        = x_divergences_factor
        self.z_divergences_factor                                        = z_divergences_factor
        self.combine_strategy                                            = combine_strategy
        self.distribution_source                                         = distribution_source
        self.energy_step                                                 = energy_step
        self.power_step                                                  = power_step
        self.compute_power                                               = compute_power
        self.integrated_flux                                             = integrated_flux
        self.power_density                                               = power_density


class HybridUndulatorOutputParameters:
    def __init__(self,
                 moment_x                  = 0.0,
                 moment_y                  = 0.0,
                 moment_z                  = 0.0,
                 moment_xp                 = 0.0,
                 moment_yp                 = 0.0,
                 cumulated_energies        = None,
                 cumulated_integrated_flux = None,
                 cumulated_power_density   = None,
                 cumulated_power           = None,
                 initial_flux              = 0.0,
                 total_power               = 0.0
    ):
        self.moment_x                  = moment_x
        self.moment_y                  = moment_y
        self.moment_z                  = moment_z
        self.moment_xp                 = moment_xp
        self.moment_yp                 = moment_yp
        self.cumulated_energies        = cumulated_energies
        self.cumulated_integrated_flux = cumulated_integrated_flux
        self.cumulated_power_density   = cumulated_power_density
        self.cumulated_power           = cumulated_power
        self.initial_flux              = initial_flux
        self.total_power               = total_power

class HybridUndulatorCalculator:

    def __init__(self, 
                 input_parameters: HybridUndulatorInputParameters, 
                 listener: HybridUndulatorListener = None):
        if not SRW_INSTALLED: raise ImportError("Please install SRW to use the hybrid methods")

        self.__input_parameters  = input_parameters
        self.__listener          = listener if not listener is None else DefaultHybridUndulatorListener()
        self.__output_parameters = None

    def get_input_parameters(self) -> HybridUndulatorInputParameters: return self.__input_parameters
    def get_output_parameters(self) -> HybridUndulatorOutputParameters: return self.__output_parameters

    def run_hybrid_undulator_simulation(self, do_cumulated_calculations = False):
        self.__output_parameters = HybridUndulatorOutputParameters()

        self.__listener.receive_messages(["Generating Initial Ray-Tracing beam"], data={"progress": 10})
        output_beam = self._generate_initial_beam()
        self.__listener.receive_messages(["Starting Wave-Optics Calculations"], data={"progress": 20})
        total_power = self.__apply_undulator_distributions_calculation(output_beam , do_cumulated_calculations)
        self.__output_parameters.total_power = total_power

        return output_beam


    # ABSTRACT METHODS ###############################################
    #
    def _generate_initial_beam(self): raise NotImplementedError
    def _get_rays_from_beam(self, output_beam: Any): raise NotImplementedError
    def _get_k_from_energy(self, energies: numpy.ndarray): raise NotImplementedError
    def _retrace_output_beam(self, output_beam: Any, distance: float): raise NotImplementedError
    #
    # ###############################################################
    
    ''' -> to the widget
    def __check_fields(widget):
    widget.number_of_rays = congruence.checkPositiveNumber(widget.number_of_rays, "Number of rays")
    widget.seed = congruence.checkPositiveNumber(widget.seed, "Seed")

    if widget.use_harmonic == 0:
        if widget.distribution_source != 0: raise Exception("Harmonic Energy can be computed only for explicit SRW Calculation")

        widget.harmonic_number = congruence.checkStrictlyPositiveNumber(widget.harmonic_number, "Harmonic Number")
    elif widget.use_harmonic == 2:
        if widget.distribution_source != 0: raise Exception("Energy Range can be computed only for explicit SRW Calculation")

        widget.energy = congruence.checkStrictlyPositiveNumber(widget.energy, "Photon Energy From")
        widget.energy_to = congruence.checkStrictlyPositiveNumber(widget.energy_to, "Photon Energy To")
        widget.energy_points = congruence.checkStrictlyPositiveNumber(widget.energy_points, "Nr. Energy Values")
        congruence.checkGreaterThan(widget.energy_to, widget.energy, "Photon Energy To", "Photon Energy From")
    else:
        widget.energy = congruence.checkStrictlyPositiveNumber(widget.energy, "Photon Energy")

    if widget.optimize_source > 0:
        widget.max_number_of_rejected_rays = congruence.checkPositiveNumber(widget.max_number_of_rejected_rays,
                                                                            "Max number of rejected rays")
        congruence.checkFile(widget.optimize_file_name)
    '''
    
    
    def __apply_undulator_distributions_calculation(self, output_beam, do_cumulated_calculations):
        input_parameters: HybridUndulatorInputParameters   = self.__input_parameters
        output_parameters: HybridUndulatorOutputParameters = self.__output_parameters
        listener: HybridUndulatorListener                  = self.__listener

        if input_parameters.use_harmonic == 2: # range
            energy_points = int(input_parameters.energy_points)

            x_array = numpy.full(energy_points, None)
            z_array = numpy.full(energy_points, None)

            intensity_source_dimension_array = numpy.full(energy_points, None)

            x_first_array = numpy.full(energy_points, None)
            z_first_array = numpy.full(energy_points, None)

            intensity_angular_distribution_array = numpy.full(energy_points, None)
            energies = numpy.linspace(input_parameters.energy, input_parameters.energy_to, energy_points)

            total_power = None
            delta_e = energies[1] - energies[0]

            if input_parameters.use_stokes != 1: raise ValueError("multi energy calculation is possible with calculation with Stokes only")

            listener.receive_messages(["Computing integrated flux from Radiation Stokes Parameters"], data={"progress":25})

            flux_from_stokes = _get_integrated_flux_from_stokes(input_parameters, output_parameters, energies)

            integrated_flux_array = numpy.divide(flux_from_stokes * delta_e, 0.001 * energies)  # switch to BW = energy step
            nr_rays_array         = input_parameters.number_of_rays * integrated_flux_array / numpy.sum(integrated_flux_array)
            prog_bars             = numpy.linspace(30, 80, energy_points)

            current_seed = time.time() if input_parameters.seed == 0 else input_parameters.seed
            random.seed(current_seed)

            output_rays = self._get_rays_from_beam(output_beam)

            first_index = 0
            last_index  = 0
            for energy, i in zip(energies, range(energy_points)):
                last_index = min(first_index + int(nr_rays_array[i]), len(output_rays))
                rays       = output_rays[first_index:last_index]

                listener.receive_messages([f"Running SRW for energy: {energy}"], data={})

                x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution, integrated_flux, _ = _run_SRW_calculation(input_parameters,
                                                                                                                                              output_parameters,
                                                                                                                                              energy,
                                                                                                                                              flux_from_stokes=float(flux_from_stokes[i]),
                                                                                                                                              do_cumulated_calculations=False)

                x_array[i]       = x
                z_array[i]       = z
                x_first_array[i] = x_first
                z_first_array[i] = z_first
                intensity_source_dimension_array[i]     = intensity_source_dimension
                intensity_angular_distribution_array[i] = intensity_angular_distribution

                rays[:, 10] = self._get_k_from_energy(numpy.random.uniform(energy, energy + delta_e, size=len(rays)))

                listener.receive_messages([f"Applying new Spatial/Angular Distribution for energy: {energy}"], data={})
                

                _generate_user_defined_distribution_from_srw(rays=rays,
                                                              coord_x=x_array[i],
                                                              coord_z=z_array[i],
                                                              intensity=intensity_source_dimension_array[i],
                                                              distribution_type=Distribution.POSITION,
                                                              kind_of_sampler=input_parameters.kind_of_sampler,
                                                              seed=current_seed + 1)
                _generate_user_defined_distribution_from_srw(rays=rays,
                                                              coord_x=x_first_array[i],
                                                              coord_z=z_first_array[i],
                                                              intensity=intensity_angular_distribution_array[i],
                                                              distribution_type=Distribution.DIVERGENCE,
                                                              kind_of_sampler=input_parameters.kind_of_sampler,
                                                              seed=current_seed + 2)

                listener.receive_messages([], data={"progress": prog_bars[i]})
                
                first_index = last_index
                current_seed += 2

            if not last_index == len(output_rays):
                excluded_rays = output_rays[last_index:]
                excluded_rays[:, 9] = -999
            
            output_parameters.initial_flux = None
        else:
            integrated_flux = None

            energy = input_parameters.energy if input_parameters.use_harmonic == 1 else _resonance_energy(input_parameters, 
                                                                                                          harmonic=input_parameters.harmonic_number)

            if input_parameters.distribution_source == 0:
                listener.receive_messages(["Running SRW"], data={})


                if input_parameters.use_stokes == 1: flux_from_stokes = _get_integrated_flux_from_stokes(input_parameters, output_parameters,[energy])[0]
                else:                                flux_from_stokes = 0.0

                x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution, integrated_flux, total_power = _run_SRW_calculation(input_parameters,
                                                                                                                                                        output_parameters,
                                                                                                                                                        energy,
                                                                                                                                                        flux_from_stokes=flux_from_stokes,
                                                                                                                                                        do_cumulated_calculations=do_cumulated_calculations)
            elif input_parameters.distribution_source == 1:
                listener.receive_messages(["Loading SRW files"], data={})

                x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution = _load_SRW_files(input_parameters)
                total_power = None
            elif input_parameters.distribution_source == 2:  # ASCII FILES
                listener.receive_messages(["Loading Ascii files"], data={})

                x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution = _load_ASCII_files(input_parameters)
                total_power = None
            else:
                raise ValueError(f"Distribution source not valid {input_parameters.distribution_source}")
            
            output_parameters.initial_flux = integrated_flux

            listener.receive_messages(["Applying new Spatial/Angular Distribution"], data={"progress": 50})
            
            _generate_user_defined_distribution_from_srw(rays=output_beam._beam.rays,
                                                          coord_x=x,
                                                          coord_z=z,
                                                          intensity=intensity_source_dimension,
                                                          distribution_type=Distribution.POSITION,
                                                          kind_of_sampler=input_parameters.kind_of_sampler,
                                                          seed=time.time() if input_parameters.seed == 0 else input_parameters.seed + 1)

            listener.receive_messages([], data={"progress": 70})

            _generate_user_defined_distribution_from_srw(rays=output_beam._beam.rays,
                                                          coord_x=x_first,
                                                          coord_z=z_first,
                                                          intensity=intensity_angular_distribution,
                                                          distribution_type=Distribution.DIVERGENCE,
                                                          kind_of_sampler=input_parameters.kind_of_sampler,
                                                          seed=time.time() if input_parameters.seed == 0 else input_parameters.seed + 2)

        if input_parameters.distribution_source == 0 and _is_canted_undulator(input_parameters) and output_parameters.waist_position != 0.0:
            self._retrace_output_beam(output_beam, -output_parameters.waist_position)
            
        return total_power

####################################################################################
# SRW CALCULATION
####################################################################################


def __get_source_slit_data(input_parameters: HybridUndulatorInputParameters, direction="b"):
    if input_parameters.auto_expand == 1:
        source_dimension_wf_h_slit_points = int(numpy.ceil(0.55 * input_parameters.source_dimension_wf_h_slit_points) * 2)
        source_dimension_wf_v_slit_points = int(numpy.ceil(0.55 * input_parameters.source_dimension_wf_v_slit_points) * 2)
        source_dimension_wf_h_slit_gap    = input_parameters.source_dimension_wf_h_slit_gap * 1.1
        source_dimension_wf_v_slit_gap    = input_parameters.source_dimension_wf_v_slit_gap * 1.1
    else:
        source_dimension_wf_h_slit_points = input_parameters.source_dimension_wf_h_slit_points
        source_dimension_wf_v_slit_points = input_parameters.source_dimension_wf_v_slit_points
        source_dimension_wf_h_slit_gap    = input_parameters.source_dimension_wf_h_slit_gap
        source_dimension_wf_v_slit_gap    = input_parameters.source_dimension_wf_v_slit_gap

    if direction == "h":
        return source_dimension_wf_h_slit_points, source_dimension_wf_h_slit_gap
    elif direction == "v":
        return source_dimension_wf_v_slit_points, source_dimension_wf_v_slit_gap
    else:
        return source_dimension_wf_h_slit_points, source_dimension_wf_h_slit_gap, source_dimension_wf_v_slit_points, source_dimension_wf_v_slit_gap

def __set_which_waist(input_parameters: HybridUndulatorInputParameters):
    if input_parameters.which_waist == 0:  # horizontal
        input_parameters.waist_position_auto = round(input_parameters.waist_position_auto_h, 4)
    elif input_parameters.which_waist == 1:  # vertical
        input_parameters.waist_position_auto = round(input_parameters.waist_position_auto_v, 4)
    else:  # middle point
        input_parameters.waist_position_auto = round(0.5 * (input_parameters.waist_position_auto_h + input_parameters.waist_position_auto_v), 4)

def __gamma(input_parameters: HybridUndulatorInputParameters):
    return 1e9 * input_parameters.electron_energy_in_GeV / (codata.m_e * codata.c ** 2 / codata.e)

def _resonance_energy(input_parameters: HybridUndulatorInputParameters, theta_x=0.0, theta_z=0.0, harmonic=1):
    g = __gamma(input_parameters)

    wavelength = ((input_parameters.undulator_period / (2.0 * g ** 2)) *
                  (1 + input_parameters.Kv ** 2 / 2.0 + input_parameters.Kh ** 2 / 2.0 +
                   g ** 2 * (theta_x ** 2 + theta_z ** 2))) / harmonic

    return m2ev / wavelength

def __get_default_initial_z(input_parameters: HybridUndulatorInputParameters):
    return input_parameters.longitudinal_central_position - 0.5 * input_parameters.undulator_period * (input_parameters.number_of_periods + 8)  # initial Longitudinal Coordinate (set before the ID)

def _is_canted_undulator(input_parameters: HybridUndulatorInputParameters):
    return input_parameters.longitudinal_central_position != 0.0

def __get_minimum_propagation_distance(input_parameters: HybridUndulatorInputParameters):
    return round(__get_source_length(input_parameters) * 1.01, 6)

def __get_source_length(input_parameters: HybridUndulatorInputParameters):
    return input_parameters.undulator_period * input_parameters.number_of_periods

def __magnetic_field_from_K(input_parameters: HybridUndulatorInputParameters):
    Bv = input_parameters.Kv * 2 * pi * codata.m_e * codata.c / (codata.e * input_parameters.undulator_period)
    Bh = input_parameters.Kh * 2 * pi * codata.m_e * codata.c / (codata.e * input_parameters.undulator_period)

    return Bv, Bh

def __create_undulator(input_parameters: HybridUndulatorInputParameters, no_shift=False):
    # ***********Undulator
    if input_parameters.magnetic_field_from == 0:
        By, Bx = __magnetic_field_from_K(input_parameters)  # Peak Vertical field [T]
    else:
        By = input_parameters.Bv
        Bx = input_parameters.Bh

    symmetry_vs_longitudinal_position_horizontal = 1 if input_parameters.symmetry_vs_longitudinal_position_horizontal == 0 else -1
    symmetry_vs_longitudinal_position_vertical = 1 if input_parameters.symmetry_vs_longitudinal_position_vertical == 0 else -1

    if Bx == 0.0:
        und = SRWLMagFldU(_arHarm=[SRWLMagFldH(_n=1,
                                               _h_or_v='v',
                                               _B=By,
                                               _ph=input_parameters.initial_phase_vertical,
                                               _a=1)],
                          _per=input_parameters.undulator_period,
                          _nPer=input_parameters.number_of_periods)  # Planar Undulator
    else:
        und = SRWLMagFldU(_arHarm=[SRWLMagFldH(_n=1,
                                               _h_or_v='h',
                                               _B=Bx,
                                               _ph=input_parameters.initial_phase_horizontal,
                                               _s=symmetry_vs_longitudinal_position_horizontal,
                                               _a=1),
                                   SRWLMagFldH(_n=1,
                                               _h_or_v='v',
                                               _B=By,
                                               _ph=input_parameters.initial_phase_vertical,
                                               _s=symmetry_vs_longitudinal_position_vertical,
                                               _a=1)],
                          _per=input_parameters.undulator_period,
                          _nPer=input_parameters.number_of_periods)  # Planar Undulator

    if no_shift:
        magFldCnt = SRWLMagFldC(_arMagFld=[und],
                                _arXc=srw_array('d', [0.0]),
                                _arYc=srw_array('d', [0.0]),
                                _arZc=srw_array('d', [0.0]))  # Container of all Field Elements
    else:
        magFldCnt = SRWLMagFldC(_arMagFld=[und],
                                _arXc=srw_array('d', [input_parameters.horizontal_central_position]),
                                _arYc=srw_array('d', [input_parameters.vertical_central_position]),
                                _arZc=srw_array('d', [input_parameters.longitudinal_central_position]))  # Container of all Field Elements

    return magFldCnt

def __create_electron_beam(input_parameters: HybridUndulatorInputParameters,
                           output_parameters: HybridUndulatorOutputParameters,
                           distribution_type=Distribution.DIVERGENCE,
                           position=0.0,
                           use_nominal=False):
    # ***********Electron Beam
    elecBeam = SRWLPartBeam()

    electron_beam_size_h = input_parameters.electron_beam_size_h if use_nominal else \
        numpy.sqrt(input_parameters.electron_beam_size_h ** 2 + (numpy.abs(input_parameters.longitudinal_central_position + position) * numpy.tan(input_parameters.electron_beam_divergence_h)) ** 2)
    electron_beam_size_v = input_parameters.electron_beam_size_v if use_nominal else \
        numpy.sqrt(input_parameters.electron_beam_size_v ** 2 + (numpy.abs(input_parameters.longitudinal_central_position + position) * numpy.tan(input_parameters.electron_beam_divergence_v)) ** 2)

    if input_parameters.type_of_initialization == 0:  # zero
        output_parameters.moment_x  = 0.0
        output_parameters.moment_y  = 0.0
        output_parameters.moment_z  = __get_default_initial_z(input_parameters)
        output_parameters.moment_xp = 0.0
        output_parameters.moment_yp = 0.0
    elif input_parameters.type_of_initialization == 2:  # sampled
        output_parameters.moment_x  = numpy.random.normal(0.0, electron_beam_size_h)
        output_parameters.moment_y  = numpy.random.normal(0.0, electron_beam_size_v)
        output_parameters.moment_z  = __get_default_initial_z(input_parameters)
        output_parameters.moment_xp = numpy.random.normal(0.0, input_parameters.electron_beam_divergence_h)
        output_parameters.moment_yp = numpy.random.normal(0.0, input_parameters.electron_beam_divergence_v)

    elecBeam.partStatMom1.x  = output_parameters.moment_x
    elecBeam.partStatMom1.y  = output_parameters.moment_y
    elecBeam.partStatMom1.z  = output_parameters.moment_z
    elecBeam.partStatMom1.xp = output_parameters.moment_xp
    elecBeam.partStatMom1.yp = output_parameters.moment_yp
    elecBeam.partStatMom1.gamma = __gamma(input_parameters)

    elecBeam.Iavg = input_parameters.ring_current  # Average Current [A]

    # 2nd order statistical moments
    elecBeam.arStatMom2[0] = 0 if distribution_type == Distribution.DIVERGENCE else (electron_beam_size_h) ** 2  # <(x-x0)^2>
    elecBeam.arStatMom2[1] = 0
    elecBeam.arStatMom2[2] = (input_parameters.electron_beam_divergence_h) ** 2  # <(x'-x'0)^2>
    elecBeam.arStatMom2[3] = 0 if distribution_type == Distribution.DIVERGENCE else (electron_beam_size_v) ** 2  # <(y-y0)^2>
    elecBeam.arStatMom2[4] = 0
    elecBeam.arStatMom2[5] = (input_parameters.electron_beam_divergence_v) ** 2  # <(y'-y'0)^2>
    # energy spread
    elecBeam.arStatMom2[10] = (input_parameters.electron_energy_spread) ** 2  # <(E-E0)^2>/E0^2

    return elecBeam


def __create_initial_wavefront_mesh(input_parameters: HybridUndulatorInputParameters, elecBeam, energy):
    # ****************** Initial Wavefront
    wfr = SRWLWfr()  # For intensity distribution at fixed photon energy

    source_dimension_wf_h_slit_points, \
    source_dimension_wf_h_slit_gap, \
    source_dimension_wf_v_slit_points, \
    source_dimension_wf_v_slit_gap = __get_source_slit_data(input_parameters, direction="b")

    wfr.allocate(1, source_dimension_wf_h_slit_points, source_dimension_wf_v_slit_points)  # Numbers of points vs Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.zStart = input_parameters.source_dimension_wf_distance + input_parameters.longitudinal_central_position  # Longitudinal Position [m] from Center of Straight Section at which SR has to be calculated
    wfr.mesh.eStart = energy  # Initial Photon Energy [eV]
    wfr.mesh.eFin = wfr.mesh.eStart  # Final Photon Energy [eV]

    wfr.mesh.xStart = -0.5 * source_dimension_wf_h_slit_gap + input_parameters.source_dimension_wf_h_slit_c  # Initial Horizontal Position [m]
    wfr.mesh.xFin   =  0.5 * source_dimension_wf_h_slit_gap + input_parameters.source_dimension_wf_h_slit_c  # 0.00015 #Final Horizontal Position [m]
    wfr.mesh.yStart = -0.5 * source_dimension_wf_v_slit_gap + input_parameters.source_dimension_wf_v_slit_c  # Initial Vertical Position [m]
    wfr.mesh.yFin   =  0.5 * source_dimension_wf_v_slit_gap + input_parameters.source_dimension_wf_v_slit_c  # 0.00015 #Final Vertical Position [m]

    wfr.partBeam = elecBeam

    return wfr


def __get_calculation_precision_settings(input_parameters: HybridUndulatorInputParameters, no_shift=False):
    # ***********Precision Parameters for SR calculation
    meth = 1  # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    relPrec = 0.01  # relative precision

    if (input_parameters.longitudinal_central_position < 0 and not no_shift):
        zStartInteg = input_parameters.longitudinal_central_position - ((0.5 * input_parameters.number_of_periods + 8) * input_parameters.undulator_period)  # longitudinal position to start integration (effective if < zEndInteg)
        zEndInteg = input_parameters.longitudinal_central_position + ((0.5 * input_parameters.number_of_periods + 8) * input_parameters.undulator_period)  # longitudinal position to finish integration (effective if > zStartInteg)
    else:
        zStartInteg = 0  # longitudinal position to start integration (effective if < zEndInteg)
        zEndInteg = 0  # longitudinal position to finish integration (effective if > zStartInteg)

    npTraj = 100000  # Number of points for trajectory calculation
    useTermin = 1  # Use "terminating terms" (i.e. asymptotic expansions at zStartInteg and zEndInteg) or not (1 or 0 respectively)
    # This is the convergence parameter. Higher is more accurate but slower!!
    sampFactNxNyForProp = 0.0  # 0.6 #sampling factor for adjusting nx, ny (effective if > 0)

    return [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp]


def __calculate_automatic_waste_position(input_parameters: HybridUndulatorInputParameters,
                                         output_parameters: HybridUndulatorOutputParameters,
                                         energy):
    magFldCnt     = __create_undulator(input_parameters, no_shift=True)
    arPrecParSpec = __get_calculation_precision_settings(input_parameters, no_shift=True)

    undulator_length = input_parameters.number_of_periods * input_parameters.undulator_period
    wavelength       = (codata.h * codata.c / codata.e) / energy

    gauss_sigma_ph  = numpy.sqrt(2 * wavelength * undulator_length) / (2 * numpy.pi)
    gauss_sigmap_ph = numpy.sqrt(wavelength / (2 * undulator_length))

    positions     = numpy.linspace(start=-0.5 * undulator_length, stop=0.5 * undulator_length, num=input_parameters.number_of_waist_fit_points)
    sizes_e_x     = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_e_y     = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_ph_x    = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_ph_y    = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_ph_an_x = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_ph_an_y = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_tot_x   = numpy.zeros(input_parameters.number_of_waist_fit_points)
    sizes_tot_y   = numpy.zeros(input_parameters.number_of_waist_fit_points)

    for i in range(input_parameters.number_of_waist_fit_points):
        position = float(positions[i])

        elecBeam    = __create_electron_beam(input_parameters, distribution_type=Distribution.POSITION, position=position, use_nominal=False)
        elecBeam_Ph = __create_electron_beam(input_parameters, distribution_type=Distribution.POSITION, use_nominal=True)
        wfr         = __create_initial_wavefront_mesh(input_parameters, elecBeam_Ph, energy)
        optBLSouDim = __create_beamline_source_dimension(input_parameters,
                                                         back_position=(input_parameters.source_dimension_wf_distance + input_parameters.longitudinal_central_position - position),
                                                         waist_calculation=input_parameters.waist_back_propagation_parameters == 1)

        srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecParSpec)
        srwl.PropagElecField(wfr, optBLSouDim)

        arI = srw_array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)  # SINGLE ELECTRON!

        x, y, intensity_distribution = __transform_srw_array(arI, wfr.mesh)

        def get_size(position, coord, intensity_distribution, projection_axis, ebeam_index):
            sigma_e  = numpy.sqrt(elecBeam.arStatMom2[ebeam_index])
            histo    = numpy.sum(intensity_distribution, axis=projection_axis)
            sigma    = get_sigma(histo, coord) if input_parameters.use_sigma_or_fwhm == 0 else get_fwhm(histo, coord)[0] / 2.355
            sigma_an = numpy.sqrt(gauss_sigma_ph ** 2 + (position * numpy.tan(gauss_sigmap_ph)) ** 2)

            if numpy.isnan(sigma): sigma = 0.0

            return sigma_e, sigma, sigma_an, numpy.sqrt(sigma ** 2 + sigma_e ** 2)

        sizes_e_x[i], sizes_ph_x[i], sizes_ph_an_x[i], sizes_tot_x[i] = get_size(position, x, intensity_distribution, 1, 0)
        sizes_e_y[i], sizes_ph_y[i], sizes_ph_an_y[i], sizes_tot_y[i] = get_size(position, y, intensity_distribution, 0, 3)

    ''' -> to widget
    def plot(widget, direction, positions, sizes_e, sizes_ph, size_ph_an, sizes_tot, waist_position, waist_size):
        widget.waist_axes[direction].clear()
        widget.waist_axes[direction].set_title(("Horizontal" if direction == 0 else "Vertical") + " Direction\n" +
                                               "Source size: " + str(round(waist_size * 1e6, 2)) + " " + r'$\mu$' + "m \n" +
                                               "at " + str(round(waist_position * 1e3, 1)) + " mm from the ID center")

        widget.waist_axes[direction].plot(positions * 1e3, sizes_e * 1e6, label='electron', color='g')
        widget.waist_axes[direction].plot(positions * 1e3, sizes_ph * 1e6, label='photon', color='b')
        widget.waist_axes[direction].plot(positions * 1e3, size_ph_an * 1e6, '--', label='photon (analytical)', color='b')
        widget.waist_axes[direction].plot(positions * 1e3, sizes_tot * 1e6, label='total', color='r')
        widget.waist_axes[direction].plot([waist_position * 1e3], [waist_size * 1e6], 'bo', label="waist")
        widget.waist_axes[direction].set_xlabel("Position relative to ID center [mm]")
        widget.waist_axes[direction].set_ylabel("Sigma [um]")
        widget.waist_axes[direction].legend()
    '''

    def get_minimum(positions, sizes):
        coeffiecients = numpy.polyfit(positions, sizes, deg=input_parameters.degree_of_waist_fit)
        p = numpy.poly1d(coeffiecients)
        bounds = [positions[0], positions[-1]]

        critical_points = numpy.array(bounds + [x for x in p.deriv().r if x.imag == 0 and bounds[0] < x.real < bounds[1]])
        critical_sizes = p(critical_points)

        minimum_value = numpy.inf
        minimum_position = numpy.nan

        for i in range(len(critical_points)):
            if critical_sizes[i] <= minimum_value:
                minimum_value = critical_sizes[i]
                minimum_position = critical_points[i]

        return minimum_position, minimum_value

    waist_position_x, waist_size_x = get_minimum(positions, sizes_tot_x)
    waist_position_y, waist_size_y = get_minimum(positions, sizes_tot_y)

    # TO OUTPUT data
    '''
    if do_plot:
        plot(widget, 0, positions, sizes_e_x, sizes_ph_x, sizes_ph_an_x, sizes_tot_x, waist_position_x, waist_size_x)
        plot(widget, 1, positions, sizes_e_y, sizes_ph_y, sizes_ph_an_y, sizes_tot_y, waist_position_y, waist_size_y)

        try:
            widget.waist_figure.draw()
        except ValueError as e:
            if "Image size of " in str(e):
                pass
            else:
                raise e
    '''

    return waist_position_x, waist_position_y


def __create_beamline_source_dimension(input_parameters: HybridUndulatorInputParameters, back_position=0.0, waist_calculation=False):
    # ***************** Optical Elements and Propagation Parameters

    opDrift = SRWLOptD(-back_position)  # back to waist position
    if not waist_calculation:
        ppDrift = [0, 0, 1., 1, 0,
                   input_parameters.horizontal_range_modification_factor_at_resizing,
                   input_parameters.horizontal_resolution_modification_factor_at_resizing,
                   input_parameters.vertical_range_modification_factor_at_resizing,
                   input_parameters.vertical_resolution_modification_factor_at_resizing,
                   0, 0, 0]
    else:
        ppDrift = [0, 0, 1., 1, 0,
                   input_parameters.waist_horizontal_range_modification_factor_at_resizing,
                   input_parameters.waist_horizontal_resolution_modification_factor_at_resizing,
                   input_parameters.waist_vertical_range_modification_factor_at_resizing,
                   input_parameters.waist_vertical_resolution_modification_factor_at_resizing,
                   0, 0, 0]

    return SRWLOptC([opDrift], [ppDrift])


def __transform_srw_array(output_array, mesh):
    h_array = numpy.linspace(mesh.xStart, mesh.xFin, mesh.nx)
    v_array = numpy.linspace(mesh.yStart, mesh.yFin, mesh.ny)

    intensity_array = numpy.zeros((h_array.size, v_array.size))

    tot_len = int(mesh.ny * mesh.nx)
    len_output_array = len(output_array)

    if len_output_array > tot_len:
        output_array = numpy.array(output_array[0:tot_len])
    elif len_output_array < tot_len:
        aux_array = srw_array('d', [0] * len_output_array)
        for i in range(len_output_array): aux_array[i] = output_array[i]
        output_array = numpy.array(aux_array)
    else:
        output_array = numpy.array(output_array)

    output_array = output_array.reshape(mesh.ny, mesh.nx)

    for ix in range(mesh.nx):
        for iy in range(mesh.ny):
            intensity_array[ix, iy] = output_array[iy, ix]

    intensity_array[numpy.where(numpy.isnan(intensity_array))] = 0.0

    return h_array, v_array, intensity_array


def __calculate_waist_position(input_parameters: HybridUndulatorInputParameters,
                               output_parameters : HybridUndulatorOutputParameters,
                               energy):
    if input_parameters.distribution_source == 0:  # SRW calculation
        if _is_canted_undulator(input_parameters):
            if input_parameters.waist_position_calculation == 0:  # None
                input_parameters.waist_position = 0.0
            elif input_parameters.waist_position_calculation == 1:  # Automatic
                if input_parameters.use_harmonic == 2: raise ValueError("Automatic calculation of the waist position for canted undulator is not allowed when Photon Energy Setting: Range")
                if input_parameters.compute_power: raise ValueError("Automatic calculation of the waist position for canted undulator is not allowed while running a thermal load loop")

                input_parameters.waist_position_auto_h, input_parameters.waist_position_auto_v = __calculate_automatic_waste_position(input_parameters, output_parameters, energy)

                __set_which_waist(input_parameters)

                output_parameters.waist_position = input_parameters.waist_position_auto

            elif input_parameters.waist_position_calculation == 2:  # User Defined
                output_parameters.waist_position = input_parameters.waist_position_user_defined
        else:
            output_parameters.waist_position = 0.0
    else:
        output_parameters.waist_position = 0.0

def _get_integrated_flux_from_stokes(input_parameters: HybridUndulatorInputParameters,
                                     output_parameters: HybridUndulatorOutputParameters,
                                     energies):
    eStart = energies[0]
    eFin = energies[-1]
    ne = len(energies)

    magFldCnt = __create_undulator(input_parameters)
    elecBeam  = __create_electron_beam(input_parameters, distribution_type=Distribution.DIVERGENCE, position=output_parameters.waist_position)
    wfr       = __create_initial_wavefront_mesh(input_parameters, elecBeam, energies[0])

    h_max = int(2.5 * eFin / _resonance_energy(input_parameters, harmonic=1))

    arPrecF = [0] * 5  # for spectral flux vs photon energy
    arPrecF[0] = 1  # initial UR harmonic to take into account
    arPrecF[1] = h_max  # final UR harmonic to take into account
    arPrecF[2] = 1.5  # longitudinal integration precision parameter
    arPrecF[3] = 1.5  # azimuthal integration precision parameter
    arPrecF[4] = 1  # calculate flux (1) or flux per unit surface (2)

    stkF = SRWLStokes()  # for spectral flux vs photon energy
    stkF.allocate(ne, 1, 1)  # numbers of points vs photon energy, horizontal and vertical positions
    stkF.mesh.zStart = input_parameters.source_dimension_wf_distance  # longitudinal position [m] at which UR has to be calculated
    stkF.mesh.eStart = eStart  # initial photon energy [eV]
    stkF.mesh.eFin = eFin  # final photon energy [eV]
    stkF.mesh.xStart = wfr.mesh.xStart  # initial horizontal position [m]
    stkF.mesh.xFin = wfr.mesh.xFin  # final horizontal position [m]
    stkF.mesh.yStart = wfr.mesh.yStart  # initial vertical position [m]
    stkF.mesh.yFin = wfr.mesh.yFin  # final vertical position [m]

    srwl.CalcStokesUR(stkF, elecBeam, magFldCnt.arMagFld[0], arPrecF)

    return numpy.array(stkF.arS[0:ne])


def _run_SRW_calculation(input_parameters: HybridUndulatorInputParameters,
                          output_parameters: HybridUndulatorOutputParameters,
                          energy, flux_from_stokes=0.0, do_cumulated_calculations=False):
    __calculate_waist_position(input_parameters, output_parameters, energy)

    magFldCnt = __create_undulator(input_parameters)
    elecBeam  = __create_electron_beam(input_parameters, distribution_type=Distribution.DIVERGENCE, position=output_parameters.waist_position)
    wfr       = __create_initial_wavefront_mesh(input_parameters, elecBeam, energy)

    arPrecParSpec = __get_calculation_precision_settings(input_parameters)

    # 1 calculate intensity distribution ME convoluted for dimension size
    srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecParSpec)

    arI = srw_array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
    srwl.CalcIntFromElecField(arI, wfr, 6, 1, 3, wfr.mesh.eStart, 0, 0)

    # from radiation at the slit we can calculate Angular Distribution and Power

    x, z, intensity_angular_distribution = __transform_srw_array(arI, wfr.mesh)

    dx = (x[1] - x[0]) * 1e3  # mm for power computations
    dy = (z[1] - z[0]) * 1e3

    if input_parameters.use_stokes == 0:
        integrated_flux = intensity_angular_distribution.sum() * dx * dy  # this is single electron -> no emittance
    else:
        integrated_flux = flux_from_stokes  # recompute the flux with the whole beam (no single electron)

    if input_parameters.compute_power:
        total_power = input_parameters.power_step if input_parameters.power_step > 0 else integrated_flux * (1e3 * input_parameters.energy_step * codata.e)
    else:
        total_power = None

    if input_parameters.compute_power and do_cumulated_calculations:
        current_energy = numpy.ones(1) * energy
        current_integrated_flux = numpy.ones(1) * integrated_flux
        current_power_density = intensity_angular_distribution.copy() * (1e3 * input_parameters.energy_step * codata.e)
        current_power = total_power

        if output_parameters.cumulated_energies is None:
            output_parameters.cumulated_energies = current_energy
            output_parameters.cumulated_integrated_flux = current_integrated_flux
            output_parameters.cumulated_power_density = current_power_density
            output_parameters.cumulated_power = numpy.ones(1) * (current_power)
        else:
            output_parameters.cumulated_energies = numpy.append(output_parameters.cumulated_energies, current_energy)
            output_parameters.cumulated_integrated_flux = numpy.append(output_parameters.cumulated_integrated_flux, current_integrated_flux)
            output_parameters.cumulated_power_density += current_power_density
            output_parameters.cumulated_power = numpy.append(output_parameters.cumulated_power, numpy.ones(1) * (output_parameters.cumulated_power[-1] + current_power))

    distance = input_parameters.source_dimension_wf_distance - output_parameters.waist_position  # relative to the center of the undulator

    x_first = numpy.arctan(x / distance)
    z_first = numpy.arctan(z / distance)

    wfrAngDist = __create_initial_wavefront_mesh(input_parameters, elecBeam, energy)
    wfrAngDist.mesh.xStart = numpy.arctan(wfr.mesh.xStart / distance)
    wfrAngDist.mesh.xFin = numpy.arctan(wfr.mesh.xFin / distance)
    wfrAngDist.mesh.yStart = numpy.arctan(wfr.mesh.yStart / distance)
    wfrAngDist.mesh.yFin = numpy.arctan(wfr.mesh.yFin / distance)

    if input_parameters.save_srw_result == 1: srwl_uti_save_intens_ascii(arI, wfrAngDist.mesh, input_parameters.angular_distribution_srw_file)

    # for source dimension, back propagation to the source position
    elecBeam    = __create_electron_beam(input_parameters, distribution_type=Distribution.POSITION, position=output_parameters.waist_position)
    wfr         = __create_initial_wavefront_mesh(input_parameters, elecBeam, energy)
    optBLSouDim = __create_beamline_source_dimension(input_parameters, back_position=(input_parameters.source_dimension_wf_distance - output_parameters.waist_position))

    srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecParSpec)
    srwl.PropagElecField(wfr, optBLSouDim)

    arI = srw_array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
    srwl.CalcIntFromElecField(arI, wfr, 6, 1, 3, wfr.mesh.eStart, 0, 0)

    if input_parameters.save_srw_result == 1: srwl_uti_save_intens_ascii(arI, wfr.mesh, input_parameters.source_dimension_srw_file)

    x, z, intensity_source_dimension = __transform_srw_array(arI, wfr.mesh)

    return x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution, integrated_flux, total_power

def _generate_user_defined_distribution_from_srw(rays: numpy. ndarray,
                                                 coord_x : numpy. ndarray,
                                                 coord_z : numpy. ndarray,
                                                 intensity : numpy. ndarray,
                                                 distribution_type=Distribution.POSITION,
                                                 kind_of_sampler=1,
                                                 seed=0):
    if kind_of_sampler == 2:
        s2d = Sampler2D(intensity, coord_x, coord_z)

        samples_x, samples_z = s2d.get_n_sampled_points(len(rays))

        if distribution_type == Distribution.POSITION:
            rays[:, 0] = samples_x
            rays[:, 2] = samples_z

        elif distribution_type == Distribution.DIVERGENCE:
            alpha_x = samples_x
            alpha_z = samples_z

            rays[:, 3] = numpy.cos(alpha_z) * numpy.sin(alpha_x)
            rays[:, 4] = numpy.cos(alpha_z) * numpy.cos(alpha_x)
            rays[:, 5] = numpy.sin(alpha_z)
    elif kind_of_sampler == 0:
        pdf = numpy.abs(intensity / numpy.max(intensity))
        pdf /= pdf.sum()

        distribution = CustomDistribution(pdf, seed=seed)

        sampled = distribution(len(rays))

        min_value_x = numpy.min(coord_x)
        step_x = numpy.abs(coord_x[1] - coord_x[0])
        min_value_z = numpy.min(coord_z)
        step_z = numpy.abs(coord_z[1] - coord_z[0])

        if distribution_type == Distribution.POSITION:
            rays[:, 0] = min_value_x + sampled[0, :] * step_x
            rays[:, 2] = min_value_z + sampled[1, :] * step_z

        elif distribution_type == Distribution.DIVERGENCE:
            alpha_x = min_value_x + sampled[0, :] * step_x
            alpha_z = min_value_z + sampled[1, :] * step_z

            rays[:, 3] = numpy.cos(alpha_z) * numpy.sin(alpha_x)
            rays[:, 4] = numpy.cos(alpha_z) * numpy.cos(alpha_x)
            rays[:, 5] = numpy.sin(alpha_z)
    elif kind_of_sampler == 1:
        min_x = numpy.min(coord_x)
        max_x = numpy.max(coord_x)
        delta_x = max_x - min_x

        min_z = numpy.min(coord_z)
        max_z = numpy.max(coord_z)
        delta_z = max_z - min_z

        dim_x = len(coord_x)
        dim_z = len(coord_z)

        grid = Grid2D((dim_x, dim_z))
        grid[..., ...] = intensity.tolist()

        d = Distribution2D(distribution_from_grid(grid, dim_x, dim_z), (0, 0), (dim_x, dim_z))

        samples = d.get_samples(len(rays), seed)

        if distribution_type == Distribution.POSITION:
            rays[:, 0] = min_x + samples[:, 0] * delta_x
            rays[:, 2] = min_z + samples[:, 1] * delta_z

        elif distribution_type == Distribution.DIVERGENCE:
            alpha_x = min_x + samples[:, 0] * delta_x
            alpha_z = min_z + samples[:, 1] * delta_z

            rays[:, 3] = numpy.cos(alpha_z) * numpy.sin(alpha_x)
            rays[:, 4] = numpy.cos(alpha_z) * numpy.cos(alpha_x)
            rays[:, 5] = numpy.sin(alpha_z)
    else:
        raise ValueError("Sampler not recognized")


####################################################################################
# SRW FILES
####################################################################################

def _load_SRW_files(input_parameters: HybridUndulatorInputParameters):
    x, z, intensity_source_dimension = __load_numpy_format(input_parameters.source_dimension_srw_file)
    x_first, z_first, intensity_angular_distribution = __load_numpy_format(input_parameters.angular_distribution_srw_file)

    return x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution


def __file_load(_fname, _read_labels=1):  # FROM SRW
    nLinesHead = 11
    hlp = []

    with open(_fname, 'r') as f:
        for i in range(nLinesHead):
            hlp.append(f.readline())

    ne, nx, ny = [int(hlp[i].replace('#', '').split()[0]) for i in [3, 6, 9]]
    ns = 1
    testStr = hlp[nLinesHead - 1]
    if testStr[0] == '#':
        ns = int(testStr.replace('#', '').split()[0])

    e0, e1, x0, x1, y0, y1 = [float(hlp[i].replace('#', '').split()[0]) for i in [1, 2, 4, 5, 7, 8]]

    data = numpy.squeeze(numpy.loadtxt(_fname, dtype=numpy.float64))  # get data from file (C-aligned flat)

    allrange = e0, e1, ne, x0, x1, nx, y0, y1, ny

    arLabels = ['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Intensity']
    arUnits = ['eV', 'm', 'm', 'ph/s/.1%bw/mm^2']

    if _read_labels:
        arTokens = hlp[0].split(' [')
        arLabels[3] = arTokens[0].replace('#', '')
        arUnits[3] = '';
        if len(arTokens) > 1:
            arUnits[3] = arTokens[1].split('] ')[0]

        for i in range(3):
            arTokens = hlp[i * 3 + 1].split()
            nTokens = len(arTokens)
            nTokensLabel = nTokens - 3
            nTokensLabel_mi_1 = nTokensLabel - 1
            strLabel = ''
            for j in range(nTokensLabel):
                strLabel += arTokens[j + 2]
                if j < nTokensLabel_mi_1: strLabel += ' '
            arLabels[i] = strLabel
            arUnits[i] = arTokens[nTokens - 1].replace('[', '').replace(']', '')

    return data, None, allrange, arLabels, arUnits


def __load_numpy_format(filename):
    data, dump, allrange, arLabels, arUnits = __file_load(filename)

    dim_x = allrange[5]
    dim_y = allrange[8]
    np_array = data.reshape((dim_y, dim_x))
    np_array = np_array.transpose()
    x_coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
    y_coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)

    return x_coordinates, y_coordinates, np_array


####################################################################################
# ASCII FILES
####################################################################################

def _load_ASCII_files(input_parameters: HybridUndulatorInputParameters):
    x_positions = __extract_distribution_from_file(distribution_file_name=input_parameters.x_positions_file)
    z_positions = __extract_distribution_from_file(distribution_file_name=input_parameters.z_positions_file)

    x_positions[:, 0] *= input_parameters.x_positions_factor
    z_positions[:, 0] *= input_parameters.z_positions_factor

    x_divergences = __extract_distribution_from_file(distribution_file_name=input_parameters.x_divergences_file)
    z_divergences = __extract_distribution_from_file(distribution_file_name=input_parameters.z_divergences_file)

    x_divergences[:, 0] *= input_parameters.x_divergences_factor
    z_divergences[:, 0] *= input_parameters.z_divergences_factor

    x, z, intensity_source_dimension                 = __combine_distributions(input_parameters, x_positions,   z_positions)
    x_first, z_first, intensity_angular_distribution = __combine_distributions(input_parameters, x_divergences, z_divergences)

    return x, z, intensity_source_dimension, x_first, z_first, intensity_angular_distribution

def __extract_distribution_from_file(distribution_file_name):
    distribution = []

    try:
        distribution_file = open(distribution_file_name, "r")

        rows = distribution_file.readlines()

        for index in range(0, len(rows)):
            row = rows[index]

            if not row.strip() == "":
                values = row.split()

                if not len(values) == 2: raise Exception("Malformed file, must be: <value> <spaces> <frequency>")

                value = float(values[0].strip())
                frequency = float(values[1].strip())

                distribution.append([value, frequency])

    except Exception as err:
        raise Exception("Problems reading distribution file: {0}".format(err))
    except:
        raise Exception("Unexpected error reading distribution file: ", sys.exc_info()[0])

    return numpy.array(distribution)


def __combine_distributions(input_parameters: HybridUndulatorInputParameters, distribution_x, distribution_y):
    coord_x = distribution_x[:, 0]
    coord_y = distribution_y[:, 0]

    intensity_x = numpy.tile(distribution_x[:, 1], (len(coord_y), 1)).transpose()
    intensity_y = numpy.tile(distribution_y[:, 1], (len(coord_x), 1))

    if   input_parameters.combine_strategy == 0: convoluted_intensity = numpy.sqrt(intensity_x * intensity_y)
    elif input_parameters.combine_strategy == 1: convoluted_intensity = numpy.sqrt(intensity_x ** 2 + intensity_y ** 2)
    elif input_parameters.combine_strategy == 2: convoluted_intensity = convolve2d(intensity_x, intensity_y, boundary='fill', mode='same', fillvalue=0)
    elif input_parameters.combine_strategy == 3: convoluted_intensity = 0.5 * (intensity_x + intensity_y)
    else: raise Exception("Unexpected combination strategy: ", input_parameters.combine_strategy)

    return coord_x, coord_y, convoluted_intensity

