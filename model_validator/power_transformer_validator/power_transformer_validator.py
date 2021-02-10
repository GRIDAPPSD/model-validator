# -------------------------------------------------------------------------------
# Copyright (c) 2020, Battelle Memorial Institute All rights reserved.
# Battelle Memorial Institute (hereinafter Battelle) hereby grants permission to any person or entity
# lawfully obtaining a copy of this software and associated documentation files (hereinafter the
# Software) to redistribute and use the Software in source and binary forms, with or without modification.
# Such person or entity may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and may permit others to do so, subject to the following conditions:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the
# following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided with the distribution.
# Other than as used herein, neither the name Battelle Memorial Institute or Battelle may be used in any
# form whatsoever without the express written consent of Battelle.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# General disclaimer for use with OSS licenses
#
# This material was prepared as an account of work sponsored by an agency of the United States Government.
# Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any
# of their employees, nor any jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for
# the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process
# disclosed, or represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer,
# or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United
# States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed
# herein do not necessarily state or reflect those of the United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the
# UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
# -------------------------------------------------------------------------------
"""
Created on Feb 15, 2021

@author: Gary Black, Shiva Poudel
"""""

#from shared.sparql import SPARQLManager

import math
import argparse
import json
import sys
import os
import importlib
import numpy as np
from tabulate import tabulate

from gridappsd import GridAPPSD

global logfile


def diffColor(colorIdx, colorFlag):
    if colorIdx == 0:
        return '\u001b[32m\u25cf\u001b[37m' if colorFlag else '\u25cb'
    elif colorIdx == 1:
        return '\u001b[33m\u25cf\u001b[37m' if colorFlag else '\u25d1'
    else:
        return '\u001b[31m\u25cf\u001b[37m' if colorFlag else '\u25cf'


def diffColorRealIdx(absDiff, perDiff):
    global greenCountReal, yellowCountReal, redCountReal

    if absDiff<1e-3 and perDiff<0.01:
        greenCountReal += 1
        return 0
    elif absDiff>=1e-2 or perDiff>=0.1:
        redCountReal += 1
        return 2
    else:
        yellowCountReal += 1
        return 1


def diffColorImagIdx(absDiff, perDiff):
    global greenCountImag, yellowCountImag, redCountImag

    if absDiff<1e-3 and perDiff<0.01:
        greenCountImag += 1
        return 0
    elif absDiff>=1e-2 or perDiff>=0.1:
        redCountImag += 1
        return 2
    else:
        yellowCountImag += 1
        return 1


def diffPercentReal(YcompValue, YbusValue):
    global minPercentDiffReal, maxPercentDiffReal

    if YbusValue == 0.0:
        return 0.0

    ratio = YcompValue/YbusValue

    if ratio > 1.0:
        percent = 100.0*(ratio - 1.0)
    else:
        percent = 100.0*(1.0 - ratio)

    minPercentDiffReal = min(minPercentDiffReal, percent)
    maxPercentDiffReal = max(maxPercentDiffReal, percent)

    return percent


def diffPercentImag(YcompValue, YbusValue):
    global minPercentDiffImag, maxPercentDiffImag

    if YbusValue == 0.0:
        return 0.0

    ratio = YcompValue/YbusValue

    if ratio > 1.0:
        percent = 100.0*(ratio - 1.0)
    else:
        percent = 100.0*(1.0 - ratio)

    minPercentDiffImag = min(minPercentDiffImag, percent)
    maxPercentDiffImag = max(maxPercentDiffImag, percent)

    return percent


def compareY(pair_b1, pair_b2, YcompValue, Ybus):
    noEntryFlag = False
    if pair_b1 in Ybus and pair_b2 in Ybus[pair_b1]:
        row = pair_b1
        col = pair_b2
        YbusValue = Ybus[row][col]
    elif pair_b2 in Ybus and pair_b1 in Ybus[pair_b2]:
        row = pair_b2
        col = pair_b1
        YbusValue = Ybus[row][col]
    else:
        row = pair_b1
        col = pair_b2
        YbusValue = complex(0.0, 0.0)
        noEntryFlag = True

    print("    between i: " + row + ", and j: " + col, flush=True)
    print("    between i: " + row + ", and j: " + col, file=logfile)

    if noEntryFlag:
        print('        *** WARNING: Entry NOT FOUND for Ybus[' + row + '][' + col + ']', flush=True)
        print('        *** WARNING: Entry NOT FOUND for Ybus[' + row + '][' + col + ']', file=logfile)

    realAbsDiff = abs(YcompValue.real - YbusValue.real)
    realPerDiff = diffPercentReal(YcompValue.real, YbusValue.real)
    realColorIdx = diffColorRealIdx(realAbsDiff, realPerDiff)
    print("        Real Ybus[i,j]:" + "{:13.6f}".format(YbusValue.real) + ", computed:" + "{:13.6f}".format(YcompValue.real) + "  " + diffColor(realColorIdx, True), flush=True)
    print("        Real Ybus[i,j]:" + "{:13.6f}".format(YbusValue.real) + ", computed:" + "{:13.6f}".format(YcompValue.real) + "  " + diffColor(realColorIdx, False), file=logfile)

    imagAbsDiff = abs(YcompValue.imag - YbusValue.imag)
    imagPerDiff = diffPercentImag(YcompValue.imag, YbusValue.imag)
    imagColorIdx = diffColorImagIdx(imagAbsDiff, imagPerDiff)
    print("        Imag Ybus[i,j]:" + "{:13.6f}".format(YbusValue.imag) + ", computed:" + "{:13.6f}".format(YcompValue.imag) + "  " + diffColor(imagColorIdx, True), flush=True)
    print("        Imag Ybus[i,j]:" + "{:13.6f}".format(YbusValue.imag) + ", computed:" + "{:13.6f}".format(YcompValue.imag) + "  " + diffColor(imagColorIdx, False), file=logfile)

    return max(realColorIdx, imagColorIdx)


def validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus):
    print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance validation...', flush=True)
    print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance validation...', file=logfile)

    # return # of lines validated
    line_count = 0

    bindings = sparql_mgr.PerLengthPhaseImpedance_line_configs()
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_configs query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_configs query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', file=logfile)
        return line_count

    Zabc = {}
    for obj in bindings:
        line_config = obj['line_config']['value']
        count = int(obj['count']['value'])
        row = int(obj['row']['value'])
        col = int(obj['col']['value'])
        r_ohm_per_m = float(obj['r_ohm_per_m']['value'])
        x_ohm_per_m = float(obj['x_ohm_per_m']['value'])
        #b_S_per_m = float(obj['b_S_per_m']['value'])
        #print('line_config: ' + line_config + ', count: ' + str(count) + ', row: ' + str(row) + ', col: ' + str(col) + ', r_ohm_per_m: ' + str(r_ohm_per_m) + ', x_ohm_per_m: ' + str(x_ohm_per_m) + ', b_S_per_m: ' + str(b_S_per_m))

        if line_config not in Zabc:
            if count == 1:
                Zabc[line_config] = np.zeros((1,1), dtype=complex)
            elif count == 2:
                Zabc[line_config] = np.zeros((2,2), dtype=complex)
            elif count == 3:
                Zabc[line_config] = np.zeros((3,3), dtype=complex)

        Zabc[line_config][row-1,col-1] = complex(r_ohm_per_m, x_ohm_per_m)
        if row != col:
            Zabc[line_config][col-1,row-1] = complex(r_ohm_per_m, x_ohm_per_m)

    #for line_config in Zabc:
    #    print('Zabc[' + line_config + ']: ' + str(Zabc[line_config]))
    #print('')

    bindings = sparql_mgr.PerLengthPhaseImpedance_line_names()
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', file=logfile)
        return line_count

    # map line_name query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1': '.1', 's2': '.2'}

    global minPercentDiffReal, maxPercentDiffReal
    minPercentDiffReal = sys.float_info.max
    maxPercentDiffReal = -sys.float_info.max
    global minPercentDiffImag, maxPercentDiffImag
    minPercentDiffImag = sys.float_info.max
    maxPercentDiffImag = -sys.float_info.max
    global greenCountReal, yellowCountReal, redCountReal
    greenCountReal = yellowCountReal = redCountReal = 0
    global greenCountImag, yellowCountImag, redCountImag
    greenCountImag = yellowCountImag = redCountImag = 0
    global greenCount, yellowCount, redCount
    greenCount = yellowCount = redCount = 0

    last_name = ''
    for obj in bindings:
        line_name = obj['line_name']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        line_config = obj['line_config']['value']
        phase = obj['phase']['value']
        #print('line_name: ' + line_name + ', line_config: ' + line_config + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', phase: ' + phase)

        if line_name!=last_name and line_config in Zabc:
            print("\nValidating PerLengthPhaseImpedance line_name: " + line_name, flush=True)
            print("\nValidating PerLengthPhaseImpedance line_name: " + line_name, file=logfile)

            last_name = line_name
            line_idx = 0
            line_count += 1

            # multiply by scalar length
            lenZabc = Zabc[line_config] * length
            # invert the matrix
            invZabc = np.linalg.inv(lenZabc)
            # test if the inverse * original = identity
            #identityTest = np.dot(lenZabc, invZabc)
            #print('identity test for ' + line_name + ': ' + str(identityTest))
            # negate the matrix and assign it to Ycomp
            Ycomp = invZabc * -1

        # we now have the negated inverted matrix for comparison
        line_idx += 1

        if Ycomp.size == 1:
            # do comparisons now
            colorIdx = compareY(bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ycomp[0,0], Ybus)

            if colorIdx == 0:
                greenCount += 1
            elif colorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1

        elif Ycomp.size == 4:
            if line_idx == 1:
                pair_i0b1 = bus1 + ybusPhaseIdx[phase]
                pair_i0b2 = bus2 + ybusPhaseIdx[phase]
            else:
                pair_i1b1 = bus1 + ybusPhaseIdx[phase]
                pair_i1b2 = bus2 + ybusPhaseIdx[phase]

                # do comparisons now
                colorIdx00 = compareY(pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                colorIdx10 = compareY(pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                colorIdx11 = compareY(pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                colorIdx = max(colorIdx00, colorIdx10, colorIdx11)

                if colorIdx == 0:
                    greenCount += 1
                elif colorIdx == 1:
                    yellowCount += 1
                else:
                    redCount += 1

        elif Ycomp.size == 9:
            if line_idx == 1:
                pair_i0b1 = bus1 + ybusPhaseIdx[phase]
                pair_i0b2 = bus2 + ybusPhaseIdx[phase]
            elif line_idx == 2:
                pair_i1b1 = bus1 + ybusPhaseIdx[phase]
                pair_i1b2 = bus2 + ybusPhaseIdx[phase]
            else:
                pair_i2b1 = bus1 + ybusPhaseIdx[phase]
                pair_i2b2 = bus2 + ybusPhaseIdx[phase]

                # do comparisons now
                colorIdx00 = compareY(pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                colorIdx10 = compareY(pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                colorIdx11 = compareY(pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                colorIdx20 = compareY(pair_i2b1, pair_i0b2, Ycomp[2,0], Ybus)
                colorIdx21 = compareY(pair_i2b1, pair_i1b2, Ycomp[2,1], Ybus)
                colorIdx22 = compareY(pair_i2b1, pair_i2b2, Ycomp[2,2], Ybus)
                colorIdx = max(colorIdx00, colorIdx10, colorIdx11, colorIdx20, colorIdx21, colorIdx22)

                if colorIdx == 0:
                    greenCount += 1
                elif colorIdx == 1:
                    yellowCount += 1
                else:
                    redCount += 1

    print("\nSummary for PerLengthPhaseImpedance lines:", flush=True)
    print("\nSummary for PerLengthPhaseImpedance lines:", file=logfile)

    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nReal \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountReal), flush=True)
    print("\nReal \u25cb  count: " + str(greenCountReal), file=logfile)
    print("Real \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountReal), flush=True)
    print("Real \u25d1  count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31m\u25cf\u001b[37m  count: " + str(redCountReal), flush=True)
    print("Real \u25cf  count: " + str(redCountReal), file=logfile)

    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), file=logfile)

    print("\nImag \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountImag), flush=True)
    print("\nImag \u25cb  count: " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountImag), flush=True)
    print("Imag \u25d1  count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31m\u25cf\u001b[37m  count: " + str(redCountImag), flush=True)
    print("Imag \u25cf  count: " + str(redCountImag), file=logfile)

    print("\nFinished validation for PerLengthPhaseImpedance lines", flush=True)
    print("\nFinished validation for PerLengthPhaseImpedance lines", file=logfile)

    return line_count


def validate_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus):
    print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd validation...', flush=True)
    print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd validation...', file=logfile)

    # return # of xfmrs validated
    xfmrs_count = 0

    bindings = sparql_mgr.PowerTransformerEnd_xfmr_impedances()
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_impedances query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_impedances query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', flush=True)
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', file=logfile)
        return xfmrs_count

    Mesh_x_ohm = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        #from_end = int(obj['from_end']['value'])
        #to_end = int(obj['to_end']['value'])
        #r_ohm = float(obj['r_ohm']['value'])
        Mesh_x_ohm[xfmr_name] = float(obj['mesh_x_ohm']['value'])
        #print('xfmr_name: ' + xfmr_name + ', from_end: ' + str(from_end) + ', to_end: ' + str(to_end) + ', r_ohm: ' + str(r_ohm) + ', mesh_x_ohm: ' + str(Mesh_x_ohm[xfmr_name]))

    # Admittances query not currently used
    #bindings = sparql_mgr.PowerTransformerEnd_xfmr_admittances()
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', file=logfile)
    #print(bindings, file=logfile)

    #if len(bindings) == 0:
    #    print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', flush=True)
    #    print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', file=logfile)
    #    return xfmrs_count

    bindings = sparql_mgr.PowerTransformerEnd_xfmr_names()
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', flush=True)
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd: NO TRANSFORMER MATCHES', file=logfile)
        return xfmrs_count

    Bus = {}
    Connection = {}
    RatedS = {}
    RatedU = {}
    R_ohm = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        #vector_group = obj['vector_group']['value']
        end_number = int(obj['end_number']['value'])
        if xfmr_name not in Bus:
            Bus[xfmr_name] = {}
            Connection[xfmr_name] = {}
            RatedS[xfmr_name] = {}
            RatedU[xfmr_name] = {}
            R_ohm[xfmr_name] = {}

        Bus[xfmr_name][end_number] = obj['bus']['value']
        #base_voltage = int(obj['base_voltage']['value'])
        Connection[xfmr_name][end_number] = obj['connection']['value']
        RatedS[xfmr_name][end_number] = int(obj['ratedS']['value'])
        RatedU[xfmr_name][end_number] = int(obj['ratedU']['value'])
        R_ohm[xfmr_name][end_number] = float(obj['r_ohm']['value'])
        #angle = int(obj['angle']['value'])
        #grounded = obj['grounded']['value']
        #r_ground = obj['r_ground']['value']
        #x_ground = obj['x_ground']['value']
        #print('xfmr_name: ' + xfmr_name + ', vector_group: ' + vector_group + ', end_number: ' + str(end_number) + ', bus: ' + bus + ', base_voltage: ' + str(base_voltage) + ', connection: ' + connection + ', ratedS: ' + str(ratedS) + ', ratedU: ' + str(ratedU) + ', r_ohm: ' + str(r_ohm) + ', angle: ' + str(angle) + ', grounded: ' + grounded)
        print('xfmr_name: ' + xfmr_name + ', end_number: ' + str(end_number) + ', bus: ' + Bus[xfmr_name][end_number] + ', connection: ' + Connection[xfmr_name][end_number] + ', ratedS: ' + str(RatedS[xfmr_name][end_number]) + ', ratedU: ' + str(RatedU[xfmr_name][end_number]) + ', r_ohm: ' + str(R_ohm[xfmr_name][end_number]))

    # initialize B upfront because it's constant
    B = np.zeros((6,3))
    B[0,0] = B[2,1] = B[4,2] =  1.0
    B[1,0] = B[3,1] = B[5,2] = -1.0
    #print(B)

    # initialize Y and D matrices, also constant, used to set A later
    Y1 = np.zeros((4,12))
    Y1[0,0] = Y1[1,4] = Y1[2,8] = Y1[3,1] = Y1[3,5] = Y1[3,9] = 1.0
    Y2 = np.zeros((4,12))
    Y2[0,2] = Y2[1,6] = Y2[2,10] = Y2[3,3] = Y2[3,7] = Y2[3,11] = 1.0
    D1 = np.zeros((4,12))
    D1[0,0] = D1[0,9] = D1[1,1] = D1[1,4] = D1[2,5] = D1[2,8] = 1.0
    D2 = np.zeros((4,12))
    D2[0,2] = D2[0,11] = D2[1,3] = D2[1,6] = D2[2,7] = D2[2,10] = 1.0

    global minPercentDiffReal, maxPercentDiffReal
    minPercentDiffReal = sys.float_info.max
    maxPercentDiffReal = -sys.float_info.max
    global minPercentDiffImag, maxPercentDiffImag
    minPercentDiffImag = sys.float_info.max
    maxPercentDiffImag = -sys.float_info.max
    global greenCountReal, yellowCountReal, redCountReal
    greenCountReal = yellowCountReal = redCountReal = 0
    global greenCountImag, yellowCountImag, redCountImag
    greenCountImag = yellowCountImag = redCountImag = 0
    global greenCount, yellowCount, redCount
    greenCount = yellowCount = redCount = 0

    for xfmr_name in Bus:
        # Note that division is always floating point in Python 3 even if
        # operands are integer
        zBaseP = (RatedU[xfmr_name][1]*RatedU[xfmr_name][1])/RatedS[xfmr_name][1]
        #zBaseS = (RatedU[xfmr_name][2]*RatedU[xfmr_name][2])/RatedS[xfmr_name][2]
        r_ohm_pu = R_ohm[xfmr_name][1]/zBaseP
        mesh_x_ohm_pu = Mesh_x_ohm[xfmr_name]/zBaseP
        zsc_1V = complex(2.0*r_ohm_pu, mesh_x_ohm_pu) * (3.0/RatedS[xfmr_name][1])
        print('xfmr_name: ' + xfmr_name + ', zBaseP: ' + str(zBaseP) + ', r_ohm_pu: ' + str(r_ohm_pu) + ', mesh_x_ohm_pu: ' + str(mesh_x_ohm_pu) + ', zsc_1V: ' + str(zsc_1V))

        # initialize ZB
        ZB = np.zeros((3,3), dtype=complex)
        ZB[0,0] = ZB[1,1] = ZB[2,2] = zsc_1V
        #print(ZB)

        # set both Vp/Vs for N and top/bottom for A
        if Connection[xfmr_name][1] == 'Y':
            Vp = RatedU[xfmr_name][1]/math.sqrt(3.0)
            top = Y1
        else:
            Vp = RatedU[xfmr_name][1]
            top = D1

        if Connection[xfmr_name][2] == 'Y':
            Vs = RatedU[xfmr_name][2]/math.sqrt(3.0)
            bottom = Y2
        else:
            Vs = RatedU[xfmr_name][2]
            bottom = D2

        # initialize N
        N = np.zeros((12,6))
        N[0,0] = N[4,2] = N[8,4] =   1.0/Vp
        N[1,0] = N[5,2] = N[9,4] =  -1.0/Vp
        N[2,1] = N[6,3] = N[10,5] =  1.0/Vs
        N[3,1] = N[7,3] = N[11,5] = -1.0/Vs
        #print(N)

        # initialize A
        A = np.vstack((top, bottom))
        #print(A)

        # compute Ycomp = A x N x B x inv(ZB) x B' x N' x A'
        # there are lots of ways to break this up including not at all, but
        # here's one way that keeps it from looking overly complex
        ANB = np.matmul(np.matmul(A, N), B)
        ANB_invZB = np.matmul(ANB, np.linalg.inv(ZB))
        ANB_invZB_Bp = np.matmul(ANB_invZB, np.transpose(B))
        ANB_invZB_BpNp = np.matmul(ANB_invZB_Bp, np.transpose(N))
        Ycomp = np.matmul(ANB_invZB_BpNp, np.transpose(A))
        print(Ycomp)

        print('Need to compare ' + Bus[xfmr_name][1] + '.1 to ' + Bus[xfmr_name][2] + '.1')
        print('Need to compare ' + Bus[xfmr_name][1] + '.2 to ' + Bus[xfmr_name][2] + '.2')
        print('Need to compare ' + Bus[xfmr_name][1] + '.3 to ' + Bus[xfmr_name][2] + '.3' + '\n')
        #colorIdx11 = compareY(Bus[xfmr_name][1]+'.1', Bus[xfmr_name][2]+'.1', Yprim[0,0], Ybus)
        #colorIdx = max(colorIdx11, colorIdx22, colorIdx33)

        xfmrs_count += 1


    return xfmrs_count


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    print("\nPOWER_TRANSFORMER_VALIDATOR starting!!!----------------------------------------------------")
    print("\nPOWER_TRANSFORMER_VALIDATOR starting!!!----------------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    ysparse,nodelist = sparql_mgr.ybus_export()

    idx = 1
    nodes = {}
    for obj in nodelist:
        nodes[idx] = obj.strip('\"')
        idx += 1
    #print(nodes)

    Ybus = {}
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row':
            continue
        if nodes[int(items[0])] not in Ybus:
            Ybus[nodes[int(items[0])]] = {}
        Ybus[nodes[int(items[0])]][nodes[int(items[1])]] = complex(float(items[2]), float(items[3]))
    #print(Ybus)

    # list of lists for the tabular report
    report = []

    PerLengthPhaseImpedance_lines = validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus)

    if PerLengthPhaseImpedance_lines > 0:
        count = greenCount + yellowCount + redCount
        VI = float(count - redCount)/float(count)
        report.append(["PerLengthPhaseImpedance", PerLengthPhaseImpedance_lines, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
    else:
        report.append(["PerLengthPhaseImpedance", PerLengthPhaseImpedance_lines])
    PowerTransformerEnd_xfmrs = validate_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus)

    print('\n', flush=True)
    print(tabulate(report, headers=["Line Type", "# Lines", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
    print('\n', file=logfile)
    print(tabulate(report, headers=["Line Type", "# Lines", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

    print('\nPOWER_TRANSFORMER_VALIDATOR DONE!!!', flush=True)
    print('\nPOWER_TRANSFORMER_VALIDATOR DONE!!!', file=logfile)



def _main():
    # for loading modules
    if (os.path.isdir('shared')):
        sys.path.append('.')
    elif (os.path.isdir('../shared')):
        sys.path.append('..')

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    log_file = open('line_model_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)    


if __name__ == "__main__":
    _main()
