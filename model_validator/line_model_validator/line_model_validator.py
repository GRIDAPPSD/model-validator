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
Created on Dec 16, 2020

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


def fillYsysUnique(bus1, bus2, Yval, Ysys):
    if Yval == 0j:
        return

    if bus1 not in Ysys:
        Ysys[bus1] = {}

    if bus2 in Ysys[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling line model value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling line model value\n', file=logfile)

    Ysys[bus1][bus2] = Yval


def fillYsysUniqueUpper(bus1, bus2, Yval, Ysys):
    if Yval == 0j:
        return

    if bus1 not in Ysys:
        Ysys[bus1] = {}

    if bus2 in Ysys[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling line model value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling line model value\n', file=logfile)

    # extract the node and phase from bus1 and bus2
    node1,phase1 = bus1.split('.')
    node2,phase2 = bus2.split('.')
    bus3 = node1 + '.' + phase2
    bus4 = node2 + '.' + phase1

    if bus3 not in Ysys:
        Ysys[bus3] = {}

    Ysys[bus1][bus2] = Ysys[bus3][bus4] = Yval


def fillYsysAdd(bus1, bus2, Yval, Ysys):
    if Yval == 0j:
        return

    if bus1 not in Ysys:
        Ysys[bus1] = {}

    if bus2 in Ysys[bus1]:
        Ysys[bus1][bus2] += Yval
    else:
        Ysys[bus1][bus2] = Yval

    #if bus1 != bus2:
    #    if bus2 not in Ysys:
    #        Ysys[bus2] = {}

    #    Ysys[bus2][bus1] = Ysys[bus1][bus2]


def fillYsysNoSwap(bus1, bus2, Yval, Ysys):
    #print('fillYsysNoSwap bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
    fillYsysUnique(bus1, bus2, Yval, Ysys)
    fillYsysAdd(bus1, bus1, -Yval, Ysys)
    fillYsysAdd(bus2, bus2, -Yval, Ysys)


def fillYsysSwap(bus1, bus2, Yval, Ysys):
    #print('fillYsysSwap bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
    fillYsysUniqueUpper(bus1, bus2, Yval, Ysys)

    # extract the node and phase from bus1 and bus2
    node1,phase1 = bus1.split('.')
    node2,phase2 = bus2.split('.')

    # mix-and-match nodes and phases for filling Ysys
    fillYsysAdd(bus1, node1+'.'+phase2, -Yval, Ysys)
    fillYsysAdd(node2+'.'+phase1, bus2, -Yval, Ysys)


def validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus, cmpFlag, Ysys):
    if cmpFlag:
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
        if cmpFlag:
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
        if cmpFlag:
            print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', flush=True)
            print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', file=logfile)
        return line_count

    # map line_name query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1': '.1', 's2': '.2'}

    if cmpFlag:
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
            if cmpFlag:
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
            if cmpFlag:
                # do comparisons now
                colorIdx = compareY(bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ycomp[0,0], Ybus)

                if colorIdx == 0:
                    greenCount += 1
                elif colorIdx == 1:
                    yellowCount += 1
                else:
                    redCount += 1
            else:
                fillYsysNoSwap(bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ycomp[0,0], Ysys)

        elif Ycomp.size == 4:
            if line_idx == 1:
                pair_i0b1 = bus1 + ybusPhaseIdx[phase]
                pair_i0b2 = bus2 + ybusPhaseIdx[phase]
            else:
                pair_i1b1 = bus1 + ybusPhaseIdx[phase]
                pair_i1b2 = bus2 + ybusPhaseIdx[phase]

                if cmpFlag:
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
                else:
                    fillYsysNoSwap(pair_i0b1, pair_i0b2, Ycomp[0,0], Ysys)
                    fillYsysSwap(pair_i1b1, pair_i0b2, Ycomp[1,0], Ysys)
                    fillYsysNoSwap(pair_i1b1, pair_i1b2, Ycomp[1,1], Ysys)

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

                if cmpFlag:
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
                else:
                    fillYsysNoSwap(pair_i0b1, pair_i0b2, Ycomp[0,0], Ysys)
                    fillYsysSwap(pair_i1b1, pair_i0b2, Ycomp[1,0], Ysys)
                    fillYsysNoSwap(pair_i1b1, pair_i1b2, Ycomp[1,1], Ysys)
                    fillYsysSwap(pair_i2b1, pair_i0b2, Ycomp[2,0], Ysys)
                    fillYsysSwap(pair_i2b1, pair_i1b2, Ycomp[2,1], Ysys)
                    fillYsysNoSwap(pair_i2b1, pair_i2b2, Ycomp[2,2], Ysys)

    if cmpFlag:
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


def validate_PerLengthSequenceImpedance_lines(sparql_mgr, Ybus, cmpFlag, Ysys):
    if cmpFlag:
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance validation...', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance validation...', file=logfile)

    # return # of lines validated
    line_count = 0

    bindings = sparql_mgr.PerLengthSequenceImpedance_line_configs()
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_configs query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_configs query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', flush=True)
            print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', file=logfile)
        return line_count

    Zabc = {}
    for obj in bindings:
        line_config = obj['line_config']['value']
        r1 = float(obj['r1_ohm_per_m']['value'])
        x1 = float(obj['x1_ohm_per_m']['value'])
        #b1 = float(obj['b1_S_per_m']['value'])
        r0 = float(obj['r0_ohm_per_m']['value'])
        x0 = float(obj['x0_ohm_per_m']['value'])
        #b0 = float(obj['b0_S_per_m']['value'])
        #print('line_config: ' + line_config + ', r1: ' + str(r1) + ', x1: ' + str(x1) + ', b1: ' + str(b1) + ', r0: ' + str(r0) + ', x0: ' + str(x0) + ', b0: ' + str(b0))

        Zs = complex((r0 + 2.0*r1)/3.0, (x0 + 2.0*x1)/3.0)
        Zm = complex((r0 - r1)/3.0, (x0 - x1)/3.0)

        Zabc[line_config] = np.array([(Zs, Zm, Zm), (Zm, Zs, Zm), (Zm, Zm, Zs)], dtype=complex)

    #for line_config in Zabc:
    #    print('Zabc[' + line_config + ']: ' + str(Zabc[line_config]))
    #print('')

    bindings = sparql_mgr.PerLengthSequenceImpedance_line_names()
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', flush=True)
            print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', file=logfile)
        return line_count

    if cmpFlag:
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

    for obj in bindings:
        line_name = obj['line_name']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        line_config = obj['line_config']['value']
        #print('line_name: ' + line_name + ', line_config: ' + line_config + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2)

        if cmpFlag:
            print("\nValidating PerLengthSequenceImpedance line_name: " + line_name, flush=True)
            print("\nValidating PerLengthSequenceImpedance line_name: " + line_name, file=logfile)

        # multiply by scalar length
        lenZabc = Zabc[line_config] * length
        # invert the matrix
        invZabc = np.linalg.inv(lenZabc)
        # test if the inverse * original = identity
        #identityTest = np.dot(lenZabc, invZabc)
        #print('identity test for ' + line_name + ': ' + str(identityTest))
        # negate the matrix and assign it to Ycomp
        Ycomp = invZabc * -1

        line_count += 1

        if cmpFlag:
            # do comparisons now
            colorIdx00 = compareY(bus1+'.1', bus2+'.1', Ycomp[0,0], Ybus)
            colorIdx10 = compareY(bus1+'.2', bus2+'.1', Ycomp[1,0], Ybus)
            colorIdx11 = compareY(bus1+'.2', bus2+'.2', Ycomp[1,1], Ybus)
            colorIdx20 = compareY(bus1+'.3', bus2+'.1', Ycomp[2,0], Ybus)
            colorIdx21 = compareY(bus1+'.3', bus2+'.2', Ycomp[2,1], Ybus)
            colorIdx22 = compareY(bus1+'.3', bus2+'.3', Ycomp[2,2], Ybus)
            colorIdx = max(colorIdx00, colorIdx10, colorIdx11, colorIdx20, colorIdx21, colorIdx22)

            if colorIdx == 0:
                greenCount += 1
            elif colorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1
        else:
            fillYsysNoSwap(bus1+'.1', bus2+'.1', Ycomp[0,0], Ysys)
            fillYsysSwap(bus1+'.2', bus2+'.1', Ycomp[1,0], Ysys)
            fillYsysNoSwap(bus1+'.2', bus2+'.2', Ycomp[1,1], Ysys)
            fillYsysSwap(bus1+'.3', bus2+'.1', Ycomp[2,0], Ysys)
            fillYsysSwap(bus1+'.3', bus2+'.2', Ycomp[2,1], Ysys)
            fillYsysNoSwap(bus1+'.3', bus2+'.3', Ycomp[2,2], Ysys)

    if cmpFlag:
        print("\nSummary for PerLengthSequenceImpedance lines:", flush=True)
        print("\nSummary for PerLengthSequenceImpedance lines:", file=logfile)

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

        print("\nFinished validation for PerLengthSequenceImpedance lines", flush=True)
        print("\nFinished validation for PerLengthSequenceImpedance lines", file=logfile)

    return line_count


def validate_ACLineSegment_lines(sparql_mgr, Ybus, cmpFlag, Ysys):
    if cmpFlag:
        print('\nLINE_MODEL_VALIDATOR ACLineSegment validation...', flush=True)
        print('\nLINE_MODEL_VALIDATOR ACLineSegment validation...', file=logfile)

    # return # of lines validated
    line_count = 0

    bindings = sparql_mgr.ACLineSegment_line_names()
    #print('LINE_MODEL_VALIDATOR ACLineSegment line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR ACLineSegment line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nLINE_MODEL_VALIDATOR ACLineSegment: NO LINE MATCHES', flush=True)
            print('\nLINE_MODEL_VALIDATOR ACLineSegment: NO LINE MATCHES', file=logfile)
        return line_count

    if cmpFlag:
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

    for obj in bindings:
        line_name = obj['line_name']['value']
        #basev = float(obj['basev']['value'])
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        r1 = float(obj['r1_Ohm']['value'])
        x1 = float(obj['x1_Ohm']['value'])
        #b1 = float(obj['b1_S']['value'])
        r0 = float(obj['r0_Ohm']['value'])
        x0 = float(obj['x0_Ohm']['value'])
        #b0 = float(obj['b0_S']['value'])
        #print('line_name: ' + line_name + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', r1: ' + str(r1) + ', x1: ' + str(x1) + ', r0: ' + str(r0) + ', x0: ' + str(x0))

        if cmpFlag:
            print("\nValidating ACLineSegment line_name: " + line_name, flush=True)
            print("\nValidating ACLineSegment line_name: " + line_name, file=logfile)

        Zs = complex((r0 + 2.0*r1)/3.0, (x0 + 2.0*x1)/3.0)
        Zm = complex((r0 - r1)/3.0, (x0 - x1)/3.0)

        Zabc = np.array([(Zs, Zm, Zm), (Zm, Zs, Zm), (Zm, Zm, Zs)], dtype=complex)
        #print('Zabc: ' + str(Zabc) + '\n')

        # multiply by scalar length
        lenZabc = Zabc * length
        #lenZabc = Zabc * length * 3.3 # Kludge to get arount units issue (ft vs. m)
        # invert the matrix
        invZabc = np.linalg.inv(lenZabc)
        # test if the inverse * original = identity
        #identityTest = np.dot(lenZabc, invZabc)
        #print('identity test for ' + line_name + ': ' + str(identityTest))
        # negate the matrix and assign it to Ycomp
        Ycomp = invZabc * -1
        #print('Ycomp: ' + str(Ycomp) + '\n')

        line_count += 1

        if cmpFlag:
            # do comparisons now
            colorIdx00 = compareY(bus1+'.1', bus2+'.1', Ycomp[0,0], Ybus)
            colorIdx10 = compareY(bus1+'.2', bus2+'.1', Ycomp[1,0], Ybus)
            colorIdx11 = compareY(bus1+'.2', bus2+'.2', Ycomp[1,1], Ybus)
            colorIdx20 = compareY(bus1+'.3', bus2+'.1', Ycomp[2,0], Ybus)
            colorIdx21 = compareY(bus1+'.3', bus2+'.2', Ycomp[2,1], Ybus)
            colorIdx22 = compareY(bus1+'.3', bus2+'.3', Ycomp[2,2], Ybus)
            colorIdx = max(colorIdx00, colorIdx10, colorIdx11, colorIdx20, colorIdx21, colorIdx22)

            if colorIdx == 0:
                greenCount += 1
            elif colorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1
        else:
            fillYsysNoSwap(bus1+'.1', bus2+'.1', Ycomp[0,0], Ysys)
            fillYsysSwap(bus1+'.2', bus2+'.1', Ycomp[1,0], Ysys)
            fillYsysNoSwap(bus1+'.2', bus2+'.2', Ycomp[1,1], Ysys)
            fillYsysSwap(bus1+'.3', bus2+'.1', Ycomp[2,0], Ysys)
            fillYsysSwap(bus1+'.3', bus2+'.2', Ycomp[2,1], Ysys)
            fillYsysNoSwap(bus1+'.3', bus2+'.3', Ycomp[2,2], Ysys)

    if cmpFlag:
        print("\nSummary for ACLineSegment lines:", flush=True)
        print("\nSummary for ACLineSegment lines:", file=logfile)

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

        print("\nFinished validation for ACLineSegment lines", flush=True)
        print("\nFinished validation for ACLineSegment lines", file=logfile)

    return line_count


def CN_dist_R(dim, i, j, wire_spacing_info, wire_cn_ts, XCoord, YCoord, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket):
    dist = (CN_diameter_jacket[wire_cn_ts] - CN_strand_radius[wire_cn_ts]*2.0)/2.0
    return dist


def CN_dist_D(dim, i, j, wire_spacing_info, wire_cn_ts, XCoord, YCoord, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket):
    ii,jj = CN_dist_ij[dim][i][j]
    dist = math.sqrt(math.pow(XCoord[wire_spacing_info][ii]-XCoord[wire_spacing_info][jj],2) + math.pow(YCoord[wire_spacing_info][ii]-YCoord[wire_spacing_info][jj],2))
    return dist


def CN_dist_DR(dim, i, j, wire_spacing_info, wire_cn_ts, XCoord, YCoord, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket):
    ii,jj = CN_dist_ij[dim][i][j]
    d = math.sqrt(math.pow(XCoord[wire_spacing_info][ii]-XCoord[wire_spacing_info][jj],2) + math.pow(YCoord[wire_spacing_info][ii]-YCoord[wire_spacing_info][jj],2))
    k = CN_strand_count[wire_cn_ts]
    R = (CN_diameter_jacket[wire_cn_ts] - CN_strand_radius[wire_cn_ts]*2.0)/2.0
    dist = math.pow(math.pow(d,k) - math.pow(R,k), 1.0/k)

    return dist

# global constants for determining Zprim values
u0 = math.pi * 4.0e-7
w = math.pi*2.0 * 60.0
p = 100.0
f = 60.0
Rg = (u0 * w)/8.0
X0 = (u0 * w)/(math.pi*2.0)
Xg = X0 * math.log(658.5 * math.sqrt(p/f))

CN_dist_func = {}
CN_dist_ij = {}

# 2x2 distance function mappings
CN_dist_func[1] = {}
CN_dist_func[1][2] = {}
CN_dist_func[1][2][1] = CN_dist_R

# 4x4 distance function mappings
CN_dist_func[2] = {}
CN_dist_ij[2] = {}
CN_dist_func[2][2] = {}
CN_dist_ij[2][2] = {}
CN_dist_func[2][2][1] = CN_dist_D
CN_dist_ij[2][2][1] = (2,1)
CN_dist_func[2][3] = {}
CN_dist_ij[2][3] = {}
CN_dist_func[2][3][1] = CN_dist_R
CN_dist_func[2][3][2] = CN_dist_DR
CN_dist_ij[2][3][2] = (2,1)
CN_dist_func[2][4] = {}
CN_dist_ij[2][4] = {}
CN_dist_func[2][4][1] = CN_dist_DR
CN_dist_ij[2][4][1] = (2,1)
CN_dist_func[2][4][2] = CN_dist_R
CN_dist_func[2][4][3] = CN_dist_D
CN_dist_ij[2][4][3] = (2,1)

# 6x6 distance function mappings
CN_dist_func[3] = {}
CN_dist_ij[3] = {}
CN_dist_func[3][2] = {}
CN_dist_ij[3][2] = {}
CN_dist_func[3][2][1] = CN_dist_D
CN_dist_ij[3][2][1] = (2,1)
CN_dist_func[3][3] = {}
CN_dist_ij[3][3] = {}
CN_dist_func[3][3][1] = CN_dist_D
CN_dist_ij[3][3][1] = (3,1)
CN_dist_func[3][3][2] = CN_dist_D
CN_dist_ij[3][3][2] = (3,2)
CN_dist_func[3][4] = {}
CN_dist_ij[3][4] = {}
CN_dist_func[3][4][1] = CN_dist_R
CN_dist_func[3][4][2] = CN_dist_DR
CN_dist_ij[3][4][2] = (2,1)
CN_dist_func[3][4][3] = CN_dist_DR
CN_dist_ij[3][4][3] = (3,1)
CN_dist_func[3][5] = {}
CN_dist_ij[3][5] = {}
CN_dist_func[3][5][1] = CN_dist_DR
CN_dist_ij[3][5][1] = (2,1)
CN_dist_func[3][5][2] = CN_dist_R
CN_dist_func[3][5][3] = CN_dist_DR
CN_dist_ij[3][5][3] = (3,2)
CN_dist_func[3][5][4] = CN_dist_D
CN_dist_ij[3][5][4] = (2,1)
CN_dist_func[3][6] = {}
CN_dist_ij[3][6] = {}
CN_dist_func[3][6][1] = CN_dist_DR
CN_dist_ij[3][6][1] = (3,1)
CN_dist_func[3][6][2] = CN_dist_DR
CN_dist_ij[3][6][2] = (3,2)
CN_dist_func[3][6][3] = CN_dist_R
CN_dist_func[3][6][4] = CN_dist_D
CN_dist_ij[3][6][4] = (3,1)
CN_dist_func[3][6][5] = CN_dist_D
CN_dist_ij[3][6][5] = (3,2)


def diagZprim(wireinfo, wire_cn_ts, neutralFlag, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen):
    if wireinfo=='ConcentricNeutralCableInfo' and neutralFlag:
        R = (CN_diameter_jacket[wire_cn_ts] - CN_strand_radius[wire_cn_ts]*2.0)/2.0
        k = CN_strand_count[wire_cn_ts]
        dist = math.pow(CN_strand_gmr[wire_cn_ts]*k*math.pow(R,k-1),1.0/k)
        Zprim = complex(CN_strand_rdc[wire_cn_ts]/k + Rg, X0*math.log(1.0/dist) + Xg)

    # this situation won't normally occur so we are just using neutralFlag to recognize the
    # row 2 diagonal for the shield calculation vs. row1 and row3 that are handled below
    elif wireinfo=='TapeShieldCableInfo' and neutralFlag:
        T = TS_tape_thickness[wire_cn_ts]
        ds = TS_diameter_screen[wire_cn_ts] + 2.0*T
        Rshield = 0.3183 * 2.3718e-8/(ds*T*math.sqrt(50.0/(100.0-20.0)))
        Dss = 0.5*(ds - T)
        Zprim = complex(Rshield + Rg, X0*math.log(1.0/Dss) + Xg)

    else:
        Zprim = complex(R25[wire_cn_ts] + Rg, X0*math.log(1.0/GMR[wire_cn_ts]) + Xg)

    return Zprim


def offDiagZprim(i, j, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen):
    if wireinfo == 'OverheadWireInfo':
        dist = math.sqrt(math.pow(XCoord[wire_spacing_info][i]-XCoord[wire_spacing_info][j],2) + math.pow(YCoord[wire_spacing_info][i]-YCoord[wire_spacing_info][j],2))

    elif wireinfo == 'ConcentricNeutralCableInfo':
        dim = len(XCoord[wire_spacing_info]) # 1=2x2, 2=4x4, 3=6x6
        dist = CN_dist_func[dim][i][j](dim, i, j, wire_spacing_info, wire_cn_ts, XCoord, YCoord, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket)

    elif wireinfo == 'TapeShieldCableInfo':
        # this should only be hit for i==2
        T = TS_tape_thickness[wire_cn_ts]
        ds = TS_diameter_screen[wire_cn_ts] + 2.0*T
        dist = 0.5*(ds - T)

    Zprim = complex(Rg, X0*math.log(1.0/dist) + Xg)

    return Zprim


def validate_WireInfo_and_WireSpacingInfo_lines(sparql_mgr, Ybus, cmpFlag, Ysys):
    if cmpFlag:
        print('\nLINE_MODEL_VALIDATOR WireInfo_and_WireSpacingInfo validation...', flush=True)
        print('\nLINE_MODEL_VALIDATOR WireInfo_and_WireSpacingInfo validation...', file=logfile)

    # return # of lines validated
    line_count = 0

    # WireSpacingInfo query
    bindings = sparql_mgr.WireInfo_spacing()
    #print('LINE_MODEL_VALIDATOR WireInfo spacing query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo spacing query results:', file=logfile)
    #print(bindings, file=logfile)

    XCoord = {}
    YCoord = {}
    for obj in bindings:
        wire_spacing_info = obj['wire_spacing_info']['value']
        cableFlag = obj['cable']['value'].upper() == 'TRUE' # don't depend on lowercase
        #usage = obj['usage']['value']
        #bundle_count = int(obj['bundle_count']['value'])
        #bundle_sep = int(obj['bundle_sep']['value'])
        seq = int(obj['seq']['value'])
        if seq == 1:
            XCoord[wire_spacing_info] = {}
            YCoord[wire_spacing_info] = {}

        XCoord[wire_spacing_info][seq] = float(obj['xCoord']['value'])
        YCoord[wire_spacing_info][seq] = float(obj['yCoord']['value'])
        #print('wire_spacing_info: ' + wire_spacing_info + ', cable: ' + str(cableFlag) + ', seq: ' + str(seq) + ', XCoord: ' + str(XCoord[wire_spacing_info][seq]) + ', YCoord: ' + str(YCoord[wire_spacing_info][seq]))

    # OverheadWireInfo specific query
    bindings = sparql_mgr.WireInfo_overhead()
    #print('LINE_MODEL_VALIDATOR WireInfo overhead query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo overhead query results:', file=logfile)
    #print(bindings, file=logfile)

    GMR = {}
    R25 = {}
    for obj in bindings:
        wire_cn_ts = obj['wire_cn_ts']['value']
        #radius = float(obj['radius']['value'])
        #coreRadius = float(obj['coreRadius']['value'])
        GMR[wire_cn_ts] = float(obj['gmr']['value'])
        #rdc = float(obj['rdc']['value'])
        R25[wire_cn_ts] = float(obj['r25']['value'])
        #r50 = float(obj['r50']['value'])
        #r75 = float(obj['r75']['value'])
        #amps = int(obj['amps']['value'])
        #print('overhead wire_cn_ts: ' + wire_cn_ts + ', gmr: ' + str(GMR[wire_cn_ts]) + ', r25: ' + str(R25[wire_cn_ts]))

    # ConcentricNeutralCableInfo specific query
    bindings = sparql_mgr.WireInfo_concentricNeutral()
    #print('LINE_MODEL_VALIDATOR WireInfo concentricNeutral query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo concentricNeutral query results:', file=logfile)
    #print(bindings, file=logfile)

    CN_diameter_jacket = {}
    CN_strand_count = {}
    CN_strand_radius = {}
    CN_strand_gmr = {}
    CN_strand_rdc = {}
    for obj in bindings:
        wire_cn_ts = obj['wire_cn_ts']['value']
        #radius = float(obj['radius']['value'])
        #coreRadius = float(obj['coreRadius']['value'])
        GMR[wire_cn_ts] = float(obj['gmr']['value'])
        #rdc = float(obj['rdc']['value'])
        R25[wire_cn_ts] = float(obj['r25']['value'])
        #r50 = float(obj['r50']['value'])
        #r75 = float(obj['r75']['value'])
        #amps = int(obj['amps']['value'])
        #insulationFlag = obj['amps']['value'].upper() == 'TRUE'
        #insulation_thickness = float(obj['insulation_thickness']['value'])
        #diameter_core = float(obj['diameter_core']['value'])
        #diameter_insulation = float(obj['diameter_insulation']['value'])
        #diameter_screen = float(obj['diameter_screen']['value'])
        CN_diameter_jacket[wire_cn_ts] = float(obj['diameter_jacket']['value'])
        #diameter_neutral = float(obj['diameter_neutral']['value'])
        #sheathneutral = obj['sheathneutral']['value'].upper()=='TRUE'
        CN_strand_count[wire_cn_ts] = int(obj['strand_count']['value'])
        CN_strand_radius[wire_cn_ts] = float(obj['strand_radius']['value'])
        CN_strand_gmr[wire_cn_ts] = float(obj['strand_gmr']['value'])
        CN_strand_rdc[wire_cn_ts] = float(obj['strand_rdc']['value'])
        #print('concentric wire_cn_ts: ' + wire_cn_ts + ', gmr: ' + str(GMR[wire_cn_ts]) + ', r25: ' + str(R25[wire_cn_ts]) + ', diameter_jacket: ' + str(CN_diameter_jacket[wire_cn_ts]) + ', strand_count: ' + str(CN_strand_count[wire_cn_ts]) + ', strand_radius: ' + str(CN_strand_radius[wire_cn_ts]) + ', strand_gmr: ' + str(CN_strand_gmr[wire_cn_ts]) + ', strand_rdc: ' + str(CN_strand_rdc[wire_cn_ts]))

    # TapeShieldCableInfo specific query
    bindings = sparql_mgr.WireInfo_tapeShield()
    #print('LINE_MODEL_VALIDATOR WireInfo tapeShield query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo tapeShield query results:', file=logfile)
    #print(bindings, file=logfile)

    TS_diameter_screen = {}
    TS_tape_thickness = {}
    for obj in bindings:
        wire_cn_ts = obj['wire_cn_ts']['value']
        #radius = float(obj['radius']['value'])
        #coreRadius = float(obj['coreRadius']['value'])
        GMR[wire_cn_ts] = float(obj['gmr']['value'])
        #rdc = float(obj['rdc']['value'])
        R25[wire_cn_ts] = float(obj['r25']['value'])
        #r50 = float(obj['r50']['value'])
        #r75 = float(obj['r75']['value'])
        #amps = int(obj['amps']['value'])
        #insulationFlag = obj['amps']['value'].upper() == 'TRUE'
        #insulation_thickness = float(obj['insulation_thickness']['value'])
        #diameter_core = float(obj['diameter_core']['value'])
        #diameter_insulation = float(obj['diameter_insulation']['value'])
        TS_diameter_screen[wire_cn_ts] = float(obj['diameter_screen']['value'])
        #diameter_jacket = float(obj['diameter_jacket']['value'])
        #sheathneutral = obj['sheathneutral']['value'].upper()=='TRUE'
        #tapelap = int(obj['tapelap']['value'])
        TS_tape_thickness[wire_cn_ts] = float(obj['tapethickness']['value'])
        #print('tape wire_cn_ts: ' + wire_cn_ts + ', gmr: ' + str(GMR[wire_cn_ts]) + ', r25: ' + str(R25[wire_cn_ts]) + ', diameter_screen: ' + str(TS_diameter_screen[wire_cn_ts]) + ', tape_thickness: ' + str(TS_tape_thickness[wire_cn_ts]))

    # line_names query for all types
    bindings = sparql_mgr.WireInfo_line_names()
    #print('LINE_MODEL_VALIDATOR WireInfo line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nLINE_MODEL_VALIDATOR WireInfo_and_WireSpacingInfo: NO LINE MATCHES', flush=True)
            print('\nLINE_MODEL_VALIDATOR WireInfo_and_WireSpacingInfo: NO LINE MATCHES', file=logfile)
        return line_count

    if cmpFlag:
        # initialize summary statistics
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

    # map line_name query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 'N': '.4', 's1': '.1', 's2': '.2'}

    # map between 0-base numpy array indices and 1-based formulas so everything lines up
    i1 = j1 = 0
    i2 = j2 = 1
    i3 = j3 = 2
    i4 = j4 = 3
    i5 = j5 = 4
    i6 = j6 = 5

    tape_line = None
    tape_skip = False
    phaseIdx = 0
    CN_done = False
    for obj in bindings:
        line_name = obj['line_name']['value']
        #basev = float(obj['basev']['value'])
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        wire_spacing_info = obj['wire_spacing_info']['value']
        phase = obj['phase']['value']
        wire_cn_ts = obj['wire_cn_ts']['value']
        wireinfo = obj['wireinfo']['value']
        #print('line_name: ' + line_name + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', length: ' + str(length) + ', wire_spacing_info: ' + wire_spacing_info + ', phase: ' + phase + ', wire_cn_ts: ' + wire_cn_ts + ', wireinfo: ' + wireinfo)

        # TapeShieldCableInfo is special so it needs some special processing
        # first, the wireinfo isn't always TapeShieldCableInfo so need to match on line_name instead
        # second, only a single phase is implemented so need a way to skip processing multiple phases
        if wireinfo=='TapeShieldCableInfo' or line_name==tape_line:
            tape_line = line_name
            if tape_skip:
                continue
        else:
            tape_line = None
            tape_skip = False

        if phaseIdx == 0:
            pair_i0b1 = bus1 + ybusPhaseIdx[phase]
            pair_i0b2 = bus2 + ybusPhaseIdx[phase]

            dim = len(XCoord[wire_spacing_info])
            if wireinfo == 'OverheadWireInfo':
                if dim == 2:
                    Zprim = np.empty((2,2), dtype=complex)
                elif dim == 3:
                    Zprim = np.empty((3,3), dtype=complex)
                elif dim == 4:
                    Zprim = np.empty((4,4), dtype=complex)

            elif wireinfo == 'ConcentricNeutralCableInfo':
                if dim == 1:
                    Zprim = np.empty((2,2), dtype=complex)
                elif dim == 2:
                    Zprim = np.empty((4,4), dtype=complex)
                elif dim == 3:
                    Zprim = np.empty((6,6), dtype=complex)

            elif wireinfo == 'TapeShieldCableInfo':
                if dim == 2:
                    Zprim = np.empty((3,3), dtype=complex)
                else:
                    if cmpFlag:
                        print('WARNING: TapeShieldCableInfo implementation only supports 1 phase and not the number found: ' + str(dim-1), flush=True)
                        print('WARNING: TapeShieldCableInfo implementation only supports 1 phase and not the number found: ' + str(dim-1), file=logfile)
                    tape_skip = True
                    continue

            # row 1
            Zprim[i1,j1] = diagZprim(wireinfo, wire_cn_ts, False, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

            if wireinfo=='ConcentricNeutralCableInfo' and dim==1:
                CN_done = True

                # row 2
                Zprim[i2,j1] = Zprim[i1,j2] = offDiagZprim(2, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i2,j2] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

            elif wireinfo == 'TapeShieldCableInfo':
                # row 2
                Zprim[i2,j1] = Zprim[i1,j2] = offDiagZprim(2, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                # neutralFlag is passed as True as a flag indicating to use the 2nd row shield calculation
                Zprim[i2,j2] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

        elif phaseIdx == 1:
            pair_i1b1 = bus1 + ybusPhaseIdx[phase]
            pair_i1b2 = bus2 + ybusPhaseIdx[phase]

            # row 2
            if line_name != tape_line:
                Zprim[i2,j1] = Zprim[i1,j2] = offDiagZprim(2, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i2,j2] = diagZprim(wireinfo, wire_cn_ts, False, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

            if wireinfo=='ConcentricNeutralCableInfo' and dim==2:
                CN_done = True

                # row 3
                Zprim[i3,j1] = Zprim[i1,j3] = offDiagZprim(3, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i3,j2] = Zprim[i2,j3] = offDiagZprim(3, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i3,j3] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

                # row 4
                Zprim[i4,j1] = Zprim[i1,j4] = offDiagZprim(4, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j2] = Zprim[i2,j4] = offDiagZprim(4, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j3] = Zprim[i3,j4] = offDiagZprim(4, 3, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j4] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

            elif line_name == tape_line:
                # row 3
                # coordinates for neutral are stored in index 2 for TapeShieldCableInfo
                Zprim[i3,j1] = Zprim[i1,j3] = Zprim[i3,j2] = Zprim[i2,j3] = offDiagZprim(2, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i3,j3] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

        elif phaseIdx == 2:
            pair_i2b1 = bus1 + ybusPhaseIdx[phase]
            pair_i2b2 = bus2 + ybusPhaseIdx[phase]

            # row 3
            Zprim[i3,j1] = Zprim[i1,j3] = offDiagZprim(3, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
            Zprim[i3,j2] = Zprim[i2,j3] = offDiagZprim(3, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
            Zprim[i3,j3] = diagZprim(wireinfo, wire_cn_ts, False, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

            if wireinfo == 'ConcentricNeutralCableInfo':
                CN_done = True

                # row 4
                Zprim[i4,j1] = Zprim[i1,j4] = offDiagZprim(4, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j2] = Zprim[i2,j4] = offDiagZprim(4, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j3] = Zprim[i3,j4] = offDiagZprim(4, 3, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i4,j4] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                # row 5
                Zprim[i5,j1] = Zprim[i1,j5] = offDiagZprim(5, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i5,j2] = Zprim[i2,j5] = offDiagZprim(5, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i5,j3] = Zprim[i3,j5] = offDiagZprim(5, 3, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i5,j4] = Zprim[i4,j5] = offDiagZprim(5, 4, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i5,j5] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

                # row 6
                Zprim[i6,j1] = Zprim[i1,j6] = offDiagZprim(6, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i6,j2] = Zprim[i2,j6] = offDiagZprim(6, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i6,j3] = Zprim[i3,j6] = offDiagZprim(6, 3, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i6,j4] = Zprim[i4,j6] = offDiagZprim(6, 4, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i6,j5] = Zprim[i5,j6] = offDiagZprim(6, 5, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
                Zprim[i6,j6] = diagZprim(wireinfo, wire_cn_ts, True, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

        elif phaseIdx == 3:
            # this can only be phase 'N' so no need to store 'pair' values
            # row 4
            Zprim[i4,j1] = Zprim[i1,j4] = offDiagZprim(4, 1, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
            Zprim[i4,j2] = Zprim[i2,j4] = offDiagZprim(4, 2, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
            Zprim[i4,j3] = Zprim[i3,j4] = offDiagZprim(4, 3, wireinfo, wire_spacing_info, wire_cn_ts, XCoord, YCoord, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)
            Zprim[i4,j4] = diagZprim(wireinfo, wire_cn_ts, phase, R25, GMR, CN_strand_count, CN_strand_rdc, CN_strand_gmr, CN_strand_radius, CN_diameter_jacket, TS_tape_thickness, TS_diameter_screen)

        # for OverheadWireInfo, take advantage that there is always a phase N
        # and it's always the last item processed for a line_name so a good way
        # to know when to trigger the Ybus comparison code
        # for ConcentricNeutralCableInfo, a flag is the easiest
        if (wireinfo=='OverheadWireInfo' and phase == 'N') or (wireinfo=='ConcentricNeutralCableInfo' and CN_done):
            if cmpFlag:
                if line_name == tape_line:
                    print("\nValidating TapeShieldCableInfo line_name: " + line_name, flush=True)
                    print("\nValidating TapeShieldCableInfo line_name: " + line_name, file=logfile)
                else:
                    print("\nValidating " + wireinfo + " line_name: " + line_name, flush=True)
                    print("\nValidating " + wireinfo + " line_name: " + line_name, file=logfile)

            if wireinfo == 'ConcentricNeutralCableInfo':
                # the Z-hat slicing below is based on having an 'N' phase so need to
                # account for that when it doesn't exist
                phaseIdx += 1
                CN_done = False

            # create the Z-hat matrices to then compute Zabc for Ybus comparisons
            Zij = Zprim[:phaseIdx,:phaseIdx]
            Zin = Zprim[:phaseIdx,phaseIdx:]
            Znj = Zprim[phaseIdx:,:phaseIdx]
            #Znn = Zprim[phaseIdx:,phaseIdx:]
            invZnn = np.linalg.inv(Zprim[phaseIdx:,phaseIdx:])

            # finally, compute Zabc from Z-hat matrices
            Zabc = np.subtract(Zij, np.matmul(np.matmul(Zin, invZnn), Znj))

            # multiply by scalar length
            lenZabc = Zabc * length
            # invert the matrix
            invZabc = np.linalg.inv(lenZabc)
            # test if the inverse * original = identity
            #identityTest = np.dot(lenZabc, invZabc)
            #print('identity test for ' + line_name + ': ' + str(identityTest))
            # negate the matrix and assign it to Ycomp
            Ycomp = invZabc * -1

            line_count += 1

            if Ycomp.size == 1:
                if cmpFlag:
                    colorIdx = compareY(pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                else:
                    fillYsysNoSwap(pair_i0b1, pair_i0b2, Ycomp[0,0], Ysys)

            elif Ycomp.size == 4:
                if cmpFlag:
                    colorIdx00 = compareY(pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                    colorIdx10 = compareY(pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                    colorIdx11 = compareY(pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                    colorIdx = max(colorIdx00, colorIdx10, colorIdx11)
                else:
                    fillYsysNoSwap(pair_i0b1, pair_i0b2, Ycomp[0,0], Ysys)
                    fillYsysSwap(pair_i1b1, pair_i0b2, Ycomp[1,0], Ysys)
                    fillYsysNoSwap(pair_i1b1, pair_i1b2, Ycomp[1,1], Ysys)

            elif Ycomp.size == 9:
                if cmpFlag:
                    colorIdx00 = compareY(pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                    colorIdx10 = compareY(pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                    colorIdx11 = compareY(pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                    colorIdx20 = compareY(pair_i2b1, pair_i0b2, Ycomp[2,0], Ybus)
                    colorIdx21 = compareY(pair_i2b1, pair_i1b2, Ycomp[2,1], Ybus)
                    colorIdx22 = compareY(pair_i2b1, pair_i2b2, Ycomp[2,2], Ybus)
                    colorIdx = max(colorIdx00, colorIdx10, colorIdx11, colorIdx20, colorIdx21, colorIdx22)
                else:
                    fillYsysNoSwap(pair_i0b1, pair_i0b2, Ycomp[0,0], Ysys)
                    fillYsysSwap(pair_i1b1, pair_i0b2, Ycomp[1,0], Ysys)
                    fillYsysNoSwap(pair_i1b1, pair_i1b2, Ycomp[1,1], Ysys)
                    fillYsysSwap(pair_i2b1, pair_i0b2, Ycomp[2,0], Ysys)
                    fillYsysSwap(pair_i2b1, pair_i1b2, Ycomp[2,1], Ysys)
                    fillYsysNoSwap(pair_i2b1, pair_i2b2, Ycomp[2,2], Ysys)

            if cmpFlag:
                if colorIdx == 0:
                    greenCount += 1
                elif colorIdx == 1:
                    yellowCount += 1
                else:
                    redCount += 1

            phaseIdx = 0
        else:
            phaseIdx += 1

    if cmpFlag:
        print("\nSummary for WireInfo_and_WireSpacingInfo lines:", flush=True)
        print("\nSummary for WireInfo_and_WireSpacingInfo lines:", file=logfile)

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

        print("\nFinished validation for WireInfo_and_WireSpacingInfo lines", flush=True)
        print("\nFinished validation for WireInfo_and_WireSpacingInfo lines", file=logfile)

    return line_count


def start(log_file, feeder_mrid, model_api_topic, cmpFlag=True, Ysys=None):
    global logfile
    logfile = log_file

    if cmpFlag:
        print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------")
        print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    if cmpFlag:
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
    else:
        Ybus = None

    PerLengthPhaseImpedance_lines = validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus, cmpFlag, Ysys)
    if cmpFlag:
        if PerLengthPhaseImpedance_lines > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["PerLengthPhaseImpedance", PerLengthPhaseImpedance_lines, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["PerLengthPhaseImpedance", PerLengthPhaseImpedance_lines])

    PerLengthSequenceImpedance_lines = validate_PerLengthSequenceImpedance_lines(sparql_mgr, Ybus, cmpFlag, Ysys)
    if cmpFlag:
        if PerLengthSequenceImpedance_lines > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["PerLengthSequenceImpedance", PerLengthSequenceImpedance_lines, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["PerLengthSequenceImpedance", PerLengthSequenceImpedance_lines])

    ACLineSegment_lines = validate_ACLineSegment_lines(sparql_mgr, Ybus, cmpFlag, Ysys)
    if cmpFlag:
        if ACLineSegment_lines > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["ACLineSegment", ACLineSegment_lines, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["ACLineSegment", ACLineSegment_lines])

    WireInfo_and_WireSpacingInfo_lines = validate_WireInfo_and_WireSpacingInfo_lines(sparql_mgr, Ybus, cmpFlag, Ysys)
    if cmpFlag:
        if WireInfo_and_WireSpacingInfo_lines > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["WireInfo_and_WireSpacingInfo", WireInfo_and_WireSpacingInfo_lines, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["WireInfo_and_WireSpacingInfo", WireInfo_and_WireSpacingInfo_lines])

        print('\n', flush=True)
        print(tabulate(report, headers=["Line Type", "# Lines", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
        print('\n', file=logfile)
        print(tabulate(report, headers=["Line Type", "# Lines", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

        print('\nLINE_MODEL_VALIDATOR DONE!!!', flush=True)
        print('\nLINE_MODEL_VALIDATOR DONE!!!', file=logfile)


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
