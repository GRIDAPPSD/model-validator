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

import networkx as nx
import pandas as pd
import math
import argparse
import json
import sys
import os
import importlib
import numpy as np
import time
from tabulate import tabulate

from gridappsd import GridAPPSD

global logfile


def diffColorReal(absDiff, perDiff, colorFlag):
    global greenCountReal, yellowCountReal, redCountReal

    if absDiff<1e-3 and perDiff<0.01:
        if colorFlag: greenCountReal += 1
        return '\u001b[32mGREEN\u001b[37m' if colorFlag else 'GREEN'
    elif absDiff>=1e-2 or perDiff>=0.1:
        if colorFlag: redCountReal += 1
        return '\u001b[31mRED\u001b[37m' if colorFlag else 'RED'
    else:
        if colorFlag: yellowCountReal += 1
        return '\u001b[33mYELLOW\u001b[37m' if colorFlag else 'YELLOW'


def diffColorImag(absDiff, perDiff, colorFlag):
    global greenCountImag, yellowCountImag, redCountImag

    if absDiff<1e-3 and perDiff<0.01:
        if colorFlag: greenCountImag += 1
        return '\u001b[32mGREEN\u001b[37m' if colorFlag else 'GREEN'
    elif absDiff>=1e-2 or perDiff>=0.1:
        if colorFlag: redCountImag += 1
        return '\u001b[31mRED\u001b[37m' if colorFlag else 'RED'
    else:
        if colorFlag: yellowCountImag += 1
        return '\u001b[33mYELLOW\u001b[37m' if colorFlag else 'YELLOW'


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


def compareY(line_name, pair_b1, pair_b2, YcompValue, Ybus):
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
    print("        Real Ybus[i,j]:" + "{:10.6f}".format(YbusValue.real) + ", computed:" + "{:10.6f}".format(YcompValue.real) + " => " + diffColorReal(realAbsDiff, realPerDiff, True), flush=True)
    print("        Real Ybus[i,j]:" + "{:10.6f}".format(YbusValue.real) + ", computed:" + "{:10.6f}".format(YcompValue.real) + " => " + diffColorReal(realAbsDiff, realPerDiff, False), file=logfile)

    imagAbsDiff = abs(YcompValue.imag - YbusValue.imag)
    imagPerDiff = diffPercentImag(YcompValue.imag, YbusValue.imag)
    print("        Imag Ybus[i,j]:" + "{:10.6f}".format(YbusValue.imag) + ", computed:" + "{:10.6f}".format(YcompValue.imag) + " => " + diffColorImag(imagAbsDiff, imagPerDiff, True), flush=True)
    print("        Imag Ybus[i,j]:" + "{:10.6f}".format(YbusValue.imag) + ", computed:" + "{:10.6f}".format(YcompValue.imag) + " => " + diffColorImag(imagAbsDiff, imagPerDiff, False), file=logfile)


def validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus):
    print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance validation...', flush=True)
    print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance validation...', file=logfile)

    bindings = sparql_mgr.PerLengthPhaseImpedance_line_configs()
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_configs query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthPhaseImpedance line_configs query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthPhaseImpedance: NO LINE MATCHES', file=logfile)
        return

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
        return

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
            compareY(line_name, bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ycomp[0,0], Ybus)

        elif Ycomp.size == 4:
            if line_idx == 1:
                pair_i0b1 = bus1 + ybusPhaseIdx[phase]
                pair_i0b2 = bus2 + ybusPhaseIdx[phase]
            else:
                pair_i1b1 = bus1 + ybusPhaseIdx[phase]
                pair_i1b2 = bus2 + ybusPhaseIdx[phase]

                # do comparisons now
                compareY(line_name, pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)

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
                compareY(line_name, pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                compareY(line_name, pair_i2b1, pair_i0b2, Ycomp[2,0], Ybus)
                compareY(line_name, pair_i2b1, pair_i1b2, Ycomp[2,1], Ybus)
                compareY(line_name, pair_i2b1, pair_i2b2, Ycomp[2,2], Ybus)

    print("\nSummary for PerLengthPhaseImpedance lines:", flush=True)
    print("\nSummary for PerLengthPhaseImpedance lines:", file=logfile)

    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nReal \u001b[32mGREEN\u001b[37m count:  " + str(greenCountReal), flush=True)
    print("\nReal GREEN count:  " + str(greenCountReal), file=logfile)
    print("Real \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountReal), flush=True)
    print("Real YELLOW count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31mRED\u001b[37m count:    " + str(redCountReal), flush=True)
    print("Real RED count:    " + str(redCountReal), file=logfile)

    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), file=logfile)

    print("\nImag \u001b[32mGREEN\u001b[37m count:  " + str(greenCountImag), flush=True)
    print("\nImag GREEN count:  " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountImag), flush=True)
    print("Imag YELLOW count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31mRED\u001b[37m count:    " + str(redCountImag), flush=True)
    print("Imag RED count:    " + str(redCountImag), file=logfile)

    print("\nFinished validation for PerLengthPhaseImpedance lines", flush=True)
    print("\nFinished validation for PerLengthPhaseImpedance lines", file=logfile)

    return


def validate_PerLengthSequenceImpedance_lines(sparql_mgr, Ybus):
    print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance validation...', flush=True)
    print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance validation...', file=logfile)

    bindings = sparql_mgr.PerLengthSequenceImpedance_line_configs()
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_configs query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR PerLengthSequenceImpedance line_configs query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', file=logfile)
        return

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
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR PerLengthSequenceImpedance: NO LINE MATCHES', file=logfile)
        return

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

    for obj in bindings:
        line_name = obj['line_name']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        line_config = obj['line_config']['value']
        #print('line_name: ' + line_name + ', line_config: ' + line_config + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2)

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

        # do comparisons now
        compareY(line_name, bus1+'.1', bus2+'.1', Ycomp[0,0], Ybus)
        compareY(line_name, bus1+'.2', bus2+'.1', Ycomp[1,0], Ybus)
        compareY(line_name, bus1+'.2', bus2+'.2', Ycomp[1,1], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.1', Ycomp[2,0], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.2', Ycomp[2,1], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.3', Ycomp[2,2], Ybus)

    print("\nSummary for PerLengthSequenceImpedance lines:", flush=True)
    print("\nSummary for PerLengthSequenceImpedance lines:", file=logfile)

    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nReal \u001b[32mGREEN\u001b[37m count:  " + str(greenCountReal), flush=True)
    print("\nReal GREEN count:  " + str(greenCountReal), file=logfile)
    print("Real \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountReal), flush=True)
    print("Real YELLOW count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31mRED\u001b[37m count:    " + str(redCountReal), flush=True)
    print("Real RED count:    " + str(redCountReal), file=logfile)

    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), file=logfile)

    print("\nImag \u001b[32mGREEN\u001b[37m count:  " + str(greenCountImag), flush=True)
    print("\nImag GREEN count:  " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountImag), flush=True)
    print("Imag YELLOW count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31mRED\u001b[37m count:    " + str(redCountImag), flush=True)
    print("Imag RED count:    " + str(redCountImag), file=logfile)

    print("\nFinished validation for PerLengthSequenceImpedance lines", flush=True)
    print("\nFinished validation for PerLengthSequenceImpedance lines", file=logfile)

    return


def validate_ACLineSegment_lines(sparql_mgr, Ybus):
    print('\nLINE_MODEL_VALIDATOR ACLineSegment validation...', flush=True)
    print('\nLINE_MODEL_VALIDATOR ACLineSegment validation...', file=logfile)

    bindings = sparql_mgr.ACLineSegment_line_names()
    #print('LINE_MODEL_VALIDATOR ACLineSegment line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR ACLineSegment line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR ACLineSegment: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR ACLineSegment: NO LINE MATCHES', file=logfile)
        return

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

        # do comparisons now
        compareY(line_name, bus1+'.1', bus2+'.1', Ycomp[0,0], Ybus)
        compareY(line_name, bus1+'.2', bus2+'.1', Ycomp[1,0], Ybus)
        compareY(line_name, bus1+'.2', bus2+'.2', Ycomp[1,1], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.1', Ycomp[2,0], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.2', Ycomp[2,1], Ybus)
        compareY(line_name, bus1+'.3', bus2+'.3', Ycomp[2,2], Ybus)

    print("\nSummary for ACLineSegment lines:", flush=True)
    print("\nSummary for ACLineSegment lines:", file=logfile)

    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nReal \u001b[32mGREEN\u001b[37m count:  " + str(greenCountReal), flush=True)
    print("\nReal GREEN count:  " + str(greenCountReal), file=logfile)
    print("Real \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountReal), flush=True)
    print("Real YELLOW count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31mRED\u001b[37m count:    " + str(redCountReal), flush=True)
    print("Real RED count:    " + str(redCountReal), file=logfile)

    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), file=logfile)

    print("\nImag \u001b[32mGREEN\u001b[37m count:  " + str(greenCountImag), flush=True)
    print("\nImag GREEN count:  " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountImag), flush=True)
    print("Imag YELLOW count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31mRED\u001b[37m count:    " + str(redCountImag), flush=True)
    print("Imag RED count:    " + str(redCountImag), file=logfile)

    print("\nFinished validation for ACLineSegment lines", flush=True)
    print("\nFinished validation for ACLineSegment lines", file=logfile)

    return


def validate_WireInfo_lines(sparql_mgr, Ybus):
    print('\nLINE_MODEL_VALIDATOR WireInfo validation...', flush=True)
    print('\nLINE_MODEL_VALIDATOR WireInfo validation...', file=logfile)

    # define all the constants needed for WireInfo
    u0 = math.pi * 4.0e-7
    w = math.pi*2.0 * 60.0
    p = 100.0
    f = 60.0
    Rg = (u0 * w)/8.0
    X0 = (u0 * w)/(math.pi*2.0)
    Xg = X0 * math.log(658.5 * math.sqrt(p/f))

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
        #print('wire_spacing_info: ' + wire_spacing_info + ', cable: ' + str(cableFlag) + ', seq: ' + str(seq) + ', xCoord: ' + str(xCoord) + ', yCoord: ' + str(yCoord))

    bindings = sparql_mgr.WireInfo_overhead()
    #print('LINE_MODEL_VALIDATOR WireInfo overhead query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo overhead query results:', file=logfile)
    #print(bindings, file=logfile)

    OH_gmr = {}
    OH_r25 = {}
    for obj in bindings:
        wire_cn_ts = obj['wire_cn_ts']['value']
        #radius = float(obj['radius']['value'])
        #coreRadius = float(obj['coreRadius']['value'])
        OH_gmr[wire_cn_ts] = float(obj['gmr']['value'])
        #rdc = float(obj['rdc']['value'])
        OH_r25[wire_cn_ts] = float(obj['r25']['value'])
        #r50 = float(obj['r50']['value'])
        #r75 = float(obj['r75']['value'])
        #amps = int(obj['amps']['value'])
        #print('wire_cn_ts: ' + wire_cn_ts + ', gmr: ' + str(OH_gmr[wire_cn_ts]) + ', r25: ' + str(OH_r25[wire_cn_ts]))

    bindings = sparql_mgr.WireInfo_concentric()
    #print('LINE_MODEL_VALIDATOR WireInfo concentric query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo concentric query results:', file=logfile)
    #print(bindings, file=logfile)

    CN_gmr = {}
    CN_r25 = {}
    CN_diameter_jacket = {}
    CN_strand_count = {}
    CN_strand_radius = {}
    CN_strand_gmr = {}
    CN_strand_rdc = {}
    for obj in bindings:
        wire_cn_ts = obj['wire_cn_ts']['value']
        #radius = float(obj['radius']['value'])
        #coreRadius = float(obj['coreRadius']['value'])
        CN_gmr[wire_cn_ts] = float(obj['gmr']['value'])
        #rdc = float(obj['rdc']['value'])
        CN_r25[wire_cn_ts] = float(obj['r25']['value'])
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
        print('concentric wire_cn_ts: ' + wire_cn_ts + ', gmr: ' + str(CN_gmr[wire_cn_ts]) + ', r25: ' + str(CN_r25[wire_cn_ts]) + ', diameter_jacket: ' + str(CN_diameter_jacket[wire_cn_ts]) + ', strand_count: ' + str(CN_strand_count[wire_cn_ts]) + ', strand_radius: ' + str(CN_strand_radius[wire_cn_ts]) + ', strand_gmr: ' + str(CN_strand_gmr[wire_cn_ts]) + ', strand_rdc: ' + str(CN_strand_rdc[wire_cn_ts]))


    # map line_name query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 'N': '.4', 's1': '.1', 's2': '.2'}

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

    bindings = sparql_mgr.WireInfo_line_names()
    #print('LINE_MODEL_VALIDATOR WireInfo line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR WireInfo line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        print('\nLINE_MODEL_VALIDATOR WireInfo: NO LINE MATCHES', flush=True)
        print('\nLINE_MODEL_VALIDATOR WireInfo: NO LINE MATCHES', file=logfile)
        return

    tape_line = None
    phaseIdx = 0
    for obj in bindings:
        line_name = obj['line_name']['value']
        #basev = float(obj['basev']['value'])
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        wire_spacing_info = obj['wire_spacing_info']['value']
        phase = obj['phase']['value'].upper()
        wire_cn_ts = obj['wire_cn_ts']['value']
        wireinfo = obj['wireinfo']['value']
        print('line_name: ' + line_name + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', length: ' + str(length) + ', wire_spacing_info: ' + wire_spacing_info + ', phase: ' + phase + ', wire_cn_ts: ' + wire_cn_ts + ', wireinfo: ' + wireinfo)

        # simple way to keep from processing TapeShieldCableInfo for now
        if wireinfo=='TapeShieldCableInfo' or line_name==tape_line:
            tape_line = line_name
            continue

        tape_line = None

        if wireinfo == 'ConcentricNeutralCableInfo':
            break

        if phaseIdx == 0:
            pair_i0b1 = bus1 + ybusPhaseIdx[phase]
            pair_i0b2 = bus2 + ybusPhaseIdx[phase]

            if len(XCoord[wire_spacing_info]) == 2:
                if wireinfo == 'OverheadWireInfo':
                    Zprim = np.empty((2,2), dtype=complex)
                elif wireinfo == 'ConcentricNeutralCableInfo':
                    Zprim = np.empty((4,4), dtype=complex)
            elif len(XCoord[wire_spacing_info]) == 3:
                if wireinfo == 'OverheadWireInfo':
                    Zprim = np.empty((3,3), dtype=complex)
                elif wireinfo == 'ConcentricNeutralCableInfo':
                    Zprim = np.empty((5,5), dtype=complex)
            elif len(XCoord[wire_spacing_info]) == 4:
                if wireinfo == 'OverheadWireInfo':
                    Zprim = np.empty((4,4), dtype=complex)
                elif wireinfo == 'ConcentricNeutralCableInfo':
                    Zprim = np.empty((6,6), dtype=complex)

            Zprim[0,0] = complex(OH_r25[wire_cn_ts] + Rg, X0*math.log(1.0/OH_gmr[wire_cn_ts]) + Xg)

        elif phaseIdx == 1:
            pair_i1b1 = bus1 + ybusPhaseIdx[phase]
            pair_i1b2 = bus2 + ybusPhaseIdx[phase]

            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][2]-XCoord[wire_spacing_info][1],2) + math.pow(YCoord[wire_spacing_info][2]-YCoord[wire_spacing_info][1],2))
            Zprim[1,0] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[0,1] = Zprim[1,0]
            Zprim[1,1] = complex(OH_r25[wire_cn_ts] + Rg, X0*math.log(1.0/OH_gmr[wire_cn_ts]) + Xg)

        elif phaseIdx == 2:
            pair_i2b1 = bus1 + ybusPhaseIdx[phase]
            pair_i2b2 = bus2 + ybusPhaseIdx[phase]

            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][3]-XCoord[wire_spacing_info][1],2) + math.pow(YCoord[wire_spacing_info][3]-YCoord[wire_spacing_info][1],2))
            Zprim[2,0] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[0,2] = Zprim[2,0]
            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][3]-XCoord[wire_spacing_info][2],2) + math.pow(YCoord[wire_spacing_info][3]-YCoord[wire_spacing_info][2],2))
            Zprim[2,1] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[1,2] = Zprim[2,1]
            Zprim[2,2] = complex(OH_r25[wire_cn_ts] + Rg, X0*math.log(1.0/OH_gmr[wire_cn_ts]) + Xg)

        elif phaseIdx == 3:
            # this can only be phase 'N' so no need to store 'pair' values
            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][4]-XCoord[wire_spacing_info][1],2) + math.pow(YCoord[wire_spacing_info][4]-YCoord[wire_spacing_info][1],2))
            Zprim[3,0] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[0,3] = Zprim[3,0]
            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][4]-XCoord[wire_spacing_info][2],2) + math.pow(YCoord[wire_spacing_info][4]-YCoord[wire_spacing_info][2],2))
            Zprim[3,1] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[1,3] = Zprim[3,1]
            dist = math.sqrt(math.pow(XCoord[wire_spacing_info][4]-XCoord[wire_spacing_info][3],2) + math.pow(YCoord[wire_spacing_info][4]-YCoord[wire_spacing_info][3],2))
            Zprim[3,2] = complex(Rg, X0*math.log(1.0/dist) + Xg)
            Zprim[2,3] = Zprim[3,2]
            Zprim[3,3] = complex(OH_r25[wire_cn_ts] + Rg, X0*math.log(1.0/OH_gmr[wire_cn_ts]) + Xg)

        # take advantage that there is always a phase N and it's always the last
        # item processed for a line_name so a good way to know when to trigger
        # the Ybus comparison code
        if phase == 'N':
            print("\nValidating OverheadWireInfo line_name: " + line_name, flush=True)
            print("\nValidating OverheadWireInfo line_name: " + line_name, file=logfile)

            # create the Z-hat matrices to then compute Zabc for Ybus comparisons
            Zij = Zprim[:phaseIdx,:phaseIdx]
            Zin = Zprim[:phaseIdx,phaseIdx:]
            Znj = Zprim[phaseIdx:,:phaseIdx]
            invZnn = np.linalg.inv(Zprim[phaseIdx:,phaseIdx:])

            # finally, compute Zabc from Z-hat matrices
            Zabc = Zij - Zin*invZnn*Znj
            # note this can also be done as follows through numpy methods, but
            # the end results are the same so stick with the one that looks simpler
            #Zabc = np.subtract(Zij, np.matmul(np.matmul(Zin, invZnn), Znj))

            # multiply by scalar length
            lenZabc = Zabc * length
            # invert the matrix
            invZabc = np.linalg.inv(lenZabc)
            # test if the inverse * original = identity
            #identityTest = np.dot(lenZabc, invZabc)
            #print('identity test for ' + line_name + ': ' + str(identityTest))
            # negate the matrix and assign it to Ycomp
            Ycomp = invZabc * -1

            if Ycomp.size == 1:
                compareY(line_name, pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)

            elif Ycomp.size == 4:
                compareY(line_name, pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)

            elif Ycomp.size == 9:
                compareY(line_name, pair_i0b1, pair_i0b2, Ycomp[0,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i0b2, Ycomp[1,0], Ybus)
                compareY(line_name, pair_i1b1, pair_i1b2, Ycomp[1,1], Ybus)
                compareY(line_name, pair_i2b1, pair_i0b2, Ycomp[2,0], Ybus)
                compareY(line_name, pair_i2b1, pair_i1b2, Ycomp[2,1], Ybus)
                compareY(line_name, pair_i2b1, pair_i2b2, Ycomp[2,2], Ybus)

            phaseIdx = 0
        else:
            phaseIdx += 1

    print("\nSummary for WireInfo lines:", flush=True)
    print("\nSummary for WireInfo lines:", file=logfile)

    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:11.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:11.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nReal \u001b[32mGREEN\u001b[37m count:  " + str(greenCountReal), flush=True)
    print("\nReal GREEN count:  " + str(greenCountReal), file=logfile)
    print("Real \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountReal), flush=True)
    print("Real YELLOW count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31mRED\u001b[37m count:    " + str(redCountReal), flush=True)
    print("Real RED count:    " + str(redCountReal), file=logfile)

    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:11.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:11.6f}".format(maxPercentDiffImag), file=logfile)

    print("\nImag \u001b[32mGREEN\u001b[37m count:  " + str(greenCountImag), flush=True)
    print("\nImag GREEN count:  " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33mYELLOW\u001b[37m count: " + str(yellowCountImag), flush=True)
    print("Imag YELLOW count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31mRED\u001b[37m count:    " + str(redCountImag), flush=True)
    print("Imag RED count:    " + str(redCountImag), file=logfile)

    print("\nFinished validation for WireInfo lines", flush=True)
    print("\nFinished validation for WireInfo lines", file=logfile)

    return


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------")
    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------", file=logfile)

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

    #validate_PerLengthPhaseImpedance_lines(sparql_mgr, Ybus)

    #validate_PerLengthSequenceImpedance_lines(sparql_mgr, Ybus)

    #validate_ACLineSegment_lines(sparql_mgr, Ybus)

    validate_WireInfo_lines(sparql_mgr, Ybus)

    print('\nLINE_MODEL_VALIDATOR DONE!!!', flush=True)
    print('\nLINE_MODEL_VALIDATOR DONE!!!', file=logfile)

    return


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
