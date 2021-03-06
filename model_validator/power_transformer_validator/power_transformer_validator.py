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
import inspect
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

    #if bus2 not in Ysys:
    #    Ysys[bus2] = {}

    if bus2 in Ysys[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling power transformer value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling power transformer value\n', file=logfile)

    # if needed for debugging, here's how to find the calling functions
    #if bus2=='X2673305B.1' and bus1=='X2673305B.2':
    #    print('*** fillYsysUnique bus1: ' + bus1 + ', bus2: ' + bus2 + ', caller: ' + str(inspect.stack()[1].function) + ', ' + str(inspect.stack()[2].function), flush=True)

    #Ysys[bus1][bus2] = Ysys[bus2][bus1] = Yval
    Ysys[bus1][bus2] = Yval


def fillYsysAdd(bus1, bus2, Yval, Ysys):
    if Yval == 0j:
        return

    if bus1 not in Ysys:
        Ysys[bus1] = {}

    # if needed for debugging, here's how to find the calling functions
    #if bus2=='X2673305B.1' and bus1=='X2673305B.2':
    #    print('*** fillYsysAdd bus1: ' + bus1 + ', bus2: ' + bus2 + ', caller: ' + str(inspect.stack()[1].function) + ', ' + str(inspect.stack()[2].function), flush=True)

    if bus2 in Ysys[bus1]:
        Ysys[bus1][bus2] += Yval
    else:
        Ysys[bus1][bus2] = Yval


def fillYsys_6x6(bus1, bus2, DY_flag, Ycomp, Ysys):
    # fill Ysys directly from Ycomp
    # first fill the ones that are independent of DY_flag
    # either because the same bus is used or the same phase
    fillYsysAdd(bus1+'.1', bus1+'.1', Ycomp[0,0], Ysys)
    fillYsysAdd(bus1+'.2', bus1+'.1', Ycomp[1,0], Ysys)
    fillYsysAdd(bus1+'.2', bus1+'.2', Ycomp[1,1], Ysys)
    fillYsysAdd(bus1+'.3', bus1+'.1', Ycomp[2,0], Ysys)
    fillYsysAdd(bus1+'.3', bus1+'.2', Ycomp[2,1], Ysys)
    fillYsysAdd(bus1+'.3', bus1+'.3', Ycomp[2,2], Ysys)
    fillYsysUnique(bus2+'.1', bus1+'.1', Ycomp[3,0], Ysys)
    fillYsysAdd(bus2+'.1', bus2+'.1', Ycomp[3,3], Ysys)
    fillYsysUnique(bus2+'.2', bus1+'.2', Ycomp[4,1], Ysys)
    fillYsysAdd(bus2+'.2', bus2+'.1', Ycomp[4,3], Ysys)
    fillYsysAdd(bus2+'.2', bus2+'.2', Ycomp[4,4], Ysys)
    fillYsysUnique(bus2+'.3', bus1+'.3', Ycomp[5,2], Ysys)
    fillYsysAdd(bus2+'.3', bus2+'.1', Ycomp[5,3], Ysys)
    fillYsysAdd(bus2+'.3', bus2+'.2', Ycomp[5,4], Ysys)
    fillYsysAdd(bus2+'.3', bus2+'.3', Ycomp[5,5], Ysys)

    # now fill the ones that are dependent on DY_flag, which
    # are different bus and different phase
    if DY_flag:
        fillYsysUnique(bus2+'.1', bus1+'.2', Ycomp[4,0], Ysys)
        fillYsysUnique(bus2+'.1', bus1+'.3', Ycomp[5,0], Ysys)
        fillYsysUnique(bus2+'.2', bus1+'.1', Ycomp[3,1], Ysys)
        fillYsysUnique(bus2+'.2', bus1+'.3', Ycomp[5,1], Ysys)
        fillYsysUnique(bus2+'.3', bus1+'.1', Ycomp[3,2], Ysys)
        fillYsysUnique(bus2+'.3', bus1+'.2', Ycomp[4,2], Ysys)

    else:
        fillYsysUnique(bus2+'.2', bus1+'.1', Ycomp[4,0], Ysys)
        fillYsysUnique(bus2+'.3', bus1+'.1', Ycomp[5,0], Ysys)
        fillYsysUnique(bus2+'.1', bus1+'.2', Ycomp[3,1], Ysys)
        fillYsysUnique(bus2+'.3', bus1+'.2', Ycomp[5,1], Ysys)
        fillYsysUnique(bus2+'.1', bus1+'.3', Ycomp[3,2], Ysys)
        fillYsysUnique(bus2+'.2', bus1+'.3', Ycomp[4,2], Ysys)


def validate_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus, cmpFlag, Ysys, Unsupported):
    if cmpFlag:
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd validation...\n', flush=True)
        print('\nPOWER_TRANSFORMER_VALIDATOR PowerTransformerEnd validation...\n', file=logfile)

    # return # of xfmrs validated
    xfmrs_count = 0

    bindings = sparql_mgr.PowerTransformerEnd_xfmr_impedances()
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_impedances query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR PowerTransformerEnd xfmr_impedances query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
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
        if cmpFlag:
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
        # can't handle 3-winding transformers so issue a warning and skip
        # to the next transformer in that case
        if end_number == 3:
            if cmpFlag:
                print('    *** WARNING: 3-winding, 3-phase PowerTransformerEnd transformers are not supported: ' + xfmr_name + '\n', flush=True)
                print('    *** WARNING: 3-winding, 3-phase PowerTransformerEnd transformers are not supported: ' + xfmr_name + '\n', file=logfile)
            else:
                bus1 = Bus[xfmr_name][1]
                bus2 = Bus[xfmr_name][2]
                bus3 = obj['bus']['value'].upper()
                print('\n*** WARNING: 3-winding, 3-phase PowerTransformerEnd transformers are not supported, xfmr: ' + xfmr_name + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', bus3: ' + bus3 + '\n', flush=True)
                print('\n*** WARNING: 3-winding, 3-phase PowerTransformerEnd transformers are not supported, xfmr: ' + xfmr_name + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', bus3: ' + bus3 + '\n', file=logfile)
                Unsupported[bus1] = Unsupported[bus2] = Unsupported[bus3] = [(bus1, bus2, bus3), '3-winding 3-phase transformer']

            # need to clear out the previous dictionary entries for this
            # 3-winding transformer so it isn't processed below
            Bus.pop(xfmr_name, None)
            Connection.pop(xfmr_name, None)
            RatedS.pop(xfmr_name, None)
            RatedU.pop(xfmr_name, None)
            R_ohm.pop(xfmr_name, None)
            continue

        if xfmr_name not in Bus:
            Bus[xfmr_name] = {}
            Connection[xfmr_name] = {}
            RatedS[xfmr_name] = {}
            RatedU[xfmr_name] = {}
            R_ohm[xfmr_name] = {}

        Bus[xfmr_name][end_number] = obj['bus']['value'].upper()
        #base_voltage = int(obj['base_voltage']['value'])
        Connection[xfmr_name][end_number] = obj['connection']['value']
        RatedS[xfmr_name][end_number] = int(float(obj['ratedS']['value']))
        RatedU[xfmr_name][end_number] = int(obj['ratedU']['value'])
        R_ohm[xfmr_name][end_number] = float(obj['r_ohm']['value'])
        #angle = int(obj['angle']['value'])
        #grounded = obj['grounded']['value']
        #r_ground = obj['r_ground']['value']
        #x_ground = obj['x_ground']['value']
        #print('xfmr_name: ' + xfmr_name + ', end_number: ' + str(end_number) + ', bus: ' + Bus[xfmr_name][end_number] + ', connection: ' + Connection[xfmr_name][end_number] + ', ratedS: ' + str(RatedS[xfmr_name][end_number]) + ', ratedU: ' + str(RatedU[xfmr_name][end_number]) + ', r_ohm: ' + str(R_ohm[xfmr_name][end_number]))

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

    for xfmr_name in Bus:
        # Note that division is always floating point in Python 3 even if
        # operands are integer
        zBaseP = (RatedU[xfmr_name][1]*RatedU[xfmr_name][1])/RatedS[xfmr_name][1]
        #zBaseS = (RatedU[xfmr_name][2]*RatedU[xfmr_name][2])/RatedS[xfmr_name][2]
        r_ohm_pu = R_ohm[xfmr_name][1]/zBaseP
        mesh_x_ohm_pu = Mesh_x_ohm[xfmr_name]/zBaseP
        zsc_1V = complex(2.0*r_ohm_pu, mesh_x_ohm_pu) * (3.0/RatedS[xfmr_name][1])
        #print('xfmr_name: ' + xfmr_name + ', zBaseP: ' + str(zBaseP) + ', r_ohm_pu: ' + str(r_ohm_pu) + ', mesh_x_ohm_pu: ' + str(mesh_x_ohm_pu) + ', zsc_1V: ' + str(zsc_1V))

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
        #print(Ycomp)

        bus1 = Bus[xfmr_name][1]
        bus2 = Bus[xfmr_name][2]

        # set special case flag that indicates if we need to swap the phases
        # for each bus to do the Ybus matching
        connect_DY_flag = Connection[xfmr_name][1]=='D' and Connection[xfmr_name][2]=='Y'

        if cmpFlag:
            print('Validating PowerTranformerEnd transformer_name: ' + xfmr_name, flush=True)
            print('Validating PowerTranformerEnd transformer_name: ' + xfmr_name, file=logfile)

            # do Ybus comparisons and determine overall transformer status color
            xfmrColorIdx = 0
            for row in range(4, 7):
                for col in range(0, 3):
                    Yval = Ycomp[row,col]
                    if Yval != 0j:
                        if connect_DY_flag:
                            colorIdx = compareY(bus1+'.'+str(row-3), bus2+'.'+str(col+1), Yval, Ybus)
                        else:
                            colorIdx = compareY(bus1+'.'+str(col+1), bus2+'.'+str(row-3), Yval, Ybus)

                        xfmrColorIdx = max(xfmrColorIdx, colorIdx)

            xfmrs_count += 1

            if xfmrColorIdx == 0:
                greenCount += 1
            elif xfmrColorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1

            print("", flush=True)
            print("", file=logfile)

        else:
            # delete row and column 8 and 4 making a 6x6 matrix
            Ycomp = np.delete(Ycomp, 7, 0)
            Ycomp = np.delete(Ycomp, 7, 1)
            Ycomp = np.delete(Ycomp, 3, 0)
            Ycomp = np.delete(Ycomp, 3, 1)

            fillYsys_6x6(bus1, bus2, connect_DY_flag, Ycomp, Ysys)

    if cmpFlag:
        print("\nSummary for PowerTransformerEnd transformers:", flush=True)
        print("\nSummary for PowerTransformerEnd transformers:", file=logfile)

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

        print("\nFinished validation for PowerTransformerEnd transformers", flush=True)
        print("\nFinished validation for PowerTransformerEnd transformers", file=logfile)

    return xfmrs_count


def validate_TransformerTank_xfmrs(sparql_mgr, Ybus, cmpFlag, Ysys, Unsupported):
    if cmpFlag:
        print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank validation...\n', flush=True)
        print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank validation...\n', file=logfile)

    # return # of xfmrs validated
    xfmrs_count = 0

    bindings = sparql_mgr.TransformerTank_xfmr_rated()
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_rated query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_rated query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank: NO TRANSFORMER MATCHES', flush=True)
            print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank: NO TRANSFORMER MATCHES', file=logfile)
        return xfmrs_count

    RatedS = {}
    RatedU = {}
    Connection = {}
    R_ohm = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        #xfmr_code = obj['xfmr_code']['value']
        enum = int(obj['enum']['value'])

        if xfmr_name not in RatedS:
            RatedS[xfmr_name] = {}
            RatedU[xfmr_name] = {}
            Connection[xfmr_name] = {}
            R_ohm[xfmr_name] = {}

        RatedS[xfmr_name][enum] = int(float(obj['ratedS']['value']))
        RatedU[xfmr_name][enum] = int(obj['ratedU']['value'])
        Connection[xfmr_name][enum] = obj['connection']['value']
        #angle = int(obj['angle']['value'])
        R_ohm[xfmr_name][enum] = float(obj['r_ohm']['value'])
        #print('xfmr_name: ' + xfmr_name + ', xfmr_code: ' + xfmr_code + ', enum: ' + str(enum) + ', ratedS: ' + str(RatedS[xfmr_name][enum]) + ', ratedU: ' + str(RatedU[xfmr_name][enum]) + ', connection: ' + Connection[xfmr_name][enum] + ', angle: ' + str(angle) + ', r_ohm: ' + str(R_ohm[xfmr_name][enum]))

    bindings = sparql_mgr.TransformerTank_xfmr_sct()
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_sct query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_sct query results:', file=logfile)
    #print(bindings, file=logfile)

    Leakage_z = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        enum = int(obj['enum']['value'])
        #gnum = int(obj['gnum']['value'])
        if xfmr_name not in Leakage_z:
            Leakage_z[xfmr_name] = {}

        Leakage_z[xfmr_name][enum] = float(obj['leakage_z']['value'])
        #loadloss = float(obj['loadloss']['value'])
        #print('xfmr_name: ' + xfmr_name + ', enum: ' + str(enum) + ', gnum: ' + str(gnum) + ', leakage_z: ' + str(Leakage_z[xfmr_name][enum]) + ', loadloss: ' + str(loadloss))

    #bindings = sparql_mgr.TransformerTank_xfmr_nlt()
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_nlt query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_nlt query results:', file=logfile)
    #print(bindings, file=logfile)

    bindings = sparql_mgr.TransformerTank_xfmr_names()
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('POWER_TRANSFORMER_VALIDATOR TransformerTank xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank: NO TRANSFORMER MATCHES', flush=True)
            print('\nPOWER_TRANSFORMER_VALIDATOR TransformerTank: NO TRANSFORMER MATCHES', file=logfile)
        return xfmrs_count

    Bus = {}
    Phase = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        #xfmr_code = obj['xfmr_code']['value']
        #vector_group = obj['vector_group']['value']
        enum = int(obj['enum']['value'])

        if xfmr_name not in Bus:
            Bus[xfmr_name] = {}
            Phase[xfmr_name] = {}

        Bus[xfmr_name][enum] = obj['bus']['value'].upper()
        #baseV = int(obj['baseV']['value'])
        Phase[xfmr_name][enum] = obj['phase']['value']
        #grounded = obj['grounded']['value']
        #rground = obj['rground']['value']
        #xground = obj['xground']['value']
        #print('xfmr_name: ' + xfmr_name + ', enum: ' + str(enum) + ', bus: ' + Bus[xfmr_name][enum] + ', phase: ' + Phase[xfmr_name][enum])
        #print('xfmr_name: ' + xfmr_name + ', xfmr_code: ' + xfmr_code + ', vector_group: ' + vector_group + ', enum: ' + str(enum) + ', bus: ' + Bus[xfmr_name][enum] + ', baseV: ' + str(baseV) + ', phase: ' + Phase[xfmr_name][enum] + ', grounded: ' + grounded)

    # initialize different variations of B upfront and then figure out later
    # which to use for each transformer
    # 3-phase
    B = {}
    B['3p'] = np.zeros((6,3))
    B['3p'][0,0] = B['3p'][2,1] = B['3p'][4,2] =  1.0
    B['3p'][1,0] = B['3p'][3,1] = B['3p'][5,2] = -1.0
    #print(B['3p'])
    # 1-phase, 2-windings
    B['2w'] = np.zeros((2,1))
    B['2w'][0,0] =  1.0
    B['2w'][1,0] = -1.0
    # 1-phase, 3-windings
    B['3w'] = np.zeros((3,2))
    B['3w'][0,0] = B['3w'][0,1] = B['3w'][2,1] = 1.0
    B['3w'][1,0] = -1.0

    # initialize Y and D matrices, also constant, used to set A for
    # 3-phase transformers
    Y1_3p = np.zeros((4,12))
    Y1_3p[0,0] = Y1_3p[1,4] = Y1_3p[2,8] = Y1_3p[3,1] = Y1_3p[3,5] = Y1_3p[3,9] = 1.0
    Y2_3p = np.zeros((4,12))
    Y2_3p[0,2] = Y2_3p[1,6] = Y2_3p[2,10] = Y2_3p[3,3] = Y2_3p[3,7] = Y2_3p[3,11] = 1.0
    D1_3p = np.zeros((4,12))
    D1_3p[0,0] = D1_3p[0,9] = D1_3p[1,1] = D1_3p[1,4] = D1_3p[2,5] = D1_3p[2,8] = 1.0
    D2_3p = np.zeros((4,12))
    D2_3p[0,2] = D2_3p[0,11] = D2_3p[1,3] = D2_3p[1,6] = D2_3p[2,7] = D2_3p[2,10] = 1.0

    # initialize A for each transformer variation
    A = {}
    A['2w'] = np.identity(4)
    A['3w'] = np.identity(6)
    A['3p_YY'] = np.vstack((Y1_3p, Y2_3p))
    A['3p_DD'] = np.vstack((D1_3p, D2_3p))
    A['3p_YD'] = np.vstack((Y1_3p, D2_3p))
    A['3p_DY'] = np.vstack((D1_3p, Y2_3p))

    # map transformer query phase values to nodelist indexes
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

    for xfmr_name in Bus:
        # determine the type of transformer to drive the computation
        if Phase[xfmr_name][1] == 'ABC':
            # 3-phase
            Bkey = '3p'
            Akey = Bkey + '_' + Connection[xfmr_name][1] + Connection[xfmr_name][2]
        elif 3 in Phase[xfmr_name]:
            # 1-phase, 3-winding
            Bkey = '3w'
            Akey = Bkey
        else:
            # 1-phase, 2-winding
            Bkey = '2w'
            Akey = Bkey

        # note that division is always floating point in Python 3 even if
        # operands are integer
        zBaseP = (RatedU[xfmr_name][1]*RatedU[xfmr_name][1])/RatedS[xfmr_name][1]
        zBaseS = (RatedU[xfmr_name][2]*RatedU[xfmr_name][2])/RatedS[xfmr_name][2]
        r_ohm_pu = R_ohm[xfmr_name][1]/zBaseP
        mesh_x_ohm_pu = Leakage_z[xfmr_name][1]/zBaseP

        if Bkey == '3p':
            zsc_1V = complex(2.0*r_ohm_pu, mesh_x_ohm_pu) * (3.0/RatedS[xfmr_name][1])
            # initialize ZB
            ZB = np.zeros((3,3), dtype=complex)
            ZB[0,0] = ZB[1,1] = ZB[2,2] = zsc_1V

            # initialize N
            if Connection[xfmr_name][1] == 'Y':
                Vp = RatedU[xfmr_name][1]/math.sqrt(3.0)
            else:
                Vp = RatedU[xfmr_name][1]

            if Connection[xfmr_name][2] == 'Y':
                Vs = RatedU[xfmr_name][2]/math.sqrt(3.0)
            else:
                Vs = RatedU[xfmr_name][2]

            N = np.zeros((12,6))
            N[0,0] = N[4,2] = N[8,4] =   1.0/Vp
            N[1,0] = N[5,2] = N[9,4] =  -1.0/Vp
            N[2,1] = N[6,3] = N[10,5] =  1.0/Vs
            N[3,1] = N[7,3] = N[11,5] = -1.0/Vs

        elif Bkey == '3w':
            zsc_1V = complex(3.0*r_ohm_pu, mesh_x_ohm_pu) * (1.0/RatedS[xfmr_name][1])
            zod_1V = complex(2.0*R_ohm[xfmr_name][2], Leakage_z[xfmr_name][2])/zBaseS * (1.0/RatedS[xfmr_name][2])
            # initialize ZB
            ZB = np.zeros((2,2), dtype=complex)
            ZB[0,0] = ZB[1,1] = zsc_1V
            ZB[1,0] = ZB[0,1] = 0.5*(zsc_1V + zsc_1V - zod_1V)

            #initialize N
            Vp = RatedU[xfmr_name][1]
            Vs1 = RatedU[xfmr_name][2]
            Vs2 = RatedU[xfmr_name][3]

            N = np.zeros((6,3))
            N[0,0] =  1.0/Vp
            N[1,0] = -1.0/Vp
            N[2,1] =  1.0/Vs1
            N[3,1] = -1.0/Vs1
            N[4,2] = -1.0/Vs2
            N[5,2] =  1.0/Vs2

        else:
            zsc_1V = complex(2.0*r_ohm_pu, mesh_x_ohm_pu) * (1.0/RatedS[xfmr_name][1])
            # initialize ZB
            ZB = np.zeros((1,1), dtype=complex)
            ZB[0,0] = zsc_1V

            #initialize N
            Vp = RatedU[xfmr_name][1]
            Vs = RatedU[xfmr_name][2]

            N = np.zeros((4,2))
            N[0,0] =  1.0/Vp
            N[1,0] = -1.0/Vp
            N[2,1] =  1.0/Vs
            N[3,1] = -1.0/Vs

        # compute Ycomp = A x N x B x inv(ZB) x B' x N' x A'
        # there are lots of ways to break this up including not at all, but
        # here's one way that keeps it from looking overly complex
        ANB = np.matmul(np.matmul(A[Akey], N), B[Bkey])
        ANB_invZB = np.matmul(ANB, np.linalg.inv(ZB))
        ANB_invZB_Bp = np.matmul(ANB_invZB, np.transpose(B[Bkey]))
        ANB_invZB_BpNp = np.matmul(ANB_invZB_Bp, np.transpose(N))
        Ycomp = np.matmul(ANB_invZB_BpNp, np.transpose(A[Akey]))
        #print(Ycomp)

        # do Ybus comparisons and determine overall transformer status color
        xfmrColorIdx = 0

        if cmpFlag:
            print('Validating TranformerTank transformer_name: ' + xfmr_name, flush=True)
            print('Validating TranformerTank transformer_name: ' + xfmr_name, file=logfile)

        if Bkey == '3p':
            bus1 = Bus[xfmr_name][1]
            bus2 = Bus[xfmr_name][2]

            if cmpFlag:
                for row in range(4, 7):
                    for col in range(0, 3):
                        Yval = Ycomp[row,col]
                        if Yval != 0j:
                            # special case where we need to swap the phases
                            # for each bus to do the Ybus matching
                            if Akey == '3p_DY':
                                colorIdx = compareY(bus1+'.'+str(row-3), bus2+'.'+str(col+1), Yval, Ybus)
                            else:
                                colorIdx = compareY(bus1+'.'+str(col+1), bus2+'.'+str(row-3), Yval, Ybus)

                            xfmrColorIdx = max(xfmrColorIdx, colorIdx)
            else:
                # delete row and column 8 and 4 making a 6x6 matrix
                Ycomp = np.delete(Ycomp, 7, 0)
                Ycomp = np.delete(Ycomp, 7, 1)
                Ycomp = np.delete(Ycomp, 3, 0)
                Ycomp = np.delete(Ycomp, 3, 1)

                fillYsys_6x6(bus1, bus2, Akey=='3p_DY', Ycomp, Ysys)

        elif Bkey == '3w':
            bus1 = Bus[xfmr_name][1] + ybusPhaseIdx[Phase[xfmr_name][1]]
            bus2 = Bus[xfmr_name][2] + ybusPhaseIdx[Phase[xfmr_name][2]]
            bus3 = Bus[xfmr_name][3] + ybusPhaseIdx[Phase[xfmr_name][3]]

            if cmpFlag:
                Yval = Ycomp[2,0]
                colorIdx20 = compareY(bus1, bus2, Yval, Ybus)

                Yval = Ycomp[5,0]
                colorIdx50 = compareY(bus1, bus3, Yval, Ybus)
                xfmrColorIdx = max(colorIdx20, colorIdx50)

            else:
                # split phase transformers are a bit tricky, but Shiva
                # figured out how it needs to be done with reducing the
                # matrix and how the 3 buses come into it

                # delete row and column 5, 4, and 2 making a 3x3 matrix
                Ycomp = np.delete(Ycomp, 4, 0)
                Ycomp = np.delete(Ycomp, 4, 1)
                Ycomp = np.delete(Ycomp, 3, 0)
                Ycomp = np.delete(Ycomp, 3, 1)
                Ycomp = np.delete(Ycomp, 1, 0)
                Ycomp = np.delete(Ycomp, 1, 1)

                fillYsysAdd(bus1, bus1, Ycomp[0,0], Ysys)
                fillYsysUnique(bus2, bus1, Ycomp[1,0], Ysys)
                fillYsysAdd(bus2, bus2, Ycomp[1,1], Ysys)
                fillYsysUnique(bus3, bus1, Ycomp[2,0], Ysys)
                fillYsysAdd(bus3, bus2, Ycomp[2,1], Ysys)
                fillYsysAdd(bus3, bus3, Ycomp[2,2], Ysys)

        else:
            bus1 = Bus[xfmr_name][1] + ybusPhaseIdx[Phase[xfmr_name][1]]
            bus2 = Bus[xfmr_name][2] + ybusPhaseIdx[Phase[xfmr_name][2]]
            Yval = Ycomp[2,0]

            if cmpFlag:
                xfmrColorIdx = compareY(bus1, bus2, Yval, Ybus)
            else:
                # delete row and column 4 and 2 making a 2x2 matrix
                Ycomp = np.delete(Ycomp, 3, 0)
                Ycomp = np.delete(Ycomp, 3, 1)
                Ycomp = np.delete(Ycomp, 1, 0)
                Ycomp = np.delete(Ycomp, 1, 1)

                fillYsysAdd(bus1, bus1, Ycomp[0,0], Ysys)
                fillYsysUnique(bus2, bus1, Ycomp[1,0], Ysys)
                fillYsysAdd(bus2, bus2, Ycomp[1,1], Ysys)

        if cmpFlag:
            xfmrs_count += 1

            if xfmrColorIdx == 0:
                greenCount += 1
            elif xfmrColorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1

            print("", flush=True)
            print("", file=logfile)

    if cmpFlag:
        print("\nSummary for TransformerTank transformers:", flush=True)
        print("\nSummary for TransformerTank transformers:", file=logfile)

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

        print("\nFinished validation for TransformerTank transformers", flush=True)
        print("\nFinished validation for TransformerTank transformers", file=logfile)

    return xfmrs_count


def start(log_file, feeder_mrid, model_api_topic, cmpFlag=True, Ysys=None, Unsupported=None):
    global logfile
    logfile = log_file

    if cmpFlag:
        print("\nPOWER_TRANSFORMER_VALIDATOR starting!!!-----------------------------------------")
        print("\nPOWER_TRANSFORMER_VALIDATOR starting!!!-----------------------------------------", file=logfile)

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

    #PowerTransformerEnd_xfmrs = 0
    PowerTransformerEnd_xfmrs = validate_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus, cmpFlag, Ysys, Unsupported)
    if cmpFlag:
        if PowerTransformerEnd_xfmrs > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["PowerTransformerEnd", PowerTransformerEnd_xfmrs, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["PowerTransformerEnd", PowerTransformerEnd_xfmrs])

    TransformerTank_xfmrs = validate_TransformerTank_xfmrs(sparql_mgr, Ybus, cmpFlag, Ysys, Unsupported)
    if cmpFlag:
        if TransformerTank_xfmrs > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append(["TransformerTank", TransformerTank_xfmrs, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append(["TransformerTank", TransformerTank_xfmrs])

        print('\n', flush=True)
        print(tabulate(report, headers=["Transformer Type", "# Transformers", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
        print('\n', file=logfile)
        print(tabulate(report, headers=["Transformer Type", "# Transformers", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

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
    log_file = open('power_transformer_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)


if __name__ == "__main__":
    _main()

