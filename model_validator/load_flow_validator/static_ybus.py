# ------------------------------------------------------------------------------
# Copyright (c) 2022, Battelle Memorial Institute All rights reserved.
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
# ------------------------------------------------------------------------------
"""
Created on Jan 6, 2022

@author: Gary Black
"""""

import sys
import os
import argparse
import json
import importlib
import math
import numpy as np

from gridappsd import GridAPPSD

# START TRANSFORMERS
def fillYbusUnique_xfmrs(bus1, bus2, Yval, Ybus):
    if Yval == 0j:
        return

    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ybus[' + bus1 + '][' + bus2 + '] when filling power transformer value\n', flush=True)

    Ybus[bus1][bus2] = Yval


def fillYbusAdd_xfmrs(bus1, bus2, Yval, Ybus):
    if Yval == 0j:
        return

    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        Ybus[bus1][bus2] += Yval
    else:
        Ybus[bus1][bus2] = Yval


def fillYbus_6x6_xfmrs(bus1, bus2, DY_flag, Ycomp, Ybus):
    # fill Ybus directly from Ycomp
    # first fill the ones that are independent of DY_flag
    # either because the same bus is used or the same phase
    fillYbusAdd_xfmrs(bus1+'.1', bus1+'.1', Ycomp[0,0], Ybus)
    fillYbusAdd_xfmrs(bus1+'.2', bus1+'.1', Ycomp[1,0], Ybus)
    fillYbusAdd_xfmrs(bus1+'.2', bus1+'.2', Ycomp[1,1], Ybus)
    fillYbusAdd_xfmrs(bus1+'.3', bus1+'.1', Ycomp[2,0], Ybus)
    fillYbusAdd_xfmrs(bus1+'.3', bus1+'.2', Ycomp[2,1], Ybus)
    fillYbusAdd_xfmrs(bus1+'.3', bus1+'.3', Ycomp[2,2], Ybus)
    fillYbusUnique_xfmrs(bus2+'.1', bus1+'.1', Ycomp[3,0], Ybus)
    fillYbusAdd_xfmrs(bus2+'.1', bus2+'.1', Ycomp[3,3], Ybus)
    fillYbusUnique_xfmrs(bus2+'.2', bus1+'.2', Ycomp[4,1], Ybus)
    fillYbusAdd_xfmrs(bus2+'.2', bus2+'.1', Ycomp[4,3], Ybus)
    fillYbusAdd_xfmrs(bus2+'.2', bus2+'.2', Ycomp[4,4], Ybus)
    fillYbusUnique_xfmrs(bus2+'.3', bus1+'.3', Ycomp[5,2], Ybus)
    fillYbusAdd_xfmrs(bus2+'.3', bus2+'.1', Ycomp[5,3], Ybus)
    fillYbusAdd_xfmrs(bus2+'.3', bus2+'.2', Ycomp[5,4], Ybus)
    fillYbusAdd_xfmrs(bus2+'.3', bus2+'.3', Ycomp[5,5], Ybus)

    # now fill the ones that are dependent on DY_flag, which
    # are different bus and different phase
    if DY_flag:
        fillYbusUnique_xfmrs(bus2+'.1', bus1+'.2', Ycomp[4,0], Ybus)
        fillYbusUnique_xfmrs(bus2+'.1', bus1+'.3', Ycomp[5,0], Ybus)
        fillYbusUnique_xfmrs(bus2+'.2', bus1+'.1', Ycomp[3,1], Ybus)
        fillYbusUnique_xfmrs(bus2+'.2', bus1+'.3', Ycomp[5,1], Ybus)
        fillYbusUnique_xfmrs(bus2+'.3', bus1+'.1', Ycomp[3,2], Ybus)
        fillYbusUnique_xfmrs(bus2+'.3', bus1+'.2', Ycomp[4,2], Ybus)

    else:
        fillYbusUnique_xfmrs(bus2+'.2', bus1+'.1', Ycomp[4,0], Ybus)
        fillYbusUnique_xfmrs(bus2+'.3', bus1+'.1', Ycomp[5,0], Ybus)
        fillYbusUnique_xfmrs(bus2+'.1', bus1+'.2', Ycomp[3,1], Ybus)
        fillYbusUnique_xfmrs(bus2+'.3', bus1+'.2', Ycomp[5,1], Ybus)
        fillYbusUnique_xfmrs(bus2+'.1', bus1+'.3', Ycomp[3,2], Ybus)
        fillYbusUnique_xfmrs(bus2+'.2', bus1+'.3', Ycomp[4,2], Ybus)


def fill_Ybus_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus):
    bindings = sparql_mgr.PowerTransformerEnd_xfmr_impedances()
    #print('POWER_TRANSFORMER_FILL_YBUS PowerTransformerEnd xfmr_impedances query results:', flush=True)
    #print(bindings, flush=True)

    if len(bindings) == 0:
        return

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
        return

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

        # delete row and column 8 and 4 making a 6x6 matrix
        Ycomp = np.delete(Ycomp, 7, 0)
        Ycomp = np.delete(Ycomp, 7, 1)
        Ycomp = np.delete(Ycomp, 3, 0)
        Ycomp = np.delete(Ycomp, 3, 1)

        fillYbus_6x6_xfmrs(bus1, bus2, connect_DY_flag, Ycomp, Ybus)


def fill_Ybus_TransformerTank_xfmrs(sparql_mgr, Ybus):
    bindings = sparql_mgr.TransformerTank_xfmr_rated()
    #print('POWER_TRANSFORMER_FILL_YBUS TransformerTank xfmr_rated query results:', flush=True)
    #print(bindings, flush=True)

    if len(bindings) == 0:
        return

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
    #print('POWER_TRANSFORMER_FILL_YBUS TransformerTank xfmr_sct query results:', flush=True)
    #print(bindings, flush=True)

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
    #print('POWER_TRANSFORMER_FILL_YBUS TransformerTank xfmr_nlt query results:', flush=True)
    #print(bindings, flush=True)

    bindings = sparql_mgr.TransformerTank_xfmr_names()
    #print('POWER_TRANSFORMER_FILL_YBUS TransformerTank xfmr_names query results:', flush=True)
    #print(bindings, flush=True)

    if len(bindings) == 0:
        return

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

        if Bkey == '3p':
            bus1 = Bus[xfmr_name][1]
            bus2 = Bus[xfmr_name][2]

            # delete row and column 8 and 4 making a 6x6 matrix
            Ycomp = np.delete(Ycomp, 7, 0)
            Ycomp = np.delete(Ycomp, 7, 1)
            Ycomp = np.delete(Ycomp, 3, 0)
            Ycomp = np.delete(Ycomp, 3, 1)

            fillYbus_6x6_xfmrs(bus1, bus2, Akey=='3p_DY', Ycomp, Ybus)

        elif Bkey == '3w':
            bus1 = Bus[xfmr_name][1] + ybusPhaseIdx[Phase[xfmr_name][1]]
            bus2 = Bus[xfmr_name][2] + ybusPhaseIdx[Phase[xfmr_name][2]]
            bus3 = Bus[xfmr_name][3] + ybusPhaseIdx[Phase[xfmr_name][3]]

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

            fillYbusAdd_xfmrs(bus1, bus1, Ycomp[0,0], Ybus)
            fillYbusUnique_xfmrs(bus2, bus1, Ycomp[1,0], Ybus)
            fillYbusAdd_xfmrs(bus2, bus2, Ycomp[1,1], Ybus)
            fillYbusUnique_xfmrs(bus3, bus1, Ycomp[2,0], Ybus)
            fillYbusAdd_xfmrs(bus3, bus2, Ycomp[2,1], Ybus)
            fillYbusAdd_xfmrs(bus3, bus3, Ycomp[2,2], Ybus)

        else:
            bus1 = Bus[xfmr_name][1] + ybusPhaseIdx[Phase[xfmr_name][1]]
            bus2 = Bus[xfmr_name][2] + ybusPhaseIdx[Phase[xfmr_name][2]]
            Yval = Ycomp[2,0]

            # delete row and column 4 and 2 making a 2x2 matrix
            Ycomp = np.delete(Ycomp, 3, 0)
            Ycomp = np.delete(Ycomp, 3, 1)
            Ycomp = np.delete(Ycomp, 1, 0)
            Ycomp = np.delete(Ycomp, 1, 1)

            fillYbusAdd_xfmrs(bus1, bus1, Ycomp[0,0], Ybus)
            fillYbusUnique_xfmrs(bus2, bus1, Ycomp[1,0], Ybus)
            fillYbusAdd_xfmrs(bus2, bus2, Ycomp[1,1], Ybus)
# FINISH TRANSFORMERS

# START SWITCHES
def fillYbusUnique_switches(bus1, bus2, Ybus):
    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ybus[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ybus[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', file=logfile)

    # if needed, here's how to find the two immediate calling functions
    #if bus2=='X2673305B.1' and bus1=='X2673305B.2':
    #    print('*** fillYbusUnique bus1: ' + bus1 + ', bus2: ' + bus2 + ', caller: ' + str(inspect.stack()[1].function) + ', ' + str(inspect.stack()[2].function), flush=True)

    Ybus[bus1][bus2] = complex(-500.0, 500.0)


def fillYbusAdd_switches(bus1, bus2, Ybus):
    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        Ybus[bus1][bus2] += complex(500.0, -500.0)
    else:
        Ybus[bus1][bus2] = complex(500.0, -500.0)


def fillYbusNoSwap_switches(bus1, bus2, is_Open, Ybus):
    #print('fillYbusNoSwap bus1: ' + bus1 + ', bus2: ' + bus2 + ', is_Open: ' + str(is_Open), flush=True)
    if not is_Open:
        fillYbusUnique_switches(bus2, bus1, Ybus)
        fillYbusAdd_switches(bus1, bus1, Ybus)
        fillYbusAdd_switches(bus2, bus2, Ybus)

def fill_Ybus_SwitchingEquipment_switches(sparql_mgr, Ybus):
    bindings = sparql_mgr.SwitchingEquipment_switch_names()
    #print('SWITCHING_EQUIPMENT_FILL_YBUS switch_names query results:', flush=True)
    #print(bindings, flush=True)

    if len(bindings) == 0:
        return

    # map transformer query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3'}

    for obj in bindings:
        sw_name = obj['sw_name']['value']
        #base_V = int(obj['base_V']['value'])
        is_Open = obj['is_Open']['value'].upper() == 'TRUE'
        #rated_Current = int(obj['rated_Current']['value'])
        #breaking_Capacity = int(obj['breaking_Capacity']['value'])
        #sw_ph_status = obj['sw_ph_status']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        phases_side1 = obj['phases_side1']['value']
        #phases_side2 = obj['phases_side2']['value']
        #print('sw_name: ' + sw_name + ', is_Open: ' + str(is_Open) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', phases_side1: (' + phases_side1 + ')' + ', phases_side2: (' + phases_side2 + ')')

        if phases_side1 == '':
            # 3-phase switch
            #print('3-phase switch found bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
            fillYbusNoSwap_switches(bus1+'.1', bus2+'.1', is_Open, Ybus)
            fillYbusNoSwap_switches(bus1+'.2', bus2+'.2', is_Open, Ybus)
            fillYbusNoSwap_switches(bus1+'.3', bus2+'.3', is_Open, Ybus)

        else:
            # 1- or 2-phase switch
            switchColorIdx = 0
            for phase in phases_side1:
                #print('1/2-phase switch found phase: ' + phase + ', bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
                if phase in ybusPhaseIdx:
                    fillYbusNoSwap_switches(bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], is_Open, Ybus)
# FINISH SWITCHES


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    print('\nStarting to build static Ybus...', flush=True)

    Ybus = {}
    Unsupported = {}

    mod_import = importlib.import_module('line_model_validator.line_model_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ybus, Unsupported)
    #print('line_model_validator static Ybus...')
    #print(Ybus)
    line_count = 0
    for bus1 in Ybus:
        line_count += len(Ybus[bus1])
    print('\nLine_model # entries: ' + str(line_count), flush=True)

    fill_Ybus_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus)
    fill_Ybus_TransformerTank_xfmrs(sparql_mgr, Ybus)
    #print('power_transformer_validator static Ybus...')
    #print(Ybus)
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    xfmr_count = count - line_count
    print('\nPower_transformer # entries: ' + str(xfmr_count), flush=True)

    fill_Ybus_SwitchingEquipment_switches(sparql_mgr, Ybus)
    #print('switching_equipment_validator (final) static Ybus...')
    #print(Ybus)
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    switch_count = count - line_count - xfmr_count
    print('\nSwitching_equipment # entries: ' + str(switch_count), flush=True)

    print('\nFull static Ybus:')
    for bus1 in Ybus:
        for bus2 in Ybus[bus1]:
            print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag))

    ysysCount = 0
    for bus1 in Ybus:
        ysysCount += len(Ybus[bus1])
    print('\nTotal static Ybus # entries: ' + str(ysysCount) + '\n', flush=True)


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
    log_file = open('static_ybus.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)


if __name__ == "__main__":
    _main()

