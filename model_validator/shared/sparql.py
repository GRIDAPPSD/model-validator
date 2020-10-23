"""Module for querying and parsing SPARQL through GridAPPS-D"""
import logging
import pandas as pd
import numpy as np
import re
from gridappsd import GridAPPSD, topics, utils

class SPARQLManager:
    """Class for querying and parsing SPARQL in GridAPPS-D.
    """

    
    def __init__(self, gapps, feeder_mrid, model_api_topic, timeout=30):
        """Connect to the platform.

        :param feeder_mrid: unique identifier for the feeder in
            question. Since PyVVO works on a per feeder basis, this is
            required, and all queries will be executed for the specified
            feeder.
        :param gapps: gridappsd_object
        :param timeout: timeout for querying the blazegraph database.
        """

        # Connect to the platform.
        self.gad = gapps
       
        # Assign feeder mrid.
        self.feeder_mrid = feeder_mrid

        # Timeout for SPARQL queries.
        self.timeout = timeout

        # Powergridmodel API topic
        self.topic = model_api_topic

    def query_transformers(self):
        """Get information on transformers in the feeder."""
        # Perform the query.
        XFMR_QUERY = """
        PREFIX r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c: <http://iec.ch/TC57/CIM100#>
        SELECT ?pname ?tname ?xfmrcode ?bus ?vgrp ?enum ?ratedS ?ratedU WHERE {
        VALUES ?fdrid {"%s"}
        ?p r:type c:PowerTransformer.
        ?p c:Equipment.EquipmentContainer ?fdr.
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?p c:IdentifiedObject.name ?pname.
        ?p c:PowerTransformer.vectorGroup ?vgrp.
        ?t c:TransformerTank.PowerTransformer ?p.
        ?t c:IdentifiedObject.name ?tname.
        ?asset c:Asset.PowerSystemResources ?t.
        ?asset c:Asset.AssetInfo ?inf.
        ?inf c:IdentifiedObject.name ?xfmrcode.
        ?end c:TransformerTankEnd.TransformerTank ?t.
        ?end c:TransformerTankEnd.phases ?phsraw.
        ?end c:TransformerEnd.endNumber ?enum.
        ?end c:TransformerEnd.Terminal ?trm.
        ?trm c:Terminal.ConnectivityNode ?cn. 
        ?cn c:IdentifiedObject.name ?bus.
        ?asset c:Asset.PowerSystemResources ?t.
        ?asset c:Asset.AssetInfo ?tinf.
        ?einf c:TransformerEndInfo.TransformerTankInfo ?tinf.
        ?einf c:TransformerEndInfo.endNumber ?enum.
        ?einf c:TransformerEndInfo.ratedS ?ratedS.
        ?einf c:TransformerEndInfo.ratedU ?ratedU.
        }
        ORDER BY ?pname ?tname ?enum
        """% self.feeder_mrid
        results = self.gad.query_data(XFMR_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        output = pd.DataFrame(list_of_dicts)
        return output
    
    def query_energyconsumer(self):
        """Get information on loads in the feeder."""
        # Perform the query.
        LOAD_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus ?basev ?p ?q ?conn ?cnt ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe WHERE {
        ?s r:type c:EnergyConsumer.        
        VALUES ?fdrid {"%s"}         
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?s c:IdentifiedObject.name ?name.
        ?s c:ConductingEquipment.BaseVoltage ?bv.
        ?bv c:BaseVoltage.nominalVoltage ?basev.
        ?s c:EnergyConsumer.customerCount ?cnt.
        ?s c:EnergyConsumer.p ?p.
        ?s c:EnergyConsumer.q ?q.
        ?s c:EnergyConsumer.phaseConnection ?connraw.        
        bind(strafter(str(?connraw),"PhaseShuntConnectionKind.") as ?conn)
        ?s c:EnergyConsumer.LoadResponse ?lr.
        ?lr c:LoadResponseCharacteristic.pConstantImpedance ?pz.
        ?lr c:LoadResponseCharacteristic.qConstantImpedance ?qz.
        ?lr c:LoadResponseCharacteristic.pConstantCurrent ?pi.
        ?lr c:LoadResponseCharacteristic.qConstantCurrent ?qi.
        ?lr c:LoadResponseCharacteristic.pConstantPower ?pp.
        ?lr c:LoadResponseCharacteristic.qConstantPower ?qp.
        ?lr c:LoadResponseCharacteristic.pVoltageExponent ?pe.
        ?lr c:LoadResponseCharacteristic.qVoltageExponent ?qe.
        OPTIONAL {?ecp c:EnergyConsumerPhase.EnergyConsumer ?s.
        ?ecp c:EnergyConsumerPhase.phase ?phsraw.
        bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
        ?t c:Terminal.ConductingEquipment ?s.
        ?t c:Terminal.ConnectivityNode ?cn. 
        ?cn c:IdentifiedObject.name ?bus
        }
        GROUP BY ?name ?bus ?basev ?p ?q ?cnt ?conn ?pz ?qz ?pi ?qi ?pp ?qp ?pe ?qe 
        ORDER by ?name
        """% self.feeder_mrid
        results = self.gad.query_data(LOAD_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        output = pd.DataFrame(list_of_dicts)
        return output

    def acline_measurements(self):
        message = {
            "modelId": self.feeder_mrid,
            "requestType": "QUERY_OBJECT_MEASUREMENTS",
            "resultFormat": "JSON",
            "objectType": "ACLineSegment"}     
        try:
            acline_meas = self.gad.get_response(self.topic, message, timeout=180) 
            acline_meas = acline_meas['data']
            acline_measA = [m for m in acline_meas if m['type'] == 'A']
            df_acline_measA = pd.DataFrame(acline_measA)
            del df_acline_measA['trmid']
            del df_acline_measA['eqid']
            df_acline_measA = df_acline_measA.assign(rating = np.zeros(df_acline_measA.shape[0]))
            return df_acline_measA
        except:
            print('ACLINE_MEASUREMENTS current Measurements (A) are missing for ACLineSegment', flush = True)
            return 
        
    def acline_rating_query(self):
        # Getting the Ratings for ACLineSegment
        AC_LINE_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?eqtype ?eqname ?eqid ?val ?cn1id WHERE {
        VALUES ?fdrid {"%s"} 
        VALUES ?dur {"5E9"}
        # VALUES ?seq {"1"}
        VALUES ?eqtype {"ACLineSegment"}
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?eq c:Equipment.EquipmentContainer ?fdr.
        ?eq c:IdentifiedObject.name ?eqname.
        ?eq c:IdentifiedObject.mRID ?eqid.
        ?eq r:type ?rawtype.
        #bind(strafter(str(?rawtype),"#") as ?eqtype)
        ?t c:Terminal.ConductingEquipment ?eq.
        ?t c:ACDCTerminal.OperationalLimitSet ?ols.
        ?t c:ACDCTerminal.sequenceNumber ?seq.
        ?t c:Terminal.ConnectivityNode ?cn1.
        ?cn1 c:IdentifiedObject.mRID ?cn1id.
        ?clim c:OperationalLimit.OperationalLimitSet ?ols.
        ?clim r:type c:CurrentLimit.
        ?clim c:OperationalLimit.OperationalLimitType ?olt.
        ?olt c:OperationalLimitType.acceptableDuration ?dur.
        ?clim c:CurrentLimit.value ?val.
        }
        ORDER by ?eqtype ?eqname ?eqid ?val
        """% self.feeder_mrid
        results = self.gad.query_data(AC_LINE_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        rating_query = pd.DataFrame(list_of_dicts)
        return rating_query

    def graph_query(self):
        TERMINAL_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT  ?eqname ?id ?bus1 ?bus2 WHERE {
        VALUES ?fdrid {"%s"}  
        VALUES ?cimraw {c:LoadBreakSwitch c:Recloser c:Breaker c:PowerTransformer c:ACLineSegment c:Fuse}
        ?eq r:type ?cimraw.
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?eq c:Equipment.EquipmentContainer ?fdr.
        ?eq c:IdentifiedObject.mRID ?id.
        ?t1 c:Terminal.ConductingEquipment ?eq.
        ?t1 c:ACDCTerminal.sequenceNumber "1".
        ?t1 c:Terminal.ConnectivityNode ?cn1. 
        ?cn1 c:IdentifiedObject.name ?bus1.
        ?t2 c:Terminal.ConductingEquipment ?eq.
        ?t2 c:ACDCTerminal.sequenceNumber "2".
        ?t2 c:Terminal.ConnectivityNode ?cn2. 
        ?cn2 c:IdentifiedObject.name ?bus2.
        ?eq c:IdentifiedObject.name ?eqname.
        ?eq a ?classraw.
        }
        ORDER by  ?eqname 
        """% self.feeder_mrid
        results = self.gad.query_data(TERMINAL_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        graph_query = pd.DataFrame(list_of_dicts)
        return list_of_dicts
    
    def sourcebus_query(self):
        SOURCE_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus ?basev ?nomv ?vmag ?vang ?r1 ?x1 ?r0 ?x0 WHERE {
        ?s r:type c:EnergySource.
        VALUES ?fdrid {"%s"} 
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?s c:IdentifiedObject.name ?name.
        ?s c:ConductingEquipment.BaseVoltage ?bv.
        ?bv c:BaseVoltage.nominalVoltage ?basev.
        ?s c:EnergySource.nominalVoltage ?nomv. 
        ?s c:EnergySource.voltageMagnitude ?vmag. 
        ?s c:EnergySource.voltageAngle ?vang. 
        ?s c:EnergySource.r ?r1. 
        ?s c:EnergySource.x ?x1. 
        ?s c:EnergySource.r0 ?r0. 
        ?s c:EnergySource.x0 ?x0. 
        ?t c:Terminal.ConductingEquipment ?s.
        ?t c:Terminal.ConnectivityNode ?cn. 
        ?cn c:IdentifiedObject.name ?bus
        }
        ORDER by ?name
        """% self.feeder_mrid
        results = self.gad.query_data(SOURCE_QUERY)
        bindings = results['data']['results']['bindings']
        sourcebus = bindings[0]['bus']['value']
        return sourcebus
    
    def switch_query(self):
        LBS_QUERY = """
        PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX c:  <http://iec.ch/TC57/CIM100#>
        SELECT ?name ?bus1 ?bus2 ?id WHERE {
        SELECT ?cimtype ?name ?bus1 ?bus2 ?phs ?id WHERE {
        VALUES ?fdrid {"%s"}  # 9500 node
        VALUES ?cimraw {c:LoadBreakSwitch c:Recloser c:Breaker}
        ?fdr c:IdentifiedObject.mRID ?fdrid.
        ?s r:type ?cimraw.
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?s c:IdentifiedObject.name ?name.
        ?s c:IdentifiedObject.mRID ?id.
        ?t1 c:Terminal.ConductingEquipment ?s.
        ?t1 c:ACDCTerminal.sequenceNumber "1".
        ?t1 c:Terminal.ConnectivityNode ?cn1. 
        ?cn1 c:IdentifiedObject.name ?bus1.
        ?t2 c:Terminal.ConductingEquipment ?s.
        ?t2 c:ACDCTerminal.sequenceNumber "2".
        ?t2 c:Terminal.ConnectivityNode ?cn2. 
        ?cn2 c:IdentifiedObject.name ?bus2
            OPTIONAL {?swp c:SwitchPhase.Switch ?s.
            ?swp c:SwitchPhase.phaseSide1 ?phsraw.
            bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) }
        } ORDER BY ?name ?phs
        }
        GROUP BY ?cimtype ?name ?bus1 ?bus2 ?id
        ORDER BY ?cimtype ?name
        """% self.feeder_mrid
        results = self.gad.query_data(LBS_QUERY)
        bindings = results['data']['results']['bindings']
        list_of_dicts = []
        for obj in bindings:
            list_of_dicts.append({k:v['value'] for (k, v) in obj.items()})
        return list_of_dicts

    def switch_meas_query(self):
        message = {
        "modelId": self.feeder_mrid,
        "requestType": "QUERY_OBJECT_MEASUREMENTS",
        "resultFormat": "JSON",
        "objectType": "LoadBreakSwitch"}     
        obj_msr_loadsw = self.gad.get_response(self.topic, message, timeout=30) 
        obj_msr_loadsw = obj_msr_loadsw['data']
        obj_msr_loadsw = [d for d in obj_msr_loadsw if d['type'] == 'Pos']
        return obj_msr_loadsw
