"""Module for querying and parsing SPARQL through GridAPPS-D"""
import logging
import pandas as pd
import numpy as np
import re
from gridappsd import GridAPPSD, topics, utils

# Map CIM booleans (come back as strings) to Python booleans.
BOOLEAN_MAP = {'true': True, 'false': False}
REGEX_1 = re.compile(r'^\s*\{\s*"data"\s*:\s*')
REGEX_2 = re.compile(r'\s*,\s*"responseComplete".+$')

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
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?name ?wnum ?bus ?phs ?eqid  WHERE { 
        VALUES ?fdrid {"%s"}
        ?s c:Equipment.EquipmentContainer ?fdr.
        ?fdr c:IdentifiedObject.mRID ?fdrid. 
        ?s r:type c:PowerTransformer.
        ?end c:TransformerTankEnd.TransformerTank ?tank.
        ?tank c:TransformerTank.PowerTransformer ?s.
        ?s c:IdentifiedObject.name ?name.
        ?s c:IdentifiedObject.mRID ?eqid.
        ?end c:TransformerEnd.Terminal ?trm.
        ?end c:TransformerEnd.endNumber ?wnum.
        ?trm c:IdentifiedObject.mRID ?trmid. 
        ?trm c:Terminal.ConnectivityNode ?cn. 
        ?cn c:IdentifiedObject.name ?bus.
        OPTIONAL {?end c:TransformerTankEnd.phases ?phsraw.
        bind(strafter(str(?phsraw),"PhaseCode.") as ?phs)}
        }
        ORDER BY ?name ?wnum ?phs
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
        acline_meas = self.gad.get_response(self.topic, message, timeout=180) 
        acline_meas = acline_meas['data']
        acline_measA = [m for m in acline_meas if m['type'] == 'A']
        df_acline_measA = pd.DataFrame(acline_measA)
        del df_acline_measA['trmid']
        del df_acline_measA['eqid']
        del df_acline_measA['measid']
        return df_acline_measA

    def get_glm(self):
        """Given a model ID, get a GridLAB-D (.glm) model."""
        payload = {'configurationType': 'GridLAB-D Base GLM',
                   'parameters': {'model_id': self.feeder_mrid}}
        response = self.gad.get_response(topic=topics.CONFIG, message=payload,
                                         timeout=self.timeout)
        # Fix bad json return.
        # TODO: remove when platform is fixed.
        glm = REGEX_2.sub('', REGEX_1.sub('', response['message']))
        return glm