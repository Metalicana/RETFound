from typing import TypedDict, List, Dict, Any, Annotated
import operator
import numpy as np

class AgentState(TypedDict):
    patient_id: str
    metadata: Dict[str, Any]
    
    fundus_img: np.ndarray
    oct_img: np.ndarray
    clinical_narrative: str
    equity_opinion: str 
    
    oct_diagnosis: Any
    slo_diagnosis: Any  
    
    vision_opinion: Dict[str, Any]
    vision_opinion_oct: Dict[str, Any]
    vision_opinion_slo: Dict[str, Any]
    
    functional_opinion: Dict[str, Any]
    
    guidelines: str
    safety_output: str
    counterfactual_trace: Dict[str, Any]
    
    final_diagnosis: Dict[str, Any]
    fairness_flag: bool
