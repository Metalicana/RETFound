from typing import TypedDict, List, Dict, Any, Annotated
import operator
import numpy as np

class AgentState(TypedDict):
    patient_id: str
    metadata: Dict[str, Any]
    
    fundus_img: np.ndarray
    oct_img: np.ndarray
    clinical_narrative: str
    oct_diagnosis: Any
    slo_diagnosis: Any  
    
    vision_opinion: Dict[str, Any]
    functional_opinion: Dict[str, Any]
    
    final_diagnosis: Dict[str, Any]
    fairness_flag: bool