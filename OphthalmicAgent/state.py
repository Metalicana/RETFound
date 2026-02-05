from typing import TypedDict, List, Dict, Any, Annotated
import operator
import numpy as np

class AgentState(TypedDict):
    patient_id: str
    metadata: Dict[str, Any]
    
    fundus_img: np.ndarray
    clinical_narrative: str
    vision_features: Any  

    specialist_opinions: Annotated[List[Dict[str, Any]], operator.add]
    final_diagnosis: str
    fairness_flag: bool