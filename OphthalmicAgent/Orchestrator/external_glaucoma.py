"""Isolated orchestrator for external glaucoma datasets."""
import json, os, re
from openai import AzureOpenAI

class ExternalGlaucomaOrchestrator:
    def __init__(self, evidence_modality, model_client=None):
        if evidence_modality not in {"oct","cfp"}: raise ValueError(evidence_modality)
        self.modality=evidence_modality
        self.deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-5.1")
        self.client=model_client or AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),api_key=os.getenv("AZURE_OPENAI_API_KEY"),api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-12-01-preview"))

    def analyze(self, probability, cfp_report, cdr, counterfactual_trace=None):
        if self.modality=="oct":
            relationship=("RETFound is an OCT-derived disease-probability tool. The visual report and CDR come from the paired CFP and provide complementary optic-nerve evidence. "
                          "Do not describe the probability as CFP-derived. A normal-appearing CFP does not automatically exclude glaucoma detected from OCT; resolve disagreement using strength, quality, and corroboration.")
        else:
            relationship=("RETFound, the visual report, and CDR are all derived from the CFP. They are different analyses of the same image, so do not treat them as independent votes or double-count agreement.")
        response=self.client.chat.completions.create(model=self.deployment,temperature=.2,messages=[
            {"role":"system","content":("You are the final glaucoma diagnostic orchestrator for an external-dataset evaluation. "+relationship+
              " The counterfactual trace is a dependency audit, not new disease evidence and not a vote. Do not invent findings. Return exactly:\n[LABELS]\nGLAUCOMA_DETECTED: [0 or 1]\n[/LABELS]\n\nReasoning:\n[brief evidence-based explanation].")},
            {"role":"user","content":f"RETFound-{self.modality.upper()} normalized glaucoma score: {probability}%\n\nPaired CFP specialist report: {cfp_report}\n\nVertical CDR: {cdr if cdr is not None else 'Not Available'}\n\nCounterfactual trace: {json.dumps(counterfactual_trace or {},sort_keys=True)}"}])
        raw=response.choices[0].message.content
        match=re.search(r"\[LABELS\](.*?)\[/LABELS\]",raw,re.S)
        return {"agent":"External_Glaucoma_Orchestrator","decision":raw,"labels":match.group(1).strip() if match else "GLAUCOMA_DETECTED: -1"}
