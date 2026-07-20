"""Isolated modality-aware counterfactual audit for external datasets."""
from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent

class ExternalCounterfactualAgent(CounterfactualCFPAgent):
    def __init__(self, evidence_modality, *args, **kwargs):
        if evidence_modality not in {"oct","cfp"}: raise ValueError(evidence_modality)
        self.modality=evidence_modality
        super().__init__(*args,**kwargs)

    def _messages(self,evidence):
        scenarios="full_evidence, without_retfound_probability, without_visual_interpretation, without_cdr_tool"
        relationship=("The RETFound probability comes from OCT; the visual report and CDR come from the paired CFP. These are complementary modalities. A normal CFP does not automatically negate OCT evidence."
                      if self.modality=="oct" else
                      "RETFound, the visual report, and CDR come from the same CFP and must not be treated as independent votes.")
        return [{"role":"system","content":("You are a glaucoma counterfactual evidence-audit agent for external evaluation. "+relationship+
                 " Return JSON only with scenarios (name, diagnosis, confidence, reasoning) and interpretation. Use exactly: "+scenarios+". Diagnosis is 1, 0, or -1. Missing evidence means unavailable, not normal.")},
                {"role":"user","content":"Audit this modality-labelled evidence:\n"+self._canonical(evidence)}]

    def analyze(self,*,case_id,retfound_probability,cfp_report,cdr):
        evidence={f"retfound_{self.modality}_glaucoma_probability_percent":retfound_probability,
                  "paired_cfp_specialist_report":cfp_report,"vertical_cup_to_disc_ratio":cdr,"retfound_modality":self.modality}
        fingerprint=self._fingerprint(case_id,evidence);cached=self._cache.get(fingerprint)
        if cached is not None:
            result=dict(cached);result["cache_hit"]=True;return result
        response=self.model_client.chat.completions.create(model=self.deployment,messages=self._messages(evidence),temperature=0,response_format={"type":"json_object"})
        from CounterfactualAgent.counterfactual_agent import _canonical_json,_extract_json_object
        raw=response.choices[0].message.content
        try:
            trace=self._validate_trace(_extract_json_object(raw),case_id)
        except Exception as exc:
            raise ValueError(f"{exc}; raw_response={raw[:4000]!r}") from exc
        result={**trace,"fingerprint":fingerprint,"prompt_version":f"external_glaucoma_{self.modality}_ablation_v1","deployment":self.deployment,"evidence":evidence,"raw_response":raw}
        self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        with self.cache_path.open("a",encoding="utf-8") as handle:handle.write(_canonical_json(result)+"\n")
        self._cache[fingerprint]=result;returned=dict(result);returned["cache_hit"]=False;return returned
