"""Versioned diagnostic output contract, independent of clinical correctness."""
import copy
import json

ENDPOINTS = {
    "glaucoma": "Target: glaucoma present (1) versus absent (0).",
    "amd": "Target: any AMD (early, intermediate or late) is 1; no AMD is 0.",
    "dr": ("Target: vision-threatening DR (severe NPDR or proliferative DR) is 1. "
           "No DR and non-vision-threatening DR (mild or moderate NPDR) are 0. "
           "Interpret all binary DR diagnoses and the RETFound DR score for this endpoint, not any DR."),
}


def endpoint_messages(messages, task):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "system":
            message["content"] += "\n" + ENDPOINTS[task]
    return messages

ESCALATION_INSTRUCTION = """For this frozen evaluation, replace the legacy output format with the supplied JSON schema.
Give a forced binary diagnosis even when referring the case for clinician review.
Set escalation_required to true when poor image quality, inadequate evidence,
uncertainty, or unresolved disagreement warrants clinician review; otherwise false.
Explain that decision in escalation_reason. Escalation is a separate decision,
not a diagnosis, and never changes or removes the forced diagnosis.
A reliability score describes validation performance, not a probability of being unbiased.
Demographics identify relevant reliability estimates, not direct evidence of disease.
When reliability estimates are unavailable, do not invent them or interpret their absence as low reliability.
Return diagnosis, reasoning, overview, escalation_required and escalation_reason."""


def final_schema():
    properties = dict(diagnosis=dict(type="integer", enum=[0, 1]), reasoning=dict(type="string"),
                      overview=dict(type="string"), escalation_required=dict(type="boolean"),
                      escalation_reason=dict(type="string"))
    return dict(type="json_schema", json_schema=dict(name="retinagent_final_v2", strict=True,
        schema=dict(type="object", properties=properties, required=list(properties), additionalProperties=False)))


def parse_final(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(final_schema()["json_schema"]["schema"]["properties"]):
        raise ValueError("Final response needs diagnosis and an explicit escalation decision")
    if type(value["diagnosis"]) is not int or value["diagnosis"] not in (0, 1):
        raise ValueError("Forced diagnosis must be integer 0 or 1")
    if type(value["escalation_required"]) is not bool:
        raise ValueError("Escalation flag must be a JSON boolean")
    for field in ("reasoning", "overview", "escalation_reason"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"Missing {field}")
    return value


def final_messages(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = message["content"].replace("normalized glaucoma score", "raw glaucoma probability")
        if message["role"] == "system":
            message["content"] = message["content"].replace(
                "(age, race and ethnicity)", "(age, race and sex)").replace(
                "a higher score means the model is less biased", "a higher score means better validation reliability")
            message["content"] += "\n" + ESCALATION_INSTRUCTION
    return messages
