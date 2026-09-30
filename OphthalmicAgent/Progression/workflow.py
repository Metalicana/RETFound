"""Independent specialist calls, cached evidence ablation, and direct final labels."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from pathlib import Path

from .prompts import ENDPOINTS, SCENARIOS, SYSTEM_PROMPTS, VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def binary(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"Expected an integer binary label, got {value!r}")
    return value


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing nonempty report/reasoning")
    return value.strip()


def response_format(stage):
    if stage != "orchestrator":
        return {"type": "json_object"}

    def object_schema(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    endpoint = object_schema({
        "prediction": {"type": "integer", "enum": [0, 1]},
        "reasoning": {"type": "string"},
        "review_required": {"type": "boolean"},
    })
    schema = object_schema({"predictions": object_schema({t: endpoint for t in ENDPOINTS})})
    return {"type": "json_schema", "json_schema": {
        "name": "gdp_progression_orchestrator", "strict": True, "schema": schema,
    }}


class ResponseRefusal(RuntimeError):
    """A refusal is not a malformed prediction to repair or retry."""


def validate(stage, value):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    if stage not in {"counterfactual", "orchestrator"}:
        return {"report": text(value.get("report"))}
    if stage == "counterfactual":
        scenarios = value.get("scenarios", {})
        if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIOS):
            raise ValueError("Counterfactual response must contain all five scenarios")
        for name in SCENARIOS:
            scenario = scenarios[name]
            if not isinstance(scenario, dict) or not isinstance(scenario.get("predictions"), dict):
                raise ValueError(f"Invalid scenario {name}")
            if set(scenario["predictions"]) != set(ENDPOINTS):
                raise ValueError(f"Missing or extra endpoints in {name}")
            for label in scenario["predictions"].values():
                binary(label)
            text(scenario.get("reasoning"))
        text(value.get("interpretation"))
    else:
        predictions = value.get("predictions", {})
        if not isinstance(predictions, dict) or set(predictions) != set(ENDPOINTS):
            found = sorted(predictions) if isinstance(predictions, dict) else type(predictions).__name__
            raise ValueError("Final response must contain exactly all six endpoints under 'predictions'; "
                             f"expected={sorted(ENDPOINTS)} found={found} top_level_keys={sorted(value)}")
        for target, prediction in predictions.items():
            if not isinstance(prediction, dict):
                raise ValueError(f"Invalid endpoint {target}")
            binary(prediction.get("prediction"))
            text(prediction.get("reasoning"))
            if type(prediction.get("review_required")) is not bool:
                raise ValueError(f"Missing boolean review_required: {target}")
    return value


def messages(stage, evidence, image_path=None):
    content = [{"type": "text", "text": json.dumps(evidence, sort_keys=True, allow_nan=False)}]
    if image_path is not None:
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    return [{"role": "system", "content": SYSTEM_PROMPTS[stage]}, {"role": "user", "content": content}]


class CachedCaller:
    def __init__(self, root, deployment, client_factory, *, max_attempts=3, retry_sleep=5):
        self.root = Path(root)
        self.deployment = deployment
        self.client_factory = client_factory
        self.client = None
        self.max_attempts = max_attempts
        self.retry_sleep = retry_sleep

    def call(self, case_id, stage, evidence, image_path=None):
        request = {"model": self.deployment, "messages": messages(stage, evidence, image_path),
                   "temperature": 0.2, "response_format": response_format(stage),
                   "max_completion_tokens": 8000}
        signature = digest({"request": request, "prompt_version": VERSION})
        path = self.root / "stage_cache" / case_id / f"{stage}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["fingerprint"] != signature:
                raise ValueError(f"Cached evidence changed: {path}; use a new output directory")
            return validate(stage, json.loads(cached["raw_response"]))
        if self.client is None:
            self.client = self.client_factory()
        log = self.root / "attempts.jsonl"
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            record = {"case_id": case_id, "stage": stage, "attempt": attempt, "fingerprint": signature,
                      "response_format": request["response_format"]}
            try:
                response = self.client.chat.completions.create(**request)
                raw = response.choices[0].message.content or ""
                record["raw_response"] = raw
                usage = response.usage
                record["usage"] = {k: int(getattr(usage, k, 0) or 0) for k in (
                    "prompt_tokens", "completion_tokens", "total_tokens")}
                record["finish_reason"] = response.choices[0].finish_reason
                refusal = getattr(response.choices[0].message, "refusal", None)
                if refusal:
                    record["refusal"] = refusal
                    raise ResponseRefusal(f"{case_id}/{stage}: API refusal: {refusal}")
                if response.choices[0].finish_reason != "stop":
                    raise ValueError(f"Incomplete response: {response.choices[0].finish_reason}")
                result = validate(stage, json.loads(raw))
                write_json(path, {**record, "request": request, "prompt_version": VERSION})
                return result
            except Exception as error:
                last_error = error
                record["error"] = str(error)
                if isinstance(error, ResponseRefusal) or getattr(error, "status_code", None) in {400, 401, 403, 404}:
                    raise
                if attempt < self.max_attempts:
                    time.sleep(self.retry_sleep)
            finally:
                with log.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        raise RuntimeError(f"{case_id}/{stage} failed after {self.max_attempts} attempts: {last_error}") from last_error

    def close(self):
        if self.client is not None:
            self.client.close()


def ablation_packets(full):
    packets = {name: copy.deepcopy(full) for name in SCENARIOS}
    packets["without_helper_predictions"].pop("helper_predictions")
    packets["without_structural_interpretation"].pop("structural_reports")
    packets["without_functional_interpretation"].pop("functional_report")
    for key in ("patient_narrative", "reliability"):
        packets["without_demographic_reliability"].pop(key)
    return packets


def run_case(case, caller):
    """Only whitelisted baseline evidence enters this function; no labels or outcomes."""
    case_id = case["case_id"]
    bio = caller.call(case_id, "bio_profiler", {
        "demographics": case["demographics"], "modalities": case["modalities"],
    })
    structural = {"rnflt": caller.call(case_id, "rnflt_specialist", {
        "baseline_only": True, "statistics_source_units": case["rnflt_statistics"],
    }, case["rnflt_image"])["report"]}
    if "oct_image" in case:
        structural["oct"] = caller.call(case_id, "oct_specialist", {
            "baseline_only": True, "montage": "Five ordered slices from a single baseline volume",
        }, case["oct_image"])["report"]
    functional = caller.call(case_id, "functional_specialist", {
        "baseline_only": True, "total_deviation_db": case["td_values"],
        "descriptive_summary": case["td_summary"],
    })
    full = {"endpoints": ENDPOINTS, "patient_narrative": bio["report"],
            "helper_predictions": case["helper_predictions"], "reliability": case["reliability"],
            "structural_reports": structural, "functional_report": functional["report"]}
    audit = caller.call(case_id, "counterfactual", {
        "endpoints": ENDPOINTS, "scenario_evidence_packets": ablation_packets(full),
    })
    final = caller.call(case_id, "orchestrator", {**full, "counterfactual_trace": audit})
    return {"case_id": case_id, "predictions": final["predictions"],
            "bio_profiler": bio, "structural_reports": structural,
            "functional_specialist": functional, "counterfactual": audit}
