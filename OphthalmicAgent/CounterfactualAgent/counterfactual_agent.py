"""Cached leave-one-evidence-source-out reasoning for glaucoma.

The counterfactual trace is an audit of evidence dependence. It is not a new
clinical measurement and must not be converted into a majority-vote diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROMPT_VERSION = "glaucoma_evidence_ablation_v1"
SCENARIOS = (
    "full_evidence",
    "without_retfound_probability",
    "without_cdr_tool",
    "without_visual_interpretation",
    "without_demographic_reliability",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Counterfactual agent did not return a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Counterfactual agent response must be a JSON object")
    return value


class CounterfactualAgent:
    """Generate each evidence-ablation trace once and reuse it from JSONL."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        model_client: Any | None = None,
        deployment: str | None = None,
    ) -> None:
        self.cache_path = Path(
            cache_path
            or os.getenv(
                "COUNTERFACTUAL_CACHE_PATH",
                "outputs/counterfactual/glaucoma_counterfactual_traces.jsonl",
            )
        )
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        if model_client is None:
            from openai import AzureOpenAI

            model_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        self.model_client = model_client
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            return rows
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid counterfactual cache JSON at {self.cache_path}:{line_number}"
                    ) from exc
                fingerprint = row.get("fingerprint")
                if isinstance(fingerprint, str):
                    rows[fingerprint] = row
        return rows

    def _fingerprint(self, case_id: str, evidence: dict[str, Any]) -> str:
        payload = {
            "case_id": case_id,
            "deployment": self.deployment,
            "evidence": evidence,
            "prompt_version": PROMPT_VERSION,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_trace(trace: dict[str, Any], case_id: str) -> dict[str, Any]:
        scenarios = trace.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("Counterfactual trace is missing scenarios")

        by_name: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            name = scenario.get("name")
            diagnosis = scenario.get("diagnosis")
            if name in SCENARIOS and diagnosis in (-1, 0, 1):
                by_name[str(name)] = {
                    "name": str(name),
                    "diagnosis": int(diagnosis),
                    "confidence": str(scenario.get("confidence", "uncertain")),
                    "reasoning": str(scenario.get("reasoning", "")).strip(),
                }

        missing = [name for name in SCENARIOS if name not in by_name]
        if missing:
            raise ValueError(f"Counterfactual response omitted scenarios: {missing}")

        full_label = by_name["full_evidence"]["diagnosis"]
        flips = [
            name
            for name in SCENARIOS[1:]
            if by_name[name]["diagnosis"] != full_label
        ]
        return {
            "case_id": case_id,
            "task": "glaucoma",
            "scenarios": [by_name[name] for name in SCENARIOS],
            "full_evidence_diagnosis": full_label,
            "label_flip_scenarios": flips,
            "label_flip_count": len(flips),
            "evidence_sensitive": bool(flips),
            "interpretation": str(trace.get("interpretation", "")).strip(),
        }

    def _messages(self, evidence: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a glaucoma counterfactual evidence-audit agent. Produce diagnoses under the "
                    "full evidence and four leave-one-source-out scenarios. 'Without' means unavailable, "
                    "not normal and not negative. Do not invent missing findings. Demography and its trust "
                    "score modify confidence in RETFound; they are not anatomical disease evidence. Each "
                    "scenario must be judged only from evidence remaining in that scenario. A diagnosis is "
                    "1 for glaucoma, 0 for no glaucoma, and -1 only when genuinely inconclusive. Return JSON "
                    "only with: scenarios (array of name, diagnosis, confidence, reasoning) and interpretation. "
                    f"Use exactly these scenario names: {', '.join(SCENARIOS)}. The interpretation must "
                    "briefly identify which removed source, if any, changes the diagnosis."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Run the evidence-ablation audit on this case. For without_retfound_probability, omit "
                    "the numeric RETFound probability but retain the other evidence. For without_cdr_tool, "
                    "treat CDR as unavailable. For without_visual_interpretation, omit both OCT and SLO "
                    "specialist reports. For without_demographic_reliability, omit the patient narrative and "
                    "trust score and use no subgroup adjustment.\n\nEVIDENCE_JSON:\n" + _canonical_json(evidence)
                ),
            },
        ]

    def analyze(
        self,
        *,
        case_id: str,
        patient_narrative: str,
        retfound_probability: float,
        oct_report: Any,
        slo_report: Any,
        cdr: Any,
        trust_score: float,
    ) -> dict[str, Any]:
        evidence = {
            "patient_narrative": patient_narrative,
            "retfound_glaucoma_probability_percent": retfound_probability,
            "oct_specialist_report": oct_report,
            "slo_specialist_report": slo_report,
            "vertical_cup_to_disc_ratio": cdr,
            "demographic_reliability_trust_score": trust_score,
        }
        fingerprint = self._fingerprint(case_id, evidence)
        cached = self._cache.get(fingerprint)
        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            return result

        response = self.model_client.chat.completions.create(
            model=self.deployment,
            messages=self._messages(evidence),
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_response = response.choices[0].message.content
        trace = self._validate_trace(_extract_json_object(raw_response), case_id)
        result = {
            **trace,
            "fingerprint": fingerprint,
            "prompt_version": PROMPT_VERSION,
            "deployment": self.deployment,
            "evidence": evidence,
            "raw_response": raw_response,
        }

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(result) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._cache[fingerprint] = result
        returned = dict(result)
        returned["cache_hit"] = False
        return returned

    @staticmethod
    def concise_trace(trace: dict[str, Any]) -> dict[str, Any]:
        """Return only the audit fields safe to pass to the final orchestrator."""
        return {
            "full_evidence_diagnosis": trace.get("full_evidence_diagnosis"),
            "scenarios": trace.get("scenarios", []),
            "label_flip_scenarios": trace.get("label_flip_scenarios", []),
            "evidence_sensitive": bool(trace.get("evidence_sensitive")),
            "interpretation": trace.get("interpretation", ""),
        }
