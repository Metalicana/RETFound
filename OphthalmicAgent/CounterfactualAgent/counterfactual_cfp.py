from CounterfactualAgent.counterfactual_agent import CounterfactualAgent


SCENARIO_NAMES = (
    "full_evidence",
    "without_retfound_probability",
    "without_visual_interpretation",
    "without_cdr_tool",
)


def _canonical_scenario_name(value):
    import re

    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "full": "full_evidence",
        "with_full_evidence": "full_evidence",
        "without_retfound": "without_retfound_probability",
        "without_probability": "without_retfound_probability",
        "without_retfound_score": "without_retfound_probability",
        "without_visual": "without_visual_interpretation",
        "without_cfp": "without_visual_interpretation",
        "without_cfp_interpretation": "without_visual_interpretation",
        "without_cdr": "without_cdr_tool",
    }
    return aliases.get(name, name)


def _scenario_rows(trace):
    raw = trace.get("scenarios", [])
    if isinstance(raw, dict):
        rows = []
        for name, value in raw.items():
            row = dict(value) if isinstance(value, dict) else {"diagnosis": value}
            row.setdefault("name", name)
            rows.append(row)
        if rows:
            return rows
    if isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]
        if rows:
            return rows

    # Some JSON responses place the four named scenarios at the top level.
    rows = []
    for name in SCENARIO_NAMES:
        value = trace.get(name)
        if value is None:
            continue
        row = dict(value) if isinstance(value, dict) else {"diagnosis": value}
        row.setdefault("name", name)
        rows.append(row)
    return rows


def _diagnosis(value):
    if isinstance(value, bool):
        return int(value)
    try:
        diagnosis = int(value)
    except (TypeError, ValueError):
        normalized = str(value or "").strip().lower()
        labels = {
            "glaucoma": 1,
            "glaucoma_detected": 1,
            "positive": 1,
            "no_glaucoma": 0,
            "normal": 0,
            "negative": 0,
            "inconclusive": -1,
            "indeterminate": -1,
            "uncertain": -1,
        }
        diagnosis = labels.get(_canonical_scenario_name(normalized), -2)
    if diagnosis not in (-1, 0, 1):
        raise ValueError(f"Invalid counterfactual diagnosis: {value!r}")
    return diagnosis


class CounterfactualCFPAgent(CounterfactualAgent):
    """CFP-only evidence ablation using the existing validated cache machinery."""

    def _messages(self, evidence):
        scenarios = "full_evidence, without_retfound_probability, without_visual_interpretation, without_cdr_tool"
        return [
            {
                "role": "system",
                "content": (
                    "You are a glaucoma counterfactual evidence-audit agent for a CFP-only pipeline. "
                    "Return JSON only with scenarios (name, diagnosis, confidence, reasoning) and "
                    f"interpretation. Use exactly: {scenarios}. Diagnosis is 1, 0, or -1. Missing evidence "
                    "means unavailable, not normal. Never introduce OCT, SLO, demographics, or history."
                ),
            },
            {"role": "user", "content": "Audit this CFP evidence:\n" + self._canonical(evidence)},
        ]

    @staticmethod
    def _canonical(value):
        import json
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _validate_trace(trace, case_id):
        by_name = {}
        for item in _scenario_rows(trace):
            supplied_name = item.get("name", item.get("scenario", item.get("scenario_name")))
            name = _canonical_scenario_name(supplied_name)
            if name in SCENARIO_NAMES:
                by_name[name] = item
        missing = [name for name in SCENARIO_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"Counterfactual response omitted scenarios: {missing}")
        scenarios = []
        for name in SCENARIO_NAMES:
            item = by_name[name]
            diagnosis = _diagnosis(
                item.get("diagnosis", item.get("label", item.get("prediction", -1)))
            )
            scenarios.append({"name": name, "diagnosis": diagnosis,
                              "confidence": str(item.get("confidence", "uncertain")),
                              "reasoning": str(item.get("reasoning", ""))})
        full = scenarios[0]["diagnosis"]
        flips = [item["name"] for item in scenarios[1:] if item["diagnosis"] != full]
        return {"case_id": case_id, "task": "glaucoma_cfp", "scenarios": scenarios,
                "full_evidence_diagnosis": full, "label_flip_scenarios": flips,
                "label_flip_count": len(flips), "evidence_sensitive": bool(flips),
                "interpretation": str(trace.get("interpretation", ""))}

    def analyze(self, *, case_id, retfound_probability, cfp_report, cdr):
        evidence = {
            "retfound_cfp_glaucoma_probability_percent": retfound_probability,
            "cfp_specialist_report": cfp_report,
            "vertical_cup_to_disc_ratio": cdr,
        }
        fingerprint = self._fingerprint(case_id, evidence)
        cached = self._cache.get(fingerprint)
        if cached is not None:
            result = dict(cached); result["cache_hit"] = True; return result
        response = self.model_client.chat.completions.create(
            model=self.deployment, messages=self._messages(evidence), temperature=0,
            response_format={"type": "json_object"},
        )
        from CounterfactualAgent.counterfactual_agent import _canonical_json, _extract_json_object
        raw = response.choices[0].message.content
        trace = self._validate_trace(_extract_json_object(raw), case_id)
        result = {**trace, "fingerprint": fingerprint, "prompt_version": "glaucoma_cfp_ablation_v1",
                  "deployment": self.deployment, "evidence": evidence, "raw_response": raw}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(result) + "\n")
        self._cache[fingerprint] = result
        returned = dict(result); returned["cache_hit"] = False; return returned
