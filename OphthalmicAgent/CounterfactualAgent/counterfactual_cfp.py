from CounterfactualAgent.counterfactual_agent import CounterfactualAgent


class CounterfactualCFPAgent(CounterfactualAgent):
    """CFP-only evidence ablation using the existing validated cache machinery."""

    def _messages(self, evidence):
        scenarios = "full_evidence, without_retfound_probability, without_visual_interpretation"
        return [
            {
                "role": "system",
                "content": (
                    "You are a glaucoma counterfactual evidence-audit agent for a CFP-only pipeline. "
                    "Return JSON only with scenarios (name, diagnosis, confidence, reasoning) and "
                    f"interpretation. Use exactly: {scenarios}. Diagnosis is 1, 0, or -1. Missing evidence "
                    "means unavailable, not normal. Never introduce OCT, SLO, demographics, history, or CDR."
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
        names = (
            "full_evidence",
            "without_retfound_probability",
            "without_visual_interpretation",
        )
        by_name = {item.get("name"): item for item in trace.get("scenarios", []) if isinstance(item, dict)}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Counterfactual response omitted scenarios: {missing}")
        scenarios = []
        for name in names:
            item = by_name[name]
            diagnosis = int(item.get("diagnosis", -1))
            if diagnosis not in (-1, 0, 1):
                raise ValueError(f"Invalid diagnosis for {name}: {diagnosis}")
            scenarios.append({"name": name, "diagnosis": diagnosis,
                              "confidence": str(item.get("confidence", "uncertain")),
                              "reasoning": str(item.get("reasoning", ""))})
        full = scenarios[0]["diagnosis"]
        flips = [item["name"] for item in scenarios[1:] if item["diagnosis"] != full]
        return {"case_id": case_id, "task": "glaucoma_cfp", "scenarios": scenarios,
                "full_evidence_diagnosis": full, "label_flip_scenarios": flips,
                "label_flip_count": len(flips), "evidence_sensitive": bool(flips),
                "interpretation": str(trace.get("interpretation", ""))}

    def analyze(self, *, case_id, retfound_probability, cfp_report):
        evidence = {
            "retfound_cfp_glaucoma_probability_percent": retfound_probability,
            "cfp_specialist_report": cfp_report,
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
