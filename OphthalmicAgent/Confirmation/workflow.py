"""Resumable frozen inference. Shared images never imply shared demographic decisions."""
from __future__ import annotations
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import run_fairvision_ablation as base
from Ablation import fairvision_live as live
from Confirmation.contract import parse_final
from Confirmation.design import demographics_profiles, locked_json, read_protocol


class BudgetClient:
    """Record total actual requests and lock the returned deployment identity."""
    def __init__(self, client, root, limit):
        self.client, self.root, self.limit = client, root, limit
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        ledger = self.root / "api_attempt_count.json"
        used = json.loads(ledger.read_text())["attempts"] if ledger.exists() else 0
        base.require(used < self.limit, "API attempt budget reached; valid stages are cached")
        base.require(not (self.root / "model_drift.json").exists(), "Returned model changed; inspect model_drift.json")
        base.write_json(ledger, dict(attempts=used + 1, limit=self.limit))
        response = self.client.chat.completions.create(**request)
        identity = dict(deployment=request["model"], returned_model=response.model)
        path = self.root / "model_identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            base.write_json(self.root / "model_drift.json", identity)
            raise RuntimeError("Azure returned a different model ID; run stopped")
        locked_json(path, identity)
        return response


def make_client(root, config, limit):
    from openai import AzureOpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE")
    base.require(endpoint and os.environ.get("AZURE_OPENAI_API_KEY"), "Missing Azure credentials")
    os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
    locked_json(root / "api_endpoint.json", dict(endpoint=endpoint, deployment=config["deployment"],
                                                  api_version=config["api_version"]))
    sdk = AzureOpenAI(azure_endpoint=endpoint, api_key=os.environ["AZURE_OPENAI_API_KEY"],
                      api_version=config["api_version"], max_retries=0, timeout=180)
    return BudgetClient(sdk, root, limit)


def decide(root, config, case, evidence, trust, orchestrator, client, receipt, variant, scenario="original"):
    path = root / "agent" / case["task"] / variant / f"{case['case_id']}.json"
    if scenario != "original":
        path = root / "demographic_audit" / case["task"] / case["case_id"] / f"{scenario}.json"
    identity = base.digest(dict(receipt=receipt, evidence=evidence, trust=trust, variant=variant, scenario=scenario))
    if path.exists():
        saved = json.loads(path.read_text())
        base.require(saved["evidence_fingerprint"] == identity and saved["fingerprint"] == config["fingerprint"]
                     and saved["case_id"] == case["case_id"] and saved["task"] == case["task"]
                     and saved["variant"] == variant and saved["scenario"] == scenario, "Decision input changed")
        final = parse_final(saved["decision"])
        base.require(final["diagnosis"] == saved["prediction"] and
                     final["escalation_required"] is saved["escalation_required"], "Cached decision/flag mismatch")
        return saved
    api_root = root / "api" / case["task"] / case["case_id"] / variant / scenario
    cf = live.CachedClient(client, api_root, config, case["task"], "counterfactual", live.validate_trace)
    response = cf.create(messages=live.counterfactual_messages(case["task"], evidence, trust))
    trace = live.validate_trace(response.choices[0].message.content)
    orchestrator.model_client = live.CachedClient(client, api_root, config, case["task"], "orchestrator",
                                                  parse_final, trust is None)
    state = dict(clinical_narrative=evidence["narrative"], vision_opinion_oct=evidence["oct_report"],
                 vision_opinion_slo=evidence["slo_report"])
    decision = orchestrator.analyze(state, evidence["probability_percent"], evidence["cdr"], trust, trace)
    final = parse_final(decision["decision"])
    saved = dict(fingerprint=config["fingerprint"], receipt=receipt, scenario=scenario,
        evidence_fingerprint=identity, task=case["task"],
        case_id=case["case_id"], variant=variant, prediction=final["diagnosis"],
        escalation_required=final["escalation_required"], escalation_reason=final["escalation_reason"],
        decision=decision["decision"], counterfactual_trace=trace, shared_evidence_sha256=base.digest(evidence))
    base.write_json(path, saved)
    return saved


def anchor_fitted(root, validation, task, receipt):
    scored = []
    for row in validation:
        if row["task"] != task:
            continue
        path = root / "anchor_validation" / task / f"{row['image_id']}.json"
        saved = json.loads(path.read_text())
        base.require(saved["fingerprint"] == base.digest(dict(receipt=receipt, row=row)), "Changed anchor validation row")
        scored.append({**row, "y_prob": saved["y_prob"], "y_pred": saved["y_pred"]})
    return base.fit_reliability(scored, task, "retfound_oct", fixed_threshold=.5)


def fairvision(root, config, cases, validation, receipt, client, device):
    from data.loader import ExcelEyeLoader
    manifest = root / "locked_manifest.csv"
    base.audit.write_csv(manifest, [dict(filename=c["filename"], Task_Folder=c["task"],
                                        Ground_Truth=c["truth"], **c["metadata"]) for c in cases])
    loader = ExcelEyeLoader(str(manifest))
    trusts = live.anchor_priors(root, config, cases, validation, loader, device, receipt)
    for task in base.TASKS:
        fitted = anchor_fitted(root, validation, task, receipt)
        oct_agent = live.make_oct(task, Path(config["oct_weights"]), device)
        slo = importlib.import_module(f"VisionAgent.vision_slo_{task}").VisionSpecialistSlo(None, device=device)
        locked_json(root / "cdr_model.json", dict(model=slo.cdr_model_name,
                    revision=getattr(slo.model_cdr.config, "_commit_hash", None)))
        base.require(json.loads((root / "cdr_model.json").read_text()) ==
                     json.loads((root / "external_cdr_model.json").read_text()), "FairVision/external CDR revisions differ")
        bio = importlib.import_module(f"BioProfilerAgent.bio_profiler_{task}").BioProfiler()
        orchestrator = importlib.import_module(f"Orchestrator.fairvision_{task}").Orchestrator()
        for i, case in enumerate(c for c in cases if c["task"] == task):
            evidence = live.shared_evidence(root, config, case, loader, oct_agent, slo, bio, client, receipt)
            arms = list(base.VARIANTS[2:])
            if int(base.digest([task, case["case_id"]])[:8], 16) % 2:
                arms.reverse()
            for variant in arms:
                trust = trusts[task][case["case_id"]] if variant == "retinagent_full" else None
                decide(root, config, case, evidence, trust, orchestrator, client, receipt, variant)
            for profile in demographics_profiles(case):
                changed = {**case, **{k: profile[k] for k in ("metadata", "age_group", "race", "sex_gender")}}
                trust = base.trust_for(changed, task, "retfound_oct", fitted)
                api_root = root / "api" / task / case["case_id"] / "demographics" / profile["name"]
                bio.model_client = live.CachedClient(client, api_root, config, task, "bioprofiler")
                # Both imaging specialists are metadata-blind. Reuse their reports,
                # not the original narrative, reliability, ablation or final decision.
                perturbed = {**evidence, "narrative": bio.generate_narrative(changed["metadata"])}
                saved = decide(root, config, changed, perturbed, trust, orchestrator, client, receipt,
                               "retinagent_full", profile["name"])
                locked_json(root / "demographic_audit" / task / case["case_id"] / f"{profile['name']}_profile.json",
                            dict(profile=profile, image_evidence_sha256=base.digest({k: v for k, v in evidence.items() if k != "narrative"}),
                                 decision_fingerprint=saved["evidence_fingerprint"]))
            print(f"complete {task} {i+1}/250 case={case['case_id']} paired arms + six live profiles", flush=True)
        del oct_agent, slo, bio, orchestrator


def external_tools(root, cases, receipt, device):
    import torch
    from PIL import Image, ImageOps
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from precompute_external_cdr import cdr_from_mask
    model_id = read_protocol()["cdr_model"]
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    locked_json(root / "external_cdr_model.json", dict(model=model_id, revision=getattr(model.config, "_commit_hash", None)))
    for case in cases:
        path = root / "external_cdr" / case["dataset"] / f"{case['case_id']}.json"
        identity = base.digest(dict(receipt=receipt, image=case["cfp_path"]))
        if path.exists():
            base.require(json.loads(path.read_text())["fingerprint"] == identity, "CDR cache changed")
            continue
        image = ImageOps.equalize(Image.open(case["cfp_path"]).convert("RGB"))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        vertical, _ = cdr_from_mask(logits.argmax(dim=1)[0].cpu().numpy())
        base.write_json(path, dict(fingerprint=identity, cdr=vertical))
    del model


def external_trace(raw):
    from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent, SCENARIO_NAMES
    parsed = json.loads(raw)
    base.require(isinstance(parsed.get("scenarios"), dict) and set(parsed["scenarios"]) == set(SCENARIO_NAMES),
                 "External trace must contain all four scenarios")
    for item in parsed["scenarios"].values():
        base.require(type(item.get("diagnosis")) is int and item["diagnosis"] in (-1, 0, 1), "Invalid external scenario")
    return CounterfactualCFPAgent._validate_trace(parsed, "external")


def external(root, config, cases, receipt, client):
    from CounterfactualAgent.external_glaucoma import ExternalCounterfactualAgent
    from CounterfactualAgent.counterfactual_cfp import SCENARIO_NAMES
    from Orchestrator.external_glaucoma import ExternalGlaucomaOrchestrator
    from run_external_glaucoma_agent import data_url
    for i, case in enumerate(cases):
        dataset, key = case["dataset"], case["case_id"]
        path = root / "external" / dataset / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            base.require(saved["receipt"] == receipt and saved["fingerprint"] == config["fingerprint"] and
                         saved["dataset"] == dataset and saved["case_id"] == key, "External source changed")
            final = parse_final(saved["decision"])
            base.require(final["diagnosis"] == saved["prediction"] and
                         final["escalation_required"] is saved["escalation_required"], "Cached external flag mismatch")
            continue
        api_root = root / "api" / dataset / key
        cdr = json.loads((root / "external_cdr" / dataset / f"{key}.json").read_text())["cdr"]
        imaging = live.CachedClient(client, api_root, config, "glaucoma", "cfp_report")
        response = imaging.create(messages=[dict(role="system", content=(
            "You are an ophthalmic imaging specialist reviewing one color fundus photograph for glaucoma. "
            "Assess gradability, vertical cupping, neuroretinal-rim thinning or notching, superior-inferior asymmetry, "
            "vessel displacement or bayoneting, laminar dots, disc hemorrhage, RNFL defects, and peripapillary atrophy. "
            "Do not invent a numerical cup-to-disc ratio and do not infer findings from any AI score. "
            "End with IMPRESSION: supports glaucoma, supports normal, or indeterminate, and name the features driving it.")),
            dict(role="user", content=[dict(type="text", text="Analyze this CFP for glaucoma-related structural findings."),
                                       dict(type="image_url", image_url=dict(url=data_url(Path(case["cfp_path"]))))])])
        report = response.choices[0].message.content
        template = ExternalCounterfactualAgent.__new__(ExternalCounterfactualAgent)
        template.modality = "cfp"
        evidence = dict(retfound_cfp_glaucoma_probability_percent=case["probability"] * 100,
                        paired_cfp_specialist_report=report, vertical_cup_to_disc_ratio=cdr, retfound_modality="cfp")
        messages = template._messages(evidence)
        messages[0]["content"] += " Return scenarios as an object keyed by the four specified names, not an array."
        schema = copy.deepcopy(live.response_format("counterfactual"))
        scenario = next(iter(schema["json_schema"]["schema"]["properties"]["scenarios"]["properties"].values()))
        schema["json_schema"]["schema"]["properties"]["scenarios"].update(
            properties={n: scenario for n in SCENARIO_NAMES}, required=list(SCENARIO_NAMES))
        cf = live.CachedClient(client, api_root, config, "glaucoma", "external_counterfactual", external_trace)
        raw = cf.create(messages=messages, response_format=schema).choices[0].message.content
        trace = external_trace(raw)
        orchestrator = ExternalGlaucomaOrchestrator.__new__(ExternalGlaucomaOrchestrator)
        orchestrator.modality, orchestrator.deployment = "cfp", config["deployment"]
        orchestrator.client = live.CachedClient(client, api_root, config, "glaucoma", "orchestrator", parse_final)
        raw = orchestrator.analyze(case["probability"] * 100, report, cdr, trace)["decision"]
        result = parse_final(raw)
        base.write_json(path, dict(fingerprint=config["fingerprint"], receipt=receipt, dataset=dataset,
            case_id=key, prediction=result["diagnosis"],
            escalation_required=result["escalation_required"], escalation_reason=result["escalation_reason"],
            decision=raw, counterfactual_trace=trace))
        print(f"complete external {i+1}/451 {dataset}/{key}", flush=True)
