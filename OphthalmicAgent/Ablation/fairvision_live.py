"""Paired reliability ablation using the task-specific OphthalmicAgent components.

Only inference is performed. Labels enter validation-prior estimation and final
evaluation, never the API evidence. Calls and local image reports are resumable.
"""
from __future__ import annotations

import copy
import importlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from CounterfactualAgent.counterfactual_agent import CounterfactualAgent, SCENARIOS
from run_fairvision_ablation import (
    ROOT, TASKS, VARIANTS, canonical, cdr_value, digest, fit_reliability, probability,
    require, require_images, resolve_image_path, selected_tasks, sha, trust_for, write_json,
)

DISEASES = {"amd": "age-related macular degeneration (AMD)", "dr": "diabetic retinopathy",
            "glaucoma": "glaucoma"}
HEADS = {"amd": "AMD", "dr": "DR", "glaucoma": "Glaucoma"}


def response_format(stage):
    def obj(properties):
        return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)
    string = {"type": "string"}
    scenario = obj(dict(diagnosis={"type": "integer", "enum": [-1, 0, 1]}, confidence=string, reasoning=string))
    schema = (obj(dict(scenarios=obj({name: scenario for name in SCENARIOS}), interpretation=string))
              if stage == "counterfactual" else
              obj(dict(diagnosis={"type": "integer", "enum": [0, 1]}, reasoning=string, overview=string)))
    return dict(type="json_schema", json_schema=dict(name="fairvision_" + stage, strict=True, schema=schema))


def final_label(raw, task):
    if raw.lstrip().startswith("{"):
        value = json.loads(raw)
        require(set(value) == {"diagnosis", "reasoning", "overview"}, "Invalid final JSON fields")
        require(type(value["diagnosis"]) is int and value["diagnosis"] in (0, 1), "Invalid final label")
        require(all(isinstance(value[k], str) and value[k].strip() for k in ("reasoning", "overview")), "Missing final explanation")
        return value["diagnosis"]
    blocks = re.findall(r"\[LABELS\](.*?)\[/LABELS\]", raw, re.S)
    require(len(blocks) == 1, "Expected exactly one LABELS block; no negative fallback")
    match = re.fullmatch(rf"\s*{task.upper()}_DETECTED:\s*([01])\s*", blocks[0])
    require(match is not None, f"Invalid {task} label block")
    return int(match.group(1))


def validate_trace(raw):
    value = json.loads(raw)
    require(isinstance(value, dict), "Counterfactual response must be an object")
    scenarios = value.get("scenarios", [])
    if isinstance(scenarios, dict):
        require(set(scenarios) == set(SCENARIOS), "Scenario names differ")
        scenarios = [dict(**scenarios[name], name=name) for name in SCENARIOS]
        value["scenarios"] = scenarios
    require(isinstance(scenarios, list) and len(scenarios) == 5, "Need all five scenarios")
    require(all(isinstance(r, dict) for r in scenarios), "Invalid scenario objects")
    require({r.get("name") for r in scenarios} == set(SCENARIOS), "Scenario names differ")
    for row in scenarios:
        require(type(row.get("diagnosis")) is int and row["diagnosis"] in (-1, 0, 1), "Invalid scenario diagnosis")
        require(isinstance(row.get("reasoning"), str) and row["reasoning"].strip(), "Missing scenario rationale")
    require(isinstance(value.get("interpretation"), str), "Missing interpretation")
    return value


def counterfactual_messages(task, evidence, trust):
    # Adapt the existing evidence-ablation prompt to its actual diagnostic task.
    template = CounterfactualAgent.__new__(CounterfactualAgent)
    payload = dict(patient_narrative=evidence["narrative"],
                   retfound_probability_percent=evidence["probability_percent"],
                   oct_specialist_report=evidence["oct_report"],
                   slo_specialist_report=evidence["slo_report"],
                   vertical_cup_to_disc_ratio=evidence["cdr"])
    if trust is not None:
        payload["demographic_reliability_trust_score"] = trust
    messages = template._messages(payload)
    messages[0]["content"] = messages[0]["content"].replace("glaucoma", DISEASES[task])
    messages[0]["content"] += (
        " An unavailable reliability estimate is neither low reliability nor proof of disease."
        " Do not invent validation performance. Keep each rationale to one short sentence."
        " For the response schema, scenarios is an object keyed by the five scenario names,"
        " rather than an array. Each value contains diagnosis, confidence and reasoning."
    )
    return messages


def redact_images(value):
    if isinstance(value, dict):
        return {k: redact_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_images(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "image_sha256:" + digest(value)
    return value


def orchestrator_messages(messages, no_priors):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "system":
            message["content"] += (
                "\nA trust score describes validation performance, not the probability that a model is unbiased."
                " If no reliability estimate is supplied, do not invent one or treat its absence as low reliability."
                "\nFor this experiment, replace the legacy LABELS text format above with the supplied JSON schema:"
                " diagnosis (0 or 1), reasoning and overview. Retain the diagnostic instructions above."
            )
        elif no_priors:
            message["content"] = re.sub(r"(?m)^.*\*\*Trust Score\*\*:.*\n?", "", message["content"])
    return messages


class CachedClient:
    """Small SDK-compatible adapter; validate before caching any model response."""

    def __init__(self, client, root, config, task, stage, validator=None, no_priors=False):
        self.client, self.root, self.config = client, root, config
        self.task, self.stage = task, stage
        self.validator = validator
        self.no_priors = no_priors
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        request = copy.deepcopy(request)
        request["model"] = self.config["deployment"]
        request.setdefault("temperature", .2)
        request["max_completion_tokens"] = 8192 if self.stage == "counterfactual" else 4096
        if self.stage in ("orchestrator", "counterfactual"):
            request["response_format"] = response_format(self.stage)
        if self.stage == "orchestrator":
            request["messages"] = orchestrator_messages(request["messages"], self.no_priors)
        frozen = self.config.get("frozen_generation")
        if frozen:
            from Confirmation.contract import endpoint_messages, final_messages, final_schema
            request["temperature"] = frozen["temperature"]
            request["top_p"] = frozen["top_p"]
            request["max_completion_tokens"] = frozen["max_completion_tokens"]
            request.pop("seed", None)
            request.pop("max_tokens", None)
            if self.stage != "bioprofiler":
                request["messages"] = endpoint_messages(request["messages"], self.task)
            if self.stage == "orchestrator":
                request["messages"] = final_messages(request["messages"])
                request["response_format"] = final_schema()
        fingerprint = digest(dict(run=self.config["fingerprint"], task=self.task,
                                  stage=self.stage, request=request))
        path = self.root / (fingerprint + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            require(record["fingerprint"] == fingerprint, "API cache fingerprint mismatch")
            content = record["content"]
            require(isinstance(content, str) and content.strip(), "Empty cached response")
            if self.validator:
                self.validator(content)
        else:
            last = None
            for attempt in range(1, 4):
                record = dict(fingerprint=fingerprint, task=self.task, stage=self.stage,
                              request=redact_images(request), attempt=attempt)
                try:
                    record["request_started_utc"] = datetime.now(timezone.utc).isoformat()
                    response = self.client.chat.completions.create(**request)
                    choice = response.choices[0]
                    content = choice.message.content
                    record.update(content=content, finish_reason=choice.finish_reason,
                                  model=response.model,
                                  response_id=getattr(response, "id", None),
                                  completed_utc=datetime.now(timezone.utc).isoformat(),
                                  usage=response.usage.model_dump() if response.usage else None)
                    require(choice.finish_reason == "stop", f"Incomplete response: {choice.finish_reason}")
                    require(not getattr(choice.message, "refusal", None), "API refusal")
                    require(isinstance(content, str) and content.strip(), "Empty response")
                    if self.validator:
                        self.validator(content)
                    write_json(path, record)
                    break
                except Exception as exc:
                    last = exc
                    # Do not serialize exception bodies, which can contain request credentials.
                    record["error_type"] = type(exc).__name__
                    write_json(self.root / "failed" / f"{fingerprint}_{time.time_ns()}.json", record)
                    if attempt < 3:
                        time.sleep(5)
            else:
                raise RuntimeError(f"{self.task}/{self.stage} failed after 3 attempts; rerun to resume") from last
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@contextmanager
def working_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def image_path(config, row):
    return resolve_image_path(config["data_root"], row)


def load_images(loader, config, row):
    import pandas as pd

    # The loader reads a label for bookkeeping; no loader metadata is sent to an agent.
    series = pd.Series(dict(filename=str(image_path(config, row)), Task_Folder=row["task"],
                            Ground_Truth=row.get("truth", row.get("y_true"))))
    with redirect_stdout(io.StringIO()):
        return loader.load_patient_from_excel_row(series)


def make_oct(task, weights, device):
    module = importlib.import_module(f"VisionAgent.vision_oct_{task}")
    from VisionAgent.linear_probing_oct3 import RETFoundMultiHead, RETFound_mae

    # The full saved state dict contains the backbone. Avoid a redundant load from
    # the historical hard-coded Lustre pretraining path; strict load occurs in __init__.
    original = module.get_model_oct
    module.get_model_oct = lambda: RETFoundMultiHead(
        RETFound_mae(img_size=224, num_classes=0, drop_path_rate=.2, global_pool=""))
    try:
        return module.VisionSpecialistOct(str(weights), device=device)
    finally:
        module.get_model_oct = original


def anchor_priors(root, config, cases, validation, loader, device, receipt, tasks=TASKS):
    """Run the actual agent checkpoint on validation only, with zero API calls."""
    result = {}
    path = root / "anchor_trust.json"
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved["receipt"] == receipt, "Prior checkpoint/input identity changed")
        result = saved["scores"]
    for task in selected_tasks(tasks):
        if task in result:
            require(set(result[task]) == {c["case_id"] for c in cases if c["task"] == task},
                    f"Incomplete cached anchor priors for {task}")
            continue
        oct_agent = make_oct(task, config["oct_weights"], device)
        scored = []
        for i, row in enumerate(r for r in validation if r["task"] == task):
            key = row["image_id"]
            path = root / "anchor_validation" / task / f"{key}.json"
            identity = digest(dict(receipt=receipt, row=row))
            if path.exists():
                saved = json.loads(path.read_text())
                require(saved["fingerprint"] == identity, "Validation cache changed")
            else:
                patient = load_images(loader, config, row)
                output = oct_agent.get_features_oct(patient["oct_img"])[HEADS[task]]
                saved = dict(fingerprint=identity, y_prob=probability(output["Probability"]),
                             y_pred=int(output["Detected"]))
                write_json(path, saved)
            scored.append({**row, "y_prob": saved["y_prob"], "y_pred": saved["y_pred"]})
            if (i + 1) % 100 == 0:
                print(f"anchor validation {task}: {i+1}/1000 (no API calls)", flush=True)
        fitted = fit_reliability(scored, task, "retfound_oct", fixed_threshold=.5)
        result[task] = {c["case_id"]: trust_for(c, task, "retfound_oct", fitted)
                        for c in cases if c["task"] == task}
        del oct_agent
        write_json(root / "anchor_trust.json", dict(receipt=receipt, scores=result))
    return result


def shared_evidence(root, config, case, loader, oct_agent, slo_agent, bio, client, receipt):
    path = root / "shared" / case["task"] / f"{case['case_id']}.json"
    identity = digest(dict(receipt=receipt, task=case["task"], case_id=case["case_id"], metadata=case["metadata"]))
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved["fingerprint"] == identity, "Shared evidence cache changed")
        return saved["evidence"]
    patient = load_images(loader, config, case)
    api_root = root / "api" / case["task"] / case["case_id"] / "shared"
    for component, stage in ((oct_agent, "oct_report"), (slo_agent, "slo_report"), (bio, "bioprofiler")):
        component.model_client = CachedClient(client, api_root, config, case["task"], stage)
    state = {"clinical_narrative": bio.generate_narrative(case["metadata"])}
    scratch = root / "images" / case["task"] / case["case_id"]
    with working_directory(scratch):
        for folder in ("octs", "slo"):
            Path(f"outputs/google_form/{case['task']}/{folder}").mkdir(parents=True, exist_ok=True)
        _, oct_report = oct_agent.analyze(patient["oct_img"], patient["middle_oct"], state)
        state["vision_opinion_oct"] = oct_report
        slo_report, cdr = slo_agent.analyze(patient["fundus_img"], state)
    evidence = dict(narrative=state["clinical_narrative"], oct_report=oct_report, slo_report=slo_report,
                    probability_percent=state["oct_diagnosis"][HEADS[case["task"]]]["Prob_Pct"], cdr=cdr_value(cdr))
    probability(evidence["probability_percent"] / 100)
    write_json(path, dict(fingerprint=identity, evidence=evidence))
    return evidence


def run_pair(root, config, case, evidence, trust, orchestrator, client, receipt):
    task, case_id = case["task"], case["case_id"]
    arms = list(VARIANTS[2:])
    if int(digest([task, case_id])[:8], 16) % 2:
        arms.reverse()
    for variant in arms:
        path = root / "agent" / task / variant / f"{case_id}.json"
        identity = digest(dict(receipt=receipt, evidence=evidence, trust=trust, variant=variant))
        if path.exists():
            saved = json.loads(path.read_text())
            require(saved["evidence_fingerprint"] == identity and saved["fingerprint"] == config["fingerprint"],
                    "Agent result cache changed")
            continue
        arm_trust = trust if variant == "retinagent_full" else None
        api_root = root / "api" / task / case_id / variant
        cf = CachedClient(client, api_root, config, task, "counterfactual", validate_trace)
        response = cf.create(messages=counterfactual_messages(task, evidence, arm_trust), temperature=0)
        trace = validate_trace(response.choices[0].message.content)
        orchestrator.model_client = CachedClient(client, api_root, config, task, "orchestrator",
                                                 lambda raw: final_label(raw, task), arm_trust is None)
        state = dict(clinical_narrative=evidence["narrative"], vision_opinion_oct=evidence["oct_report"],
                     vision_opinion_slo=evidence["slo_report"])
        decision = orchestrator.analyze(state, evidence["probability_percent"], evidence["cdr"], arm_trust, trace)
        write_json(path, dict(fingerprint=config["fingerprint"], evidence_fingerprint=identity,
                             task=task, case_id=case_id, variant=variant,
                             prediction=final_label(decision["decision"], task), decision=decision["decision"],
                             counterfactual_trace=trace, shared_evidence_sha256=digest(evidence)))


def run(args, config, cases):
    import fcntl
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "OphthalmicAgent/.env")
    root = args.run_root
    with (root / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An ablation process is already running in this directory") from exc
        _run_locked(args, config, cases)


def _run_locked(args, config, cases):
    from openai import AzureOpenAI
    from data.loader import ExcelEyeLoader
    from run_fairvision_ablation import audit

    root = args.run_root
    tasks = selected_tasks(args.tasks)
    count = len(tasks) if args.stage == "smoke" else sum(c["task"] in tasks for c in cases)
    print(f"Selected tasks: {', '.join(tasks)}; {count} paired cases. "
          f"Uncached scope: {count * 7} successful API calls before retries; matching calls are reused.", flush=True)
    require(args.deployment == config["deployment"] and args.api_version == config["api_version"], "API settings differ from prepare")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE")
    require(endpoint and os.environ.get("AZURE_OPENAI_API_KEY"), "Missing Azure endpoint or API key")
    os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
    validation = json.loads((root / "validation_cases.json").read_text())
    require(digest(validation) == config["validation_sha256"], "Validation manifest changed")
    weights = Path(config["oct_weights"])
    require(weights.is_file(), f"Missing OCT checkpoint: {weights}; prepare with --oct-weights PATH")
    resolved = require_images(Path(config["data_root"]), cases + validation)
    print("Checking checkpoint and 3750 image fingerprints before any API calls", flush=True)
    receipt = dict(run=config["fingerprint"], endpoint=endpoint, weights_sha256=sha(weights),
                   images={f"{r['task']}/{Path(r['filename']).stem}": sha(resolved[r["filename"]]) for r in cases + validation})
    receipt_path = root / "live_receipt.json"
    if receipt_path.exists():
        require(json.loads(receipt_path.read_text()) == receipt, "Images, endpoint or checkpoint changed; use a new run root")
    else:
        write_json(receipt_path, receipt)
    receipt_id = digest(receipt)
    manifest_path = root / "locked_manifest.csv"
    audit.write_csv(manifest_path, [dict(filename=c["filename"], Task_Folder=c["task"], Ground_Truth=c["truth"], **c["metadata"]) for c in cases])
    loader = ExcelEyeLoader(str(manifest_path))
    trust = anchor_priors(root, config, cases, validation, loader, args.device, receipt_id, tasks)
    write_json(root / "execution_tasks.json", dict(run_fingerprint=config["fingerprint"],
               stage=args.stage, tasks=list(tasks), paired_cases=count))
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=os.environ["AZURE_OPENAI_API_KEY"],
                         api_version=config["api_version"], max_retries=0, timeout=180)
    for task in tasks:
        selected = sorted((c for c in cases if c["task"] == task), key=lambda r: r["case_id"])
        if args.stage == "smoke":
            selected = selected[:1]
        oct_agent = make_oct(task, weights, args.device)
        slo_agent = importlib.import_module(f"VisionAgent.vision_slo_{task}").VisionSpecialistSlo(None, device=args.device)
        bio = importlib.import_module(f"BioProfilerAgent.bio_profiler_{task}").BioProfiler()
        orchestrator = importlib.import_module(f"Orchestrator.fairvision_{task}").Orchestrator()
        cdr_identity = dict(model=slo_agent.cdr_model_name,
                            revision=getattr(slo_agent.model_cdr.config, "_commit_hash", None))
        cdr_path = root / "cdr_model.json"
        if cdr_path.exists():
            require(json.loads(cdr_path.read_text()) == cdr_identity, "CDR model revision changed")
        else:
            write_json(cdr_path, cdr_identity)
        for i, case in enumerate(selected, 1):
            evidence = shared_evidence(root, config, case, loader, oct_agent, slo_agent, bio, client, receipt_id)
            run_pair(root, config, case, evidence, trust[task][case["case_id"]], orchestrator, client, receipt_id)
            print(f"paired {task} {i}/{len(selected)} case={case['case_id']}", flush=True)
        del oct_agent, slo_agent, bio, orchestrator
    print(f"{count} paired {'smoke ' if args.stage == 'smoke' else ''}cases complete for {', '.join(tasks)}."
          + (f" Resume with --stage run --tasks {' '.join(tasks)}." if args.stage == "smoke" else ""), flush=True)
