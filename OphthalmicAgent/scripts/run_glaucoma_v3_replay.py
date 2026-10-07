"""V3: new OCT presentation, then fresh dependent reasoning. Offline by default."""
from __future__ import annotations

import argparse
import copy
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "OphthalmicAgent"), str(ROOT / "OphthalmicAgent/scripts")]
import run_glaucoma_v2_replay as replay
from CounterfactualAgent import counterfactual_agent as cf
from ReplayV3 import oct_presentation as oct_view

base = replay.base
VERSION = "glaucoma_oct_framing_v3_20261007"
STAGES = ("oct", "counterfactual", "final")
API_CAP = 147
ORIGINAL_RUNNER_SHA256 = "2a9323b26fd165de289a3d179699783246d9c2fd23fc8b0105621342b255abd4"
OCT_REPAIR_LIMIT = 2000
OCT_REPAIR_STAGE = "oct_completion_repair"
AMENDMENT_FILE = "oct_completion_amendment.json"
DEFAULT_DIR = ROOT / "OphthalmicAgent/outputs/glaucoma_v3_replay"
DEFAULT_DATA = ROOT / "Datasets/FairVision/Glaucoma/Test"
SCOPE = ("Exploratory historical-error-selected OCT and downstream replay. "
         "Cached SLO, CDR, narrative, probability and reliability; not a full-pipeline or held-out evaluation.")


def code_hashes():
    paths = [Path(__file__), Path(oct_view.__file__), Path(cf.__file__), Path(replay.__file__)]
    return {**replay.code_hashes(), **{str(p.relative_to(ROOT)): base.sha(p) for p in paths}}


def filename(case_id):
    path = PurePosixPath(case_id)
    base.require(path.parent == PurePosixPath("data/Glaucoma/Test") and
                 replay.re.fullmatch(r"data_\d+\.npz", path.name), "Invalid Glaucoma/Test case ID")
    return path.name


def selected_errors(source):
    evaluation = [copy.deepcopy(r) for r in source["evaluation"] if r["group"] == "historical_error"]
    base.require(len({r["case_id"] for r in evaluation}) == len(evaluation), "Duplicate evaluation ID")
    base.require(all(r["truth"] != r["historical_prediction"] for r in evaluation), "Invalid error selection")
    wanted = {r["case_id"] for r in evaluation if r["eligible"]}
    # Preserve the pre-existing order; do not select using V2 outcomes.
    cases = [dict(case_id=c["case_id"], evidence=copy.deepcopy(c["evidence"]),
                  source_line=c["source_line"], source_fingerprint=c["source_fingerprint"])
             for c in source["cases"] if c["case_id"] in wanted]
    base.require(len(cases) == len(wanted) and {c["case_id"] for c in cases} == wanted, "Incomplete source cases")
    for c in evaluation:
        filename(c["case_id"])
    base.require(0 < 3 * len(cases) <= API_CAP, "V3 request plan exceeds its 147-attempt ceiling")
    return cases, evaluation


def source_payloads(args, all_ids):
    names = {filename(k) for k in all_ids}
    if args.archive:
        # Read only, never extract paths. Task-qualified member names avoid AMD/DR ID collisions.
        seen = set()
        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                base.require(member.isfile() and path.parent == PurePosixPath("Datasets/FairVision/Glaucoma/Test")
                             and path.name in names and path.name not in seen and
                             0 < member.size < 128 * 1024 * 1024, "Unexpected/duplicate image archive member")
                seen.add(path.name)
                yield path.name, archive.extractfile(member).read()
        base.require(seen == names, "Image archive does not cover the historical error cohort")
    else:
        root = args.data_root.resolve()
        base.require(root.name == "Test" and root.parent.name == "Glaucoma", "Use the Glaucoma/Test source directory")
        for name in sorted(names):
            path = root / name
            base.require(path.is_file() and not path.is_symlink(), f"Missing or indirect source file: {path}")
            yield name, path.read_bytes()


def final_request(case, settings, systems, oct_report, trace):
    evidence = {**case["evidence"], "oct_specialist_report": oct_report}
    current = dict(evidence=evidence, trace=cf.CounterfactualAgent.concise_trace(trace))
    request = replay.requests_for(current, settings, systems)["v2"]
    # Keep V2's decision instructions and contract. Correct only the presentation provenance.
    old = replay.measurement_checks(evidence)
    new = {k: v for k, v in old.items() if k != "original_images_reassessed"}
    new.update(oct_images_reassessed=True, slo_images_reassessed=False,
               oct_acquisition=case["image"]["metadata"], historical_api_image_recovered=False)
    content = request["messages"][1]["content"]
    tail = "\n- Measurement provenance: " + json.dumps(old, sort_keys=True)
    base.require(content.endswith(tail), "V2 provenance format changed")
    request["messages"][1]["content"] = content[:-len(tail)] + "\n- Measurement provenance: " + json.dumps(new, sort_keys=True)
    return request


def request_for(stage, case, bundle, directory, parsed, amendment=None):
    if stage == "oct":
        image = (directory / case["image"]["path"]).read_bytes()
        base.require(hashlib.sha256(image).hexdigest() == case["image"]["sha256"], "OCT request image changed")
        request = oct_view.request(image, case["image"]["metadata"], bundle["settings"]["deployment"])
        if amendment:
            request["max_completion_tokens"] = amendment["oct_max_completion_tokens"]
        return request
    if stage == "counterfactual":
        evidence = {**case["evidence"], "oct_specialist_report": parsed["oct"]["report"]}
        # Pure prompt builder: no constructor, client, cache or filesystem side effects.
        return dict(model=bundle["settings"]["deployment"], temperature=0,
                    response_format={"type": "json_object"},
                    messages=cf.CounterfactualAgent._messages(None, evidence))
    base.require(stage == "final", "Unknown replay stage")
    return final_request(case, bundle["settings"], bundle["systems"], parsed["oct"]["report"], parsed["counterfactual"])


def parse_response(stage, raw, case_id):
    base.require(isinstance(raw, str) and raw.strip(), "Empty/refused response")
    if stage == "oct":
        return dict(report=raw)
    if stage == "final":
        return replay.parse_response("v2", raw)
    value = json.loads(raw)
    base.require(isinstance(value, dict), "Counterfactual response must be an object")
    scenarios = value.get("scenarios")
    base.require(isinstance(scenarios, list) and len(scenarios) == len(cf.SCENARIOS), "Incomplete counterfactual response")
    for s in scenarios:
        base.require(isinstance(s, dict) and type(s.get("diagnosis")) is int and s["diagnosis"] in (-1, 0, 1)
                     and isinstance(s.get("reasoning"), str) and s["reasoning"].strip(), "Invalid scenario")
    base.require({s.get("name") for s in scenarios} == set(cf.SCENARIOS), "Duplicate or unknown scenario")
    return cf.CounterfactualAgent._validate_trace(value, case_id)


def prepare(args):
    source = replay.load_bundle(args.source_bundle)
    cases, evaluation = selected_errors(source)
    base.require(not args.experiment_dir.exists(), "Experiment directory exists; do not overwrite frozen V3 inputs")
    saved_rows = base.read_csv(args.v2_results)
    baseline = {r["case_id"]: r for r in saved_rows}
    base.require(len(baseline) == len(saved_rows) and set(baseline) == {r["case_id"] for r in evaluation},
                 "V2 results must cover each historical error exactly once")
    for row in evaluation:
        old = baseline[row["case_id"]]
        base.require(base.binary(old["truth"]) == row["truth"] and
                     base.binary(old["historical_prediction"]) == row["historical_prediction"], "V2 reference mismatch")
        row["v2_status"] = old["v2_status"]
        row["v2_prediction"] = base.binary(old["v2_prediction"]) if old["v2_status"] == "valid" else None
    args.experiment_dir.mkdir(parents=True)
    (args.experiment_dir / "images").mkdir()
    by_name = {filename(c["case_id"]): c for c in cases}
    for name, payload in source_payloads(args, [r["case_id"] for r in evaluation]):
        if name not in by_name:
            continue  # Raw images do not replace missing saved non-OCT evidence.
        with oct_view.np.load(io.BytesIO(payload), allow_pickle=False) as data:
            volume = data["oct_bscans"]  # No reference labels or other patient fields are read.
        image, metadata = oct_view.render(volume)
        image_bytes = oct_view.jpeg_bytes(image)
        relative = f"images/{Path(name).stem}.jpg"
        (args.experiment_dir / relative).write_bytes(image_bytes)
        by_name[name]["image"] = dict(path=relative, sha256=hashlib.sha256(image_bytes).hexdigest(),
                                     npz_sha256=hashlib.sha256(payload).hexdigest(), metadata=metadata)
    base.require(all("image" in c for c in cases), "Missing OCT inputs")
    bundle = dict(version=VERSION, scope=SCOPE, cases=cases, evaluation=evaluation,
                  settings=source["settings"], systems=source["systems"], api_budget=3 * len(cases),
                  source_bundle_fingerprint=source["fingerprint"],
                  source_sha256=dict(v2_bundle=base.sha(args.source_bundle), v2_results=base.sha(args.v2_results)),
                  runtime_code_sha256=code_hashes(), oct_system=oct_view.SYSTEM_PROMPT,
                  renderer_versions=dict(numpy=oct_view.np.__version__, opencv=oct_view.cv2.__version__,
                                         pillow=oct_view.Image.__version__))
    for case in cases:
        case["oct_request_sha256"] = base.digest(request_for("oct", case, bundle, args.experiment_dir, {}))
    bundle["fingerprint"] = base.digest(bundle)
    base.write_json(args.experiment_dir / "bundle.json", bundle)
    return load_bundle(args.experiment_dir)


def load_bundle(directory):
    bundle = json.loads((directory / "bundle.json").read_text())
    base.require(bundle["fingerprint"] == base.digest({k: v for k, v in bundle.items() if k != "fingerprint"}),
                 "V3 bundle checksum mismatch")
    current = code_hashes()
    original = {**current, str(Path(__file__).relative_to(ROOT)): ORIGINAL_RUNNER_SHA256}
    # Accept only this explicitly identified pre-amendment runner, not arbitrary code drift.
    base.require(bundle["version"] == VERSION and bundle["runtime_code_sha256"] in (current, original), "V3 runtime changed")
    base.require(bundle["oct_system"] == oct_view.SYSTEM_PROMPT, "OCT prompt changed")
    base.require(0 < bundle["api_budget"] == 3 * len(bundle["cases"]) <= API_CAP, "Invalid API budget")
    ids = [c["case_id"] for c in bundle["cases"]]
    eligible = {r["case_id"] for r in bundle["evaluation"] if r["eligible"]}
    base.require(len(ids) == len(set(ids)) and set(ids) == eligible, "Invalid eligible case set")
    for case in bundle["cases"]:
        name = filename(case["case_id"])
        meta = case["image"]["metadata"]
        base.require(case["image"]["path"] == f"images/{Path(name).stem}.jpg" and
                     meta == oct_view.slice_metadata(meta["volume_shape"]), "Image path or slice metadata changed")
        request = request_for("oct", case, bundle, directory, {})
        base.require(base.digest(request) == case["oct_request_sha256"], "Frozen OCT request changed")
    return bundle


def response_details(attempt):
    response = json.loads(attempt["response_json"]) if attempt.get("response_json") else {}
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    return dict(case_id=attempt["case_id"], stage=attempt["arm"], status=attempt["status"],
                finish_reason=choice.get("finish_reason"), usage=response.get("usage"),
                response_characters=len(attempt.get("raw") or ""),
                refusal=(choice.get("message") or {}).get("refusal"), error=attempt.get("error"))


def read_attempts(directory):
    path = directory / "run/ledger.sqlite3"
    if not path.exists():
        return [], {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY started_utc, case_id, arm")]
        metadata = dict(db.execute("SELECT name,value FROM metadata"))
    return rows, metadata


def check_initial_truncation(bundle, directory, attempt):
    base.require(attempt["case_id"] == bundle["cases"][0]["case_id"] and attempt["arm"] == "oct" and
                 attempt["status"] == "invalid" and attempt["parsed_json"] is None,
                 "Amendment requires the first OCT attempt to be invalid, without a parsed report")
    request = request_for("oct", bundle["cases"][0], bundle, directory, {})
    base.require(request["max_completion_tokens"] == 500 and attempt["request_hash"] == base.digest(request) and
                 json.loads(attempt["request_json"]) == request, "Original 500-token request receipt mismatch")
    response = json.loads(attempt["response_json"]) if attempt["response_json"] else {}
    choices = response.get("choices") or []
    details = response_details(attempt)
    base.require(len(choices) == 1 and details["finish_reason"] == "length" and not details["refusal"] and
                 not (choices[0].get("message") or {}).get("tool_calls"),
                 "Not a confirmed token-limit stop. Refusing amendment; inspect the provider receipt")
    base.require((choices[0].get("message") or {}).get("content") == attempt["raw"], "Raw response receipt mismatch")


def run_identity(bundle, amendment=None):
    identity = dict(version=VERSION, bundle_fingerprint=bundle["fingerprint"], api_budget=bundle["api_budget"])
    if amendment:
        identity.update(api_budget=amendment["api_budget"], completion_amendment=amendment["fingerprint"])
    return identity


def load_amendment(bundle, directory):
    path = directory / AMENDMENT_FILE
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    base.require(value["fingerprint"] == base.digest({k: v for k, v in value.items() if k != "fingerprint"}) and
                 value["bundle_fingerprint"] == bundle["fingerprint"] and value["version"] == "oct_completion_limit_v1",
                 "Completion amendment fingerprint mismatch")
    base.require(value["runtime_code_sha256"] == code_hashes(), "Completion amendment runtime changed")
    base.require(value["oct_max_completion_tokens"] == OCT_REPAIR_LIMIT and
                 value["api_budget"] == bundle["api_budget"] + 1 <= API_CAP + 1,
                 "Invalid completion amendment budget")
    check_initial_truncation(bundle, directory, value["original_attempt"])
    hashes = {c["case_id"]: base.digest(request_for("oct", c, bundle, directory, {}, value)) for c in bundle["cases"]}
    base.require(value["oct_request_sha256"] == hashes, "Amended OCT requests changed")
    return value


def amend_oct_limit(bundle, directory):
    """Explicit offline amendment after exactly one confirmed length failure; never resets the ledger."""
    root = directory / "run"
    base.require((root / "ledger.sqlite3").exists(), "No failed V3 run exists; nothing to amend")
    with replay.run_lock(root):
        rows, metadata = read_attempts(directory)
        amendment = load_amendment(bundle, directory)
        if amendment is None:
            base.require(len(rows) == 1, "Amendment only supports a pilot stopped at its first OCT request")
            check_initial_truncation(bundle, directory, rows[0])
            amendment = dict(version="oct_completion_limit_v1", bundle_fingerprint=bundle["fingerprint"],
                             created_utc=replay.utc(), original_attempt=rows[0],
                             oct_max_completion_tokens=OCT_REPAIR_LIMIT, api_budget=bundle["api_budget"] + 1,
                             runtime_code_sha256=code_hashes())
            amendment["oct_request_sha256"] = {
                c["case_id"]: base.digest(request_for("oct", c, bundle, directory, {}, amendment)) for c in bundle["cases"]}
            amendment["fingerprint"] = base.digest(amendment)
        original = amendment["original_attempt"]
        base.require(next((r for r in rows if (r["case_id"], r["arm"]) ==
                           (original["case_id"], "oct")), None) == original, "Original failed receipt changed")
        old_identity, new_identity = run_identity(bundle), run_identity(bundle, amendment)
        base.require(json.loads((root / "run_identity.json").read_text()) in (old_identity, new_identity),
                     "Run identity mismatch")
        base.require(metadata.get("bundle") == bundle["fingerprint"] and metadata.get("version") == VERSION and
                     metadata.get("budget") in (str(bundle["api_budget"]), str(amendment["api_budget"])) and
                     metadata.get("completion_amendment") in (None, amendment["fingerprint"]), "Ledger identity mismatch")
        if metadata.get("completion_amendment") is None:
            base.require(rows == [original], "Cannot amend a run with additional requests")
        # Persist the immutable receipt/recipe first. Repeating this command can finish
        # an interrupted local migration, but cannot grant another retry or larger cap.
        if not (directory / AMENDMENT_FILE).exists():
            base.write_json(directory / AMENDMENT_FILE, amendment)
        with closing(sqlite3.connect(root / "ledger.sqlite3")) as db:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute("UPDATE metadata SET value=? WHERE name='budget'", (str(amendment["api_budget"]),))
                db.execute("INSERT OR IGNORE INTO metadata VALUES ('completion_amendment', ?)", (amendment["fingerprint"],))
        base.write_json(root / "run_identity.json", new_identity)
    print(f"Offline amendment recorded: OCT cap 500 -> {OCT_REPAIR_LIMIT} for every eligible case. "
          f"Original failed attempt retained. V3 ceiling {amendment['api_budget']}, including that attempt. No API calls.")
    return amendment


class Ledger(replay.Ledger):
    """Reuse durable V2 reservation/receipt operations with an isolated V3 identity."""
    def __init__(self, root, bundle):
        amendment = load_amendment(bundle, root.parent)
        base.require(not root.is_symlink(), "Run root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "run_identity.json"
        identity = run_identity(bundle, amendment)
        if marker.exists():
            base.require(json.loads(marker.read_text()) == identity, "Run root belongs to another experiment")
        else:
            base.require(not any(p.name != ".run.lock" for p in root.iterdir()), "Unidentified nonempty run directory")
            base.write_json(marker, identity)
        self.db = sqlite3.connect(root / "ledger.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            case_id TEXT NOT NULL, arm TEXT NOT NULL, request_hash TEXT NOT NULL,
            request_json TEXT NOT NULL, started_utc TEXT NOT NULL, status TEXT NOT NULL,
            finished_utc TEXT, raw TEXT, response_json TEXT, parsed_json TEXT, error TEXT,
            PRIMARY KEY (case_id, arm))""")
        self.db.commit()
        try:
            self.pin("bundle", bundle["fingerprint"])
            self.pin("budget", str(identity["api_budget"]))
            self.pin("version", VERSION)
            if amendment:
                self.pin("completion_amendment", amendment["fingerprint"])
        except BaseException:
            self.db.close()
            raise
        self.limit = identity["api_budget"]


def validated_attempts(bundle, directory, attempts):
    amendment = load_amendment(bundle, directory)
    allowed = {(c["case_id"], s) for c in bundle["cases"] for s in STAGES}
    limit = bundle["api_budget"]
    if amendment:
        original = amendment["original_attempt"]
        retry_key = (original["case_id"], OCT_REPAIR_STAGE)
        allowed.add(retry_key)
        limit = amendment["api_budget"]
    mapping = {(a["case_id"], a["arm"]): a for a in attempts}
    base.require(len(mapping) == len(attempts) <= limit and set(mapping) <= allowed,
                 "Invalid attempt ledger")
    if amendment:
        key = (original["case_id"], "oct")
        base.require(mapping.pop(key, None) == original, "Original failed receipt changed or disappeared")
        if retry_key in mapping:
            mapping[key] = mapping.pop(retry_key)
    results = {}
    for case in bundle["cases"]:
        parsed = {}
        for stage in STAGES:
            attempt = mapping.get((case["case_id"], stage))
            if not attempt:
                continue
            prior = STAGES[:STAGES.index(stage)]
            base.require(all(s in parsed for s in prior), "Downstream attempt used missing or invalid evidence")
            request = request_for(stage, case, bundle, directory, parsed, amendment)
            base.require(attempt["request_hash"] == base.digest(request) and
                         json.loads(attempt["request_json"]) == request, "Dependent request receipt mismatch")
            if attempt["status"] == "valid":
                value = parse_response(stage, attempt["raw"], case["case_id"])
                base.require(value == json.loads(attempt["parsed_json"]), "Parsed receipt mismatch")
                parsed[stage] = value
        results[case["case_id"]] = parsed
    return mapping, results


def execute(bundle, directory, client_factory, max_cases):
    base.require(type(max_cases) is int and 0 < max_cases <= len(bundle["cases"]),
                 "Choose an explicit positive --max-cases within the frozen cohort")
    # The sole ledger lives with its frozen inputs, shared by pilot and continuation.
    root = directory / "run"
    with replay.run_lock(root):
        ledger, client = Ledger(root, bundle), None
        try:
            attempts = ledger.rows()
            amendment = load_amendment(bundle, directory)
            mapping, saved = validated_attempts(bundle, directory, attempts)
            base.require(all(a["status"] == "valid" for a in mapping.values()),
                         "A failed/interrupted attempt exists. Stop and inspect; no automatic retries or continuation")
            for case in bundle["cases"][:max_cases]:
                parsed = saved[case["case_id"]]
                for stage in STAGES:
                    if stage in parsed:
                        continue
                    request = request_for(stage, case, bundle, directory, parsed, amendment)
                    slot = OCT_REPAIR_STAGE if (amendment and stage == "oct" and
                        case["case_id"] == amendment["original_attempt"]["case_id"]) else stage
                    key = (case["case_id"], slot)
                    base.require(ledger.reserve(*key, request), "Unexpected existing reservation")
                    try:
                        if client is None:
                            client = client_factory(ledger)
                        response = client.chat.completions.create(**request)
                    except Exception as exc:
                        ledger.finish(key, "api_error", error=type(exc).__name__)
                        raise RuntimeError("API attempt recorded. No retries; dependent stages were not sent") from exc
                    data = response.model_dump(mode="json")
                    data["provider_request_id"] = getattr(response, "_request_id", None)
                    choice = response.choices[0] if response.choices else None
                    raw = choice.message.content if choice else None
                    ledger.save_response(key, raw, data)
                    try:
                        base.require(isinstance(response.model, str) and response.model, "Missing returned model")
                        ledger.pin("returned_model", response.model)
                    except ValueError:
                        ledger.finish(key, "model_drift", error="Returned model changed or missing")
                        raise
                    try:
                        finish = choice.finish_reason if choice else None
                        base.require(choice and finish == "stop", f"Incomplete response (finish_reason={finish!r}); "
                                     "inspect the saved receipt; do not use incomplete output")
                        parsed[stage] = parse_response(stage, raw, case["case_id"])
                    except (ValueError, TypeError) as exc:
                        ledger.finish(key, "invalid", error=str(exc))
                        raise ValueError(f"Invalid {stage} response saved (finish_reason={finish!r}). "
                                         "No retries or negative fallback. Use --stage inspect") from exc
                    ledger.finish(key, "valid", parsed=parsed[stage])
                    print(f"attempt {len(ledger.rows())}/{ledger.limit}: {Path(case['case_id']).stem} {stage}", flush=True)
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
            ledger.close()


def collect(bundle, directory):
    root = directory / "run"
    with replay.run_lock(root):
        ledger = Ledger(root, bundle)
        try:
            attempts = ledger.rows()
        finally:
            ledger.close()
    mapping, parsed = validated_attempts(bundle, directory, attempts)
    amendment = load_amendment(bundle, directory)
    rows = []
    for old in bundle["evaluation"]:
        row = dict(old)
        for stage in STAGES:
            row[stage + "_status"] = mapping.get((old["case_id"], stage), {}).get(
                "status", "not_attempted" if old["eligible"] else "missing_saved_evidence")
        current = parsed.get(old["case_id"], {}).get("final")
        row.update(v3_prediction=current["diagnosis"] if current else None,
                   v3_reasoning=current["reasoning"] if current else "",
                   escalation_required=current["escalation_required"] if current else None)
        rows.append(row)
    valid = [r for r in rows if r["final_status"] == "valid"]
    paired = [r for r in valid if r["v2_status"] == "valid"]
    repaired = sum(r["v3_prediction"] == r["truth"] for r in valid)
    caution = ("Error-selected descriptive repair counts only. No full-cohort F1 or significance claim. "
               "Correct-case regressions are not assessed. V2 is a saved, noncontemporaneous comparator.")
    report = dict(version=VERSION, scope=SCOPE, bundle_fingerprint=bundle["fingerprint"],
                  expected=len(rows), eligible=len(bundle["cases"]), api_budget=run_identity(bundle, amendment)["api_budget"],
                  attempts_reserved=len(attempts), valid=len(valid), repaired=repaired,
                  still_wrong=len(valid) - repaired, unassessed=len(rows) - len(valid),
                  saved_v2_pairs=len(paired),
                  additional_repairs_vs_saved_v2=sum(r["v3_prediction"] == r["truth"] and
                      r["v2_prediction"] != r["truth"] for r in paired),
                  lost_repairs_vs_saved_v2=sum(r["v3_prediction"] != r["truth"] and
                      r["v2_prediction"] == r["truth"] for r in paired), caution=caution,
                  unsuccessful_attempts=[response_details(a) for a in attempts if a["status"] != "valid"],
                  completion_amendment=amendment["fingerprint"] if amendment else None)
    root.mkdir(parents=True, exist_ok=True)
    base.audit.write_csv(root / "case_results.csv", rows)
    base.write_json(root / "summary.json", report)
    with (root / "api_receipts.jsonl").open("w") as handle:
        for attempt in attempts:
            handle.write(base.canonical(attempt) + "\n")
    (root / "report.txt").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def summary(bundle, amendment=None):
    limit = run_identity(bundle, amendment)["api_budget"]
    print(f"{VERSION}: {SCOPE}\n"
          f"Eligible errors: {len(bundle['cases'])}; missing saved evidence: "
          f"{sum(not r['eligible'] for r in bundle['evaluation'])}.\n"
          f"Three requests/case (OCT -> counterfactual -> final); at most {limit} attempts.\n"
          "No new SLO, CDR, Bio-Profiler, classifier or training calls. No automatic retries.\n"
          f"The downloaded V2 receipts contained 100 attempts; 100 + {limit} = {100 + limit}. "
          "This is not a global counter for other jobs.\n"
          "V1/V2 inputs, prompts and results are not overwritten.")
    if amendment:
        print(f"Explicit OCT token amendment: {OCT_REPAIR_LIMIT} tokens; one prior failed attempt remains charged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "preflight", "inspect", "amend-oct-limit", "run", "collect"), default="preflight")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--source-bundle", type=Path, default=replay.DEFAULT_BUNDLE)
    parser.add_argument("--v2-results", type=Path, default=replay.DEFAULT_RUN / "failed_case_results.csv")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    sources.add_argument("--archive", type=Path, help="Optional transferred Glaucoma-only image archive")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-cases", type=int, help="Required for run; frozen prefix, same ledger on continuation")
    args = parser.parse_args()
    bundle = prepare(args) if args.stage == "prepare" else load_bundle(args.experiment_dir)
    if args.stage == "amend-oct-limit":
        amendment = amend_oct_limit(bundle, args.experiment_dir)
    else:
        amendment = load_amendment(bundle, args.experiment_dir)
    summary(bundle, amendment)
    if args.stage in ("prepare", "preflight"):
        print("Frozen images, indices and requests verified. No API client created; no network calls.")
    elif args.stage == "inspect":
        attempts, _ = read_attempts(args.experiment_dir)
        validated_attempts(bundle, args.experiment_dir, attempts)
        print(json.dumps([response_details(a) for a in attempts], indent=2))
        print("Read-only receipt inspection. No API calls or ledger changes.")
    elif args.stage == "run":
        base.require(args.allow_api, "Paid requests require --allow-api")
        base.require(replay.os.environ.get("AZURE_OPENAI_ENDPOINT") and replay.os.environ.get("AZURE_OPENAI_API_KEY"),
                     "Set Azure credentials before running")
        try:
            execute(bundle, args.experiment_dir, replay.azure_factory(bundle), args.max_cases)
        finally:
            if (args.experiment_dir / "run/run_identity.json").exists():
                collect(bundle, args.experiment_dir)
    elif args.stage == "collect":
        collect(bundle, args.experiment_dir)


if __name__ == "__main__":
    main()
