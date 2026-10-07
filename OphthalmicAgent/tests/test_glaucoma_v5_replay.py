"""Offline V5 tests: image identity, source separation, budget and failure handling."""
import base64
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_glaucoma_v5_replay as v5


def slo_response():
    return dict(image_quality="limited", quality_note="Fine boundaries cannot be assessed confidently.",
                features={name: dict(state="not_assessable", observation="Boundary detail is limited.")
                          for name in v5.slo.FEATURES}, summary="Qualitative observations remain limited.")


def downstream_response(data, final):
    if final:
        sources = copy.deepcopy(data["qualification_review"]["source_review"])
    else:
        sources = {name: dict(finding_ids=[], limitation_ids=list(ids),
                             interpretation="Limited source observations remain uncertain.")
                   for name, ids in data["protected_limitation_ids"].items()}
    value = dict(source_review=sources)
    if final:
        value.update(cdr_use="context_only_unverified", dependence_note="Sources share acquisitions.",
                     diagnosis=0, reasoning="Forced research decision despite uncertainty.", overview="Limited evidence.",
                     escalation_required=True, escalation_reason="Unresolved uncertainty.")
    return value


class FakeClient:
    def __init__(self, mutate=None, fail_at=None):
        self.chat = SimpleNamespace(completions=self)
        self.requests, self.closed = [], False
        self.mutate, self.fail_at = mutate, fail_at

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        n = len(self.requests)
        if n == self.fail_at:
            raise TimeoutError("Synthetic timeout")
        content = request["messages"][1]["content"]
        if isinstance(content, list):
            value = slo_response()
        else:
            data = json.loads(content)
            value = downstream_response(data, "qualification_review" in data)
        envelope = dict(model="synthetic-model", choices=[dict(finish_reason="stop", message=dict(
            content=json.dumps(value), refusal=None, tool_calls=None))])
        if self.mutate:
            self.mutate(envelope, n)
        return SimpleNamespace(model=envelope["model"], choices=[SimpleNamespace(
            finish_reason=envelope["choices"][0]["finish_reason"],
            message=SimpleNamespace(**envelope["choices"][0]["message"]))],
            model_dump=lambda **_: copy.deepcopy(envelope))

    def close(self):
        self.closed = True


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        images = self.root / "originals"
        images.mkdir()
        cases, evaluation, pins = [], [], {}
        for i, key in enumerate(v5.NATIVE_SHA256):
            p = images / ("slo_fundus_" + key.removeprefix("data_") + ".jpg")
            Image.new("RGB", (512, 664), (40+i*60,)*3).save(p, format="JPEG")
            pins[key] = hashlib.sha256(p.read_bytes()).hexdigest()
            case_id = f"data/Glaucoma/Test/{key}.npz"
            evidence = dict(patient_narrative="Synthetic adult", retfound_glaucoma_probability_percent=45,
                oct_specialist_report="Excavation may be present.\nPresence cannot be confirmed.",
                slo_specialist_report="OLD SLO ONLY: possible rim change.", vertical_cup_to_disc_ratio=.5,
                demographic_reliability_trust_score=.8)
            cases.append(dict(case_id=case_id, evidence=evidence,
                              oct_acquisition=dict(acquisition_metadata_verified=False)))
            evaluation.append(dict(case_id=case_id, truth=i, historical_prediction=1-i,
                                   v3_prediction=1-i, eligible=True, v4_selected=True))
        for i in range(55):
            evaluation.append(dict(case_id=f"unassessed_{i}", truth=0, historical_prediction=1,
                                   v3_prediction=None, eligible=i<47, v4_selected=False))
        self.prior = dict(cases=cases, evaluation=evaluation, settings=dict(deployment="gpt-5.1",
            temperature=.3, api_version="2024-12-01-preview", timeout_seconds=180))
        self.snapshot = dict(v2_attempts=100, v3_attempts=7, v4_attempts=4, known_prior_attempts=111,
                             v3_metadata=dict(returned_model="synthetic-model", endpoint_sha256="endpoint"))
        self.prior_parsed = {c["case_id"]: dict(final=dict(diagnosis=1-i)) for i,c in enumerate(cases)}
        self.addCleanup(patch.stopall)
        patch.object(v5, "NATIVE_SHA256", pins).start()
        self.args = SimpleNamespace(experiment_dir=self.root / "v5", v4_dir=self.root / "v4",
            v3_dir=self.root / "v3", v2_root=self.root / "v2", v2_bundle=self.root / "bundle_v2/bundle.json", image_dir=images)
        with patch.object(v5, "read_sources", return_value=(self.prior, self.prior_parsed, self.snapshot)):
            with patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
                self.bundle = v5.prepare(self.args)
        self.out = self.args.experiment_dir
        self.case = self.bundle["cases"][0]

    def execute(self, client, count=2):
        with patch.object(v5, "check_sources"), redirect_stdout(io.StringIO()):
            v5.execute(self.bundle, self.out, lambda _: client, count)

    def collect(self):
        with redirect_stdout(io.StringIO()):
            return v5.collect(self.bundle, self.out)

    def test_original_jpeg_is_sent_byte_for_byte_without_resize_or_metadata_leak(self):
        r = v5.request_for("slo", self.case, self.bundle, self.out, {})
        image = r["messages"][1]["content"][1]["image_url"]
        sent = base64.b64decode(image["url"].split(",",1)[1])
        self.assertEqual(sent, Path(self.case["native_slo"]["source_path"]).read_bytes())
        self.assertEqual(v5.slo.image_metadata(sent)["height"], 664)
        self.assertEqual(image["detail"], "high")
        self.assertNotIn("OLD SLO", json.dumps(r))
        self.assertNotIn(self.case["case_id"], json.dumps(r))
        self.assertNotIn("patient_narrative", json.dumps(r))
        self.assertEqual((r["temperature"], r["max_completion_tokens"]), (.2, 2000))

    def test_diagnoses_and_prior_outputs_do_not_enter_requests(self):
        p = dict(slo=slo_response())
        req = v5.request_for("review", self.case, self.bundle, self.out, p)
        p["review"] = downstream_response(json.loads(req["messages"][1]["content"]), False)
        before = [v5.request_for(s, self.case, self.bundle, self.out, p) for s in v5.STAGES]
        self.case.update(truth="SECRET", v4_prediction="SECRET", trace="SECRET")
        self.bundle["evaluation"][0]["truth"] = "SECRET"
        after = [v5.request_for(s, self.case, self.bundle, self.out, p) for s in v5.STAGES]
        self.assertEqual(before, after)
        self.assertNotIn("SECRET", json.dumps(after))
        self.assertNotIn("OLD SLO", json.dumps(after))
        self.assertNotIn("other_evidence", json.loads(after[1]["messages"][1]["content"]))

    def test_downstream_contract_settings_and_other_evidence_unchanged(self):
        original = copy.deepcopy(self.case)
        p = dict(slo=slo_response())
        for stage in ("review", "final"):
            expected = v5.v4.request_for(stage, v5.updated_case(self.case, p), self.bundle, p)
            actual = v5.request_for(stage, self.case, self.bundle, self.out, p)
            self.assertEqual({k:v for k,v in actual.items() if k!='messages'},
                             {k:v for k,v in expected.items() if k!='messages'})
            self.assertEqual(actual['messages'][0], expected['messages'][0])
            a, e = (json.loads(r['messages'][1]['content']) for r in (actual,expected))
            self.assertEqual({k:v for k,v in a.items() if k!='measurement_provenance'},
                             {k:v for k,v in e.items() if k!='measurement_provenance'})
            self.assertTrue(a['measurement_provenance']['slo_images_reassessed'])
            self.assertFalse(a['measurement_provenance']['cdr_recomputed'])
            if stage=='review':p['review']=downstream_response(a,False)
        self.assertEqual(original,self.case)
        updated=v5.updated_case(self.case,p)['evidence']
        for k in original['evidence']:
            if k!='slo_specialist_report':self.assertEqual(updated[k],original['evidence'][k])

    def test_feature_states_remain_qualified_in_v4_source_units(self):
        value=slo_response()
        for i,name in enumerate(v5.slo.FEATURES):value['features'][name]['state']=v5.slo.STATES[i%4]
        p=dict(slo=value)
        case=v5.updated_case(self.case,p)
        units=v5.v4.contract.source_units(case['evidence'])
        protected=v5.v4.contract.protected_ids(units)['slo']
        for u in units['slo']:
            if any(t in u['text'] for t in ('Possible finding','Not observed','Not assessable')):
                self.assertIn(u['id'],protected)

    def test_bad_feature_states_incomplete_reports_and_extra_diagnosis_rejected(self):
        mutations=[lambda r:r['features'].pop(v5.slo.FEATURES[0]),
                   lambda r:r['features'][v5.slo.FEATURES[0]].update(state='normal'),
                   lambda r:r['features'][v5.slo.FEATURES[0]].update(observation=''),
                   lambda r:r.update(diagnosis=1), lambda r:r.update(image_quality='normal')]
        for change in mutations:
            value=slo_response();change(value)
            with self.assertRaises(ValueError):v5.slo.parse(json.dumps(value))
        value=slo_response();value['image_quality']='ungradable'
        self.assertEqual(v5.slo.parse(json.dumps(value)),value)
        for state in v5.slo.STATES[:-1]:
            value['features'][v5.slo.FEATURES[0]]['state']=state
            with self.assertRaises(ValueError):v5.slo.parse(json.dumps(value))

    def test_resume_prefix_six_call_cap_and_all_57_rows_retained(self):
        client=FakeClient()
        self.execute(client,1);self.execute(client,2);self.execute(client,2)
        self.assertEqual(len(client.requests),6)
        self.assertTrue(client.closed)
        report=self.collect()
        self.assertEqual((report['valid'],report['unassessed'],report['known_total_attempts']),(2,55,117))
        self.assertEqual((report['repaired'],report['still_wrong']),(1,1))
        rows=v5.base.read_csv(self.out/'run/case_results.csv')
        self.assertEqual(len(rows),57)
        self.assertTrue(all(r['v5_prediction']=='' for r in rows[2:]))
        ledger=v5.Ledger(self.out/'run',self.bundle)
        try:
            with self.assertRaisesRegex(ValueError,'budget'):ledger.reserve('extra','slo',{})
        finally:ledger.close()

    def test_timeout_is_charged_and_blocks_retry_and_downstream(self):
        client=FakeClient(fail_at=1)
        with self.assertRaises(RuntimeError):self.execute(client)
        with self.assertRaises(ValueError):self.execute(client)
        self.assertEqual(len(client.requests),1)
        report=self.collect()
        self.assertEqual((report['valid'],report['known_total_attempts']),(0,112))

    def test_interrupted_reservation_blocks_before_client(self):
        ledger=v5.Ledger(self.out/'run',self.bundle)
        ledger.reserve(self.case['case_id'],'slo',v5.request_for('slo',self.case,self.bundle,self.out,{}))
        ledger.close()
        client=FakeClient()
        with self.assertRaises(ValueError):self.execute(client)
        self.assertEqual(client.requests,[])

    def test_changed_image_or_upstream_run_blocks_before_client(self):
        client=FakeClient()
        with patch.object(v5,'check_sources',side_effect=ValueError('Upstream changed')):
            with self.assertRaises(ValueError):v5.execute(self.bundle,self.out,lambda _:client,2)
        (self.out/self.case['native_slo']['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute(client)
        self.assertEqual(client.requests,[])

    def test_bundle_runtime_identity_and_request_tampering_rejected(self):
        self.assertEqual(v5.load_bundle(self.out),self.bundle)
        self.bundle['cases'][0]['evidence']['oct_specialist_report']='Changed'
        v5.base.write_json(self.out/'bundle.json',self.bundle)
        with self.assertRaises(ValueError):v5.load_bundle(self.out)

    def test_invalid_receipt_is_rejected_on_read(self):
        self.execute(FakeClient(),1)
        rows=v5.read_run(self.bundle,self.out)
        rows[1]['request_json']='{}'
        with self.assertRaises(ValueError):v5.validate_attempts(self.bundle,self.out,rows)
        ledger=v5.Ledger(self.out/'run',self.bundle)
        with ledger.db:ledger.db.execute("UPDATE metadata SET value='999' WHERE name='budget'")
        ledger.close()
        with self.assertRaises(ValueError):v5.read_run(self.bundle,self.out)

    def test_prepare_cannot_overwrite_or_expand_pilot(self):
        with self.assertRaises(ValueError):v5.prepare(self.args)
        client=FakeClient()
        for count in (None,0,3,49,True):
            with self.assertRaises(ValueError):self.execute(client,count)
        self.assertEqual(client.requests,[])

    def test_truncation_refusal_model_drift_and_invalid_schema_stop_without_retry(self):
        def invalid(e,n):
            value=json.loads(e['choices'][0]['message']['content']);value['diagnosis']=1
            e['choices'][0]['message']['content']=json.dumps(value)
        mutations=[lambda e,n:e['choices'][0].update(finish_reason='length'),
                   lambda e,n:e['choices'][0]['message'].update(refusal='Refused'),
                   lambda e,n:e.update(model='changed'),invalid]
        for i,mutation in enumerate(mutations):
            with self.subTest(i=i):
                root=self.out/f'isolated_{i}'
                (root/'images').mkdir(parents=True)
                for c in self.bundle['cases']:(root/c['native_slo']['path']).write_bytes((self.out/c['native_slo']['path']).read_bytes())
                client=FakeClient(mutate=mutation)
                with patch.object(v5,'check_sources'),redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):v5.execute(self.bundle,root,lambda _:client,2)
                    with self.assertRaises(ValueError):v5.execute(self.bundle,root,lambda _:client,2)
                self.assertEqual(len(client.requests),1)

    def test_downstream_failure_never_advances_to_next_case(self):
        client=FakeClient(fail_at=2)
        with self.assertRaises(RuntimeError):self.execute(client)
        self.assertEqual(len(client.requests),2)
        self.assertEqual(self.collect()['valid'],0)

    def test_final_cannot_rewrite_review_or_treat_cdr_as_verified(self):
        p=dict(slo=slo_response())
        req=v5.request_for('review',self.case,self.bundle,self.out,p)
        p['review']=downstream_response(json.loads(req['messages'][1]['content']),False)
        req=v5.request_for('final',self.case,self.bundle,self.out,p)
        value=downstream_response(json.loads(req['messages'][1]['content']),True)
        for diagnosis in (0,1):
            value['diagnosis']=diagnosis
            self.assertEqual(v5.parse('final',json.dumps(value),self.case,p)['diagnosis'],diagnosis)
        value['source_review']['slo']['interpretation']='Confident disease claim.'
        with self.assertRaises(ValueError):v5.parse('final',json.dumps(value),self.case,p)
        value['source_review']=copy.deepcopy(p['review']['source_review']);value['cdr_use']='verified'
        with self.assertRaises(ValueError):v5.parse('final',json.dumps(value),self.case,p)

    def test_wrong_resolution_and_exif_rejected(self):
        for size,exif in [((200,200),None),((512,664),Image.Exif())]:
            b=io.BytesIO()
            options={}
            if exif is not None:
                exif[274]=6;options['exif']=exif
            Image.new('RGB',size).save(b,format='JPEG',**options)
            with self.assertRaises(ValueError):v5.slo.image_metadata(b.getvalue())


if __name__ == '__main__':
    unittest.main()
