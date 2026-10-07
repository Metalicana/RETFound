"""No-network checks for the explicit first-OCT token-limit amendment."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_glaucoma_v3_replay import FakeClient, make_bundle, v3


class RefusalClient(FakeClient):
    def create(self, **request):
        response = super().create(**request)
        data = response.model_dump(mode="json")
        data["choices"][0]["message"]["refusal"] = "Synthetic refusal"
        return response


class CompletionRepairTests(unittest.TestCase):
    def run_fake(self, b, root, client, count=2):
        with redirect_stdout(StringIO()):
            v3.execute(b, root, lambda _: client, count)

    def stopped(self, root, client=None):
        b = make_bundle(root)
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_fake(b, root, client or FakeClient(finish="length"))
        return b

    def amend(self, b, root):
        with redirect_stdout(StringIO()):
            return v3.amend_oct_limit(b, root)

    def test_amendment_preserves_original_and_changes_only_oct_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            frozen = (root / "bundle.json").read_bytes()
            old, _ = v3.read_attempts(root)
            amendment = self.amend(b, root)
            self.assertEqual((root / "bundle.json").read_bytes(), frozen)
            self.assertEqual(v3.read_attempts(root)[0], old)
            self.assertEqual(amendment["original_attempt"], old[0])
            self.assertEqual(amendment["api_budget"], b["api_budget"] + 1)
            for case in b["cases"]:
                original = v3.request_for("oct", case, b, root, {})
                amended = v3.request_for("oct", case, b, root, {}, amendment)
                self.assertEqual(original["max_completion_tokens"], 500)
                self.assertEqual(amended, {**original, "max_completion_tokens": 2000})

    def test_repair_resume_counts_failure_and_does_not_repeat_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            self.amend(b, root)
            client = FakeClient()
            self.run_fake(b, root, client, 1)
            self.run_fake(b, root, client)
            self.run_fake(b, root, client)
            attempts, _ = v3.read_attempts(root)
            self.assertEqual(len(attempts), 7)
            self.assertEqual(len(client.requests), 6)
            self.assertEqual(sum(a["status"] == "invalid" for a in attempts), 1)
            self.assertEqual(sum(a["arm"] == v3.OCT_REPAIR_STAGE for a in attempts), 1)
            with redirect_stdout(StringIO()):
                report = v3.collect(b, root)
            self.assertEqual((report["attempts_reserved"], report["api_budget"], report["valid"]), (7, 7, 2))
            self.assertEqual(report["unsuccessful_attempts"][0]["finish_reason"], "length")
            ledger = v3.Ledger(root / "run", b)
            try:
                with self.assertRaisesRegex(ValueError, "budget exhausted"):
                    ledger.reserve("extra", "oct", {})
            finally:
                ledger.close()

    def test_repeated_amendment_never_grants_more_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            original = self.amend(b, root)
            saved = (root / v3.AMENDMENT_FILE).read_bytes()
            self.assertEqual(self.amend(b, root), original)
            client = FakeClient()
            self.run_fake(b, root, client, 1)
            self.assertEqual(self.amend(b, root), original)
            self.assertEqual((root / v3.AMENDMENT_FILE).read_bytes(), saved)
            self.assertEqual(v3.read_attempts(root)[1]["budget"], str(b["api_budget"] + 1))

    def test_second_truncation_is_not_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            self.amend(b, root)
            client = FakeClient(finish="length")
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.amend(b, root)
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(len(v3.read_attempts(root)[0]), 2)

    def test_filter_refusal_timeout_and_other_finish_reasons_are_not_amended(self):
        for client in (FakeClient(finish="content_filter"), RefusalClient(finish="length"),
                       FakeClient(error=True), FakeClient(finish="tool_calls"), FakeClient(finish=None)):
            with self.subTest(client=client), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b = self.stopped(root, client)
                before = v3.read_attempts(root)
                with self.assertRaises(ValueError):
                    self.amend(b, root)
                self.assertFalse((root / v3.AMENDMENT_FILE).exists())
                self.assertEqual(v3.read_attempts(root), before)

    def test_no_amendment_for_unused_or_already_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            with self.assertRaises(ValueError):
                self.amend(b, root)
            self.run_fake(b, root, FakeClient(), 1)
            with self.assertRaises(ValueError):
                self.amend(b, root)
            self.assertFalse((root / v3.AMENDMENT_FILE).exists())

    def test_changed_or_deleted_original_receipt_is_rejected(self):
        for operation in ("UPDATE attempts SET raw='tampered' WHERE arm='oct'",
                          "DELETE FROM attempts WHERE arm='oct'"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b = self.stopped(root)
                self.amend(b, root)
                ledger = v3.Ledger(root / "run", b)
                with ledger.db:
                    ledger.db.execute(operation)
                ledger.close()
                client = FakeClient()
                with self.assertRaises(ValueError):
                    self.run_fake(b, root, client)
                self.assertEqual(client.requests, [])

    def test_tampered_amendment_is_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            amendment = self.amend(b, root)
            amendment["oct_max_completion_tokens"] = 10000
            amendment["fingerprint"] = v3.base.digest({k: v for k, v in amendment.items() if k != "fingerprint"})
            v3.base.write_json(root / v3.AMENDMENT_FILE, amendment)
            with self.assertRaises(ValueError):
                v3.load_amendment(b, root)

    def test_interrupted_offline_amendment_finishes_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = self.stopped(root)
            original_write = v3.base.write_json

            def interrupted(path, value):
                if path.name == "run_identity.json":
                    raise OSError("Synthetic interruption after database update")
                original_write(path, value)

            with patch.object(v3.base, "write_json", side_effect=interrupted), self.assertRaises(OSError):
                self.amend(b, root)
            client = FakeClient()
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(client.requests, [])
            self.amend(b, root)
            self.run_fake(b, root, client, 1)
            self.assertEqual(len(client.requests), 3)
            self.assertEqual(len(v3.read_attempts(root)[0]), 4)

    def test_only_identified_original_runtime_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            runner = str(Path(v3.__file__).relative_to(v3.ROOT))
            b["runtime_code_sha256"][runner] = v3.ORIGINAL_RUNNER_SHA256
            b["fingerprint"] = v3.base.digest({k: v for k, v in b.items() if k != "fingerprint"})
            v3.base.write_json(root / "bundle.json", b)
            self.assertEqual(v3.load_bundle(root), b)
            b["runtime_code_sha256"][runner] = "unrecognized_old_runner"
            b["fingerprint"] = v3.base.digest({k: v for k, v in b.items() if k != "fingerprint"})
            v3.base.write_json(root / "bundle.json", b)
            with self.assertRaises(ValueError):
                v3.load_bundle(root)

    def test_inspect_cli_shows_finish_reason_without_mutating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.stopped(root)
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = subprocess.run([sys.executable, str(Path(v3.__file__)), "--experiment-dir", str(root),
                                     "--stage", "inspect"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"finish_reason": "length"', result.stdout)
            self.assertIn("Read-only receipt inspection", result.stdout)
            self.assertEqual({str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
