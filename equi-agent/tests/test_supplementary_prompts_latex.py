"""Offline checks for the supplementary prompt export; no agent imports."""
import ast
import csv
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "equi-agent/scripts/export_supplementary_prompts_latex.py"
SPEC = importlib.util.spec_from_file_location("prompt_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class PromptExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = []
        for spec in exporter.source_specs():
            source = (ROOT / spec["path"]).read_text()
            cls.modules.append((spec, source, exporter.extract(source, spec)))

    def test_short_prompts_and_exclusion_of_comments_logs_and_docstrings(self):
        source = '''"""You are a fake module docstring."""
# prompt = "You are a commented-out experiment."
print("You are seeing a diagnostic log, not a prompt.")
messages = [{"role": "system", "content": "Return JSON."},
            {"role": "user", "content": [{"type": "text", "text": "Inspect."}]}]
'''
        spec = dict(path="fixture.py", functions=(), dependencies=())
        entries = exporter.extract(source, spec)
        self.assertEqual([entry["body"] for entry in entries], ["Return JSON.", "Inspect."])

    def test_multiline_dynamic_template_is_not_evaluated(self):
        source = 'prompt = f"""Inspect this case.\n{patient_metadata}\nReturn JSON."""\n'
        entries = exporter.extract(source, dict(path="fixture.py", functions=(), dependencies=()))
        self.assertEqual(entries[0]["body"], 'f"""Inspect this case.\n{patient_metadata}\nReturn JSON."""')
        self.assertIn("not evaluated", entries[0]["kind"])

    def test_conditional_fragments_have_opposite_branch_labels(self):
        entries = next(entries for spec, _, entries in self.modules if "EquityAgent/" in spec["path"])
        conditions = {entry["condition"] for entry in entries}
        self.assertIn("if calibration_blob is not None", conditions)
        self.assertIn("else of calibration_blob is not None", conditions)

    def test_selected_functions_omit_comments_not_prompt_hash_characters(self):
        source = '''def messages():
    # Obsolete draft: do not publish this as a prompt.
    return [{"role": "system", "content": "# Heading: keep me"}]
'''
        spec = dict(path="fixture.py", functions=("messages",), dependencies=())
        entries = exporter.extract(source, spec)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("Obsolete draft", entries[0]["body"])
        self.assertIn("# Heading: keep me", entries[0]["body"])
        ast.parse(entries[0]["body"])

    def test_missing_selected_function_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing functions"):
            exporter.extract("x = 1", dict(path="fixture.py", functions=("messages",), dependencies=()))

    def test_required_scopes_and_progression_roles(self):
        self.assertEqual(len(self.modules), 47)
        self.assertEqual({spec["section"] for spec, _, _ in self.modules}, set(exporter.SECTIONS))
        entries = next(entries for spec, _, entries in self.modules if spec["path"].endswith("Progression/prompts.py"))
        self.assertEqual({entry["label"] for entry in entries if entry["label"].startswith("System prompt:")}, {
            "System prompt: " + role for role in ("bio_profiler", "rnflt_specialist", "oct_specialist",
                                                "functional_specialist", "counterfactual", "orchestrator")
        })
        body = "\n".join(entry["body"] for spec, _, entries in self.modules
                         if spec["section"] == "prospective" for entry in entries)
        self.assertIn("escalation_required", body)
        self.assertIn("def final_schema", body)

    def test_all_direct_message_text_in_scope_is_covered(self):
        for spec, source, entries in self.modules:
            candidates = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Dict):
                    candidates.extend(value for key, value in zip(node.keys, node.values)
                                      if isinstance(key, ast.Constant) and key.value in {"content", "text", "system"}
                                      and exporter.is_text_expression(value))
                elif isinstance(node, ast.Call):
                    candidates.extend(kw.value for kw in node.keywords if kw.arg in {"content", "text", "system"}
                                      and exporter.is_text_expression(kw.value))
            for candidate in candidates:
                self.assertTrue(any(entry["line_start"] <= candidate.lineno and
                                    entry["line_end"] >= candidate.end_lineno for entry in entries),
                                f"Missing message: {spec['path']}:{candidate.lineno}")

    def test_all_construction_blocks_are_parseable_and_crlf_is_normalized(self):
        for spec, _, _ in self.modules:
            raw_source = (ROOT / spec["path"]).read_bytes().decode()
            for entry in exporter.extract(raw_source, spec):
                self.assertNotIn("\r", entry["body"])
                if entry["kind"].startswith("Python"):
                    ast.parse(entry["body"])

    def test_export_is_reproducible_and_hashes_match(self):
        before = {spec["path"]: exporter.sha((ROOT / spec["path"]).read_bytes())
                  for spec, _, _ in self.modules}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = exporter.export(ROOT, out)
            first = {p.name: p.read_bytes() for p in out.iterdir()}
            exporter.export(ROOT, out)
            self.assertEqual(first, {p.name: p.read_bytes() for p in out.iterdir()})
            self.assertEqual(manifest["artifact_sha256"], exporter.sha(first["supplementary_prompts.tex"]))
            self.assertEqual(manifest, json.loads(first["provenance.json"]))
            with (out / "prompt_index.csv").open(newline="") as handle:
                index = list(csv.DictReader(handle))
            self.assertEqual(len(index), manifest["block_count"])
            self.assertEqual(len({row["id"] for row in index}), len(index))
            for row in index:
                self.assertEqual(row["source_sha256"], before[row["source"]])
            document = first["supplementary_prompts.tex"].decode()
            blocks = re.findall(r"\\begin\{Verbatim\}\n(.*?)\n\\end\{Verbatim\}", document, re.S)
            self.assertEqual(len(blocks), len(index))
            self.assertEqual([exporter.sha(body.encode()) for body in blocks],
                             [row["display_sha256"] for row in index])
            self.assertEqual(document.count(r"\begin{document}"), 1)
            self.assertEqual(document.count(r"\end{document}"), 1)
            self.assertNotIn(r"\input{", document)
            self.assertNotIn(r"\include{", document)
            self.assertIn("not a verified frozen snapshot", document)
            self.assertIn(r"\appendix", document)
        for path, expected in before.items():
            self.assertEqual(exporter.sha((ROOT / path).read_bytes()), expected)

    def test_latex_escaping_and_verbatim_guard(self):
        self.assertEqual(exporter.latex_escape("x_y & 5% {z}"), r"x\_y \& 5\% \{z\}")
        with self.assertRaisesRegex(ValueError, "delimiter"):
            exporter.verbatim(r"malicious \end{Verbatim} contents")


if __name__ == "__main__":
    unittest.main()
