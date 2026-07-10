import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from CounterfactualAgent.counterfactual_agent import CounterfactualAgent, SCENARIOS


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=json.dumps(self.content))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _response():
    return {
        "scenarios": [
            {
                "name": name,
                "diagnosis": 0 if name == "without_retfound_probability" else 1,
                "confidence": "moderate",
                "reasoning": f"Reasoning for {name}.",
            }
            for name in SCENARIOS
        ],
        "interpretation": "The diagnosis depends on the RETFound probability.",
    }


class CounterfactualAgentTest(unittest.TestCase):
    def test_second_identical_case_uses_saved_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            client = _FakeClient(_response())
            agent = CounterfactualAgent(path, model_client=client, deployment="test-model")
            inputs = dict(
                case_id="case-1",
                patient_narrative="Patient context.",
                retfound_probability=72.0,
                oct_report="OCT report.",
                slo_report="SLO report.",
                cdr=0.62,
                trust_score=0.81,
            )

            first = agent.analyze(**inputs)
            second = agent.analyze(**inputs)

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(client.chat.completions.calls, 1)
            self.assertEqual(first["label_flip_scenarios"], ["without_retfound_probability"])
            self.assertEqual(len(path.read_text().splitlines()), 1)

            reloaded_client = _FakeClient(_response())
            reloaded = CounterfactualAgent(path, model_client=reloaded_client, deployment="test-model")
            third = reloaded.analyze(**inputs)
            self.assertTrue(third["cache_hit"])
            self.assertEqual(reloaded_client.chat.completions.calls, 0)

    def test_evidence_change_creates_new_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            client = _FakeClient(_response())
            agent = CounterfactualAgent(path, model_client=client, deployment="test-model")
            inputs = dict(
                case_id="case-1",
                patient_narrative="Patient context.",
                retfound_probability=72.0,
                oct_report="OCT report.",
                slo_report="SLO report.",
                cdr=0.62,
                trust_score=0.81,
            )
            agent.analyze(**inputs)
            agent.analyze(**{**inputs, "cdr": 0.70})
            self.assertEqual(client.chat.completions.calls, 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
