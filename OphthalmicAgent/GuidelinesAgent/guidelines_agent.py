from GuidelinesAgent.orchestrator import retrieve_guidelines
import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "gpt-5.1"

class GuidelinesAgent:
    def __init__(self):
        self.name = "GuidelinesAgent"
        self.client = None
        if AZURE_KEY and AZURE_ENDPOINT:
            try:
                self.client = AzureOpenAI(
                    api_key=AZURE_KEY,
                    api_version="2024-12-01-preview",
                    azure_endpoint=AZURE_ENDPOINT
                )
            except Exception as e:
                print(f"[{self.name}] AzureOpenAI init error: {e}")

    def _prune_empty_fields(self, value):
        if isinstance(value, dict):
            pruned = {}
            for key, item in value.items():
                cleaned = self._prune_empty_fields(item)
                if cleaned in (None, "", [], {}):
                    continue
                pruned[key] = cleaned
            return pruned

        if isinstance(value, list):
            pruned = []
            for item in value:
                cleaned = self._prune_empty_fields(item)
                if cleaned in (None, "", [], {}):
                    continue
                pruned.append(cleaned)
            return pruned

        return value

    def _format_evidence_for_prompt(self, evidence: dict) -> str:
        cleaned = self._prune_empty_fields(evidence)
        lines = [
            f"Original note: {cleaned.get('patient_context', {}).get('patient_note', '')}",
            f"Search query used: {cleaned.get('query', '')}",
            f"Diagnosis only: {cleaned.get('diagnosis_only', False)}",
        ]

        stats = cleaned.get("stats", {})
        if stats:
            lines.append(
                "Retrieval stats: "
                f"returned={stats.get('returned', 0)}, "
                f"high_value={stats.get('high_value', 0)}, "
                f"total_fetched={stats.get('total_fetched', 0)}, "
                f"calls_made={stats.get('calls_made', 0)}"
            )

        for index, item in enumerate(cleaned.get("evidence", []), start=1):
            lines.append(f"Evidence {index} text: {item.get('text', '')}")
            if item.get("source"):
                lines.append(f"Evidence {index} source: {item['source']}")
            if item.get("journal"):
                lines.append(f"Evidence {index} journal: {item['journal']}")
            if item.get("year"):
                lines.append(f"Evidence {index} year: {item['year']}")
            if item.get("type"):
                lines.append(f"Evidence {index} type: {item['type']}")
            if item.get("url"):
                lines.append(f"Evidence {index} url: {item['url']}")

            extraction = item.get("clinical_extraction", {})
            if extraction.get("diagnostic_criteria"):
                lines.append(
                    f"Evidence {index} diagnostic criteria: {', '.join(extraction['diagnostic_criteria'])}"
                )
            if extraction.get("diagnostic_procedures"):
                lines.append(
                    f"Evidence {index} diagnostic procedures: {', '.join(extraction['diagnostic_procedures'])}"
                )
            if extraction.get("required_clinical_info"):
                lines.append(
                    f"Evidence {index} required clinical info: {', '.join(extraction['required_clinical_info'])}"
                )
            if extraction.get("risk_factors"):
                lines.append(
                    f"Evidence {index} risk factors: {', '.join(extraction['risk_factors'])}"
                )
            if extraction.get("image_findings"):
                lines.append(
                    f"Evidence {index} image findings: {', '.join(extraction['image_findings'])}"
                )

        return "\n".join(lines)

    def extract_note_query(self, note: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )

        prompt = f"""
You are preparing a search query for an ophthalmology guidelines retrieval agent.

Rules:
- Return one plain-text search query only.
- Keep it diagnosis-oriented and as specific as the note supports.
- Include the most relevant diagnostic clues, demographics, and exam or imaging findings if they materially narrow the diagnosis.
- Do not include treatment or management language.
- Do not use JSON, markdown, bullets, or explanation.

Patient note:
{note}
"""

        response = self.client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=400,
        )

        result_text = (response.choices[0].message.content or "").strip()
        return result_text or note[:120]

    def summarize_evidence_text(self, note: str, evidence: dict) -> str:
        if self.client is None:
            raise RuntimeError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )

        prompt = f"""
You are summarizing retrieved ophthalmology guidelines for diagnosing eye diseases.

Write the output as plain text paragraphs only.

Rules:
- No JSON.
- No markdown.
- No bullet points.
- No headers.
- Keep the output very short.
- Use 2 short paragraphs maximum.
- Focus only on information that could help the model diagnose the patient.
- Prioritize likely diagnosis, strongest supporting findings, key risk factors, important exam or imaging findings, and the most useful differentiating clues.
- Exclude management, treatment, citations, source descriptions, and background that does not change the diagnostic reasoning.
- If the evidence is limited or mixed, say that in one short sentence.
- Base the summary only on the patient note and retrieved evidence below.

Patient note:
{note}

Retrieved evidence:
{self._format_evidence_for_prompt(evidence)}
"""

        response = self.client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=220,
        )

        return (response.choices[0].message.content or "").strip()

    def consult_note(
        self,
        note: str,
        max_results: int = 5,
        diagnosis_only: bool = True,
    ):
        extracted_query = self.extract_note_query(note)

        evidence = self.consult(
            query=extracted_query,
            max_results=max_results,
            patient_context={"patient_note": note},
            diagnosis_only=diagnosis_only,
        )
        return self.summarize_evidence_text(note, evidence)

    # input: suspected diagnosis + optional patient context for diagnostic guidance
    def consult(
        self,
        query: str,
        max_results: int = 5,
        race: str | None = None,
        conditions: list[str] | None = None,
        image_findings: list[str] | None = None,
        symptoms: list[str] | None = None,
        age: int | None = None,
        sex: str | None = None,
        patient_context: dict | None = None,
        diagnosis_only: bool = True,
    ):
        print(f"[{self.name}] Consulting for query: '{query}'")

        merged_context = dict(patient_context or {})
        if race is not None:
            merged_context["race"] = race
        if conditions is not None:
            merged_context["conditions"] = conditions
        if image_findings is not None:
            merged_context["image_findings"] = image_findings
        if symptoms is not None:
            merged_context["symptoms"] = symptoms
        if age is not None:
            merged_context["age"] = age
        if sex is not None:
            merged_context["sex"] = sex

        evidence = retrieve_guidelines(
            query,
            max_total=max_results,
            patient_context=merged_context,
            diagnosis_only=diagnosis_only,
        )
        evidence["evidence"] = evidence.get("evidence", [])[:max_results]

        return self._prune_empty_fields(evidence)


# test
if __name__ == "__main__":
    agent = GuidelinesAgent()

    note = (
        "62 year old Asian female with family history of glaucoma and migraine. "
        "Reports peripheral vision loss. Imaging shows inferior RNFL thinning and optic disc cupping. "
        "Concern for normal-tension glaucoma."
    )
    results = agent.consult_note(
        note,
        max_results=5,
        diagnosis_only=True,
    )

#    print("\n--- Retrieved Evidence ---")
    print(results)
