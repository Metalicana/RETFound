from GuidelinesAgent.pubmed_client import search_pubmed
from GuidelinesAgent.google_client import search_web_serpapi
from GuidelinesAgent.evidence_tool import process_evidence
from GuidelinesAgent.normalizer import normalize_results


def normalize_condition_query(query: str) -> str:
    noise_terms = [
        "diagnosis",
        "diagnostic",
        "guideline",
        "guidelines",
        "criteria",
        "criterion",
        "workup",
        "screening",
        "exam",
        "imaging",
        "treatment",
        "therapy",
        "management",
        "plan",
    ]

    normalized = query
    for term in noise_terms:
        normalized = normalized.replace(term, "")
        normalized = normalized.replace(term.title(), "")

    normalized = " ".join(normalized.split())
    return normalized or query.strip()

# check if extracted doc has high value for diagnosis support
def is_high_value(extraction: dict) -> bool:
    confidence = extraction.get("confidence", "low")
    has_diagnostic_procedures = bool(extraction.get("diagnostic_procedures", []))
    has_diagnostic_criteria = bool(extraction.get("diagnostic_criteria", []))
    has_required_info = bool(extraction.get("required_clinical_info", []))
    has_risk_factors = bool(extraction.get("risk_factors", []))
    has_image_findings = bool(extraction.get("image_findings", []))
    
    # High value if medium/high confidence and contains diagnosis-focused content.
    if confidence in ["high", "medium"]:
        return (
            has_diagnostic_procedures
            or has_diagnostic_criteria
            or has_required_info
            or has_risk_factors
            or has_image_findings
        )
    
    return False


def build_diagnostic_query(query: str, patient_context: dict | None = None) -> str:
    patient_context = patient_context or {}
    condition_query = normalize_condition_query(query)
    context_bits = []

    race = patient_context.get("race")
    age = patient_context.get("age")
    sex = patient_context.get("sex")
    conditions = patient_context.get("conditions", [])
    image_findings = patient_context.get("image_findings", [])
    symptoms = patient_context.get("symptoms", [])

    if age:
        context_bits.append(f"age {age}")
    if sex:
        context_bits.append(sex)
    if race:
        context_bits.append(race)
    if conditions:
        context_bits.extend(conditions)
    if symptoms:
        context_bits.extend(symptoms)
    if image_findings:
        context_bits.extend(image_findings)

    context_clause = f" {' '.join(context_bits)}" if context_bits else ""

    return (
        f"{condition_query}{context_clause} ophthalmology guideline diagnosis "
        "diagnostic criteria differential diagnosis workup exam imaging risk factors"
    )


def is_treatment_focused(item: dict) -> bool:
    text = " ".join(
        [
            item.get("title", "") or "",
            item.get("snippet", "") or "",
            item.get("text", "") or "",
        ]
    ).lower()

    treatment_terms = [
        "treatment",
        "therapy",
        "drug",
        "medication",
        "dose",
        "dosage",
        "surgery",
        "surgical",
        "intervention",
        "management plan",
    ]
    diagnosis_terms = ["diagnos", "screen", "criterion", "criteria", "workup", "exam", "imaging", "finding"]

    has_treatment = any(term in text for term in treatment_terms)
    has_diagnosis = any(term in text for term in diagnosis_terms)

    return has_treatment and not has_diagnosis



def retrieve_guidelines(
    query: str,
    min_high_value: int = 3,
    max_total: int = 15,
    max_calls: int = 10,
    patient_context: dict | None = None,
    diagnosis_only: bool = True,
) -> dict:
    search_query = build_diagnostic_query(query, patient_context) if diagnosis_only else query
    all_enriched = []
    high_value_count = 0
    batch_size = 5
    batches_fetched = 0
    max_batches = max_total // batch_size if batch_size else 0

    calls_made = 0

    # keep fetching until we have enough high-value results or hit the limit
    while high_value_count < min_high_value and batches_fetched < max_batches:
#        print(f"[Fetching batch {batches_fetched + 1}]...")
        retstart = batches_fetched * batch_size  # Pagination offset for PubMed

        pubmed_results = []
        web_results = []

        # PubMed call
        if calls_made < max_calls:
            pubmed_results = search_pubmed(search_query, max_results=batch_size, retstart=retstart)
            calls_made += 1

        # SerpApi call
        if calls_made < max_calls:
            web_start = batches_fetched * batch_size
            web_results = search_web_serpapi(search_query, count=batch_size, start=web_start)
            calls_made += 1

        if not pubmed_results and not web_results:
            break

        combined = []
        if pubmed_results:
            combined.extend(pubmed_results)
        if web_results:
            combined.extend(web_results)

        if diagnosis_only:
            combined = [r for r in combined if not is_treatment_focused(r)]

        enriched_batch = [process_evidence(result, query) for result in combined]
        all_enriched.extend(enriched_batch)

        high_value_count = sum(
            1 for r in all_enriched 
            if is_high_value(r.get("clinical_extraction", {}))
        )

        batches_fetched += 1
#        print(f"  High-value results so far: {high_value_count}/{min_high_value} (calls made: {calls_made}/{max_calls})")
    
    # collect high-value results (return all found, cap to max_total)
    high_value_results = [
        r for r in all_enriched 
        if is_high_value(r.get("clinical_extraction", {}))
    ]

    # If not enough high-value, include medium-quality results
    if len(high_value_results) < min_high_value:
        medium_quality = [
            r for r in all_enriched 
            if r.get("clinical_extraction", {}).get("confidence") in ["medium", "high"]
            and r not in high_value_results
        ]
        final_results = high_value_results + medium_quality
    else:
        final_results = high_value_results

    # enforce overall max_total cap
    final_results = final_results[:max_total]
    
    normalized = normalize_results(final_results)
    
    return {
        "query": query,
        "search_query": search_query,
        "diagnosis_only": diagnosis_only,
        "patient_context": patient_context or {},
        "evidence": normalized,
        "stats": {
            "total_fetched": len(all_enriched),
            "high_value": len(high_value_results),
            "returned": len(final_results),
            "calls_made": calls_made
        }
    }
