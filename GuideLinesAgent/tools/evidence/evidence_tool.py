import os
import requests
import json
import re
from openai import AzureOpenAI

AZURE_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT="gpt-5.1"
# Initialize AzureOpenAI client only if env vars are provided; otherwise disable LLM calls
client = None
if AZURE_KEY and AZURE_ENDPOINT:
    try:
        client = AzureOpenAI(
            api_key=AZURE_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_ENDPOINT
        )
    except Exception as e:
        print(f"[evidence_tool] AzureOpenAI init error: {e}")
        client = None
else:
    print("[evidence_tool] AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set; LLM calls disabled")


EMPTY_EXTRACTION = {
    "condition": "",
    "required_clinical_info": [],
    "guideline_pathway": [],
    "diagnostic_criteria": [],
    "diagnostic_procedures": [],
    "risk_factors": [],
    "image_findings": [],
    "differential_diagnoses": [],
    "confidence": "low",
}


def _empty_extraction(query: str) -> dict:
    return {
        **EMPTY_EXTRACTION,
        "condition": query,
    }


def _extract_matching_terms(text: str, terms: list[str]) -> list[str]:
    found = []
    lowered = text.lower()
    for term in terms:
        if term.lower() in lowered and term not in found:
            found.append(term)
    return found


def heuristic_extract_clinical_recommendations(text: str, query: str) -> dict:
    lowered = text.lower()

    required_info = _extract_matching_terms(
        lowered,
        [
            "age",
            "sex",
            "race",
            "ethnicity",
            "family history",
            "migraine",
            "vasospasm",
            "raynaud",
            "steroid",
            "trauma",
            "symptoms",
            "intraocular pressure",
            "iop",
            "central corneal thickness",
        ],
    )

    guideline_pathway = _extract_matching_terms(
        lowered,
        [
            "guideline",
            "consensus",
            "preferred practice pattern",
            "aao",
            "nice",
            "diagnostic workup",
            "screening",
        ],
    )

    diagnostic_criteria = _extract_matching_terms(
        lowered,
        [
            "visual field defect",
            "optic nerve damage",
            "progressive cupping",
            "normal iop",
            "open angle",
            "retinal nerve fiber layer thinning",
        ],
    )

    diagnostic_procedures = _extract_matching_terms(
        lowered,
        [
            "tonometry",
            "gonioscopy",
            "pachymetry",
            "oct",
            "optical coherence tomography",
            "perimetry",
            "visual field",
            "fundus exam",
            "disc exam",
            "optic nerve exam",
        ],
    )

    risk_factors = _extract_matching_terms(
        lowered,
        [
            "family history",
            "asian",
            "african",
            "migraine",
            "vasospasm",
            "raynaud",
            "age",
            "steroid",
            "myopia",
        ],
    )

    image_findings = _extract_matching_terms(
        lowered,
        [
            "rnfl thinning",
            "retinal nerve fiber layer thinning",
            "optic disc cupping",
            "cupping",
            "disc hemorrhage",
            "notching",
            "rim thinning",
        ],
    )

    differential_diagnoses = _extract_matching_terms(
        lowered,
        [
            "optic neuropathy",
            "compressive lesion",
            "ischemic optic neuropathy",
            "retinal disease",
            "nonglaucomatous optic neuropathy",
        ],
    )

    non_empty_groups = sum(
        bool(group)
        for group in [
            required_info,
            guideline_pathway,
            diagnostic_criteria,
            diagnostic_procedures,
            risk_factors,
            image_findings,
            differential_diagnoses,
        ]
    )

    confidence = "low"
    if non_empty_groups >= 4:
        confidence = "high"
    elif non_empty_groups >= 2:
        confidence = "medium"

    return {
        "condition": query,
        "required_clinical_info": required_info,
        "guideline_pathway": guideline_pathway,
        "diagnostic_criteria": diagnostic_criteria,
        "diagnostic_procedures": diagnostic_procedures,
        "risk_factors": risk_factors,
        "image_findings": image_findings,
        "differential_diagnoses": differential_diagnoses,
        "confidence": confidence,
    }


def _parse_llm_json(result_text: str) -> dict | None:
    if not result_text:
        return None

    candidate = result_text.strip()
    json_match = re.search(r'\{.*\}', candidate, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return {
        "condition": parsed.get("condition", ""),
        "required_clinical_info": parsed.get("required_clinical_info", []),
        "guideline_pathway": parsed.get("guideline_pathway", []),
        "diagnostic_criteria": parsed.get("diagnostic_criteria", []),
        "diagnostic_procedures": parsed.get("diagnostic_procedures", []),
        "risk_factors": parsed.get("risk_factors", []),
        "image_findings": parsed.get("image_findings", []),
        "differential_diagnoses": parsed.get("differential_diagnoses", []),
        "confidence": parsed.get("confidence", "low"),
    }

# check relevance of document before running through LLMs using keyword matching
# TO BE ADJUSTED FOR NOT JUST GLAUCOMA IN THE FUTURE
def is_relevant(title: str, abstract: str, query: str) -> tuple[bool, str]:
    """
    Quick keyword check for relevance.
    Returns: (is_relevant, reason)
    """
    text = f"{title} {abstract}".lower()
    query_lower = query.lower()
    
    medical_keywords = {
        "glaucoma": ["glaucoma", "iop", "intraocular pressure", "optic nerve", "glaucomatous"],
        "ophthalmology": ["eye", "ocular", "ophthalm", "vision", "retina", "cornea", "lens", "iris"],
        "guideline": ["guideline", "recommendation", "protocol", "standard", "consensus"],
        "diagnosis": ["diagnos", "workup", "screen", "criterion", "criteria", "exam", "imaging", "finding"],
        "treatment": ["treatment", "therapy", "drug", "medication", "management", "intervention", "surgery", "implant"],
    }
    
    relevant_keywords = set()
    for category, keywords in medical_keywords.items():
        for kw in keywords:
            if kw in text:
                relevant_keywords.add(category)
    
    if "glaucoma" in query_lower:
        if any(kw in text for kw in medical_keywords["glaucoma"]):
            # For diagnosis support, avoid treatment-only content with no diagnostic signal.
            has_diag = any(kw in text for kw in medical_keywords["diagnosis"])
            has_treat = any(kw in text for kw in medical_keywords["treatment"])
            if has_treat and not has_diag:
                return False, "Treatment-focused"
            return True, "Glaucoma-related"
        else:
            return False, "No glaucoma keywords"
    
    if any(kw in text for kw in medical_keywords["ophthalmology"]):
        has_diag = any(kw in text for kw in medical_keywords["diagnosis"])
        has_treat = any(kw in text for kw in medical_keywords["treatment"])
        if has_treat and not has_diag:
            return False, "Treatment-focused"
        return True, "Ophthalmology-related"
    
    return False, "Not medical/ophthalmology"




# fetch PMC API full doc text
def fetch_pmc_fulltext(pmcid: str) -> str:
    if not pmcid:
        return ""
    
    try:
        pmcid = pmcid.replace("PMC", "").replace("pmc", "").strip()
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/json/"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return ""
        
        data = resp.json()
        text_parts = []
        
        # doc title
        if "front" in data and "article-meta" in data["front"]:
            title = data["front"]["article-meta"].get("title-group", {}).get("article-title", {})
            if isinstance(title, dict) and "#text" in title:
                text_parts.append(title["#text"])
        
        # abstract
        if "front" in data and "article-meta" in data["front"]:
            abstract = data["front"]["article-meta"].get("abstract", {})
            if isinstance(abstract, dict) and "#text" in abstract:
                text_parts.append(abstract["#text"])
        
        # body txt
        if "body" in data:
            body=data["body"]
            for section in body.get("sec", []):
                if isinstance(section, dict) and "#text" in section:
                    text_parts.append(section["#text"])
        
        return " ".join(text_parts)
    
    except Exception as e:
        print(f"Error fetching PMC {pmcid}: {str(e)}")
        return ""


# clinical recommendations using LLM
def extract_clinical_recommendations(text: str, query: str) -> dict:
    if not text or len(text) < 50:
        return _empty_extraction(query)
    
    try:
        text = text[:3000]
        
        prompt = f"""
You are an expert ophthalmic physician. Extract structured diagnosis-support guideline info.

Original Clinical Query: {query}

Medical Text:
{text}

Return only JSON:
{{
    "condition": "condition or disease mentioned",
    "required_clinical_info": ["history or demographics needed (e.g., race, age, family history, comorbidities)"],
    "guideline_pathway": ["which diagnostic guideline or pathway to follow"],
    "diagnostic_criteria": ["diagnostic criteria or thresholds"],
    "diagnostic_procedures": ["recommended tests/exam steps for diagnosis"],
    "risk_factors": ["risk factor 1", "risk factor 2"],
    "image_findings": ["imaging/fundus/OCT findings relevant to diagnosis"],
    "differential_diagnoses": ["alternative diagnoses to rule out"],
    "confidence": "high/medium/low"
}}

Do not include treatment plans, medications, dosages, or procedures for treatment.
Return empty arrays/strings if not found.
"""
        
        if client is None:
            print("[evidence_tool] AzureOpenAI client not configured; skipping LLM extraction")
            return heuristic_extract_clinical_recommendations(text, query)

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=500
        )
        
        result_text = response.choices[0].message.content
        result = _parse_llm_json(result_text)
        if result is not None:
            return result

        print("[evidence_tool] LLM returned non-JSON output; using heuristic extraction")
        return heuristic_extract_clinical_recommendations(text, query)
    
    except Exception as e:
        print(f"Error extracting recommendations: {str(e)}")
        return heuristic_extract_clinical_recommendations(text, query)


def process_evidence(pubmed_result: dict, query: str) -> dict:
    title = pubmed_result.get("title", "")
    # Accept multiple possible text fields for different sources (pubmed vs web)
    abstract = (
        pubmed_result.get("text")
        or pubmed_result.get("abstract")
        or pubmed_result.get("snippet")
        or pubmed_result.get("description")
        or ""
    )
    
    is_rel, reason = is_relevant(title, abstract, query)
    
    if not is_rel:
        print(f"  [Skipped] {title[:50]}... ({reason})")
        return {
            **pubmed_result,
            "clinical_extraction": {
                **EMPTY_EXTRACTION,
                "skipped_reason": reason
            }
        }
    
    fulltext=""
    pmcid = pubmed_result.get("pmcid", "")
    # For web results, there may be a URL instead of pmid/pmcid
    url = pubmed_result.get("url", "")
    
    if pmcid:
        print(f"  [Fetching full-text] PMC{pmcid}...")
        fulltext = fetch_pmc_fulltext(pmcid)
    elif url and pubmed_result.get("source", "").lower() != "pubmed":
        # For web results, use the snippet/description as the text to extract from
        print(f"  [Web item] using snippet from {url}")
    
    text_to_extract = fulltext or abstract or title
    clinical_info = extract_clinical_recommendations(text_to_extract, query)
    
    enriched = {
        **pubmed_result,
        "clinical_extraction": clinical_info
    }
    
    return enriched
