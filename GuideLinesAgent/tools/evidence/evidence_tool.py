import os
import requests
import json
import re
from openai import AzureOpenAI

AZURE_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT="gpt-5.1"

client = AzureOpenAI(
    api_key="",
    api_version="2024-12-01-preview",
    azure_endpoint="")

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
        "treatment": ["treatment", "therapy", "drug", "medication", "management", "intervention", "surgery", "implant"],
        "guideline": ["guideline", "recommendation", "protocol", "standard", "consensus"],
    }
    
    relevant_keywords = set()
    for category, keywords in medical_keywords.items():
        for kw in keywords:
            if kw in text:
                relevant_keywords.add(category)
    
    if "glaucoma" in query_lower:
        if any(kw in text for kw in medical_keywords["glaucoma"]):
            return True, "Glaucoma-related"
        else:
            return False, "No glaucoma keywords"
    
    if any(kw in text for kw in medical_keywords["ophthalmology"]):
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
        return {
            "condition": query,
            "key_recommendations": [],
            "treatments": [],
            "risk_factors": [],
            "dosage_info": "",
            "confidence": "low"
        }
    
    try:
        text = text[:3000]
        
        prompt = f"""
You are an expert ophthalmic physician. Extract structured clinical info.

Original Clinical Query: {query}

Medical Text:
{text}

Return only JSON:
{{
    "condition": "condition or disease mentioned",
    "key_recommendations": ["recommendation 1", "recommendation 2"],
    "treatments": ["treatment option 1", "treatment option 2"],
    "risk_factors": ["risk factor 1", "risk factor 2"],
    "dosage_info": "any dosage/administration info",
    "confidence": "high/medium/low"
}}

Return empty arrays/strings if not found.
"""
        
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=500
        )
        
        result_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(0)
        
        result = json.loads(result_text)
        return result
    
    except Exception as e:
        print(f"Error extracting recommendations: {str(e)}")
        return {
            "condition": query,
            "key_recommendations": [],
            "treatments": [],
            "risk_factors": [],
            "dosage_info": "",
            "confidence": "low"
        }


def process_evidence(pubmed_result: dict, query: str) -> dict:
    title = pubmed_result.get("title", "")
    abstract = pubmed_result.get("text", "")
    
    is_rel, reason = is_relevant(title, abstract, query)
    
    if not is_rel:
        print(f"  [Skipped] {title[:50]}... ({reason})")
        return {
            **pubmed_result,
            "clinical_extraction": {
                "condition": "",
                "key_recommendations": [],
                "treatments": [],
                "risk_factors": [],
                "dosage_info": "",
                "confidence": "low",
                "skipped_reason": reason
            }
        }
    
    fulltext=""
    pmcid = pubmed_result.get("pmcid","")
    
    if pmcid:
        print(f"  [Fetching full-text] PMC{pmcid}...")
        fulltext = fetch_pmc_fulltext(pmcid)
    
    text_to_extract = fulltext or abstract or title
    clinical_info = extract_clinical_recommendations(text_to_extract, query)
    
    enriched = {
        **pubmed_result,
        "clinical_extraction": clinical_info
    }
    
    return enriched
