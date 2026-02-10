from tools.evidence.pubmed.pubmed_client import search_pubmed
# from tools.evidence.web.google_client import search_web
from tools.evidence.evidence_tool import process_evidence
from util.normalizer import normalize_results

# check if extracted doc has high value
def is_high_value(extraction: dict) -> bool:
    confidence = extraction.get("confidence", "low")
    has_treatments = bool(extraction.get("treatments", []))
    has_recommendations = bool(extraction.get("key_recommendations", []))
    has_risk_factors = bool(extraction.get("risk_factors", []))
    
    # high value if medium or high confidence and with recommendations/risk factors/treatments/dosage
    if confidence in ["high", "medium"]:
        return has_treatments or has_recommendations or has_risk_factors
    
    return False



def retrieve_guidelines(query: str, min_high_value: int = 3, max_total: int = 15) -> dict:
    all_enriched = []
    high_value_count = 0
    batch_size = 5
    batches_fetched = 0
    max_batches = max_total // batch_size
    
    # keep fetching until we have enough high-value results or hit the limit
    while high_value_count < min_high_value and batches_fetched < max_batches:
        print(f"[Fetching batch {batches_fetched + 1}]...")
        retstart = batches_fetched * batch_size  # Pagination offset
        pubmed_results = search_pubmed(query, max_results=batch_size, retstart=retstart)
        
        if not pubmed_results:
            break
        
        enriched_batch = [process_evidence(result, query) for result in pubmed_results]
        all_enriched.extend(enriched_batch)
        
        high_value_count = sum(
            1 for r in all_enriched 
            if is_high_value(r.get("clinical_extraction", {}))
        )
        
        batches_fetched += 1
        print(f"  High-value results so far: {high_value_count}/{min_high_value}")
    
    # return only high value results + low value ones if not enough high value
    high_value_results = [
        r for r in all_enriched 
        if is_high_value(r.get("clinical_extraction", {}))
    ]
    
    # med quality if not enough high qual
    if len(high_value_results) < min_high_value:
        medium_quality = [
            r for r in all_enriched 
            if r.get("clinical_extraction", {}).get("confidence") in ["medium", "high"]
            and r not in high_value_results
        ]
        final_results = high_value_results + medium_quality[:min_high_value - len(high_value_results)]
    else:
        final_results = high_value_results[:min_high_value]
    
    normalized = normalize_results(final_results)
    
    return {
        "query": query,
        "evidence": normalized,
        "stats": {
            "total_fetched": len(all_enriched),
            "high_value": len(high_value_results),
            "returned": len(final_results)
        }
    }