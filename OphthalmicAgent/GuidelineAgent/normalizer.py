def normalize_results(results: list) -> list:
    normalized = []

    for r in results:
        normalized.append({
            "text": r.get("snippet") or r.get("title"),
            "source": r.get("source"),
            "year": r.get("year"),
            "type": r.get("type"),
            "url": r.get("url", None),
            "journal": r.get("journal", None),
            "clinical_extraction": r.get("clinical_extraction", {})
        })

    return normalized