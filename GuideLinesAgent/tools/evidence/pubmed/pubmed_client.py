import requests
import json

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def search_pubmed(query: str, max_results: int = 5, retstart: int = 0) -> list:
    # Date filter, currently from within the last 3 years to now
    from datetime import datetime, timedelta
    three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime("%Y/%m/%d")
    today = datetime.now().strftime("%Y/%m/%d")
    query_with_date = f"{query} AND {three_years_ago}[PDAT] : {today}[PDAT]"
    
    search_resp = requests.get(
        BASE + "esearch.fcgi",
        params={
            "db" : "pubmed",
            "term": query_with_date,
            "retmode": "json",
            "retmax": max_results,
            "retstart": retstart
        }
    ).json()

    ids = search_resp["esearchresult"].get("idlist", [])
    if not ids:
        return []
    
    summary_resp = requests.get(
        BASE + "esummary.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }
    ).json()

    results = []
    for uid in summary_resp["result"].get("uids", []):
        r = summary_resp["result"][uid]
        
        # get PMCID from esummary articleids array
        pmcid = ""
        if "articleids" in r:
            for aid in r["articleids"]:
                if aid.get("idtype") == "pmc":
                    pmcid = aid.get("value", "")
                    break
        
        results.append({
            "title": r.get("title"),
            "journal": r.get("fulljournalname"),
            "year": r.get("pubdate", "")[:4],
            "pmid": uid,
            "pmcid": pmcid,
            "source": "PubMed",
            "type": "peer_reviewed"
        })

    return results