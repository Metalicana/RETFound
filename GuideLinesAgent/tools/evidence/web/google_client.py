import os
import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_NUM = 5


def search_web_serpapi(query: str, count: int = DEFAULT_NUM, start: int = 0) -> list:
    if not SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY environment variable not set")

    params = {
        "engine": "google",
        "q": query,
        "num": count,
        "start": start,
        "api_key": SERPAPI_KEY,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        results = []

        for item in organic[:count]:
            title = item.get("title") or ""
            snippet = item.get("snippet") or item.get("snippet_highlighted") or ""
            url = item.get("link") or item.get("serpapi_link") or item.get("displayed_link") or ""

            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": "SerpApi",
                "type": "web_result",
            })

        return results

    except Exception as e:
        print(f"[SerpApi] Error searching for '{query}': {e}")
        return []

    # Returns a list of dicts with keys: title, snippet, url, source, type
