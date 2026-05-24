from __future__ import annotations
import os
import time
import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class PubMedClient:
    def __init__(self):
        self.api_key = os.getenv("NCBI_API_KEY")
        self.email   = os.getenv("NCBI_EMAIL", "your_email@example.com")
        self._delay  = 0.11 if self.api_key else 0.35

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        params["tool"]  = "rag_eval_project"
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        resp = requests.get(BASE_URL + endpoint, params=params, timeout=20)
        resp.raise_for_status()
        time.sleep(self._delay)
        return resp

    def search(self, query: str, n: int) -> dict:
        data   = self._get("esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": n,
            "retmode": "json", "usehistory": "y", "sort": "relevance",
        }).json()
        result = data["esearchresult"]
        print(
            f"  pubmed search: '{query}' "
            f"{int(result['count']):,} total, fetching {len(result['idlist'])}"
        )
        return {
            "pmids":     result["idlist"],
            "query_key": result["querykey"],
            "webenv":    result["webenv"],
            "count":     result["count"],
        }

    def fetch_xml(self, query_key: str, webenv: str, n: int) -> str:
        resp = self._get("efetch.fcgi", {
            "db": "pubmed", "query_key": query_key, "WebEnv": webenv,
            "retstart": 0, "retmax": n, "rettype": "abstract", "retmode": "xml",
        })
        return resp.text