import re
import time
from typing import List

import requests
from bs4 import BeautifulSoup

_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
}


def fetch_finviz_headlines(ticker: str, timeout: float = 10.0) -> List[str]:
    sym = re.sub(r"[^A-Za-z0-9_\-]", "", ticker).upper()
    url = f"https://finviz.com/quote.ashx?t={sym}"
    r = requests.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    headlines: List[str] = []
    # FinViz news table id: news-table; rows are <tr> with <a> inside
    tbl = soup.find(id="news-table")
    if not tbl:
        return headlines
    for a in tbl.find_all("a"):
        text = (a.get_text() or "").strip()
        if text:
            headlines.append(text)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for h in headlines:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq


