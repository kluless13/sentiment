from typing import List

import feedparser


def fetch_rss_headlines(feed_url: str, limit: int = 50) -> List[str]:
    d = feedparser.parse(feed_url)
    items = d.get("entries", [])[:limit]
    out: List[str] = []
    for e in items:
        title = (e.get("title") or "").strip()
        if title:
            out.append(title)
    return out


