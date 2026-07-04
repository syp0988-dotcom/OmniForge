from __future__ import annotations

import re
from typing import Any

import requests

from agentflow.utils.logging import build_logger

logger = build_logger("search_tool")


class SearchTool:
    """DuckDuckGo-backed search tool that can be extended to other providers later."""

    def search(self, query: str) -> list[dict[str, Any]]:
        """Perform a real web search using the DuckDuckGo HTML endpoint."""
        url = "https://html.duckduckgo.com/html/"
        payload = {"q": query, "kl": "us-en"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }

        try:
            response = requests.get(url, params=payload, headers=headers, timeout=15)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            logger.exception("Search request failed: %s", exc)
            return []

        results: list[dict[str, Any]] = []
        title_matches = re.findall(r'<a rel="nofollow" class="[^\"]*result__a[^\"]*" href="([^"]+)">([^<]+)</a>', html)
        snippet_matches = re.findall(r'<a[^>]+class="[^\"]*result__snippet[^\"]*"[^>]*>(.*?)</a>', html, re.S)

        for index, (url, title) in enumerate(title_matches[:5]):
            summary = re.sub(r"<.*?>", "", snippet_matches[index] if index < len(snippet_matches) else "")
            summary = re.sub(r"\s+", " ", summary).strip()
            results.append(
                {
                    "source": "duckduckgo",
                    "title": re.sub(r"\s+", " ", title).strip(),
                    "url": url,
                    "summary": summary or "No summary available",
                }
            )

        return results
