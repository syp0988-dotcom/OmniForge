"""Search provider abstraction — multi-source search support.

Architecture::

    BaseSearchProvider  ←  abstract interface
          │
    ┌─────┼──────────┐
    │     │          │
  DuckDuckGo  Brave  Serper  Google  (future)

Usage::

    provider = DuckDuckGoProvider()
    results: list[dict] = provider.search("your query")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import unquote_plus

import requests

from agentflow.utils.logging import build_logger

logger = build_logger("search_provider")


class BaseSearchProvider(ABC):
    """Abstract search provider for multi-source search support.

    Each subclass implements a single ``search(query)`` method that
    returns a normalized ``list[dict]`` with keys:

      - ``title``   — result title
      - ``url``     — result link
      - ``snippet`` — text summary / excerpt
    """

    @abstractmethod
    def search(self, query: str) -> list[dict[str, Any]]:
        """Execute a search and return normalized results."""
        ...


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo HTML endpoint search provider.

    Scrapes the public ``html.duckduckgo.com/html/`` page (no API key
    required).  Returns up to 5 results.
    """

    BASE_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    MAX_RESULTS = 5

    def search(self, query: str) -> list[dict[str, Any]]:
        """Execute a DuckDuckGo search and return normalized results."""
        payload = {"q": query, "kl": "us-en"}
        headers = {"User-Agent": self.USER_AGENT}

        try:
            response = requests.get(
                self.BASE_URL, params=payload, headers=headers, timeout=15
            )
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:
            logger.exception("DuckDuckGo search failed: %s", exc)
            return []

        return self._parse_results(html)

    def _parse_results(self, html: str) -> list[dict[str, Any]]:
        """Extract title/url/snippet tuples from the DuckDuckGo HTML page."""
        results: list[dict[str, Any]] = []

        title_matches = re.findall(
            r'<a rel="nofollow" class="[^"]*result__a[^"]*" href="([^"]+)">([^<]+)</a>',
            html,
        )
        snippet_matches = re.findall(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.S
        )

        for index, (url, title) in enumerate(title_matches[: self.MAX_RESULTS]):
            snippet = (
                re.sub(
                    r"<.*?>",
                    "",
                    snippet_matches[index] if index < len(snippet_matches) else "",
                )
            )
            snippet = re.sub(r"\s+", " ", snippet).strip()

            results.append(
                {
                    "title": re.sub(r"\s+", " ", title).strip(),
                    "url": self._clean_url(url),
                    "snippet": snippet or "No summary available",
                }
            )

        return results

    @staticmethod
    def _clean_url(url: str) -> str:
        """Decode DuckDuckGo redirect URLs to the original target."""
        if "duckduckgo.com/l/?uddg=" in url:
            encoded = url.split("uddg=", 1)[-1]
            return unquote_plus(encoded)
        return url
