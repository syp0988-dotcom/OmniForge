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

import os
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


class TavilyProvider(BaseSearchProvider):
    """Tavily Search API provider.

    Requires an API key from https://app.tavily.com.
    Supports general, news, and finance search with configurable depth.
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def search(self, query: str, max_results: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        """Execute search via Tavily API and return normalized results."""
        if not self._api_key:
            logger.warning("Tavily API key not configured")
            return []

        payload = {
            "query": query,
            "max_results": min(max_results, 20),
            "search_depth": kwargs.get("search_depth", "basic"),
            "topic": kwargs.get("topic", "general"),
            "include_answer": False,
            "include_raw_content": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            response = requests.post(
                self.BASE_URL, json=payload, headers=headers, timeout=15
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.exception("Tavily search failed: %s", exc)
            return []

        return self._normalize(data.get("results", []))

    @staticmethod
    def _normalize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Tavily API results to standard {title, url, snippet} format."""
        results: list[dict[str, Any]] = []
        for item in raw:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        return results


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
