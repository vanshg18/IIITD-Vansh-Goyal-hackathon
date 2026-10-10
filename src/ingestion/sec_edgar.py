"""
SEC EDGAR current-filings ingestion adapter.

This source reads the public EDGAR current-filings Atom feed for 8-K filings.
A descriptive SEC_USER_AGENT with a contact email is required by SEC fair
access guidance. The adapter maps filer CIKs to current tickers when available.
"""

from __future__ import annotations

import html
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import requests

from src.engine.schemas import Article, Entity, SourceInfo
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import clean_text


ATOM_NS = "{http://www.w3.org/2005/Atom}"


class SECEdgarSource(BaseSource):
    """Fetch recent public filings from SEC EDGAR."""

    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(
        self,
        user_agent: str | None = None,
        session: requests.Session | None = None,
    ):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        self.session = session or requests.Session()
        self._company_map: dict[str, dict[str, str]] | None = None

    @property
    def name(self) -> str:
        return "SEC EDGAR"

    @property
    def source_type(self) -> str:
        return "official"

    def fetch(
        self,
        limit: int = 20,
        form_type: str = "8-K",
        timeout: int = 20,
        **kwargs: Any,
    ) -> list[Article]:
        if limit < 1:
            return []
        self._validate_user_agent()

        params = {
            "action": "getcurrent",
            "type": form_type,
            "company": "",
            "dateb": "",
            "owner": "include",
            "count": min(int(limit), 100),
            "output": "atom",
        }
        response = self.session.get(
            self.BASE_URL,
            params=params,
            timeout=timeout,
            headers=self._headers(),
        )
        response.raise_for_status()

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise RuntimeError("SEC EDGAR returned invalid Atom/XML data.") from exc

        # Metadata lookup is best-effort: the official feed remains useful if
        # ticker metadata is temporarily unavailable.
        company_map = self._load_company_map(timeout=timeout)
        articles: list[Article] = []

        for entry in root.findall(f"{ATOM_NS}entry"):
            feed_title = self._child_text(entry, "title")
            if not feed_title:
                continue

            url = self._entry_url(entry)
            timestamp = self._parse_timestamp(
                self._child_text(entry, "updated")
                or self._child_text(entry, "published")
            )
            summary = self._clean_summary(self._child_text(entry, "summary"))
            cik = self._extract_cik(url)
            mapped = company_map.get(cik or "", {})
            company_name = mapped.get("title") or self._extract_company(feed_title)
            ticker = mapped.get("ticker")

            category = entry.find(f"{ATOM_NS}category")
            actual_form = (
                clean_text(category.attrib.get("term", ""))
                if category is not None
                else form_type
            ) or form_type

            title = feed_title
            details = [f"SEC EDGAR filing, Form {actual_form}"]
            if company_name:
                details.append(f"company {company_name}")
            details.append(f"filing title: {feed_title}")
            if summary:
                details.append(summary)
            raw_text = clean_text(". ".join(details))

            identity = url or f"{cik}|{actual_form}|{feed_title}|{timestamp.isoformat()}"
            digest = sha256(identity.encode("utf-8")).hexdigest()[:20]
            entities = (
                [Entity(name=company_name, entity_type="ORGANIZATION")]
                if company_name
                else []
            )
            tickers = [ticker.upper()] if ticker else []

            articles.append(
                Article(
                    article_id=f"sec_{digest}",
                    timestamp=timestamp,
                    source=SourceInfo(
                        name="SEC EDGAR",
                        source_type="official",
                        url=url,
                    ),
                    title=title,
                    raw_text=raw_text,
                    entities=entities,
                    tickers=tickers,
                    url=url,
                )
            )

        return articles[: min(int(limit), 100)]

    def _validate_user_agent(self) -> None:
        if not self.user_agent or "@" not in self.user_agent:
            raise RuntimeError(
                "Set SEC_USER_AGENT in .env to identify your application and "
                "contact email, e.g. 'EventPulse/1.0 Your Name your.email@example.com'."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/atom+xml, application/json;q=0.9, */*;q=0.8",
        }

    def _load_company_map(self, timeout: int) -> dict[str, dict[str, str]]:
        if self._company_map is not None:
            return self._company_map

        try:
            response = self.session.get(
                self.COMPANY_TICKERS_URL,
                timeout=timeout,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.values() if isinstance(payload, dict) else payload
            company_map: dict[str, dict[str, str]] = {}
            for row in rows:
                if not isinstance(row, dict) or "cik_str" not in row:
                    continue
                cik = str(int(row["cik_str"])).zfill(10)
                company_map[cik] = {
                    "ticker": clean_text(str(row.get("ticker") or "")),
                    "title": clean_text(str(row.get("title") or "")),
                }
            self._company_map = company_map
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            self._company_map = {}

        return self._company_map

    @staticmethod
    def _child_text(entry: ET.Element, name: str) -> str:
        child = entry.find(f"{ATOM_NS}{name}")
        return clean_text(child.text if child is not None else "")

    @staticmethod
    def _entry_url(entry: ET.Element) -> str | None:
        for link in entry.findall(f"{ATOM_NS}link"):
            href = clean_text(link.attrib.get("href", ""))
            if href and link.attrib.get("rel", "alternate") == "alternate":
                return href
        for link in entry.findall(f"{ATOM_NS}link"):
            href = clean_text(link.attrib.get("href", ""))
            if href:
                return href
        return None

    @staticmethod
    def _extract_cik(url: str | None) -> str | None:
        if not url:
            return None
        match = re.search(r"/data/(\d{1,10})/", url, flags=re.IGNORECASE)
        return match.group(1).zfill(10) if match else None

    @staticmethod
    def _extract_company(feed_title: str) -> str | None:
        # Some Atom feed titles include the registrant before its 10-digit CIK.
        match = re.search(r"-\s*(.*?)\s*\(\d{10}\)", feed_title)
        return clean_text(match.group(1)) if match else None

    @staticmethod
    def _clean_summary(value: str) -> str:
        value = re.sub(r"<[^>]*>", " ", value or "")
        return clean_text(html.unescape(value))

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        candidate = value.strip()
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.now(timezone.utc)
