"""Fetch CVE/vulnerability data from the NIST NVD 2.0 API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


@dataclass
class CVERecord:
    """A single CVE entry with fields relevant to CKU-Sim."""

    cve_id: str
    published: str  # ISO date
    cvss_v3_score: float | None = None
    cvss_v3_severity: str | None = None
    cvss_v3_vector: str | None = None
    cwe_ids: list[str] | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "cve_id": self.cve_id,
            "published": self.published,
            "cvss_v3_score": self.cvss_v3_score,
            "cvss_v3_severity": self.cvss_v3_severity,
            "cvss_v3_vector": self.cvss_v3_vector,
            "cwe_ids": self.cwe_ids,
            "description": self.description,
        }


def fetch_cves_for_cpe(
    cpe_id: str,
    api_key: str | None = None,
    rate_limit: float = 6.0,
    cache_dir: Path | None = None,
) -> list[CVERecord]:
    """Fetch all CVEs associated with a CPE identifier from NVD.

    Args:
        cpe_id: CPE 2.3 identifier string.
        api_key: Optional NVD API key (increases rate limit).
        rate_limit: Seconds between requests.
        cache_dir: If set, cache raw JSON responses here.

    Returns:
        List of CVERecord objects.
    """
    if cache_dir:
        cache_file = cache_dir / f"{_safe_filename(cpe_id)}.json"
        if cache_file.exists():
            logger.info(f"Loading cached CVEs for {cpe_id}")
            with open(cache_file) as f:
                raw_items = json.load(f)
            return [_parse_cve(item) for item in raw_items]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key
        rate_limit = max(rate_limit, 0.6)  # API key allows faster

    all_items = []
    start_index = 0
    results_per_page = 2000

    while True:
        params = {
            "cpeName": cpe_id,
            "startIndex": start_index,
            "resultsPerPage": results_per_page,
        }

        logger.info(
            f"Fetching CVEs for {cpe_id} (offset={start_index})..."
        )

        try:
            resp = requests.get(
                NVD_API_BASE, params=params, headers=headers, timeout=30
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"NVD API request failed: {e}")
            break

        data = resp.json()
        vulnerabilities = data.get("vulnerabilities", [])
        all_items.extend(vulnerabilities)

        total_results = data.get("totalResults", 0)
        start_index += len(vulnerabilities)

        logger.info(
            f"  Retrieved {len(vulnerabilities)} CVEs "
            f"({start_index}/{total_results} total)"
        )

        if start_index >= total_results or not vulnerabilities:
            break

        time.sleep(rate_limit)

    # Cache raw response
    if cache_dir and all_items:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(all_items, f)
        logger.info(f"Cached {len(all_items)} CVEs to {cache_file}")

    return [_parse_cve(item) for item in all_items]


def _parse_cve(vuln_item: dict) -> CVERecord:
    """Parse a single NVD vulnerability item into a CVERecord."""
    cve = vuln_item.get("cve", {})
    cve_id = cve.get("id", "UNKNOWN")
    published = cve.get("published", "")

    # CVSS v3.1 score
    cvss_score = None
    cvss_severity = None
    cvss_vector = None
    metrics = cve.get("metrics", {})

    for key in ("cvssMetricV31", "cvssMetricV30"):
        metric_list = metrics.get(key, [])
        if metric_list:
            primary = next(
                (m for m in metric_list if m.get("type") == "Primary"),
                metric_list[0],
            )
            cvss_data = primary.get("cvssData", {})
            cvss_score = cvss_data.get("baseScore")
            cvss_severity = cvss_data.get("baseSeverity")
            cvss_vector = cvss_data.get("vectorString")
            break

    # CWE IDs
    cwe_ids = []
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-"):
                cwe_ids.append(val)

    # Description (English)
    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    return CVERecord(
        cve_id=cve_id,
        published=published,
        cvss_v3_score=cvss_score,
        cvss_v3_severity=cvss_severity,
        cvss_v3_vector=cvss_vector,
        cwe_ids=cwe_ids or None,
        description=description,
    )


def _safe_filename(cpe_id: str) -> str:
    """Convert CPE ID to a filesystem-safe filename."""
    return cpe_id.replace(":", "_").replace("*", "ALL").replace("/", "_")
