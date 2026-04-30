import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List

CACHE_PATH = os.path.join(os.path.dirname(__file__), "case_cache.json")


def _normalise(value: str) -> str:
    if not value:
        return ""

    value = value.lower().strip()
    value = value.replace("[", "").replace("]", "")
    value = value.replace("(", "").replace(")", "")
    value = re.sub(r"[.,;:]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def load_case_cache() -> List[Dict[str, Any]]:
    if not os.path.exists(CACHE_PATH):
        return []

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    return data


CASE_CACHE = load_case_cache()


def _score(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _looks_like_neutral_citation(value: str) -> bool:
    if not value:
        return False

    return bool(
        re.search(
            r"^\s*\[?\d{4}\]?\s+[a-zA-Z]{2,10}\s+\d{1,4}\s*$",
            value.strip()
        )
    )


def find_cached_case(user_input: str) -> Optional[Dict[str, Any]]:
    if not user_input:
        return None

    raw_query = user_input.strip()
    query = _normalise(raw_query)
    query_is_citation = _looks_like_neutral_citation(raw_query)

    best_case = None
    best_score = 0.0

    for case in CASE_CACHE:
        neutral = case.get("neutral_citation", "")
        report = case.get("report_citation", "")

        # Citation inputs require exact normalised citation matching.
        # This prevents typo citations like [1992] HVA 23 matching [1992] HCA 23.
        if query_is_citation:
            if query == _normalise(neutral) or query == _normalise(report):
                return case
            continue

        searchable_values = [
            case.get("case_name", ""),
            neutral,
            report
        ] + case.get("aliases", [])

        for value in searchable_values:
            if not value:
                continue

            normalised_value = _normalise(value)

            if query == normalised_value:
                return case

            if query in normalised_value or normalised_value in query:
                return case

            score = _score(query, normalised_value)
            if score > best_score:
                best_score = score
                best_case = case

    # Fuzzy matching is allowed for case names and aliases only, not citations.
    if not query_is_citation and best_score >= 0.86:
        return best_case

    return None
