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
    value = value.replace("&", " and ")
    value = value.replace("[", " ").replace("]", " ")
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_case_cache() -> List[Dict[str, Any]]:
    if not os.path.exists(CACHE_PATH):
        return []

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("case_cache.json must be a JSON array of case objects")

    return data


CASE_CACHE = load_case_cache()


def _score(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _searchable_values(case: Dict[str, Any]) -> List[str]:
    values = [
        case.get("case_name", ""),
        case.get("neutral_citation", ""),
        case.get("report_citation", ""),
        case.get("case_id", ""),
    ]

    aliases = case.get("aliases", [])
    if isinstance(aliases, list):
        values.extend([str(alias) for alias in aliases if alias])

    return [str(v) for v in values if v]


def find_cached_case(user_input: str) -> Optional[Dict[str, Any]]:
    if not user_input:
        return None

    query = _normalise(user_input)
    if not query:
        return None

    best_case = None
    best_score = 0.0

    for case in CASE_CACHE:
        for value in _searchable_values(case):
            normalised_value = _normalise(value)

            if not normalised_value:
                continue

            if query == normalised_value:
                return case

            # Allows "[1992] HCA 23" to match "1992 hca 23" and
            # "Mabo v Queensland" to match an alias within a longer input.
            if normalised_value in query or query in normalised_value:
                return case

            score = _score(query, normalised_value)
            if score > best_score:
                best_score = score
                best_case = case

    # Conservative fuzzy threshold. This catches minor typos but avoids
    # matching unrelated short inputs too aggressively.
    if best_score >= 0.86:
        return best_case

    return None
