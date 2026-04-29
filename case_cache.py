import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List

CACHE_PATH = os.path.join(os.path.dirname(__file__), "case_cache.json")


def _normalise(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[\[\]\(\),.;:]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def load_case_cache() -> List[Dict[str, Any]]:
    if not os.path.exists(CACHE_PATH):
        return []

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CASE_CACHE = load_case_cache()


def _score(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def find_cached_case(user_input: str) -> Optional[Dict[str, Any]]:
    if not user_input:
        return None

    query = _normalise(user_input)

    best_case = None
    best_score = 0.0

    for case in CASE_CACHE:
        searchable_values = [
            case.get("case_name", ""),
            case.get("neutral_citation", ""),
            case.get("report_citation", "")
        ] + case.get("aliases", [])

        for value in searchable_values:
            if not value:
                continue

            normalised_value = _normalise(value)

            if query == normalised_value:
                return case

            if normalised_value in query or query in normalised_value:
                return case

            score = _score(query, normalised_value)
            if score > best_score:
                best_score = score
                best_case = case

    if best_score >= 0.86:
        return best_case

    return None
