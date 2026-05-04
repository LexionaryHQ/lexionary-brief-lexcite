# main.py - Lexionary v3 Brief API + Lexcite AGLC Engine
# Version: 1.8.1
# Run: uvicorn main:app --host 0.0.0.0 --port 8000

import os
import re
import time
import logging
import urllib.parse
import random
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from aglc_engine import format_citation, format_freeform_line, SourceType

# ---------------- Verified case cache (CBG trust recovery) ----------------
try:
    from case_cache import find_cached_case, CASE_CACHE
except Exception as e:
    find_cached_case = None  # type: ignore
    CASE_CACHE = []  # type: ignore
    logging.warning("Case cache unavailable at startup: %s", e)


# ---------------------------------------------------------------------------
# Helper to pull out neutral citation from a longer string
# ---------------------------------------------------------------------------

def extract_neutral_citation(user_input: str) -> str | None:
    if not user_input:
        return None
    text = " ".join(user_input.split())
    pattern = r"\[\d{4}\]\s+\S+\s+\d+"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


# ---- Optional PDF extraction support
HAS_PDFMINER = False
try:
    from io import BytesIO
    from pdfminer_high_level import extract_text as pdf_extract_text  # type: ignore
    HAS_PDFMINER = True
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
        HAS_PDFMINER = True
    except Exception:
        pdf_extract_text = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("lexionary")


# ---------------- OpenAI client (brief only) ----------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class _OpenAIShim:
    def __init__(self):
        self.mode = None
        self.client = None
        try:
            from openai import OpenAI
            if OPENAI_API_KEY:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                self.mode = "new"
                log.info("OpenAI client initialised (new SDK).")
            else:
                self.mode = "none"
                log.warning("OPENAI_API_KEY not set. Summariser disabled.")
        except Exception:
            try:
                import openai  # type: ignore
                if OPENAI_API_KEY:
                    openai.api_key = OPENAI_API_KEY
                    self.client = openai
                    self.mode = "legacy"
                    log.info("OpenAI client initialised (legacy SDK).")
                else:
                    self.mode = "none"
                    log.warning("OPENAI_API_KEY not set. Summariser disabled.")
            except Exception:
                self.mode = "none"
                self.client = None
                log.error("OpenAI SDK not available.")

    def chat(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.15) -> str:
        if self.mode == "new" and self.client:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        elif self.mode == "legacy" and self.client:
            resp = self.client.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("OpenAI not configured. Set OPENAI_API_KEY or install SDK.")


_openai = _OpenAIShim()


# ---------------- FastAPI + CORS ----------------
app = FastAPI(title="Lexionary v3 - Brief API + Lexcite", version="1.8.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Models ----------------
class BriefRequest(BaseModel):
    query: Optional[str] = Field(None, description="Case name or neutral citation or direct text")
    url: Optional[str] = Field(None, description="AustLII judgment URL")
    text: Optional[str] = Field(None, description="Raw case text extract (optional)")
    pinpoints: Optional[List[str]] = Field(default_factory=list)
    depth: str = Field(default="standard")
    jurisdiction: str = Field(default="AU")
    tone: str = Field(default="neutral")


class BriefResponse(BaseModel):
    success: bool
    brief: str
    meta: Dict[str, Any] = Field(default_factory=dict)


CBG_RETRIEVAL_UNAVAILABLE_MESSAGE = (
    "We couldn’t find this case in Lexionary’s verified case cache.\n\n"
    "You can still generate a case brief by pasting the judgment text directly below.\n\n"
    "Lexionary only generates case briefs from real case text to protect accuracy."
)


# Lexcite models
class LexciteRequest(BaseModel):
    input_text: str = Field(..., description="One or more citations separated by newlines.")


class LexciteEntry(BaseModel):
    id: str
    raw: str
    source_type: str
    formatted: str
    formatted_html: str
    validated: bool
    validation_errors: List[str]
    meta: Dict[str, Any] = Field(default_factory=dict)


class LexciteResponse(BaseModel):
    api_version: str
    entries: List[LexciteEntry]
    errors: List[str] = Field(default_factory=list)


class CitationRequest(BaseModel):
    source_type: SourceType | str
    data: dict
    mode: str = "footnote"


# ---------------- AustLII constants ----------------
AUSTLII_BASE = "https://www.austlii.edu.au"
AUSTLII_SINO = f"{AUSTLII_BASE}/cgi-bin/sinosrch.cgi"
AUSTLII_MIRRORS = [
    "https://www.austlii.edu.au",
    "https://classic.austlii.edu.au",
    "https://www8.austlii.edu.au",
    "https://www7.austlii.edu.au",
]
AUSTLII_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Lexionary/1.8.1; +https://lexionary.com.au)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.austlii.edu.au/",
}


def looks_like_judgment_url(url: str) -> bool:
    return "/cgi-bin/viewdoc/au/cases/" in url and url.endswith(".html")


def is_austlii_url(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        return "austlii.edu.au" in host
    except Exception:
        return False


def rewrite_url_to_mirror(url: str, mirror: str) -> str:
    parsed = urllib.parse.urlparse(url)
    mpar = urllib.parse.urlparse(mirror)
    return urllib.parse.urlunparse(
        (mpar.scheme, mpar.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


class RateLimiter:
    def __init__(self, min_interval_sec: float = 1.2):
        self.min_interval = min_interval_sec
        self.last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.time()


limiter = RateLimiter(1.0)


def http_get(url: str, timeout: int = 22, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    limiter.wait()
    h = dict(AUSTLII_HEADERS)
    if headers:
        h.update(headers)
    return requests.get(url, headers=h, timeout=timeout)


def fetch_url_resilient(url: str, timeout: int = 20, max_total_attempts: int = 6) -> Tuple[str, str, int]:
    attempts = 0
    last_exc = None
    order = AUSTLII_MIRRORS[:]
    for attempt in range(1, max_total_attempts + 1):
        attempts = attempt
        mirror = order[(attempt - 1) % len(order)]
        try_url = rewrite_url_to_mirror(url, mirror)
        try:
            logging.info("Fetch attempt %d -> %s", attempt, try_url)
            r = http_get(try_url, timeout=timeout)
            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} server error for {try_url}")
            r.raise_for_status()
            return r.text, mirror, attempts
        except Exception as e:
            last_exc = e
            backoff = min(6.0, 0.6 * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
            logging.warning("Fetch failed (attempt %d): %s; backoff %.2fs", attempt, e, backoff)
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def clean_case_html_to_text(html: str) -> str:
    s = soup_from_html(html)
    for tag in s(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = s.find(id="content") or s.find("article") or s.find("body") or s
    for br in main.find_all("br"):
        br.replace_with("\n")
    txt = main.get_text("\n")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()


CITATION_ON_PAGE_RE = re.compile(r"\[\d{4}\]\s+[A-Z]{2,7}\s+\d{1,4}")
DATE_RE = re.compile(
    r"(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
    re.I,
)


def extract_title_citation_date(html: str) -> Tuple[str, str, Optional[str]]:
    s = soup_from_html(html)
    title = (
        s.title.string.strip()
        if s.title and s.title.string
        else (s.find("h1").get_text(" ", strip=True) if s.find("h1") else "")
    )
    body = s.get_text("\n")
    m_cit = CITATION_ON_PAGE_RE.search(body)
    citation = m_cit.group(0) if m_cit else ""
    m_date = DATE_RE.search(body)
    date_str = m_date.group(1) if m_date else None
    return title, citation, date_str


def parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%d %B %Y", "%-d %B %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None


COURT_PATHS: Dict[str, Tuple[str, str]] = {
    "HCA": ("cth", "HCA"),
    "FCA": ("cth", "FCA"),
    "FCAFC": ("cth", "FCAFC"),
    "NSWCA": ("nsw", "NSWCA"),
    "NSWSC": ("nsw", "NSWSC"),
    "VSCA": ("vic", "VSCA"),
    "VSC": ("vic", "VSC"),
    "QCA": ("qld", "QCA"),
    "QSC": ("qld", "QSC"),
    "SASCFC": ("sa", "SASCFC"),
    "SASC": ("sa", "SASC"),
    "WASCA": ("wa", "WASCA"),
    "WASC": ("wa", "WASC"),
    "TASFC": ("tas", "TASFC"),
    "TASSC": ("tas", "TASSC"),
    "ACTCA": ("act", "ACTCA"),
    "ACTSC": ("act", "ACTSC"),
    "NTCA": ("nt", "NTCA"),
    "NTSC": ("nt", "NTSC"),
}

NEUTRAL_CIT_RE = re.compile(
    r"^\s*\[?(\d{4})\]?\s+([A-Z]{2,7})\s+(\d{1,4})(?:\s*\(.*?\))?(?:\s*;.*)?\s*$",
    re.I,
)


def resolve_from_citation(q: str) -> Optional[str]:
    m = NEUTRAL_CIT_RE.match((q or "").strip())
    if not m:
        return None
    year, court_raw, num = m.group(1), m.group(2).upper(), m.group(3)
    if court_raw not in COURT_PATHS:
        return None
    jur, court = COURT_PATHS[court_raw]
    return f"{AUSTLII_BASE}/cgi-bin/viewdoc/au/cases/{jur}/{court}/{year}/{num}.html"


def austlii_name_search_first_result(query: str) -> Optional[str]:
    if not query:
        return None
    params = {"query": query, "method": "auto", "meta": "/au/cases"}
    url = f"{AUSTLII_SINO}?{urllib.parse.urlencode(params)}"
    try:
        html, _, _ = fetch_url_resilient(url, timeout=18, max_total_attempts=4)
    except Exception as e:
        log.warning("Search fetch failed: %s", e)
        return None
    s = soup_from_html(html)
    for a in s.find_all("a", href=True):
        href = a["href"]
        if "/au/cases/" in href and href.endswith(".html"):
            full = href if href.startswith("http") else urllib.parse.urljoin(AUSTLII_BASE, href)
            if looks_like_judgment_url(full):
                return full
    return None


def extract_judgment_link_from_page(html: str) -> Optional[str]:
    s = soup_from_html(html)
    for a in s.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else urllib.parse.urljoin(AUSTLII_BASE, href)
        if looks_like_judgment_url(full):
            return full
    return None


def resolve_or_search_case_url(query: Optional[str], url: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Conservative resolver:
    - direct AustLII judgment URL -> direct
    - non-direct AustLII URL -> non_direct_austlii
    - neutral citation -> citation
    - case name search -> search
    """
    if url:
        if looks_like_judgment_url(url):
            return url, "direct"
        if is_austlii_url(url):
            return url, "non_direct_austlii"
        raise HTTPException(
            status_code=400,
            detail="Only AustLII URLs are supported in 'url'. Otherwise paste case text or use a citation."
        )

    if query:
        neutral = extract_neutral_citation(query) or query
        c = resolve_from_citation(neutral)
        if c:
            return c, "citation"
        s = austlii_name_search_first_result(query)
        if s:
            return s, "search"

    return None, "none"


HCA_PDF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Lexionary/1.8.1; +https://lexionary.com.au)",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.hcourt.gov.au/",
}


def try_fetch_hca_pdf(year: str, number: str, query_hint: str = "") -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not HAS_PDFMINER:
        return None, None, "pdfminer.six not installed"

    search_q = urllib.parse.quote(f"[{year}] HCA {number}")
    search_url = f"https://www.hcourt.gov.au/search?search_api_fulltext={search_q}"
    try:
        r = requests.get(search_url, headers=HCA_PDF_HEADERS, timeout=20)
        r.raise_for_status()
        s = BeautifulSoup(r.text, "html.parser")
        pdf_links = []
        for a in s.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf") and "/eresources/" in href and "/HCA/" in href:
                pdf_links.append(urllib.parse.urljoin("https://www.hcourt.gov.au", href))
        if not pdf_links:
            for a in s.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and f"{year}" in href and "HCA" in href:
                    pdf_links.append(urllib.parse.urljoin("https://www.hcourt.gov.au", href))
        for link in pdf_links:
            try:
                rr = requests.get(link, headers=HCA_PDF_HEADERS, timeout=25)
                rr.raise_for_status()
                text = pdf_extract_text(BytesIO(rr.content))
                if re.search(rf"\[{year}\]\s+HCA\s+{number}\b", text):
                    return text, link, None
            except Exception as e:
                log.warning("HCA PDF candidate failed: %s", e)
        return None, None, "no suitable HCA PDF found"
    except Exception as e:
        log.warning("HCA search failed: %s", e)
        return None, None, "HCA search error"


def verify_case_page(html: str, resolved_url: Optional[str]) -> Dict[str, Any]:
    title, citation_on_page, date_str = extract_title_citation_date(html)
    txt = clean_case_html_to_text(html)
    ok = True
    reasons: List[str] = []

    if len(txt) < 1200:
        ok = False
        reasons.append(f"text too short ({len(txt)} chars)")
    if not citation_on_page:
        ok = False
        reasons.append("neutral citation not found on page")

    if resolved_url:
        m = re.search(r"/(\d{4})/(\d{1,4})\.html$", resolved_url)
        if m and citation_on_page:
            y_url = m.group(1)
            m2 = re.search(r"\[(\d{4})\]", citation_on_page)
            if m2 and y_url != m2.group(1):
                ok = False
                reasons.append("year in URL does not match citation on page")

    dt = parse_date_safe(date_str)
    if dt:
        now = datetime.utcnow()
        if dt > now.replace(hour=23, minute=59, second=59) and (dt - now).days > 1:
            ok = False
            reasons.append(f"decision date appears in future: {date_str}")

    return {
        "ok": ok,
        "reason": "; ".join(reasons) if reasons else "",
        "title": title,
        "citation_on_page": citation_on_page,
        "decision_date": date_str,
        "text_length": len(txt),
        "clean_text": txt,
    }


DEPTH_HINT = {
    "concise": "Output must be tight and exam ready. Use bullets. Target 120 to 180 words total.",
    "standard": "Balanced depth with short paragraphs. Target about 250 to 400 words.",
    "extended": "More depth in Rule and Application with pinpointed authorities. Target 500 to 700 words.",
}
TONE_HINT = {
    "neutral": "Neutral academic tone.",
    "exam": "Bullet first, quick recall, minimal prose.",
    "study": "Slightly explanatory with brief definitions.",
    "practical": "Practitioner tone. Ruthless relevance.",
}
JUR_HINT = {
    "AU": "Use Australian authorities and terminology. Prefer HCA and state appellate courts.",
    "AU-FED": "Bias to HCA and Federal Court authorities.",
    "AU-NSW": "Bias to NSWCA/NSWSC and HCA where relevant.",
    "AU-VIC": "Bias to VSCA/VSC and HCA.",
    "AU-QLD": "Bias to QCA/QSC and HCA.",
    "AU-WA": "Bias to WASCA/WASC and HCA.",
    "AU-SA": "Bias to SASCFC/SASC and HCA.",
    "AU-TAS": "Bias to TAS courts and HCA.",
    "AU-ACT": "Bias to ACTCA/ACTSC and HCA.",
    "AU-NT": "Bias to NTCA/NTSC and HCA.",
}

AUTHORITY_RULES = """
Authority selection rules:
• Prefer Australian primary authority. Order: HCA, relevant state or territory appellate/trial courts; foreign sources for context only.
• Do not treat the UK Bolam test as controlling for a doctor's duty to warn in Australia. If mentioned, state Rogers v Whitaker material risk standard and that professional opinion is evidentiary, not conclusive.
• Courts set standards for warnings; professional practice is evidence, not decisive.
"""


def build_irac_prompt(
    case_name_or_citation: str,
    case_text: str,
    pinpoints: List[str],
    depth: str,
    jurisdiction: str,
    tone: str,
) -> Dict[str, str]:
    depth_note = DEPTH_HINT.get(depth, "Balanced depth.")
    tone_note = TONE_HINT.get(tone, "Neutral academic tone.")
    jur_note = JUR_HINT.get(jurisdiction, "Use Australian authorities and terminology.")
    pins = f"Focus on paragraphs: {', '.join(pinpoints)}." if pinpoints else ""

    system_rules = (
        "You are Lexionary's Case Brief Generator for Australian law students. "
        "You produce accurate, structured IRAC case briefs from provided case text only. "
        "Never invent facts, holdings, quotations, paragraph numbers, legislation, judges, procedural history, or legal principles. "
        "If the supplied text is a verified cache extract rather than the full judgment, work only from that extract and say when the brief is based on the supplied extract. "
        "Your output must help a student understand the legal significance of the case, not merely repeat facts. "
        "The Application section is the most important section and must explain the court's reasoning using because-style logic. "
        "Do not draft assignment answers or provide advice to submit. This is study guidance only."
    )

    source_excerpt = case_text[:12000]

    user_task = f"""
CASE: {case_name_or_citation}

TASK:
Create a high-quality IRAC case brief strictly from the SOURCE TEXT below.
The brief must be useful for Australian law study, tutorial preparation, revision, and building case notes.

NON-NEGOTIABLE ACCURACY RULES:
• Use only the supplied SOURCE TEXT.
• Do not add facts, holdings, quotes, paragraph references, judges, statutes, procedural history, or authorities unless they appear in the text.
• If a point is not clear from the text, say "The supplied text does not state this clearly" rather than guessing.
• If the source appears to be a short extract or cache summary, briefly state that the brief is based on the supplied extract.
• Do not hallucinate pinpoint references.

{AUTHORITY_RULES}

OUTPUT FORMAT:

IRAC Summary

1. Snapshot
• Case: identify the case from the supplied information.
• Area: identify the subject area if apparent.
• Why it matters: one sentence explaining why students should know this case.

2. Issue
• State the real legal question the court had to resolve.
• Do not merely name the topic.
• If multiple issues appear, identify the main issue first and secondary issues briefly.

3. Rule
• State the governing rule, test, principle, or standard drawn from the text.
• Explain the rule in plain legal language.
• If competing rules or approaches appear, explain which approach controlled.
• Mention authorities only if they are clearly provided in the text.

4. Application
• This is the most important section.
• Explain how the court applied the rule to the facts.
• Use because-style reasoning: what mattered, why it mattered, and how it affected the outcome.
• Identify the decisive facts, policy considerations, statutory features, or reasoning steps if they appear in the text.
• If the court accepted or rejected an argument, explain the argument and why it succeeded or failed.
• Avoid generic phrases such as "the court balanced the interests" unless you explain exactly what was balanced and why.

5. Conclusion
• State the result clearly.
• State what the case stands for in one exam-ready sentence.

6. How to Use This Case in Study
• Give 2 to 3 short bullet points explaining when a student would cite or rely on this case.
• Keep this section study-focused, not assignment-writing.

STYLE REQUIREMENTS:
• Write like a strong law student creating case notes.
• Be clear, specific and legally disciplined.
• Prefer short paragraphs and bullet points.
• Do not use filler.
• Do not overstate certainty if the supplied text is limited.

CONSTRAINTS:
• {jur_note}
• {depth_note}
• {tone_note}
• {pins}

SOURCE TEXT:
<<<
{source_excerpt}
>>>
"""
    return {"system": system_rules, "user": user_task}

def call_openai(system_msg: str, user_msg: str) -> str:
    return _openai.chat(system=system_msg, user=user_msg, max_tokens=1500, temperature=0.12)


def generate_irac_from_case_text(
    *,
    case_label: str,
    case_text: str,
    pinpoints: List[str],
    depth: str,
    jurisdiction: str,
    tone: str,
) -> str:
    """Shared generation path for verified cache, retrieved sources and pasted text."""
    payload = build_irac_prompt(
        case_name_or_citation=case_label,
        case_text=case_text,
        pinpoints=pinpoints or [],
        depth=(depth or "standard").lower(),
        jurisdiction=(jurisdiction or "AU").upper(),
        tone=(tone or "neutral").lower(),
    )
    return call_openai(payload["system"], payload["user"])


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Lexionary v3 - Brief API + Lexcite",
        "endpoints": ["/health", "/brief", "/cite", "/lexcite/format"],
        "version": "1.8.1",
        "has_pdfminer": HAS_PDFMINER,
    }


@app.get("/health")
def health():
    try:
        mode = _openai.mode
    except Exception:
        mode = "unknown"
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "openai_mode": mode,
        "env_key_present": bool(OPENAI_API_KEY),
        "has_pdfminer": HAS_PDFMINER,
        "case_cache_count": len(CASE_CACHE),
    }

@app.post("/brief", response_model=BriefResponse)
def brief(req: BriefRequest, request: Request):
    t0 = time.time()

    if not (req.query or req.url or req.text):
        raise HTTPException(status_code=400, detail="Provide 'query', 'url', or 'text'.")

    # -----------------------------------------------------------------------
    # 1) Verified case cache first
    # -----------------------------------------------------------------------
    # Cache lookup intentionally runs before any live retrieval. This restores
    # trust by making known high-frequency cases reliable and fast.
    #
    # Cache lookup supports both query and text because the WordPress frontend
    # may send case names/citations as either field. Long pasted judgment text
    # is not treated as a cache lookup and is handled later as direct text.
    cache_lookup_input = ""
    if req.query and req.query.strip():
        cache_lookup_input = req.query.strip()
    elif req.text and req.text.strip() and len(req.text.strip()) <= 500:
        cache_lookup_input = req.text.strip()
    elif req.url and req.url.strip():
        cache_lookup_input = req.url.strip()

    if cache_lookup_input and find_cached_case:
        try:
            cached_case = find_cached_case(cache_lookup_input)
        except Exception as e:
            cached_case = None
            log.warning("Case cache lookup failed: %s", e)

        if cached_case:
            cached_text = (cached_case.get("text") or "").strip()

            if len(cached_text) < 120:
                log.warning(
                    "Cached case text too short for %s (%d chars).",
                    cached_case.get("case_id") or cached_case.get("case_name"),
                    len(cached_text),
                )
                return BriefResponse(
                    success=False,
                    brief="This case is in Lexionary’s verified cache, but the cached extract is too short to generate a reliable brief. Please paste judgment text directly below.",
                    meta={
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        "strategy": "verified_cache_text_too_short",
                        "verified": False,
                        "source_label": "Lexionary verified case cache",
                        "source_title": cached_case.get("case_name", ""),
                        "source_citation": cached_case.get("neutral_citation", ""),
                        "text_length": len(cached_text),
                    },
                )

            case_label = " ".join(
                part for part in [
                    cached_case.get("case_name"),
                    cached_case.get("neutral_citation"),
                    cached_case.get("report_citation"),
                ]
                if part
            )

            try:
                brief_text = generate_irac_from_case_text(
                    case_label=case_label or cache_lookup_input,
                    case_text=cached_text,
                    pinpoints=req.pinpoints or [],
                    depth=req.depth,
                    jurisdiction=req.jurisdiction,
                    tone=req.tone,
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Summariser failed: {e}")

            meta = {
                "elapsed_ms": int((time.time() - t0) * 1000),
                "resolved_url": cached_case.get("source_url", ""),
                "strategy": "verified_cache",
                "verified": True,
                "source_label": cached_case.get("source_label") or "Lexionary verified case cache",
                "source_title": cached_case.get("case_name", ""),
                "source_citation": cached_case.get("neutral_citation", ""),
                "report_citation": cached_case.get("report_citation", ""),
                "court": cached_case.get("court", ""),
                "subjects": cached_case.get("subjects", []),
                "topics": cached_case.get("topics", []),
                "source_url": cached_case.get("source_url", ""),
                "last_verified": cached_case.get("last_verified", ""),
                "depth": req.depth,
                "jurisdiction": req.jurisdiction,
                "tone": req.tone,
                "pinpoints": req.pinpoints or [],
                "text_length": len(cached_text),
                "length_chars": len(brief_text),
                "mirror_used": "",
                "attempts": 0,
                "fallback": None,
                "has_pdfminer": HAS_PDFMINER,
            }
            log.info("CBG cache hit: %s", cached_case.get("case_id") or case_label)
            return BriefResponse(success=True, brief=brief_text, meta=meta)

    resolved_url, strategy = resolve_or_search_case_url(req.query, req.url)

    html: Optional[str] = None
    mirror_used = ""
    attempts = 0
    verify_info: Dict[str, Any] = {}
    source_url_used = resolved_url

    # Conservative handling of non-direct AustLII pages
    if resolved_url and strategy == "non_direct_austlii":
        try:
            page_html, mirror_used, attempts = fetch_url_resilient(resolved_url, timeout=20, max_total_attempts=4)
            extracted_url = extract_judgment_link_from_page(page_html)
            if extracted_url:
                resolved_url = extracted_url
                source_url_used = resolved_url
                strategy = "resolved_from_non_direct_austlii"
            else:
                meta = {
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "resolved_url": resolved_url,
                    "strategy": strategy,
                    "verified": False,
                    "verify_reason": "Could not find a direct AustLII judgment page from that URL.",
                    "source_title": "",
                    "source_citation": "",
                    "decision_date": "",
                    "text_length": 0,
                    "mirror_used": mirror_used,
                    "attempts": attempts,
                    "fallback": None,
                    "has_pdfminer": HAS_PDFMINER,
                }
                return BriefResponse(
                    success=False,
                    brief=CBG_RETRIEVAL_UNAVAILABLE_MESSAGE,
                    meta=meta,
                )
        except Exception as e:
            log.warning("Non-direct AustLII fetch failed: %s", e)
            meta = {
                "elapsed_ms": int((time.time() - t0) * 1000),
                "resolved_url": resolved_url,
                "strategy": strategy,
                "verified": False,
                "verify_reason": f"Unable to fetch the AustLII page: {e}",
                "source_title": "",
                "source_citation": "",
                "decision_date": "",
                "text_length": 0,
                "mirror_used": mirror_used,
                "attempts": attempts,
                "fallback": None,
                "has_pdfminer": HAS_PDFMINER,
            }
            return BriefResponse(
                success=False,
                brief=CBG_RETRIEVAL_UNAVAILABLE_MESSAGE,
                meta=meta,
            )

    if resolved_url:
        try:
            html, mirror_used, attempts = fetch_url_resilient(resolved_url, timeout=20, max_total_attempts=6)
            verify_info = verify_case_page(html, resolved_url)
        except Exception as e_first:
            log.warning("AustLII fetch failed: %s", e_first)

    hca_fallback_used = False
    hca_fallback_reason = None
    m = NEUTRAL_CIT_RE.match((req.query or "").strip()) if req.query else None
    if ((html is None) or (verify_info and not verify_info.get("ok"))) and m and m.group(2).upper() == "HCA":
        year, number = m.group(1), m.group(3)
        extracted_text, pdf_url, reason = try_fetch_hca_pdf(year, number, query_hint=req.query or "")
        hca_fallback_reason = reason
        if extracted_text and len(extracted_text) > 1000:
            hca_fallback_used = True
            verify_info = {
                "ok": True,
                "reason": "",
                "title": f"[{year}] HCA {number} (PDF)",
                "citation_on_page": f"[{year}] HCA {number}",
                "decision_date": None,
                "text_length": len(extracted_text),
                "clean_text": extracted_text,
            }
            source_url_used = pdf_url
            strategy = "hca_pdf"

    direct_text_candidate = (req.text or "").strip()
    if not direct_text_candidate and req.query:
        q_stripped = req.query.strip()
        if len(q_stripped) > 400:
            direct_text_candidate = q_stripped

    if (not verify_info or not verify_info.get("ok")) and direct_text_candidate:
        log.info("Using direct text fallback (len=%d).", len(direct_text_candidate))
        verify_info = {
            "ok": True,
            "reason": "unverified_direct_text",
            "title": (direct_text_candidate[:80] + "...") if len(direct_text_candidate) > 80 else direct_text_candidate,
            "citation_on_page": extract_neutral_citation(direct_text_candidate) or "",
            "decision_date": None,
            "text_length": len(direct_text_candidate),
            "clean_text": direct_text_candidate,
        }
        strategy = "direct_text"
        source_url_used = None

    if not verify_info or not verify_info.get("ok"):
        meta = {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "resolved_url": resolved_url,
            "strategy": strategy,
            "verified": False,
            "verify_reason": (verify_info.get("reason") if verify_info else "Unable to fetch or verify source"),
            "source_title": (verify_info.get("title") if verify_info else ""),
            "source_citation": (verify_info.get("citation_on_page") if verify_info else ""),
            "decision_date": (verify_info.get("decision_date") if verify_info else ""),
            "text_length": (verify_info.get("text_length") if verify_info else 0),
            "mirror_used": mirror_used,
            "attempts": attempts,
            "fallback": "HCA_PDF" if hca_fallback_used else None,
            "fallback_reason": hca_fallback_reason,
            "has_pdfminer": HAS_PDFMINER,
        }
        return BriefResponse(
            success=False,
            brief=CBG_RETRIEVAL_UNAVAILABLE_MESSAGE,
            meta=meta,
        )

    case_label = req.query or source_url_used or "Unknown case"
    payload = build_irac_prompt(
        case_name_or_citation=case_label,
        case_text=verify_info["clean_text"],
        pinpoints=req.pinpoints or [],
        depth=(req.depth or "standard").lower(),
        jurisdiction=(req.jurisdiction or "AU").upper(),
        tone=(req.tone or "neutral").lower(),
    )

    try:
        brief_text = call_openai(payload["system"], payload["user"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summariser failed: {e}")

    meta = {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "resolved_url": resolved_url,
        "strategy": strategy,
        "verified": strategy != "direct_text",
        "source_title": verify_info.get("title", ""),
        "source_citation": verify_info.get("citation_on_page", ""),
        "decision_date": verify_info.get("decision_date", ""),
        "depth": req.depth,
        "jurisdiction": req.jurisdiction,
        "tone": req.tone,
        "pinpoints": req.pinpoints or [],
        "length_chars": len(brief_text),
        "mirror_used": mirror_used,
        "attempts": attempts,
        "fallback": "HCA_PDF" if hca_fallback_used else None,
        "source_url": source_url_used,
        "has_pdfminer": HAS_PDFMINER,
    }
    return BriefResponse(success=True, brief=brief_text, meta=meta)


@app.post("/cite")
async def cite(req: CitationRequest):
    try:
        result = format_citation(
            source_type=req.source_type,
            data=req.data,
            mode=req.mode,
        )
        return {
            "success": True,
            "source_type": result.source_type.value,
            "mode": result.mode,
            "text": result.text,
            "html": result.html,
        }
    except ValidationError as ve:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "validation_error",
                "messages": ve.errors(),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "server_error",
                "message": str(e),
            },
        )


# ---------------- Lexcite paste list endpoint ----------------

LEXCITE_MAX_CHARS = 8000
LEXCITE_MAX_LINES = 50
LEXCITE_MIN_LINE_LEN = 4
LEXCITE_MAX_LINE_LEN = 500


@app.post("/lexcite/format", response_model=LexciteResponse)
def lexcite_format(req: LexciteRequest, request: Request):
    api_version = datetime.utcnow().strftime("%Y-%m-%d")
    raw_text = (req.input_text or "").strip()

    if not raw_text:
        return LexciteResponse(api_version=api_version, entries=[], errors=["No input provided. Paste at least one citation."])

    total_chars = len(raw_text)
    if total_chars > LEXCITE_MAX_CHARS:
        return LexciteResponse(
            api_version=api_version,
            entries=[],
            errors=[f"Input too long. Max {LEXCITE_MAX_CHARS} characters. You submitted {total_chars} characters."],
        )

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return LexciteResponse(api_version=api_version, entries=[], errors=["No usable lines detected. Put one citation per line."])

    if len(lines) > LEXCITE_MAX_LINES:
        return LexciteResponse(
            api_version=api_version,
            entries=[],
            errors=[f"Too many lines. Max {LEXCITE_MAX_LINES} citations per run. You submitted {len(lines)} lines."],
        )

    entries: List[LexciteEntry] = []
    errors: List[str] = []

    for idx, line in enumerate(lines, start=1):
        line_len = len(line)

        if line_len < LEXCITE_MIN_LINE_LEN:
            entries.append(
                LexciteEntry(
                    id=str(idx),
                    raw=line,
                    source_type="OTHER",
                    formatted=line,
                    formatted_html=line,
                    validated=False,
                    validation_errors=[f"Line {idx} is too short to be a citation."],
                    meta={"length_violation": True},
                )
            )
            continue

        if line_len > LEXCITE_MAX_LINE_LEN:
            entries.append(
                LexciteEntry(
                    id=str(idx),
                    raw=line,
                    source_type="OTHER",
                    formatted=line,
                    formatted_html=line,
                    validated=False,
                    validation_errors=[f"Line {idx} is too long to be a single citation. Split it."],
                    meta={"length_violation": True},
                )
            )
            continue

        try:
            pe = format_freeform_line(line)
            entries.append(
                LexciteEntry(
                    id=str(idx),
                    raw=line,
                    source_type=pe.source_type,
                    formatted=pe.text,
                    formatted_html=pe.html,
                    validated=pe.validated,
                    validation_errors=pe.validation_errors,
                    meta=pe.meta,
                )
            )
        except Exception as e:
            log.exception("Lexcite processing failed for line %d: %s", idx, line)
            errors.append(f"Error processing line {idx}: {e}")

    return LexciteResponse(api_version=api_version, entries=entries, errors=errors)
