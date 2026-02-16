# main.py - Lexionary v3 Brief API + Lexcite AGLC Engine
# Version: 1.7.0
# - Keeps existing /brief IRAC endpoint.
# - Lexcite: list mode now returns BOTH text and html with italics.
# - Lexcite: neutral citations like [1992] HCA 23 are recognised and validated.

import os, re, time, logging, urllib.parse, random, json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from aglc_engine import format_citation, SourceType

# ---------------------------------------------------------------------------
# Helper to pull out neutral citation from a longer string
# ---------------------------------------------------------------------------

def extract_neutral_citation(user_input: str) -> str | None:
    if not user_input:
        return None
    text = " ".join(user_input.split())
    pattern = r"\[\d{4}\]\s+\S+\s+\d+"
    match = re.search(pattern, text)
    if match:
        return match.group(0).strip()
    return None

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

# ---------------- OpenAI client ----------------
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

    def chat(self, system: str, user: str, max_tokens: int = 900, temperature: float = 0.2) -> str:
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
app = FastAPI(title="Lexionary v3 - Brief API + Lexcite", version="1.7.0")
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


# Lexcite models
class LexciteRequest(BaseModel):
    input_text: str = Field(..., description="One or more citations separated by newlines.")


class LexciteEntry(BaseModel):
    id: str
    raw: str
    source_type: str
    text: str = ""
    html: str = ""
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
    "User-Agent": "Mozilla/5.0 (compatible; Lexionary/1.4.2; +https://lexionary.com.au)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.austlii.edu.au/",
}


def looks_like_judgment_url(url: str) -> bool:
    return "/cgi-bin/viewdoc/au/cases/" in url and url.endswith(".html")


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


def resolve_or_search_case_url(query: Optional[str], url: Optional[str]) -> Tuple[Optional[str], str]:
    if url:
        if looks_like_judgment_url(url):
            return url, "direct"
        if "austlii.edu.au" in (url or ""):
            return None, "invalid-direct"
        raise HTTPException(status_code=400, detail="Only direct AustLII judgment URLs supported in 'url'.")

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
    "User-Agent": "Mozilla/5.0 (compatible; Lexionary/1.4.2; +https://lexionary.com.au)",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.hcourt.gov.au/",
}


def try_fetch_hca_pdf(
    year: str, number: str, query_hint: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
• Prefer Australian primary authority. Order: HCA, relevant state or territory appellate or trial courts; foreign sources for context only.
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
        "You produce accurate IRAC case briefs for Australian law students. "
        "Rely on the provided case text; do not fabricate facts or holdings. "
        "If text is insufficient, identify what is missing and proceed conservatively."
    )
    user_task = f"""
CASE: {case_name_or_citation}

GOAL: Produce an IRAC summary strictly from the provided case text.

{AUTHORITY_RULES}

OUTPUT FORMAT:
IRAC Summary

Issue
• One to three lines stating the central issues.

Rule
• Governing rules and tests with brief authority references if clear.

Application
• Apply the rules to the facts as stated. Avoid speculation.

Conclusion
• Short outcome and disposition.

CONSTRAINTS:
• {jur_note}
• {depth_note}
• {tone_note}
• {pins}

SOURCE TEXT (verbatim, truncated):
\"\"\"{case_text[:12000]}\"\"\""""
    return {"system": system_rules, "user": user_task}


def call_openai(system_msg: str, user_msg: str) -> str:
    return _openai.chat(system=system_msg, user=user_msg, max_tokens=900, temperature=0.2)


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Lexionary v3 - Brief API + Lexcite",
        "endpoints": ["/health", "/brief", "/cite", "/lexcite/format"],
        "version": "1.7.0",
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
    }


@app.post("/brief", response_model=BriefResponse)
def brief(req: BriefRequest, request: Request):
    t0 = time.time()

    if not (req.query or req.url or req.text):
        raise HTTPException(status_code=400, detail="Provide 'query', 'url', or 'text'.")

    resolved_url, strategy = resolve_or_search_case_url(req.query, req.url)

    html: Optional[str] = None
    mirror_used = ""
    attempts = 0
    verify_info: Dict[str, Any] = {}
    source_url_used = resolved_url

    if resolved_url:
        try:
            html, mirror_used, attempts = fetch_url_resilient(resolved_url, timeout=20, max_total_attempts=6)
            verify_info = verify_case_page(html, resolved_url)
        except Exception as e_first:
            log.warning("AustLII fetch failed: %s", e_first)

    hca_fallback_used = False
    hca_pdf_url = None
    hca_fallback_reason = None
    m = NEUTRAL_CIT_RE.match((req.query or "").strip()) if req.query else None
    if ((html is None) or (verify_info and not verify_info.get("ok"))) and m and m.group(2).upper() == "HCA":
        year, number = m.group(1), m.group(3)
        extracted_text, pdf_url, reason = try_fetch_hca_pdf(year, number, query_hint=req.query or "")
        hca_fallback_reason = reason
        if extracted_text and len(extracted_text) > 1000:
            hca_fallback_used = True
            hca_pdf_url = pdf_url
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
            "title": (direct_text_candidate[:80] + "…") if len(direct_text_candidate) > 80 else direct_text_candidate,
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
            brief=f"Verification failed. No IRAC generated.\nReason: {meta['verify_reason']}\nChecked URL: {resolved_url or 'n/a'}",
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


# -------------------------------------------------------------------------
# Lexcite list mode parser
# -------------------------------------------------------------------------

# Neutral case:
# Mabo v Queensland (No 2) [1992] HCA 23 [150]
CASE_NEUTRAL_RE = re.compile(
    r"^(?P<name>.+?)\s+\[(?P<year>\d{4})\]\s+(?P<court>[A-Z]{2,8})\s+(?P<num>\d{1,4})"
    r"(?:\s*(?:,)?\s*\[(?P<pin>\d+)\])?\s*$"
)

# Reported case:
# Waltons Stores (Interstate) Ltd v Maher (1988) 164 CLR 387, 392
CASE_REPORTED_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<year>\d{4})\)\s+(?P<vol>\d+)\s+(?P<rep>[A-Z]{2,10})\s+(?P<page>\d+)"
    r"(?:\s*,\s*(?P<pin>\d+))?\s*$"
)

# Legislation:
# Civil Liability Act 2002 (NSW) s 5B
LEG_RE = re.compile(
    r"^(?P<title>.+?)\s+(?P<year>\d{4})\s*\(\s*(?P<jur>[A-Za-z]{2,6})\s*\)\s*(?P<prov>.+)?$"
)

# Website:
# Anything with URL
URL_RE = re.compile(r"(https?://\S+)")


def detect_source_type(raw: str) -> SourceType | None:
    t = (raw or "").strip()
    if not t:
        return None

    if URL_RE.search(t):
        return SourceType.website

    if LEG_RE.match(t) and ("Act" in t or "Regulation" in t or "Regulations" in t):
        return SourceType.legislation

    if CASE_NEUTRAL_RE.match(t) or CASE_REPORTED_RE.match(t) or (" v " in t or " v. " in t):
        return SourceType.case

    # Heuristic fallbacks:
    if "'" in t and re.search(r"\(\d{4}\)", t) and re.search(r"\b\d+\b", t):
        return SourceType.journal_article

    return None


def parse_line_to_cite_payload(raw: str) -> Tuple[SourceType | None, Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Returns (source_type, data, validation_errors, meta)
    """
    txt = (raw or "").strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    st = detect_source_type(txt)
    if not st:
        errs.append("Could not detect source type. Use Build mode for guaranteed accuracy.")
        return None, {}, errs, meta

    if st == SourceType.case:
        m = CASE_NEUTRAL_RE.match(txt)
        if m:
            name = m.group("name").strip()
            year = m.group("year")
            court = m.group("court")
            num = m.group("num")
            pin = m.group("pin")

            data = {
                "case_name": name,
                "year": year,
                "court": court,
                "decision_number": num,
                "neutral_citation_first": True,
                "unreported": True,
                "reporter_series_by_year": False,
                "volume": None,
                "reporter": None,
                "first_page": None,
                "pinpoint_type": "paragraph" if pin else None,
                "pinpoint": pin if pin else None,
            }
            return st, data, [], {"parsed_as": "case_neutral"}

        m2 = CASE_REPORTED_RE.match(txt)
        if m2:
            data = {
                "case_name": m2.group("name").strip(),
                "year": m2.group("year"),
                "volume": m2.group("vol"),
                "reporter": m2.group("rep"),
                "first_page": m2.group("page"),
                "reporter_series_by_year": False,
                "court": None,
                "decision_number": None,
                "neutral_citation_first": True,
                "unreported": False,
                "pinpoint_type": "page" if m2.group("pin") else None,
                "pinpoint": m2.group("pin") if m2.group("pin") else None,
            }
            return st, data, [], {"parsed_as": "case_reported"}

        errs.append("Case detected, but could not parse the citation. Use Build mode for cases if it is not neutral or standard reported form.")
        return st, {}, errs, {"parsed_as": "case_unparsed"}

    if st == SourceType.legislation:
        m = LEG_RE.match(txt)
        if not m:
            errs.append("Legislation detected but could not parse it. Use Build mode.")
            return st, {}, errs, meta

        title = m.group("title").strip()
        year = m.group("year")
        jur = (m.group("jur") or "").strip()
        prov = (m.group("prov") or "").strip()

        unit = None
        num = None
        if prov:
            # crude: "s 5B" / "ss 12(1), 18A"
            mprov = re.match(r"^(s|ss)\s+(.+)$", prov)
            if mprov:
                unit = mprov.group(1)
                num = mprov.group(2).strip()

        data = {
            "title": title,
            "year": year,
            "jurisdiction": jur,
            "is_bill": False,
            "pinpoint_unit": unit,
            "pinpoint_number": num,
        }
        return st, data, [], {"parsed_as": "legislation"}

    if st == SourceType.website:
        urlm = URL_RE.search(txt)
        url = urlm.group(1) if urlm else ""
        # In paste mode we cannot reliably parse title/publisher without user giving it.
        # We treat it as needs-review unless it already looks like a structured citation.
        # Minimal: use the raw line as "title", and site_name = author if we can guess nothing.
        data = {
            "author_or_org": None,
            "page_title": "Untitled page",
            "site_name": "Website",
            "date": None,
            "url": url,
            "access_date": None,
        }
        errs.append("Website citations in Paste mode need review. Use Build mode to enter title, site and access date for AGLC accuracy.")
        return st, data, errs, {"parsed_as": "website_minimal"}

    errs.append("Detected source type is not supported in Paste mode. Use Build mode.")
    return st, {}, errs, meta


# ---------------- Lexcite guardrail constants ----------------
LEXCITE_MAX_CHARS = 8000
LEXCITE_MAX_LINES = 50
LEXCITE_MIN_LINE_LEN = 4
LEXCITE_MAX_LINE_LEN = 400
LEXCITE_ESSAY_LINE_LEN = 500


def looks_like_essay_single_line(text: str) -> bool:
    if len(text) < LEXCITE_ESSAY_LINE_LEN:
        return False

    lower = text.lower()
    citation_signals = [" v ", " v. ", " act ", " regulation", "<http", "<https"]
    year_pattern = re.search(r"\(\d{4}\)", text) or re.search(r"\[\d{4}\]", text)

    if any(sig in lower for sig in citation_signals):
        return False
    if year_pattern:
        return False
    return True


def make_length_error_entry(idx: int, raw: str, reason: str) -> LexciteEntry:
    return LexciteEntry(
        id=str(idx),
        raw=raw,
        source_type="OTHER",
        text=raw,
        html=_escape_html_for_list(raw),
        validated=False,
        validation_errors=[reason],
        meta={"length_violation": True},
    )


def _escape_html_for_list(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@app.post("/lexcite/format", response_model=LexciteResponse)
def lexcite_format(req: LexciteRequest, request: Request):
    api_version = datetime.utcnow().strftime("%Y-%m-%d")
    raw_text = (req.input_text or "").strip()

    if not raw_text:
        return LexciteResponse(
            api_version=api_version,
            entries=[],
            errors=["No input provided. Paste at least one citation."],
        )

    total_chars = len(raw_text)
    if total_chars > LEXCITE_MAX_CHARS:
        msg = (
            f"Input too long. Lexcite currently supports up to {LEXCITE_MAX_CHARS} "
            f"characters across all citations. You submitted {total_chars} characters."
        )
        return LexciteResponse(api_version=api_version, entries=[], errors=[msg])

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    total_lines = len(lines)

    if total_lines == 0:
        return LexciteResponse(api_version=api_version, entries=[], errors=["No usable lines detected. Put one citation per line."])

    if total_lines > LEXCITE_MAX_LINES:
        msg = (
            f"Too many lines. Lexcite currently supports up to {LEXCITE_MAX_LINES} "
            f"citations per run. You submitted {total_lines} lines."
        )
        return LexciteResponse(api_version=api_version, entries=[], errors=[msg])

    if total_lines == 1 and looks_like_essay_single_line(lines[0]):
        msg = (
            "This looks like paragraph or assignment text, not citations. "
            "Lexcite expects one citation per line. Paste your reference list or individual citations instead."
        )
        return LexciteResponse(api_version=api_version, entries=[], errors=[msg])

    entries: List[LexciteEntry] = []
    errors: List[str] = []

    for idx, line in enumerate(lines, start=1):
        try:
            line_len = len(line)
            if line_len < LEXCITE_MIN_LINE_LEN:
                reason = f"Line {idx} is too short to be a citation (length {line_len}). Provide a complete citation."
                entries.append(make_length_error_entry(idx, line, reason))
                continue

            if line_len > LEXCITE_MAX_LINE_LEN:
                reason = (
                    f"Line {idx} is too long to be a single citation (length {line_len}). "
                    "Split this into separate citations."
                )
                entries.append(make_length_error_entry(idx, line, reason))
                continue

            st, data, parse_errors, meta = parse_line_to_cite_payload(line)

            if not st:
                entries.append(
                    LexciteEntry(
                        id=str(idx),
                        raw=line,
                        source_type="OTHER",
                        text=line,
                        html=_escape_html_for_list(line),
                        validated=False,
                        validation_errors=parse_errors,
                        meta=meta,
                    )
                )
                continue

            # Try engine formatting
            try:
                result = format_citation(source_type=st, data=data, mode="footnote")
                validated = True if not parse_errors else False
                entries.append(
                    LexciteEntry(
                        id=str(idx),
                        raw=line,
                        source_type=st.value.upper(),
                        text=result.text,
                        html=result.html,
                        validated=validated,
                        validation_errors=parse_errors,
                        meta=meta,
                    )
                )
            except Exception as e:
                entries.append(
                    LexciteEntry(
                        id=str(idx),
                        raw=line,
                        source_type=st.value.upper(),
                        text=line,
                        html=_escape_html_for_list(line),
                        validated=False,
                        validation_errors=parse_errors + [str(e)],
                        meta=meta,
                    )
                )

        except Exception as e:
            log.exception("Lexcite processing failed for line %d: %s", idx, line)
            errors.append(f"Error processing line {idx}: {e}")

    return LexciteResponse(api_version=api_version, entries=entries, errors=errors)
