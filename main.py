# main.py — Lexionary v3 Brief API + Lexcite AGLC Engine
# Version: 1.6.0 (AGLC rules upgrade)
# - Keeps existing /brief IRAC endpoint.
# - Upgrades /lexcite/format with stricter AGLC-style validation.
# Run: uvicorn main:app --host 0.0.0.0 --port 8000

import os, re, time, logging, urllib.parse, random, json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # CORSMiddleware for CORS

# ---------------------------------------------------------------------------
# Neutral citation extraction helper
# ---------------------------------------------------------------------------

def extract_neutral_citation(user_input: str) -> str | None:
    """
    Extracts a neutral citation like:
      [2025] NSWCA 243
      [1992] HCA 23
      [2010] FCAFC 75

    from any longer string that may also contain party names or extra text.
    """
    if not user_input:
        return None

    text = " ".join(user_input.split())  # normalise whitespace

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
                temperature=temperature, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        elif self.mode == "legacy" and self.client:
            resp = self.client.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens
            )
            return resp["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("OpenAI not configured. Set OPENAI_API_KEY or install SDK.")

_openai = _OpenAIShim()

# ---------------- FastAPI + CORS ----------------
app = FastAPI(title="Lexionary v3 - Brief API + Lexcite", version="1.6.0")
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
    formatted: str
    validated: bool
    validation_errors: List[str]
    meta: Dict[str, Any] = Field(default_factory=dict)

class LexciteResponse(BaseModel):
    api_version: str
    entries: List[LexciteEntry]
    errors: List[str] = Field(default_factory=list)

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
    return urllib.parse.urlunparse((mpar.scheme, mpar.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))

# ---------------- Basic rate limit ----------------
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

def http_get(url: str, timeout: int = 22, headers: Optional[Dict[str,str]] = None) -> requests.Response:
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

# ---------------- Scrape helpers ----------------
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
DATE_RE = re.compile(r"(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})", re.I)

def extract_title_citation_date(html: str) -> Tuple[str, str, Optional[str]]:
    s = soup_from_html(html)
    title = s.title.string.strip() if s.title and s.title.string else (s.find("h1").get_text(" ", strip=True) if s.find("h1") else "")
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

# ---------------- Resolve by citation or search ----------------
COURT_PATHS: Dict[str, Tuple[str, str]] = {
    # Cth
    "HCA": ("cth", "HCA"),
    "FCA": ("cth", "FCA"),
    "FCAFC": ("cth", "FCAFC"),
    # NSW
    "NSWCA": ("nsw", "NSWCA"),
    "NSWSC": ("nsw", "NSWSC"),
    # VIC
    "VSCA": ("vic", "VSCA"),
    "VSC":  ("vic", "VSC"),
    # QLD
    "QCA": ("qld", "QCA"),
    "QSC": ("qld", "QSC"),
    # SA
    "SASCFC": ("sa", "SASCFC"),
    "SASC":   ("sa", "SASC"),
    # WA
    "WASCA": ("wa", "WASCA"),
    "WASC":  ("wa", "WASC"),
    # TAS
    "TASFC": ("tas", "TASFC"),
    "TASSC": ("tas", "TASSC"),
    # ACT
    "ACTCA": ("act", "ACTCA"),
    "ACTSC": ("act", "ACTSC"),
    # NT
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

# ---------------- High Court fallback (optional) ----------------
HCA_PDF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Lexionary/1.4.2; +https://lexionary.com.au)",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.hcourt.gov.au/",
}

def try_fetch_hca_pdf(year: str, number: str, query_hint: str = "") -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (text, source_url, reason_if_none). Requires pdfminer.six.
    """
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

# ---------------- Verification ----------------
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

# ---------------- Prompting ----------------
DEPTH_HINT = {
    "concise":  "Output must be tight and exam-ready. Use bullets. Target 120–180 words total.",
    "standard": "Balanced depth with short paragraphs. Target about 250–400 words.",
    "extended": "More depth in Rule and Application with pinpointed authorities. Target 500–700 words.",
}
TONE_HINT = {
    "neutral":   "Neutral academic tone.",
    "exam":      "Bullet-first, quick recall, minimal prose.",
    "study":     "Slightly explanatory with brief definitions.",
    "practical": "Practitioner tone. Ruthless relevance.",
}
JUR_HINT = {
    "AU":     "Use Australian authorities and terminology. Prefer HCA and state appellate courts.",
    "AU-FED": "Bias to HCA and Federal Court authorities.",
    "AU-NSW": "Bias to NSWCA/NSWSC and HCA where relevant.",
    "AU-VIC": "Bias to VSCA/VSC and HCA.",
    "AU-QLD": "Bias to QCA/QSC and HCA.",
    "AU-WA":  "Bias to WASCA/WASC and HCA.",
    "AU-SA":  "Bias to SASCFC/SASC and HCA.",
    "AU-TAS": "Bias to TAS courts and HCA.",
    "AU-ACT": "Bias to ACTCA/ACTSC and HCA.",
    "AU-NT":  "Bias to NTCA/NTSC and HCA.",
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
    jur_note  = JUR_HINT.get(jurisdiction, "Use Australian authorities and terminology.")
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

# ---------------- Root + health + brief routes ----------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Lexionary v3 - Brief API + Lexcite",
        "endpoints": ["/health", "/brief", "/lexcite/format"],
        "version": "1.6.0",
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
    if not (req.query or req.url):
        raise HTTPException(status_code=400, detail="Provide 'query' or 'url'.")

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

    if (not verify_info or not verify_info.get("ok")) and (req.query and len((req.query or "").strip()) > 400):
        log.info("Using direct-text fallback for long user input (no AustLII URL resolved/verified).")
        text = req.query or ""
        verify_info = {
            "ok": True,
            "reason": "",
            "title": (text[:80] + "…") if len(text) > 80 else text,
            "citation_on_page": "",
            "decision_date": None,
            "text_length": len(text),
            "clean_text": text,
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

    payload = build_irac_prompt(
        case_name_or_citation=req.query or source_url_used or "Unknown case",
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

# -------------------------------------------------------------------------
# LEXCITE ENGINE (AGLC DETECTION + METADATA + WEBSITE SUPPORT)
# -------------------------------------------------------------------------

def normalise_ws(s: str) -> str:
    return " ".join((s or "").split())

def detect_source_type(raw: str) -> str:
    """
    Basic source type detection for Lexcite.
    Types: CASE, LEGISLATION, JOURNAL, BOOK, WEBSITE, OTHER
    """
    text = (raw or "").strip()
    if not text:
        return "OTHER"

    # Website detection – URL present, often in <...>
    if "<http" in text or "<https" in text or re.search(r"https?://", text):
        if "<" in text and ">" in text:
            return "WEBSITE"
        if re.search(r"\b(Guardian|ABC|SBS|Sydney Morning Herald|The Conversation|The Age|News\.com\.au)\b", text, re.I):
            return "WEBSITE"

    # Legislation – Act/Regulation with year + jurisdiction
    if re.search(r"\bAct\s+\d{4}\b", text) or re.search(r"\bRegulations?\s+\d{4}\b", text):
        return "LEGISLATION"

    # Journal article – quotes + (year) vol(issue) Journal page
    if "'" in text and re.search(r"\(\d{4}\)\s*\d+\(\d+\)\s+.+\s+\d+$", text):
        return "JOURNAL"

    # Book – title + (Publisher, ed, year)
    if re.search(r"\([^,]+,\s*\d+(st|nd|rd|th)\s+ed,\s*\d{4}\)", text):
        return "BOOK"

    # Case – " v " plus (year) vol reporter page or neutral style
    if " v " in text or " v. " in text:
        return "CASE"

    return "OTHER"

def extract_case_metadata_simple(raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    s = normalise_ws(raw)
    # Reported case pattern: Party v Party (year) volume reporter page
    m = re.search(
        r"^(?P<case_name>.+?)\s*\((?P<year>\d{4})\)\s+(?P<volume>\d+)\s+(?P<reporter>[A-Za-z\.]+(?:\s+[A-Za-z\.]+)*)\s+(?P<page>\d+)",
        s
    )
    if m:
        meta.update({
            "case_name": m.group("case_name").strip(),
            "year": m.group("year"),
            "volume": m.group("volume"),
            "reporter": m.group("reporter"),
            "page_start": m.group("page"),
            "pattern": "reported",
        })
        return meta

    # Neutral citation style: Party v Party [year] COURT number
    m2 = re.search(
        r"^(?P<case_name>.+?)\s*\[(?P<year>\d{4})\]\s+(?P<court>[A-Z]{2,7})\s+(?P<number>\d{1,4})",
        s
    )
    if m2:
        meta.update({
            "case_name": m2.group("case_name").strip(),
            "year": m2.group("year"),
            "court": m2.group("court"),
            "number": m2.group("number"),
            "pattern": "neutral",
        })
        return meta

    # Fallback: just try to extract case name and maybe year
    m3 = re.search(r"^(?P<case_name>.+?)(\s*\((?P<year>\d{4})\))?$", s)
    if m3:
        meta["case_name"] = (m3.group("case_name") or "").strip()
        if m3.group("year"):
            meta["year"] = m3.group("year")
        meta["pattern"] = "unknown"
    return meta

def extract_legislation_metadata_simple(raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    s = normalise_ws(raw)
    m = re.search(
        r"^(?P<title>.+?)\s+(?P<year>\d{4})\s*\((?P<jurisdiction>[A-Za-z]{2,5})\)\s*(?P<provision>.+)?$",
        s
    )
    if m:
        meta.update({
            "title": m.group("title").strip(),
            "year": m.group("year"),
            "jurisdiction": m.group("jurisdiction"),
        })
        if m.group("provision"):
            meta["provision"] = m.group("provision").strip()
    return meta

def extract_journal_metadata_simple(raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    s = normalise_ws(raw)
    m = re.search(
        r"^(?P<author>[^,]+(?:, [^,]+)*),\s*'(?P<title>[^']+)'\s*\((?P<year>\d{4})\)\s*(?P<volume>\d+)\((?P<issue>\d+)\)\s+(?P<journal>.+?)\s+(?P<page>\d+)$",
        s
    )
    if m:
        meta.update({
            "author": m.group("author").strip(),
            "article_title": m.group("title").strip(),
            "year": m.group("year"),
            "volume": m.group("volume"),
            "issue": m.group("issue"),
            "journal_title": m.group("journal").strip(),
            "page_start": m.group("page"),
        })
    return meta

def extract_book_metadata_simple(raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    s = normalise_ws(raw)
    m = re.search(
        r"^(?P<author>.+?),\s*(?P<title>.+?)\s*\((?P<publisher>[^,]+),\s*(?P<edition>[^,]+),\s*(?P<year>\d{4})\)$",
        s
    )
    if m:
        meta.update({
            "author": m.group("author").strip(),
            "title": m.group("title").strip(),
            "publisher": m.group("publisher").strip(),
            "edition": m.group("edition").strip(),
            "year": m.group("year"),
        })
    return meta

def format_today_aus() -> str:
    today = datetime.now()
    try:
        return today.strftime("%-d %B %Y")
    except Exception:
        return today.strftime("%d %B %Y").lstrip("0")

def extract_website_metadata_llm(raw: str) -> Dict[str, Any]:
    """
    Use LLM to extract website metadata for AGLC-style citation.
    Returns dict with keys: author, title, publisher, date, url, confidence.
    """
    base = {"raw": raw, "confidence": 0.0}
    text = raw.strip()
    if not text:
        return base

    url_match = re.search(r"<(https?://[^>]+)>", text)
    if not url_match:
        url_match = re.search(r"(https?://\S+)", text)
    if url_match:
        base["url"] = url_match.group(1)

    if _openai.mode not in ("new", "legacy") or not OPENAI_API_KEY:
        return base

    system = (
        "You extract metadata for website citations in AGLC4 style. "
        "Your job is ONLY to parse the input, not to invent details. "
        "If any field is missing or uncertain, leave it blank and lower the confidence. "
        "Respond ONLY with a single JSON object, no commentary."
    )
    user = f"""
Input text (may be a partial or full citation for a website article):

{raw}

Extract:
- author (personal name or organisation)
- title (article title, without quotes)
- publisher (news outlet, organisation, website)
- date (publication date in format '20 April 2020' if present)
- url (https://...)
- confidence (0.0 to 1.0)

Return JSON:

{{
  "author": "...",
  "title": "...",
  "publisher": "...",
  "date": "...",
  "url": "...",
  "confidence": 0.0
}}"""

    try:
        resp = _openai.chat(system, user, max_tokens=350, temperature=0.0)
        data = json.loads(resp)
        if isinstance(data, dict):
            for key in ("author", "title", "publisher", "date", "url", "confidence"):
                if key in data and data[key] is not None:
                    base[key] = data[key]
    except Exception as e:
        log.warning("LLM website metadata extraction failed: %s", e)

    try:
        base["confidence"] = float(base.get("confidence", 0.0))
    except Exception:
        base["confidence"] = 0.0

    if not base.get("url") and url_match:
        base["url"] = url_match.group(1)

    return base

# -------------------------------------------------------------------------
# AGLC-STYLE VALIDATION HELPERS
# -------------------------------------------------------------------------

def validate_case_aglc(raw: str, meta: Dict[str, Any]) -> List[str]:
    """
    AGLC-style checks for cases.
    We assume italics are handled by the frontend; here we enforce structure.
    """
    errors: List[str] = []
    text = normalise_ws(raw)

    # v between parties
    if " v " not in text and " v. " not in text:
        errors.append("Case name should contain 'v' between the parties (e.g. 'Mabo v Queensland (No 2)').")

    # Year brackets
    has_round_year = bool(re.search(r"\(\d{4}\)", text))
    has_square_year = bool(re.search(r"\[\d{4}\]", text))
    if has_round_year and has_square_year:
        errors.append("Do not mix round and square brackets for the year in a single case citation.")
    elif not (has_round_year or has_square_year):
        errors.append("Include a year in either round brackets for reported cases or square brackets for neutral citations.")

    # Reported vs neutral pattern
    reported_re = re.compile(
        r"\(\d{4}\)\s+\d+\s+[A-Za-z\.]+(?:\s+[A-Za-z\.]+)*\s+\d+"
    )
    neutral_re = re.compile(
        r"\[\d{4}\]\s+[A-Z]{2,7}\s+\d{1,4}"
    )

    reported_match = reported_re.search(text)
    neutral_match = neutral_re.search(text)

    if not (reported_match or neutral_match):
        errors.append(
            "AGLC case citation should follow a reported format (e.g. '(1992) 175 CLR 1') "
            "or a neutral citation format (e.g. '[1992] HCA 23')."
        )

    # Simple sanity: if reported, prefer round brackets; if neutral, prefer square
    if reported_match and has_square_year:
        errors.append("Reported law report citations should use the year in round brackets, not square brackets.")
    if neutral_match and has_round_year:
        errors.append("Neutral citations (with court abbreviation and judgment number) should use the year in square brackets.")

    # Case name presence
    if not meta.get("case_name"):
        errors.append("Include the full case name before the year (e.g. 'Mabo v Queensland (No 2)').")

    return errors

def validate_legislation_aglc(raw: str, meta: Dict[str, Any]) -> List[str]:
    """
    AGLC-style checks for legislation:
    Short Title Year (Jurisdiction) + optional pinpoint.
    """
    errors: List[str] = []
    text = normalise_ws(raw)

    if not re.search(r"\d{4}", text):
        errors.append("Legislation citations should include the year after the title (e.g. 'Fair Work Act 2009 (Cth)').")

    if "(" not in text or ")" not in text:
        errors.append("Include the jurisdiction in round brackets after the year (e.g. '(Cth)', '(NSW)').")

    if not meta.get("title"):
        errors.append("Include the full short title of the Act before the year.")
    if not meta.get("year"):
        errors.append("Include the year of the Act immediately after the title.")
    if not meta.get("jurisdiction"):
        errors.append("Include the jurisdiction abbreviation in brackets (e.g. 'Cth', 'NSW', 'Vic').")

    return errors

def validate_journal_aglc(raw: str, meta: Dict[str, Any]) -> List[str]:
    """
    AGLC-style checks for journal articles:
    Author, 'Article Title' (Year) Volume(Issue) Journal Name FirstPage.
    """
    errors: List[str] = []
    text = normalise_ws(raw)

    if "'" not in text:
        errors.append("Journal article titles should appear in single quotation marks (e.g. 'Article Title').")
    if not re.search(r"\(\d{4}\)", text):
        errors.append("Journal citations should include the year in round brackets immediately after the title.")
    if not re.search(r"\d+\(\d+\)", text):
        errors.append("Journal citations should include both volume and issue in the form volume(issue).")
    if not meta.get("page_start"):
        errors.append("Include the starting page of the article at the end of the citation.")

    required_fields = [
        ("author", "author"),
        ("article_title", "article title"),
        ("year", "year"),
        ("volume", "volume"),
        ("issue", "issue"),
        ("journal_title", "journal title"),
        ("page_start", "starting page"),
    ]
    missing = [human for key, human in required_fields if not meta.get(key)]
    if missing:
        errors.append("Missing required elements for a journal citation: " + ", ".join(missing) + ".")

    return errors

def validate_book_aglc(raw: str, meta: Dict[str, Any]) -> List[str]:
    """
    AGLC-style checks for books:
    Author, Title (Publisher, Edition, Year).
    """
    errors: List[str] = []
    text = normalise_ws(raw)

    if "(" not in text or ")" not in text:
        errors.append("Book citations should include publisher, edition and year in round brackets.")
    if text.count(",") < 2:
        errors.append("Inside the brackets, include publisher, edition and year separated by commas.")

    required_fields = [
        ("author", "author or editor"),
        ("title", "title"),
        ("publisher", "publisher"),
        ("edition", "edition"),
        ("year", "year"),
    ]
    missing = [human for key, human in required_fields if not meta.get(key)]
    if missing:
        errors.append("Missing required elements for a book citation: " + ", ".join(missing) + ".")

    return errors

def validate_website_aglc(raw: str, meta: Dict[str, Any]) -> List[str]:
    """
    AGLC-style checks for websites:
    Author, 'Title' (Publisher, Day Month Year) <URL>.
    If date is not available, accessed date is required.
    """
    errors: List[str] = []
    text = normalise_ws(raw)

    if not meta.get("author"):
        errors.append("Include an author (individual or organisation) for website citations.")
    if not meta.get("title"):
        errors.append("Include the web page or article title, which should appear in single quotation marks in the final citation.")
    if not meta.get("publisher"):
        errors.append("Include the website or organisation as the publisher (e.g. 'The Guardian', 'ABC News').")
    if not meta.get("url"):
        errors.append("Include the URL in angle brackets: <https://...>.")

    has_date = bool(meta.get("date"))
    if not has_date:
        errors.append("If no publication date is available, an accessed date is required in the final citation (e.g. 'accessed 10 December 2025').")

    return errors

# -------------------------------------------------------------------------
# Core Lexcite processing
# -------------------------------------------------------------------------

def process_lexcite_line(idx: int, raw: str) -> LexciteEntry:
  source_type = detect_source_type(raw)
  formatted = raw
  validated = False
  validation_errors: List[str] = []
  meta: Dict[str, Any] = {}

  if source_type == "CASE":
      meta = extract_case_metadata_simple(raw)
      core_missing = []

      if not meta.get("case_name"):
          core_missing.append("case name")
      if not (meta.get("year")):
          core_missing.append("year")

      # For reported, need volume, reporter, page; for neutral, need court + number
      pattern = meta.get("pattern")
      if pattern == "reported":
          if not (meta.get("volume") and meta.get("reporter") and meta.get("page_start")):
              core_missing.append("law report citation (volume, reporter, page)")
      elif pattern == "neutral":
          if not (meta.get("court") and meta.get("number")):
              core_missing.append("neutral citation details (court and judgment number)")
      else:
          # Unknown pattern – rely more heavily on AGLC validator
          pass

      if core_missing:
          validation_errors.append(
              "Looks like a case, but is missing: " + ", ".join(core_missing) +
              ". Add these details to meet AGLC structure."
          )

      aglc_errors = validate_case_aglc(raw, meta)
      validation_errors.extend(aglc_errors)

      validated = len(validation_errors) == 0
      meta["confidence"] = meta.get("confidence", 0.9)

  elif source_type == "LEGISLATION":
      meta = extract_legislation_metadata_simple(raw)
      core_missing = []
      if not meta.get("title"):
          core_missing.append("title")
      if not meta.get("year"):
          core_missing.append("year")
      if not meta.get("jurisdiction"):
          core_missing.append("jurisdiction")

      if core_missing:
          validation_errors.append(
              "Looks like legislation, but is missing: " + ", ".join(core_missing) +
              ". Include short title, year and jurisdiction to meet AGLC structure."
          )

      aglc_errors = validate_legislation_aglc(raw, meta)
      validation_errors.extend(aglc_errors)

      validated = len(validation_errors) == 0
      meta["confidence"] = meta.get("confidence", 0.9)

  elif source_type == "JOURNAL":
      meta = extract_journal_metadata_simple(raw)
      aglc_errors = validate_journal_aglc(raw, meta)
      validation_errors.extend(aglc_errors)
      validated = len(validation_errors) == 0
      meta["confidence"] = meta.get("confidence", 0.9)

  elif source_type == "BOOK":
      meta = extract_book_metadata_simple(raw)
      aglc_errors = validate_book_aglc(raw, meta)
      validation_errors.extend(aglc_errors)
      validated = len(validation_errors) == 0
      meta["confidence"] = meta.get("confidence", 0.9)

  elif source_type == "WEBSITE":
      meta = extract_website_metadata_llm(raw)
      required_fields = ["author", "title", "publisher", "url"]
      missing_keys = [f for f in required_fields if not meta.get(f)]
      pretty_name = {
          "author": "author",
          "title": "title",
          "publisher": "publisher or news source",
          "url": "URL",
      }

      has_date = bool(meta.get("date"))
      confidence = meta.get("confidence", 0.0)

      if missing_keys:
          human = [pretty_name.get(k, k) for k in missing_keys]
          validation_errors.append(
              "This appears to be a website or online article, but is missing: " +
              ", ".join(human) + ". Add these details for a complete AGLC citation."
          )

      aglc_errors = validate_website_aglc(raw, meta)
      validation_errors.extend(aglc_errors)

      if confidence < 0.6:
          validation_errors.append(
              "Low confidence in extracted website metadata. Confirm the author, title, publisher, date and URL before relying on this citation."
          )

      author = (meta.get("author") or "").strip()
      title = (meta.get("title") or "").strip()
      publisher = (meta.get("publisher") or "").strip()
      date = (meta.get("date") or "").strip()
      url = (meta.get("url") or "").strip()

      if author and title and publisher and url and not missing_keys:
          if date:
              formatted = f"{author}, '{title}' ({publisher}, {date}) <{url}>."
          else:
              accessed = format_today_aus()
              formatted = f"{author}, '{title}' ({publisher}) <{url}> accessed {accessed}."
      else:
          formatted = raw

      validated = len(validation_errors) == 0

  else:
      formatted = raw
      validated = False
      validation_errors.append(
          "Unsupported or unrecognised source type for automatic AGLC formatting in this version."
      )
      meta = {}

  return LexciteEntry(
      id=str(idx),
      raw=raw,
      source_type=source_type,
      formatted=formatted,
      validated=validated,
      validation_errors=validation_errors,
      meta=meta,
  )

@app.post("/lexcite/format", response_model=LexciteResponse)
def lexcite_format(req: LexciteRequest):
    """
    Lexcite endpoint.
    Accepts multiple citations separated by newlines in 'input_text'.
    Returns structured entries with type, formatted text, validation flags and metadata.
    """
    api_version = datetime.utcnow().strftime("%Y-%m-%d")
    text = (req.input_text or "").strip()
    if not text:
        return LexciteResponse(api_version=api_version, entries=[], errors=["No input provided."])

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entries: List[LexciteEntry] = []
    errors: List[str] = []

    for idx, line in enumerate(lines, start=1):
        try:
            entry = process_lexcite_line(idx, line)
            entries.append(entry)
        except Exception as e:
            log.exception("Lexcite processing failed for line %d: %s", idx, line)
            errors.append(f"Error processing line {idx}: {e}")

    return LexciteResponse(api_version=api_version, entries=entries, errors=errors)
