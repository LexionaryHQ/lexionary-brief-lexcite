# aglc_engine.py
# Lexionary Lexcite AGLC4 engine (formatting + validation + paste-list parsing)
# Version: 2.0.0

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


class SourceType(str, Enum):
    case = "case"
    legislation = "legislation"
    journal_article = "journal_article"
    book = "book"
    book_chapter = "book_chapter"
    media_article = "media_article"
    report = "report"
    website = "website"


@dataclass
class CitationResult:
    source_type: SourceType
    mode: str
    text: str
    html: str


# -------------------------
# Helpers
# -------------------------

_WS = re.compile(r"\s+")
_NEUTRAL = re.compile(r"\[(\d{4})\]\s+([A-Z]{2,8})\s+(\d{1,4})")
_REPORTED = re.compile(r"\((\d{4})\)\s+(\d+)\s+([A-Z][A-Z0-9]{1,10})\s+(\d+)")
_PIN_PARAS = re.compile(r"\[(\d{1,5})\]\s*$")  # trailing [150]
_PIN_PAGES = re.compile(r"\b(\d{1,5})\b$")

_JUR_MAP = {
    "CTH": "Cth",
    "COMMONWEALTH": "Cth",
    "NSW": "NSW",
    "VIC": "Vic",
    "QLD": "Qld",
    "SA": "SA",
    "WA": "WA",
    "TAS": "Tas",
    "ACT": "ACT",
    "NT": "NT",
}


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def _title_case_jur(j: str) -> str:
    if not j:
        return ""
    j2 = _norm(j)
    key = j2.upper()
    return _JUR_MAP.get(key, j2)


def _escape_min_html(s: str) -> str:
    # We only ever emit <i> tags. Escape everything else.
    if s is None:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _ital(s: str) -> Tuple[str, str]:
    # returns (text, html)
    t = s or ""
    h = f"<i>{_escape_min_html(t)}</i>"
    return t, h


def _quote_single(s: str) -> str:
    # AGLC uses single quotes for article/chapter titles
    s = s or ""
    return f"'{s}'"


def _join_authors(authors: List[str]) -> str:
    a = [x.strip() for x in (authors or []) if x and x.strip()]
    if not a:
        return ""
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return f"{a[0]} and {a[1]}"
    return ", ".join(a[:-1]) + f" and {a[-1]}"


def _pin_suffix(pinpoint: Optional[str], pin_type: Optional[str]) -> str:
    p = (pinpoint or "").strip()
    t = (pin_type or "").strip().lower()
    if not p:
        return ""
    if t == "paragraph":
        return f" [{p}]"
    if t == "page":
        return f", {p}"
    # If unknown, try to infer: bracketed looks like para, else page-ish
    if p.isdigit():
        return f", {p}"
    return f" [{p}]"


def _clean_url(url: str) -> str:
    return (url or "").strip().strip("<>").strip()


# -------------------------
# Models for builder (/cite)
# -------------------------

class CaseData(BaseModel):
    case_name: str = Field(..., min_length=1)
    year: str = Field(..., min_length=4, max_length=4)
    reporter_series_by_year: bool = False

    # reported
    volume: Optional[str] = None
    reporter: Optional[str] = None
    first_page: Optional[str] = None

    # neutral
    court: Optional[str] = None
    decision_number: Optional[str] = None

    neutral_citation_first: bool = True
    unreported: bool = False

    pinpoint_type: Optional[str] = None  # page | paragraph
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_digits(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("Year must be 4 digits")
        return v2


class LegislationData(BaseModel):
    title: str = Field(..., min_length=1)
    year: str = Field(..., min_length=4, max_length=4)
    jurisdiction: str = Field(..., min_length=2, max_length=6)
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None  # s, ss, pt, div etc
    pinpoint_number: Optional[str] = None  # 5B, 12(1) etc

    @field_validator("year")
    @classmethod
    def _year_digits(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("Year must be 4 digits")
        return v2


class JournalArticleData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    article_title: str = Field(..., min_length=1)
    year: str = Field(..., min_length=4, max_length=4)
    year_in_square_brackets: bool = False
    volume: Optional[str] = None
    issue: Optional[str] = None
    journal_title: str = Field(..., min_length=1)
    starting_page: str = Field(..., min_length=1)
    pinpoint: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_digits(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("Year must be 4 digits")
        return v2


class BookData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    title: str = Field(..., min_length=1)
    publisher: str = Field(..., min_length=1)
    year: str = Field(..., min_length=4, max_length=4)
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_digits(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("Year must be 4 digits")
        return v2


class BookChapterData(BaseModel):
    chapter_authors: List[str] = Field(default_factory=list)
    chapter_title: str = Field(..., min_length=1)
    editors: List[str] = Field(default_factory=list)
    book_title: str = Field(..., min_length=1)
    publisher: str = Field(..., min_length=1)
    year: str = Field(..., min_length=4, max_length=4)
    edition: Optional[str] = None
    starting_page: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_digits(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("Year must be 4 digits")
        return v2


class MediaArticleData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    org_as_author: Optional[str] = None
    article_title: str = Field(..., min_length=1)
    newspaper_title: str = Field(..., min_length=1)
    city: Optional[str] = None
    date: str = Field(..., min_length=4)  # accept "1 January 2025" etc
    page: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class ReportData(BaseModel):
    author_or_org: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    report_number_or_series: Optional[str] = None
    publisher: Optional[str] = None
    place: Optional[str] = None
    date: Optional[str] = None  # year or full date
    pinpoint: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class WebsiteData(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str = Field(..., min_length=1)
    site_name: str = Field(..., min_length=1)
    date: Optional[str] = None
    url: str = Field(..., min_length=6)
    access_date: str = Field(..., min_length=4)


# -------------------------
# Formatting functions (builder)
# -------------------------

def _format_case(d: CaseData, mode: str) -> CitationResult:
    case_name = _norm(d.case_name)
    year = d.year
    pin = _pin_suffix(d.pinpoint, d.pinpoint_type)

    name_text, name_html = _ital(case_name)

    neutral_text = ""
    neutral_html = ""
    if d.court and d.decision_number:
        neutral = f"[{year}] {d.court.strip().upper()} {d.decision_number.strip()}"
        neutral_text = neutral
        neutral_html = _escape_min_html(neutral)

    reported_text = ""
    reported_html = ""
    if d.volume and d.reporter and d.first_page:
        year_part = f"[{year}]" if d.reporter_series_by_year else f"({year})"
        rep = f"{year_part} {d.volume.strip()} {d.reporter.strip()} {d.first_page.strip()}"
        reported_text = rep
        reported_html = _escape_min_html(rep)

    if d.unreported:
        if not neutral_text:
            raise ValueError("Unreported cases require a neutral citation (court and decision number).")
        body_text = f"{name_text} {neutral_text}{pin}."
        body_html = f"{name_html} {neutral_html}{_escape_min_html(pin)}."
        return CitationResult(SourceType.case, mode, body_text, body_html)

    # If both provided, choose ordering
    parts_text: List[str] = []
    parts_html: List[str] = []

    if neutral_text and reported_text:
        if d.neutral_citation_first:
            parts_text = [neutral_text, reported_text]
            parts_html = [neutral_html, reported_html]
        else:
            parts_text = [reported_text, neutral_text]
            parts_html = [reported_html, neutral_html]
    elif reported_text:
        parts_text = [reported_text]
        parts_html = [reported_html]
    elif neutral_text:
        parts_text = [neutral_text]
        parts_html = [neutral_html]
    else:
        raise ValueError("Provide either a reported citation (volume, reporter, first page) or a neutral citation (court, decision number).")

    body_text = f"{name_text} " + " ".join(parts_text) + f"{pin}."
    body_html = f"{name_html} " + " ".join(parts_html) + f"{_escape_min_html(pin)}."
    return CitationResult(SourceType.case, mode, body_text, body_html)


def _format_legislation(d: LegislationData, mode: str) -> CitationResult:
    title = _norm(d.title)
    year = d.year
    jur = _title_case_jur(d.jurisdiction)

    pin = ""
    if d.pinpoint_unit and d.pinpoint_number:
        pin = f" {d.pinpoint_unit.strip()} {d.pinpoint_number.strip()}"

    bill = " Bill" if d.is_bill and not title.lower().endswith("bill") else ""
    txt = f"{title}{bill} {year} ({jur}){pin}."
    html = _escape_min_html(txt)
    return CitationResult(SourceType.legislation, mode, txt, html)


def _format_journal(d: JournalArticleData, mode: str) -> CitationResult:
    authors = _join_authors(d.authors)
    title = _quote_single(_norm(d.article_title))
    year = d.year
    vol = _norm(d.volume) if d.volume else ""
    issue = _norm(d.issue) if d.issue else ""
    journal = _norm(d.journal_title)
    start = _norm(d.starting_page)
    pin = _norm(d.pinpoint) if d.pinpoint else ""

    year_part = f"[{year}]" if d.year_in_square_brackets else f"({year})"
    vol_issue = ""
    if vol and issue:
        vol_issue = f"{vol}({issue})"
    elif vol:
        vol_issue = f"{vol}"
    elif issue:
        # uncommon, but keep it
        vol_issue = f"({issue})"

    j_text, j_html = _ital(journal)

    core = f"{year_part} {vol_issue} {j_text} {start}".strip()
    core_html = f"{_escape_min_html(year_part)} {_escape_min_html(vol_issue)} {j_html} {_escape_min_html(start)}".strip()

    if pin:
        core += f", {pin}"
        core_html += f", {_escape_min_html(pin)}"

    if d.is_online:
        url = _clean_url(d.url or "")
        ad = _norm(d.access_date or "")
        if not url or not ad:
            raise ValueError("Online journal articles require URL and access date.")
        core += f" <{url}> accessed {ad}"
        core_html += f" &lt;{_escape_min_html(url)}&gt; accessed {_escape_min_html(ad)}"

    if authors:
        txt = f"{authors}, {title} {core}."
        html = f"{_escape_min_html(authors)}, {_escape_min_html(title)} {core_html}."
    else:
        txt = f"{title} {core}."
        html = f"{_escape_min_html(title)} {core_html}."

    return CitationResult(SourceType.journal_article, mode, txt, html)


def _format_book(d: BookData, mode: str) -> CitationResult:
    authors = _join_authors(d.authors)
    title = _norm(d.title)
    pub = _norm(d.publisher)
    year = d.year
    edition = _norm(d.edition) if d.edition else ""
    pin = _norm(d.pinpoint) if d.pinpoint else ""

    t_text, t_html = _ital(title)

    inside = f"{pub}, {edition}, {year}" if edition else f"{pub}, {year}"
    txt = f"{authors}, {t_text} ({inside})"
    html = f"{_escape_min_html(authors)}, {t_html} ({_escape_min_html(inside)})"

    if not authors:
        txt = f"{t_text} ({inside})"
        html = f"{t_html} ({_escape_min_html(inside)})"

    if pin:
        txt += f" {pin}"
        html += f" {_escape_min_html(pin)}"

    txt += "."
    html += "."
    return CitationResult(SourceType.book, mode, txt, html)


def _format_book_chapter(d: BookChapterData, mode: str) -> CitationResult:
    chap_authors = _join_authors(d.chapter_authors)
    chap_title = _quote_single(_norm(d.chapter_title))
    editors = _join_authors(d.editors)
    book_title = _norm(d.book_title)
    pub = _norm(d.publisher)
    year = d.year
    edition = _norm(d.edition) if d.edition else ""
    start = _norm(d.starting_page) if d.starting_page else ""
    pin = _norm(d.pinpoint) if d.pinpoint else ""

    bt_text, bt_html = _ital(book_title)

    ed_part = ""
    if editors:
        ed_part = f"{editors} (ed)"

    inside = f"{pub}, {edition}, {year}" if edition else f"{pub}, {year}"

    tail = ""
    tail_html = ""
    if start:
        tail = f" {start}"
        tail_html = f" {_escape_min_html(start)}"
    if pin:
        if start:
            tail += f", {pin}"
            tail_html += f", {_escape_min_html(pin)}"
        else:
            tail += f" {pin}"
            tail_html += f" {_escape_min_html(pin)}"

    if chap_authors:
        txt = f"{chap_authors}, {chap_title} in {ed_part} {bt_text} ({inside}){tail}."
        html = f"{_escape_min_html(chap_authors)}, {_escape_min_html(chap_title)} in {_escape_min_html(ed_part)} {bt_html} ({_escape_min_html(inside)}){tail_html}."
    else:
        txt = f"{chap_title} in {ed_part} {bt_text} ({inside}){tail}."
        html = f"{_escape_min_html(chap_title)} in {_escape_min_html(ed_part)} {bt_html} ({_escape_min_html(inside)}){tail_html}."

    # cleanup stray "in  " if no editors
    txt = _norm(txt.replace("in  ", "in "))
    html = html.replace("in  ", "in ")
    return CitationResult(SourceType.book_chapter, mode, txt, html)


def _format_media_article(d: MediaArticleData, mode: str) -> CitationResult:
    authors = _join_authors(d.authors)
    org = _norm(d.org_as_author) if d.org_as_author else ""
    title = _quote_single(_norm(d.article_title))
    paper = _norm(d.newspaper_title)
    city = _norm(d.city) if d.city else ""
    date = _norm(d.date)
    page = _norm(d.page) if d.page else ""

    p_text, p_html = _ital(paper)

    author_part = authors or org
    if not author_part:
        author_part = ""  # allowed, but discouraged

    place_date = f"({city + ', ' if city else ''}{date})"
    place_date_html = f"({_escape_min_html(city + ', ' if city else '')}{_escape_min_html(date)})"

    core_txt = f"{p_text} {place_date}"
    core_html = f"{p_html} {place_date_html}"

    if page:
        core_txt += f" {page}"
        core_html += f" {_escape_min_html(page)}"

    if d.is_online:
        url = _clean_url(d.url or "")
        ad = _norm(d.access_date or "")
        if not url or not ad:
            raise ValueError("Online media articles require URL and access date.")
        core_txt += f" <{url}> accessed {ad}"
        core_html += f" &lt;{_escape_min_html(url)}&gt; accessed {_escape_min_html(ad)}"

    if author_part:
        txt = f"{author_part}, {title} {core_txt}."
        html = f"{_escape_min_html(author_part)}, {_escape_min_html(title)} {core_html}."
    else:
        txt = f"{title} {core_txt}."
        html = f"{_escape_min_html(title)} {core_html}."

    return CitationResult(SourceType.media_article, mode, txt, html)


def _format_report(d: ReportData, mode: str) -> CitationResult:
    author = _norm(d.author_or_org)
    title = _norm(d.title)
    series = _norm(d.report_number_or_series) if d.report_number_or_series else ""
    publisher = _norm(d.publisher) if d.publisher else ""
    place = _norm(d.place) if d.place else ""
    date = _norm(d.date) if d.date else ""
    pin = _norm(d.pinpoint) if d.pinpoint else ""

    t_text, t_html = _ital(title)

    inside_parts = [p for p in [series, publisher, place, date] if p]
    inside = ", ".join(inside_parts) if inside_parts else ""

    core_txt = f"{author}, {t_text}"
    core_html = f"{_escape_min_html(author)}, {t_html}"

    if inside:
        core_txt += f" ({inside})"
        core_html += f" ({_escape_min_html(inside)})"

    if pin:
        core_txt += f" {pin}"
        core_html += f" {_escape_min_html(pin)}"

    if d.is_online:
        url = _clean_url(d.url or "")
        ad = _norm(d.access_date or "")
        if not url or not ad:
            raise ValueError("Online reports require URL and access date.")
        core_txt += f" <{url}> accessed {ad}"
        core_html += f" &lt;{_escape_min_html(url)}&gt; accessed {_escape_min_html(ad)}"

    core_txt += "."
    core_html += "."
    return CitationResult(SourceType.report, mode, core_txt, core_html)


def _format_website(d: WebsiteData, mode: str) -> CitationResult:
    author = _norm(d.author_or_org) if d.author_or_org else ""
    title = _quote_single(_norm(d.page_title))
    site = _norm(d.site_name)
    date = _norm(d.date) if d.date else ""
    url = _clean_url(d.url)
    access = _norm(d.access_date)

    inside_parts = [p for p in [site, date] if p]
    inside = ", ".join(inside_parts)

    if author:
        txt = f"{author}, {title} ({inside}) <{url}> accessed {access}."
        html = f"{_escape_min_html(author)}, {_escape_min_html(title)} ({_escape_min_html(inside)}) &lt;{_escape_min_html(url)}&gt; accessed {_escape_min_html(access)}."
    else:
        txt = f"{title} ({inside}) <{url}> accessed {access}."
        html = f"{_escape_min_html(title)} ({_escape_min_html(inside)}) &lt;{_escape_min_html(url)}&gt; accessed {_escape_min_html(access)}."

    return CitationResult(SourceType.website, mode, txt, html)


def format_citation(source_type: SourceType | str, data: Dict[str, Any], mode: str = "footnote") -> CitationResult:
    st = SourceType(source_type) if not isinstance(source_type, SourceType) else source_type
    m = (mode or "footnote").strip().lower()
    if m not in ("footnote", "bibliography"):
        m = "footnote"

    if st == SourceType.case:
        d = CaseData(**data)
        return _format_case(d, m)
    if st == SourceType.legislation:
        d = LegislationData(**data)
        return _format_legislation(d, m)
    if st == SourceType.journal_article:
        d = JournalArticleData(**data)
        return _format_journal(d, m)
    if st == SourceType.book:
        d = BookData(**data)
        return _format_book(d, m)
    if st == SourceType.book_chapter:
        d = BookChapterData(**data)
        return _format_book_chapter(d, m)
    if st == SourceType.media_article:
        d = MediaArticleData(**data)
        return _format_media_article(d, m)
    if st == SourceType.report:
        d = ReportData(**data)
        return _format_report(d, m)
    if st == SourceType.website:
        d = WebsiteData(**data)
        return _format_website(d, m)

    raise ValueError("Unsupported source type")


# -------------------------
# Paste list parsing (best-effort, still AGLC output)
# -------------------------

@dataclass
class ParsedLine:
    source_type: str
    data: Dict[str, Any]
    warnings: List[str]


def _parse_case_line(raw: str) -> ParsedLine:
    s = _norm(raw)
    warnings: List[str] = []

    # Pinpoint at end: [150]
    pin_para = None
    m_pin = _PIN_PARAS.search(s)
    if m_pin:
        pin_para = m_pin.group(1)
        s = _PIN_PARAS.sub("", s).strip()

    # Neutral portion
    m_neu = _NEUTRAL.search(s)
    neutral_year = None
    neutral_court = None
    neutral_no = None
    if m_neu:
        neutral_year, neutral_court, neutral_no = m_neu.group(1), m_neu.group(2), m_neu.group(3)

    # Reported portion
    m_rep = _REPORTED.search(s)
    rep_year = rep_vol = rep_rep = rep_page = None
    if m_rep:
        rep_year, rep_vol, rep_rep, rep_page = m_rep.group(1), m_rep.group(2), m_rep.group(3), m_rep.group(4)

    # Case name is text before first citation token if possible
    cut_idx = None
    if m_neu:
        cut_idx = m_neu.start()
    if m_rep:
        cut_idx = m_rep.start() if (cut_idx is None or m_rep.start() < cut_idx) else cut_idx

    case_name = s if cut_idx is None else s[:cut_idx].strip()
    if not case_name:
        warnings.append("Could not confidently detect case name. Add parties like 'Smith v Jones'.")

    # Build data for engine
    year = neutral_year or rep_year or ""
    if not year:
        warnings.append("Missing year. Add a neutral citation like [1992] HCA 23 or a reported citation like (1988) 164 CLR 387.")
        # still attempt minimal
        year = "0000"

    data: Dict[str, Any] = {
        "case_name": case_name or s,
        "year": year,
        "reporter_series_by_year": False,
        "volume": rep_vol,
        "reporter": rep_rep,
        "first_page": rep_page,
        "court": neutral_court,
        "decision_number": neutral_no,
        "neutral_citation_first": True,
        "unreported": False,
        "pinpoint_type": "paragraph" if pin_para else None,
        "pinpoint": pin_para,
    }

    # Validation policy for paste mode:
    # Accept neutral-only as valid. Do not demand reported citation.
    if neutral_year and neutral_court and neutral_no:
        pass
    elif rep_year and rep_vol and rep_rep and rep_page:
        pass
    else:
        warnings.append("Missing a complete citation. Provide either [year] Court No or (year) vol Reporter page.")

    return ParsedLine("CASE", data, warnings)


def _parse_legislation_line(raw: str) -> ParsedLine:
    s = _norm(raw)
    warnings: List[str] = []

    # e.g. Civil Liability Act 2002 (NSW) s 5B
    m = re.search(r"^(.*?)\s+(\d{4})\s*\(\s*([A-Za-z]{2,6})\s*\)\s*(.*)?$", s)
    if not m:
        # try no brackets
        m = re.search(r"^(.*?)\s+(\d{4})\s+([A-Za-z]{2,6})\s*(.*)?$", s)

    if not m:
        warnings.append("Could not parse legislation. Use 'Title Year (Jur) s X'.")
        data = {"title": s, "year": "0000", "jurisdiction": "Cth", "is_bill": False, "pinpoint_unit": None, "pinpoint_number": None}
        return ParsedLine("LEGISLATION", data, warnings)

    title = _norm(m.group(1))
    year = m.group(2)
    jur = _title_case_jur(m.group(3))
    tail = _norm(m.group(4) or "")

    pin_unit = None
    pin_no = None
    if tail:
        m2 = re.search(r"^(s|ss|pt|div|sch)\s+(.+)$", tail, re.I)
        if m2:
            pin_unit = m2.group(1)
            pin_no = m2.group(2)

    data = {"title": title, "year": year, "jurisdiction": jur, "is_bill": False, "pinpoint_unit": pin_unit, "pinpoint_number": pin_no}
    return ParsedLine("LEGISLATION", data, warnings)


def _parse_website_line(raw: str) -> ParsedLine:
    s = _norm(raw)
    warnings: List[str] = []
    url = ""

    m_url = re.search(r"<(https?://[^>]+)>", s)
    if m_url:
        url = m_url.group(1)
    else:
        m_url = re.search(r"(https?://\S+)", s)
        if m_url:
            url = m_url.group(1)

    if not url:
        warnings.append("No URL detected. Add <https://...> so the website citation can be formatted properly.")

    # In paste mode, we need an access date but user may not supply.
    # We warn and still output something copyable.
    data = {
        "author_or_org": None,
        "page_title": s.replace(url, "").strip(" -") if s else "Untitled page",
        "site_name": "Website",
        "date": None,
        "url": url or "https://example.com",
        "access_date": "access date required",
    }
    warnings.append("Website citations in AGLC usually need an access date. Add 'accessed 16 February 2026' in builder mode for accuracy.")
    return ParsedLine("WEBSITE", data, warnings)


def detect_source_type_line(raw: str) -> str:
    s = _norm(raw).lower()
    if not s:
        return "OTHER"
    if " act " in f" {s} " or re.search(r"\bact\s+\d{4}\b", s) or " regulations" in s:
        return "LEGISLATION"
    if re.search(r"\[\d{4}\]\s+[A-Z]{2,8}\s+\d{1,4}", raw) or re.search(r"\(\d{4}\)\s+\d+\s+[A-Z]{2,10}\s+\d+", raw) or " v " in s or " v. " in s:
        return "CASE"
    if "http://" in s or "https://" in s or "<http" in s or "<https" in s:
        return "WEBSITE"
    return "OTHER"


def parse_line_best_effort(raw: str) -> ParsedLine:
    st = detect_source_type_line(raw)
    if st == "CASE":
        return _parse_case_line(raw)
    if st == "LEGISLATION":
        return _parse_legislation_line(raw)
    if st == "WEBSITE":
        return _parse_website_line(raw)
    return ParsedLine("OTHER", {"raw": _norm(raw)}, ["Unsupported source type in paste mode. Use the builder for this source."])


# -------------------------
# Paste mode batch formatter
# -------------------------

class LexciteEntryOut(BaseModel):
    id: str
    raw: str
    source_type: str
    formatted: str
    html: str
    validated: bool
    validation_errors: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


def format_lines_best_effort(lines: List[str], mode: str = "footnote") -> List[LexciteEntryOut]:
    out: List[LexciteEntryOut] = []

    for idx, raw in enumerate(lines, start=1):
        raw0 = raw
        parsed = parse_line_best_effort(raw0)

        validated = True
        errors: List[str] = []
        meta: Dict[str, Any] = {}

        try:
            if parsed.source_type == "CASE":
                res = format_citation(SourceType.case, parsed.data, mode=mode)
            elif parsed.source_type == "LEGISLATION":
                res = format_citation(SourceType.legislation, parsed.data, mode=mode)
            elif parsed.source_type == "WEBSITE":
                res = format_citation(SourceType.website, parsed.data, mode=mode)
            else:
                validated = False
                errors.extend(parsed.warnings)
                res = CitationResult(SourceType.website, mode, raw0, _escape_min_html(raw0))
        except Exception as e:
            validated = False
            errors.append(str(e))
            res = CitationResult(SourceType.website, mode, raw0, _escape_min_html(raw0))

        # warnings are "needs review" but we still output formatted where possible
        if parsed.warnings:
            validated = False
            errors.extend(parsed.warnings)

        meta["parsed"] = parsed.source_type

        out.append(
            LexciteEntryOut(
                id=str(idx),
                raw=raw0,
                source_type=parsed.source_type,
                formatted=res.text,
                html=res.html,
                validated=validated,
                validation_errors=errors,
                meta=meta,
            )
        )

    return out
