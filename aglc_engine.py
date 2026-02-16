# aglc_engine.py
# Lexionary Lexcite AGLC4 Engine
# Version: 2.0.0
# Outputs both plain text and HTML (italics via <i> only).
# Focus: AU law students, accuracy-first formatting, strict validation.

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


# ---------------------------
# Helpers
# ---------------------------

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _escape_html(s: str) -> str:
    # Strict escaping, then we will only inject <i> where we intend.
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _italics_text(text: str) -> Tuple[str, str]:
    """
    Returns (plain_text, html_text) where html has italicised version of `text`.
    """
    t = _clean_spaces(text)
    return t, f"<i>{_escape_html(t)}</i>"


def _fmt_pinpoint(pinpoint_type: Optional[str], pinpoint: Optional[str]) -> str:
    pt = (pinpoint_type or "").strip().lower()
    pv = (pinpoint or "").strip()

    if not pt or not pv:
        return ""

    if pt == "page":
        # AGLC: reported cases pinpoint with comma then page number
        return f", {pv}"
    if pt == "paragraph":
        # AGLC: neutral citations use [para]
        # If user gives "150" turn into "[150]"
        if re.fullmatch(r"\[\d+\]", pv):
            return f", {pv}"
        if re.fullmatch(r"\d+", pv):
            return f", [{pv}]"
        # If they typed something odd, still show it but bracket if it looks numeric-ish
        return f", [{pv}]"

    return f", {pv}"


def _join_authors(authors: List[str], mode: str) -> str:
    a = [_clean_spaces(x) for x in (authors or []) if _clean_spaces(x)]
    if not a:
        return ""

    if mode == "bibliography":
        # Very simple inversion: "Jane Smith" -> "Smith, Jane"
        inv: List[str] = []
        for name in a:
            parts = name.split()
            if len(parts) >= 2:
                inv.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
            else:
                inv.append(name)
        a = inv

    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return f"{a[0]} and {a[1]}"
    return ", ".join(a[:-1]) + f", and {a[-1]}"


def _ensure_ends_with_period(s: str) -> str:
    s = _clean_spaces(s)
    if not s:
        return s
    if s.endswith("."):
        return s
    return s + "."


# ---------------------------
# Models
# ---------------------------

class CaseCitation(BaseModel):
    case_name: str = Field(..., description="Case name, eg Mabo v Queensland (No 2)")
    year: str = Field(..., description="Decision year, eg 1992")
    # Reported (optional)
    reporter_series_by_year: bool = Field(False)
    volume: Optional[str] = Field(None)
    reporter: Optional[str] = Field(None)
    first_page: Optional[str] = Field(None)

    # Neutral (optional)
    court: Optional[str] = Field(None, description="Court code, eg HCA, NSWCA")
    decision_number: Optional[str] = Field(None, description="Neutral decision number, eg 23")

    neutral_citation_first: bool = Field(True)
    unreported: bool = Field(False)

    pinpoint_type: Optional[str] = Field(None, description="page or paragraph")
    pinpoint: Optional[str] = Field(None)

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("Year must be a 4 digit year.")
        return v

    @field_validator("case_name")
    @classmethod
    def _case_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if len(v) < 3:
            raise ValueError("Case name is required.")
        return v


class LegislationCitation(BaseModel):
    title: str
    year: str
    jurisdiction: str
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None  # s or ss
    pinpoint_number: Optional[str] = None  # 5B, 12(1), etc

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("Year must be a 4 digit year.")
        return v

    @field_validator("title")
    @classmethod
    def _title_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if len(v) < 3:
            raise ValueError("Title is required.")
        return v

    @field_validator("jurisdiction")
    @classmethod
    def _jur_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if len(v) < 2:
            raise ValueError("Jurisdiction is required, eg Cth, NSW.")
        return v


class JournalArticleCitation(BaseModel):
    authors: List[str]
    article_title: str
    year: str
    year_in_square_brackets: bool = False
    volume: Optional[str] = None
    issue: Optional[str] = None
    journal_title: str
    starting_page: str
    pinpoint: Optional[str] = None
    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("Year must be a 4 digit year.")
        return v


class BookCitation(BaseModel):
    authors: List[str]
    title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("Year must be a 4 digit year.")
        return v


class BookChapterCitation(BaseModel):
    chapter_authors: List[str]
    chapter_title: str
    editors: List[str]
    book_title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    starting_page: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = _clean_spaces(v)
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("Year must be a 4 digit year.")
        return v


class MediaArticleCitation(BaseModel):
    authors: List[str] = Field(default_factory=list)
    org_as_author: Optional[str] = None
    article_title: str
    newspaper_title: str
    city: Optional[str] = None
    date: str
    page: Optional[str] = None
    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class ReportCitation(BaseModel):
    author_or_org: str
    title: str
    report_number_or_series: Optional[str] = None
    publisher: Optional[str] = None
    place: Optional[str] = None
    date: Optional[str] = None
    pinpoint: Optional[str] = None
    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class WebsiteCitation(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str
    site_name: str
    date: Optional[str] = None
    url: str
    access_date: Optional[str] = None


# ---------------------------
# Formatters
# ---------------------------

def _format_case(data: CaseCitation, mode: str) -> CitationResult:
    case_name_plain, case_name_html = _italics_text(data.case_name)

    # Neutral citation if possible
    neutral = ""
    if data.court and data.decision_number:
        neutral = f"[{data.year}] {data.court} {data.decision_number}"

    # Reported citation if possible
    reported = ""
    if data.volume and data.reporter and data.first_page:
        if data.reporter_series_by_year:
            reported = f"[{data.year}] {data.volume} {data.reporter} {data.first_page}"
        else:
            reported = f"({data.year}) {data.volume} {data.reporter} {data.first_page}"

    # Validation logic:
    # AGLC4: case must have either (a) a reported citation OR (b) a neutral citation for unreported cases.
    if data.unreported and not neutral:
        raise ValueError("Unreported case requires a neutral citation (court and decision number).")

    if not neutral and not reported:
        raise ValueError("Provide either a reported citation (volume, reporter, first page) or a neutral citation (court and decision number).")

    pin = _fmt_pinpoint(data.pinpoint_type, data.pinpoint)

    if neutral and reported:
        if data.neutral_citation_first:
            text = f"{case_name_plain} {neutral}; {reported}{pin}"
            html = f"{case_name_html} {_escape_html(neutral)}; {_escape_html(reported)}{_escape_html(pin)}"
        else:
            text = f"{case_name_plain} {reported}; {neutral}{pin}"
            html = f"{case_name_html} {_escape_html(reported)}; {_escape_html(neutral)}{_escape_html(pin)}"
    elif neutral:
        text = f"{case_name_plain} {neutral}{pin}"
        html = f"{case_name_html} {_escape_html(neutral)}{_escape_html(pin)}"
    else:
        text = f"{case_name_plain} {reported}{pin}"
        html = f"{case_name_html} {_escape_html(reported)}{_escape_html(pin)}"

    return CitationResult(source_type=SourceType.case, mode=mode, text=_ensure_ends_with_period(text), html=_ensure_ends_with_period(html))


def _format_legislation(data: LegislationCitation, mode: str) -> CitationResult:
    title = _clean_spaces(data.title)
    yr = data.year
    jur = _clean_spaces(data.jurisdiction)
    kind = "Bill" if data.is_bill else "Act"

    base = f"{title} {yr} ({jur})"
    if data.is_bill:
        # If user wrote "Fair Work Bill", still prefer their title, but show "(Bill)"
        base = f"{title} {yr} ({jur}) ({kind})"

    if data.pinpoint_unit and data.pinpoint_number:
        base += f" {data.pinpoint_unit} {data.pinpoint_number}"

    text = _ensure_ends_with_period(base)
    html = _ensure_ends_with_period(_escape_html(base))
    return CitationResult(source_type=SourceType.legislation, mode=mode, text=text, html=html)


def _format_journal(data: JournalArticleCitation, mode: str) -> CitationResult:
    authors = _join_authors(data.authors, mode)
    if not authors:
        raise ValueError("Author(s) required for journal articles.")

    article_title = _clean_spaces(data.article_title)
    journal_plain, journal_html = _italics_text(data.journal_title)

    year_wrap = f"[{data.year}]" if data.year_in_square_brackets else f"({data.year})"

    vol_issue = ""
    if data.volume:
        vol_issue = f"{data.volume}"
        if data.issue:
            vol_issue += f"({data.issue})"

    start = _clean_spaces(data.starting_page)
    pin = f", {data.pinpoint}" if data.pinpoint else ""

    core = f"{authors}, '{article_title}' {year_wrap}"
    if vol_issue:
        core += f" {vol_issue}"
    core += f" {journal_plain} {start}{pin}"

    text = _ensure_ends_with_period(core)

    html = f"{_escape_html(authors)}, '{_escape_html(article_title)}' {_escape_html(year_wrap)}"
    if vol_issue:
        html += f" {_escape_html(vol_issue)}"
    html += f" {journal_html} {_escape_html(start)}{_escape_html(pin)}"
    html = _ensure_ends_with_period(html)

    # Online add-ons (minimal)
    if data.is_online:
        if not data.url or not data.access_date:
            raise ValueError("Online journal articles require URL and access date.")
        text = text[:-1] + f" <{data.url}> accessed {data.access_date}."
        html = html[:-1] + f" &lt;{_escape_html(data.url)}&gt; accessed {_escape_html(data.access_date)}."

    return CitationResult(source_type=SourceType.journal_article, mode=mode, text=text, html=html)


def _format_book(data: BookCitation, mode: str) -> CitationResult:
    authors = _join_authors(data.authors, mode)
    if not authors:
        raise ValueError("Author(s) required for books.")

    title_plain, title_html = _italics_text(data.title)
    pub = _clean_spaces(data.publisher)
    yr = data.year
    edition = _clean_spaces(data.edition) if data.edition else ""
    pin = _clean_spaces(data.pinpoint) if data.pinpoint else ""

    inside = f"{pub}"
    if edition:
        inside += f", {edition}"
    inside += f", {yr}"

    base_text = f"{authors}, {title_plain} ({inside})"
    base_html = f"{_escape_html(authors)}, {title_html} ({_escape_html(inside)})"

    if pin:
        base_text += f" {pin}"
        base_html += f" {_escape_html(pin)}"

    return CitationResult(source_type=SourceType.book, mode=mode, text=_ensure_ends_with_period(base_text), html=_ensure_ends_with_period(base_html))


def _format_book_chapter(data: BookChapterCitation, mode: str) -> CitationResult:
    ch_authors = _join_authors(data.chapter_authors, mode)
    if not ch_authors:
        raise ValueError("Chapter author(s) required.")

    editors = _join_authors(data.editors, mode)
    if not editors:
        raise ValueError("Editor(s) required for book chapters.")

    ch_title = _clean_spaces(data.chapter_title)
    book_plain, book_html = _italics_text(data.book_title)

    pub = _clean_spaces(data.publisher)
    yr = data.year
    edition = _clean_spaces(data.edition) if data.edition else ""
    start = _clean_spaces(data.starting_page) if data.starting_page else ""
    pin = _clean_spaces(data.pinpoint) if data.pinpoint else ""

    inside = f"{pub}"
    if edition:
        inside += f", {edition}"
    inside += f", {yr}"

    base_text = f"{ch_authors}, '{ch_title}' in {editors} (eds), {book_plain} ({inside})"
    base_html = f"{_escape_html(ch_authors)}, '{_escape_html(ch_title)}' in {_escape_html(editors)} (eds), {book_html} ({_escape_html(inside)})"

    if start:
        base_text += f" {start}"
        base_html += f" {_escape_html(start)}"
    if pin:
        base_text += f", {pin}"
        base_html += f", {_escape_html(pin)}"

    return CitationResult(source_type=SourceType.book_chapter, mode=mode, text=_ensure_ends_with_period(base_text), html=_ensure_ends_with_period(base_html))


def _format_media(data: MediaArticleCitation, mode: str) -> CitationResult:
    author = ""
    if data.org_as_author:
        author = _clean_spaces(data.org_as_author)
    else:
        author = _join_authors(data.authors, mode)

    if not author:
        raise ValueError("Provide an author or an organisation as author for media articles.")

    title = _clean_spaces(data.article_title)
    paper_plain, paper_html = _italics_text(data.newspaper_title)

    city = _clean_spaces(data.city) if data.city else ""
    date = _clean_spaces(data.date)
    page = _clean_spaces(data.page) if data.page else ""

    # AGLC: Author, 'Title' (Newspaper, City, Date) page.
    paren_bits = [paper_plain]
    if city:
        paren_bits.append(city)
    paren_bits.append(date)
    paren = ", ".join(paren_bits)

    text = f"{author}, '{title}' ({paren})"
    html = f"{_escape_html(author)}, '{_escape_html(title)}' ({paper_html}"
    if city:
        html += f", {_escape_html(city)}"
    html += f", {_escape_html(date)})"

    if page:
        text += f" {page}"
        html += f" {_escape_html(page)}"

    text = _ensure_ends_with_period(text)
    html = _ensure_ends_with_period(html)

    if data.is_online:
        if not data.url or not data.access_date:
            raise ValueError("Online media articles require URL and access date.")
        text = text[:-1] + f" <{data.url}> accessed {data.access_date}."
        html = html[:-1] + f" &lt;{_escape_html(data.url)}&gt; accessed {_escape_html(data.access_date)}."

    return CitationResult(source_type=SourceType.media_article, mode=mode, text=text, html=html)


def _format_report(data: ReportCitation, mode: str) -> CitationResult:
    author = _clean_spaces(data.author_or_org)
    title_plain, title_html = _italics_text(data.title)

    series = _clean_spaces(data.report_number_or_series) if data.report_number_or_series else ""
    publisher = _clean_spaces(data.publisher) if data.publisher else ""
    place = _clean_spaces(data.place) if data.place else ""
    date = _clean_spaces(data.date) if data.date else ""
    pin = _clean_spaces(data.pinpoint) if data.pinpoint else ""

    bits = []
    if series:
        bits.append(series)
    if publisher:
        bits.append(publisher)
    if place:
        bits.append(place)
    if date:
        bits.append(date)

    inside = ", ".join(bits) if bits else ""

    base_text = f"{author}, {title_plain}"
    base_html = f"{_escape_html(author)}, {title_html}"

    if inside:
        base_text += f" ({inside})"
        base_html += f" ({_escape_html(inside)})"

    if pin:
        base_text += f" {pin}"
        base_html += f" {_escape_html(pin)}"

    text = _ensure_ends_with_period(base_text)
    html = _ensure_ends_with_period(base_html)

    if data.is_online:
        if not data.url or not data.access_date:
            raise ValueError("Online reports require URL and access date.")
        text = text[:-1] + f" <{data.url}> accessed {data.access_date}."
        html = html[:-1] + f" &lt;{_escape_html(data.url)}&gt; accessed {_escape_html(data.access_date)}."

    return CitationResult(source_type=SourceType.report, mode=mode, text=text, html=html)


def _format_website(data: WebsiteCitation, mode: str) -> CitationResult:
    author = _clean_spaces(data.author_or_org) if data.author_or_org else ""
    title = _clean_spaces(data.page_title)
    site = _clean_spaces(data.site_name)
    date = _clean_spaces(data.date) if data.date else ""
    url = _clean_spaces(data.url)
    access = _clean_spaces(data.access_date) if data.access_date else ""

    if not url:
        raise ValueError("URL is required for website citations.")

    # AGLC-ish: Author, 'Title' (Site, Date) <url> accessed ...
    who = author if author else site

    if date:
        text = f"{who}, '{title}' ({site}, {date}) <{url}>"
        html = f"{_escape_html(who)}, '{_escape_html(title)}' ({_escape_html(site)}, {_escape_html(date)}) &lt;{_escape_html(url)}&gt;"
    else:
        text = f"{who}, '{title}' ({site}) <{url}>"
        html = f"{_escape_html(who)}, '{_escape_html(title)}' ({_escape_html(site)}) &lt;{_escape_html(url)}&gt;"

    if access:
        text += f" accessed {access}"
        html += f" accessed {_escape_html(access)}"

    return CitationResult(source_type=SourceType.website, mode=mode, text=_ensure_ends_with_period(text), html=_ensure_ends_with_period(html))


# ---------------------------
# Public entry point
# ---------------------------

def format_citation(source_type: SourceType | str, data: Dict[str, Any], mode: str = "footnote") -> CitationResult:
    st = SourceType(str(source_type))
    mode_clean = (mode or "footnote").strip().lower()
    if mode_clean not in ("footnote", "bibliography"):
        raise ValueError("Mode must be footnote or bibliography.")

    if st == SourceType.case:
        m = CaseCitation(**data)
        return _format_case(m, mode_clean)

    if st == SourceType.legislation:
        m = LegislationCitation(**data)
        return _format_legislation(m, mode_clean)

    if st == SourceType.journal_article:
        m = JournalArticleCitation(**data)
        return _format_journal(m, mode_clean)

    if st == SourceType.book:
        m = BookCitation(**data)
        return _format_book(m, mode_clean)

    if st == SourceType.book_chapter:
        m = BookChapterCitation(**data)
        return _format_book_chapter(m, mode_clean)

    if st == SourceType.media_article:
        m = MediaArticleCitation(**data)
        return _format_media(m, mode_clean)

    if st == SourceType.report:
        m = ReportCitation(**data)
        return _format_report(m, mode_clean)

    if st == SourceType.website:
        m = WebsiteCitation(**data)
        return _format_website(m, mode_clean)

    raise ValueError("Unsupported source type.")
