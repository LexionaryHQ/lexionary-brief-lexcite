# aglc_engine.py
# Lexcite AGLC4 Engine for Lexionary
# Version: 2.0.0
# - Builder-first AGLC formatting with strict required fields per source type
# - Normalises common casing issues (CLR, HCA, NSW, Cth, etc)
# - Returns BOTH plain text and safe HTML (italics via <i> only)
#
# Notes:
# - This is not a full AGLC4 treatise. It is a pragmatic, accuracy-first engine for common student citations.
# - Italics are returned as <i> tags only so the frontend can safely render and copy rich text.

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


# -------------------------------------------------------------------
# Public API types
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Normalisers
# -------------------------------------------------------------------

def _clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    x = str(s).strip()
    return x if x else None


def norm_reporter(rep: Optional[str]) -> Optional[str]:
    rep = _clean(rep)
    if not rep:
        return None
    return rep.upper()


def norm_court(court: Optional[str]) -> Optional[str]:
    court = _clean(court)
    if not court:
        return None
    return court.upper()


def norm_jurisdiction(j: Optional[str]) -> Optional[str]:
    j = _clean(j)
    if not j:
        return None
    up = j.upper()
    if up == "CTH":
        return "Cth"
    if up in {"NSW", "WA", "SA", "ACT", "NT"}:
        return up
    if up == "VIC":
        return "Vic"
    if up == "QLD":
        return "Qld"
    if up == "TAS":
        return "Tas"
    return j


def _ensure_trailing_full_stop(s: str) -> str:
    s2 = s.strip()
    if not s2.endswith("."):
        s2 += "."
    return s2


def _as_int_str(s: Optional[str]) -> Optional[str]:
    s = _clean(s)
    if not s:
        return None
    if not re.fullmatch(r"\d+", s):
        return s
    return s


def _ital(s: str) -> str:
    return f"<i>{s}</i>"


def _strip_outer_quotes(title: str) -> str:
    t = title.strip()
    t = t.strip('"').strip("'")
    return t.strip()


# -------------------------------------------------------------------
# Name formatting helpers
# -------------------------------------------------------------------

def _split_person_name(name: str) -> Tuple[str, str]:
    """
    Very small heuristic:
    "Jane Mary Smith" -> ("Jane Mary", "Smith")
    If only one token, treat as surname.
    """
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return ("", parts[0])
    given = " ".join(parts[:-1])
    surname = parts[-1]
    return (given, surname)


def format_authors_footnote(authors: List[str]) -> str:
    clean = [a.strip() for a in authors if a and a.strip()]
    return ", ".join(clean)


def format_authors_bibliography(authors: List[str]) -> str:
    clean = [a.strip() for a in authors if a and a.strip()]
    out: List[str] = []
    for a in clean:
        given, surname = _split_person_name(a)
        if surname and given:
            out.append(f"{surname}, {given}")
        else:
            out.append(a)
    return ", ".join(out)


def format_author_or_org_footnote(author_or_org: Optional[str]) -> str:
    a = _clean(author_or_org)
    return a or ""


def format_author_or_org_bibliography(author_or_org: Optional[str]) -> str:
    a = _clean(author_or_org)
    if not a:
        return ""
    given, surname = _split_person_name(a)
    if surname and given:
        return f"{surname}, {given}"
    return a


# -------------------------------------------------------------------
# Pydantic models for builder input
# -------------------------------------------------------------------

class CaseData(BaseModel):
    case_name: str
    year: str
    reporter_series_by_year: bool = False
    volume: Optional[str] = None
    reporter: Optional[str] = None
    first_page: Optional[str] = None
    court: Optional[str] = None
    decision_number: Optional[str] = None
    neutral_citation_first: bool = True
    unreported: bool = False
    pinpoint_type: Optional[str] = None   # "page" | "paragraph"
    pinpoint: Optional[str] = None

    @field_validator("case_name")
    @classmethod
    def _v_case_name(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("case_name is required")
        return v2

    @field_validator("year")
    @classmethod
    def _v_year(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("year must be 4 digits")
        return v2


class LegislationData(BaseModel):
    title: str
    year: str
    jurisdiction: str
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None   # "s" | "ss"
    pinpoint_number: Optional[str] = None

    @field_validator("title")
    @classmethod
    def _v_title(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("title is required")
        return v2

    @field_validator("year")
    @classmethod
    def _v_year(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("year must be 4 digits")
        return v2

    @field_validator("jurisdiction")
    @classmethod
    def _v_jur(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("jurisdiction is required")
        return v2


class JournalArticleData(BaseModel):
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

    @field_validator("authors")
    @classmethod
    def _v_auth(cls, v: List[str]) -> List[str]:
        if not v or not any(a.strip() for a in v):
            raise ValueError("authors is required")
        return [a.strip() for a in v if a and a.strip()]

    @field_validator("article_title", "journal_title", "starting_page")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2

    @field_validator("year")
    @classmethod
    def _v_year(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("year must be 4 digits")
        return v2


class BookData(BaseModel):
    authors: List[str]
    title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("authors")
    @classmethod
    def _v_auth(cls, v: List[str]) -> List[str]:
        if not v or not any(a.strip() for a in v):
            raise ValueError("authors is required")
        return [a.strip() for a in v if a and a.strip()]

    @field_validator("title", "publisher")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2

    @field_validator("year")
    @classmethod
    def _v_year(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("year must be 4 digits")
        return v2


class BookChapterData(BaseModel):
    chapter_authors: List[str]
    chapter_title: str
    editors: List[str]
    book_title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    starting_page: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("chapter_authors", "editors")
    @classmethod
    def _v_lists(cls, v: List[str]) -> List[str]:
        if not v or not any(a.strip() for a in v):
            raise ValueError("required list missing")
        return [a.strip() for a in v if a and a.strip()]

    @field_validator("chapter_title", "book_title", "publisher")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2

    @field_validator("year")
    @classmethod
    def _v_year(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not re.fullmatch(r"\d{4}", v2):
            raise ValueError("year must be 4 digits")
        return v2


class MediaArticleData(BaseModel):
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

    @field_validator("article_title", "newspaper_title", "date")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2


class ReportData(BaseModel):
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

    @field_validator("author_or_org", "title")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2


class WebsiteData(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str
    site_name: str
    date: Optional[str] = None
    url: str
    access_date: str

    @field_validator("page_title", "site_name", "url", "access_date")
    @classmethod
    def _v_req(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("required field missing")
        return v2


# -------------------------------------------------------------------
# Formatters
# -------------------------------------------------------------------

def _pin_case(pinpoint_type: Optional[str], pinpoint: Optional[str]) -> Tuple[str, str]:
    """
    Returns (plain_suffix, html_suffix) including leading punctuation if needed.
    """
    pt = _clean(pinpoint_type)
    pv = _clean(pinpoint)
    if not pt or not pv:
        return ("", "")
    if pt == "paragraph":
        return (f" [{pv}]", f" [{pv}]")
    if pt == "page":
        return (f" at {pv}", f" at {pv}")
    return ("", "")


def format_case(data: CaseData, mode: str) -> Tuple[str, str]:
    case_name_plain = data.case_name.strip()
    case_name_html = _ital(case_name_plain)

    year = data.year.strip()
    court = norm_court(data.court)
    decision = _clean(data.decision_number)

    reporter = norm_reporter(data.reporter)
    volume = _clean(data.volume)
    first_page = _clean(data.first_page)

    pin_plain, pin_html = _pin_case(data.pinpoint_type, data.pinpoint)

    neutral = ""
    neutral_html = ""
    if court and decision:
        neutral = f"[{year}] {court} {decision}"
        neutral_html = neutral

    reported = ""
    reported_html = ""
    if volume and reporter and first_page:
        if data.reporter_series_by_year:
            reported = f"[{year}] {volume} {reporter} {first_page}"
        else:
            reported = f"({year}) {volume} {reporter} {first_page}"
        reported_html = reported

    if data.unreported:
        if not neutral:
            raise ValueError("Unreported case requires court and decision number for neutral citation.")
        core_plain = f"{case_name_plain} {neutral}{pin_plain}"
        core_html = f"{case_name_html} {neutral_html}{pin_html}"
        return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))

    # If only one of neutral/reported exists
    if neutral and not reported:
        core_plain = f"{case_name_plain} {neutral}{pin_plain}"
        core_html = f"{case_name_html} {neutral_html}{pin_html}"
        return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))

    if reported and not neutral:
        core_plain = f"{case_name_plain} {reported}{pin_plain}"
        core_html = f"{case_name_html} {reported_html}{pin_html}"
        return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))

    # Both exist
    if data.neutral_citation_first:
        core_plain = f"{case_name_plain} {neutral}, {reported}{pin_plain}"
        core_html = f"{case_name_html} {neutral_html}, {reported_html}{pin_html}"
    else:
        core_plain = f"{case_name_plain} {reported}, {neutral}{pin_plain}"
        core_html = f"{case_name_html} {reported_html}, {neutral_html}{pin_html}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_legislation(data: LegislationData, mode: str) -> Tuple[str, str]:
    title = data.title.strip()
    year = data.year.strip()
    jur = norm_jurisdiction(data.jurisdiction) or data.jurisdiction.strip()

    title_year = f"{title} {year}"
    title_year_html = _ital(title_year)

    core_plain = f"{title_year} ({jur})"
    core_html = f"{title_year_html} ({jur})"

    pu = _clean(data.pinpoint_unit)
    pn = _clean(data.pinpoint_number)

    if pu and pn:
        pu2 = pu.strip()
        core_plain += f" {pu2} {pn}"
        core_html += f" {pu2} {pn}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_journal_article(data: JournalArticleData, mode: str) -> Tuple[str, str]:
    # Authors
    if mode == "bibliography":
        authors_plain = format_authors_bibliography(data.authors)
        authors_html = authors_plain
    else:
        authors_plain = format_authors_footnote(data.authors)
        authors_html = authors_plain

    article_title = _strip_outer_quotes(data.article_title)
    journal_title = data.journal_title.strip()

    year = data.year.strip()
    vol = _clean(data.volume)
    issue = _clean(data.issue)
    start = data.starting_page.strip()
    pin = _clean(data.pinpoint)

    year_token = f"[{year}]" if data.year_in_square_brackets else f"({year})"
    vol_token = f" {vol}" if vol else ""
    issue_token = f"({issue})" if issue else ""
    # AGLC common: (Year) Volume(Issue) Journal Page
    journal_part_plain = f"{year_token}{vol_token}{issue_token} {journal_title} {start}".strip()
    journal_part_html = f"{year_token}{vol_token}{issue_token} {_ital(journal_title)} {start}".strip()

    core_plain = f"{authors_plain}, '{article_title}' {journal_part_plain}"
    core_html = f"{authors_html}, '{article_title}' {journal_part_html}"

    if pin:
        core_plain += f", {pin}"
        core_html += f", {pin}"

    if data.is_online:
        url = _clean(data.url)
        access = _clean(data.access_date)
        if not url or not access:
            raise ValueError("Online journal article requires url and access_date.")
        core_plain += f" <{url}> accessed {access}"
        core_html += f" <{url}> accessed {access}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_book(data: BookData, mode: str) -> Tuple[str, str]:
    if mode == "bibliography":
        authors_plain = format_authors_bibliography(data.authors)
        authors_html = authors_plain
    else:
        authors_plain = format_authors_footnote(data.authors)
        authors_html = authors_plain

    title = data.title.strip()
    publisher = data.publisher.strip()
    year = data.year.strip()
    edition = _clean(data.edition)
    pin = _clean(data.pinpoint)

    paren_bits = [publisher]
    if edition:
        paren_bits.append(edition)
    paren_bits.append(year)

    paren = ", ".join(paren_bits)

    core_plain = f"{authors_plain}, {title} ({paren})"
    core_html = f"{authors_html}, {_ital(title)} ({paren})"

    if pin:
        core_plain += f" {pin}"
        core_html += f" {pin}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_book_chapter(data: BookChapterData, mode: str) -> Tuple[str, str]:
    if mode == "bibliography":
        chap_auth_plain = format_authors_bibliography(data.chapter_authors)
        eds_plain = format_authors_bibliography(data.editors)
        chap_auth_html = chap_auth_plain
        eds_html = eds_plain
    else:
        chap_auth_plain = format_authors_footnote(data.chapter_authors)
        eds_plain = format_authors_footnote(data.editors)
        chap_auth_html = chap_auth_plain
        eds_html = eds_plain

    chap_title = _strip_outer_quotes(data.chapter_title)
    book_title = data.book_title.strip()
    publisher = data.publisher.strip()
    year = data.year.strip()
    edition = _clean(data.edition)
    start = _clean(data.starting_page)
    pin = _clean(data.pinpoint)

    paren_bits = [publisher]
    if edition:
        paren_bits.append(edition)
    paren_bits.append(year)
    paren = ", ".join(paren_bits)

    start_bit_plain = f" {start}" if start else ""
    start_bit_html = start_bit_plain

    core_plain = f"{chap_auth_plain}, '{chap_title}' in {eds_plain} (ed), {book_title} ({paren}){start_bit_plain}"
    core_html = f"{chap_auth_html}, '{chap_title}' in {eds_html} (ed), {_ital(book_title)} ({paren}){start_bit_html}"

    if pin:
        core_plain += f", {pin}"
        core_html += f", {pin}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_media_article(data: MediaArticleData, mode: str) -> Tuple[str, str]:
    # AGLC: Author, 'Title', Newspaper (City, Date) page.
    authors = [a.strip() for a in (data.authors or []) if a and a.strip()]
    org = _clean(data.org_as_author)

    if org and authors:
        # If both provided, prefer explicit authors
        org = None

    if org:
        author_plain = org
        author_html = org
    else:
        if not authors:
            raise ValueError("Media article requires authors OR org_as_author.")
        author_plain = format_authors_bibliography(authors) if mode == "bibliography" else format_authors_footnote(authors)
        author_html = author_plain

    title = _strip_outer_quotes(data.article_title)
    paper = data.newspaper_title.strip()
    city = _clean(data.city)
    date = data.date.strip()
    page = _clean(data.page)

    paren_bits = []
    if city:
        paren_bits.append(city)
    paren_bits.append(date)
    paren = ", ".join(paren_bits)

    core_plain = f"{author_plain}, '{title}', {paper} ({paren})"
    core_html = f"{author_html}, '{title}', {_ital(paper)} ({paren})"

    if page:
        core_plain += f" {page}"
        core_html += f" {page}"

    if data.is_online:
        url = _clean(data.url)
        access = _clean(data.access_date)
        if not url or not access:
            raise ValueError("Online media article requires url and access_date.")
        core_plain += f" <{url}> accessed {access}"
        core_html += f" <{url}> accessed {access}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_report(data: ReportData, mode: str) -> Tuple[str, str]:
    author = data.author_or_org.strip()
    title = data.title.strip()
    series = _clean(data.report_number_or_series)
    publisher = _clean(data.publisher)
    place = _clean(data.place)
    date = _clean(data.date)
    pin = _clean(data.pinpoint)

    # Parenthetical: publisher, place, date (as available)
    paren_bits: List[str] = []
    if publisher:
        paren_bits.append(publisher)
    if place:
        paren_bits.append(place)
    if date:
        paren_bits.append(date)

    paren = ", ".join(paren_bits) if paren_bits else ""
    series_bit = f", {series}" if series else ""
    paren_bit = f" ({paren})" if paren else ""

    core_plain = f"{author}, {title}{series_bit}{paren_bit}"
    core_html = f"{author}, {_ital(title)}{series_bit}{paren_bit}"

    if pin:
        core_plain += f" {pin}"
        core_html += f" {pin}"

    if data.is_online:
        url = _clean(data.url)
        access = _clean(data.access_date)
        if not url or not access:
            raise ValueError("Online report requires url and access_date.")
        core_plain += f" <{url}> accessed {access}"
        core_html += f" <{url}> accessed {access}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


def format_website(data: WebsiteData, mode: str) -> Tuple[str, str]:
    author = _clean(data.author_or_org)
    page_title = _strip_outer_quotes(data.page_title)
    site = data.site_name.strip()
    date = _clean(data.date)
    url = data.url.strip()
    access = data.access_date.strip()

    if author:
        left_plain = f"{author}, '{page_title}'"
        left_html = left_plain
    else:
        left_plain = f"'{page_title}'"
        left_html = left_plain

    if date:
        core_plain = f"{left_plain} ({site}, {date}) <{url}> accessed {access}"
        core_html = f"{left_html} ({site}, {date}) <{url}> accessed {access}"
    else:
        core_plain = f"{left_plain} ({site}) <{url}> accessed {access}"
        core_html = f"{left_html} ({site}) <{url}> accessed {access}"

    return (_ensure_trailing_full_stop(core_plain), _ensure_trailing_full_stop(core_html))


# -------------------------------------------------------------------
# Main dispatcher
# -------------------------------------------------------------------

def format_citation(source_type: SourceType | str, data: Dict[str, Any], mode: str = "footnote") -> CitationResult:
    if isinstance(source_type, str):
        try:
            st = SourceType(source_type)
        except Exception:
            raise ValueError(f"Unsupported source_type: {source_type}")
    else:
        st = source_type

    m = (mode or "footnote").strip().lower()
    if m not in {"footnote", "bibliography"}:
        raise ValueError("mode must be 'footnote' or 'bibliography'")

    if st == SourceType.case:
        model = CaseData(**data)
        text, html = format_case(model, m)
    elif st == SourceType.legislation:
        model = LegislationData(**data)
        # normalise jurisdiction safely
        model.jurisdiction = norm_jurisdiction(model.jurisdiction) or model.jurisdiction
        text, html = format_legislation(model, m)
    elif st == SourceType.journal_article:
        model = JournalArticleData(**data)
        text, html = format_journal_article(model, m)
    elif st == SourceType.book:
        model = BookData(**data)
        text, html = format_book(model, m)
    elif st == SourceType.book_chapter:
        model = BookChapterData(**data)
        text, html = format_book_chapter(model, m)
    elif st == SourceType.media_article:
        model = MediaArticleData(**data)
        text, html = format_media_article(model, m)
    elif st == SourceType.report:
        model = ReportData(**data)
        text, html = format_report(model, m)
    elif st == SourceType.website:
        model = WebsiteData(**data)
        text, html = format_website(model, m)
    else:
        raise ValueError(f"Unsupported source_type: {st}")

    return CitationResult(source_type=st, mode=m, text=text, html=html)
