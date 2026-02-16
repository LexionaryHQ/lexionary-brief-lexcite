# aglc_engine.py
# Lexionary Lexcite AGLC4 engine (pragmatic, accuracy-first)
# Produces both plain text and safe HTML (italics via <i>).
# Version: 2.0.0

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from html import escape as html_escape
from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator


class SourceType(str, Enum):
    CASE = "case"
    LEGISLATION = "legislation"
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    MEDIA_ARTICLE = "media_article"
    REPORT = "report"
    WEBSITE = "website"


Mode = Literal["footnote", "bibliography"]


@dataclass
class CitationResult:
    source_type: SourceType
    mode: Mode
    text: str
    html: str


# -------------------------
# Helpers
# -------------------------

def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _end_full_stop(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s if s.endswith(".") else s + "."


def _italics_html(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return f"<i>{html_escape(s)}</i>"


def _plain(s: str) -> str:
    return html_escape(s)


def _join_authors_footnote(authors: List[str]) -> str:
    a = [x.strip() for x in authors if x and x.strip()]
    if not a:
        return ""
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return f"{a[0]} and {a[1]}"
    return ", ".join(a[:-1]) + f", and {a[-1]}"


def _invert_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    parts = name.split()
    if len(parts) == 1:
        return name
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _join_authors_bibliography(authors: List[str]) -> str:
    a = [x.strip() for x in authors if x and x.strip()]
    if not a:
        return ""
    inv = [_invert_name(x) for x in a]
    if len(inv) == 1:
        return inv[0]
    if len(inv) == 2:
        return f"{inv[0]} and {inv[1]}"
    return ", ".join(inv[:-1]) + f", and {inv[-1]}"


def _maybe_brackets_year(year: str, in_square: bool) -> str:
    y = year.strip()
    if not y:
        return ""
    return f"[{y}]" if in_square else f"({y})"


def _pinpoint_suffix(pinpoint_type: Optional[str], pinpoint: Optional[str]) -> str:
    pt = _clean(pinpoint_type).lower()
    pv = _clean(pinpoint)
    if not pt or not pv:
        return ""
    if pt == "paragraph":
        # AGLC paragraphs are [123]
        pv2 = pv.strip("[]").strip()
        return f" [{pv2}]"
    if pt == "page":
        # page pinpoints are just page number
        return f" {pv}"
    return ""


def _s_or_ss(unit: str) -> str:
    u = unit.strip().lower()
    if u in ("s", "sec", "section"):
        return "s"
    if u in ("ss", "secs", "sections"):
        return "ss"
    return unit.strip()  # allow custom


# -------------------------
# Models
# -------------------------

class CaseData(BaseModel):
    case_name: str = Field(..., min_length=2)
    year: Optional[str] = Field(None)
    reporter_series_by_year: bool = False
    volume: Optional[str] = None
    reporter: Optional[str] = None
    first_page: Optional[str] = None

    court: Optional[str] = None
    decision_number: Optional[str] = None

    neutral_citation_first: bool = True
    unreported: bool = False

    pinpoint_type: Optional[Literal["page", "paragraph"]] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("year must be YYYY")
        return v

    @field_validator("volume", "first_page", "decision_number")
    @classmethod
    def _digits_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not re.fullmatch(r"\d{1,5}", v):
            raise ValueError("must be numeric")
        return v

    @field_validator("court", "reporter")
    @classmethod
    def _upperish(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        return v


class LegislationData(BaseModel):
    title: str = Field(..., min_length=2)
    year: str = Field(..., min_length=4)
    jurisdiction: str = Field(..., min_length=2)
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None
    pinpoint_number: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("year must be YYYY")
        return v

    @field_validator("jurisdiction")
    @classmethod
    def _jur_ok(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("jurisdiction required")
        return v


class JournalArticleData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    article_title: str = Field(..., min_length=2)
    year: str = Field(..., min_length=4)
    year_in_square_brackets: bool = False
    volume: Optional[str] = None
    issue: Optional[str] = None
    journal_title: str = Field(..., min_length=2)
    starting_page: str = Field(..., min_length=1)
    pinpoint: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("year must be YYYY")
        return v


class BookData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    title: str = Field(..., min_length=2)
    publisher: str = Field(..., min_length=2)
    year: str = Field(..., min_length=4)
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("year must be YYYY")
        return v


class BookChapterData(BaseModel):
    chapter_authors: List[str] = Field(default_factory=list)
    chapter_title: str = Field(..., min_length=2)
    editors: List[str] = Field(default_factory=list)
    book_title: str = Field(..., min_length=2)
    publisher: str = Field(..., min_length=2)
    year: str = Field(..., min_length=4)
    edition: Optional[str] = None
    starting_page: Optional[str] = None
    pinpoint: Optional[str] = None

    @field_validator("year")
    @classmethod
    def _year_ok(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError("year must be YYYY")
        return v


class MediaArticleData(BaseModel):
    authors: List[str] = Field(default_factory=list)
    org_as_author: Optional[str] = None
    article_title: str = Field(..., min_length=2)
    newspaper_title: str = Field(..., min_length=2)
    city: Optional[str] = None
    date: str = Field(..., min_length=4)  # keep flexible: "1 January 2025"
    page: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class ReportData(BaseModel):
    author_or_org: str = Field(..., min_length=2)
    title: str = Field(..., min_length=2)
    report_number_or_series: Optional[str] = None
    publisher: Optional[str] = None
    place: Optional[str] = None
    date: Optional[str] = None
    pinpoint: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


class WebsiteData(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str = Field(..., min_length=2)
    site_name: str = Field(..., min_length=2)
    date: Optional[str] = None
    url: str = Field(..., min_length=6)
    access_date: Optional[str] = None


DataUnion = Union[
    CaseData,
    LegislationData,
    JournalArticleData,
    BookData,
    BookChapterData,
    MediaArticleData,
    ReportData,
    WebsiteData,
]


def _validate_data(source_type: SourceType, data: dict) -> DataUnion:
    if source_type == SourceType.CASE:
        return CaseData(**data)
    if source_type == SourceType.LEGISLATION:
        return LegislationData(**data)
    if source_type == SourceType.JOURNAL_ARTICLE:
        return JournalArticleData(**data)
    if source_type == SourceType.BOOK:
        return BookData(**data)
    if source_type == SourceType.BOOK_CHAPTER:
        return BookChapterData(**data)
    if source_type == SourceType.MEDIA_ARTICLE:
        return MediaArticleData(**data)
    if source_type == SourceType.REPORT:
        return ReportData(**data)
    if source_type == SourceType.WEBSITE:
        return WebsiteData(**data)
    raise ValueError("unsupported source_type")


# -------------------------
# Formatters
# -------------------------

def _format_case(d: CaseData, mode: Mode) -> CitationResult:
    case_name_txt = d.case_name.strip()
    case_name_html = _italics_html(case_name_txt)

    neutral = ""
    if d.year and d.court and d.decision_number:
        neutral = f"[{d.year}] {d.court.strip()} {d.decision_number.strip()}"

    reported = ""
    if d.year and d.volume and d.reporter and d.first_page:
        # reported year in () or []
        y = _maybe_brackets_year(d.year, d.reporter_series_by_year)
        reported = f"{y} {d.volume.strip()} {d.reporter.strip()} {d.first_page.strip()}"

    if d.unreported and not neutral:
        raise ValueError("Unreported case requires neutral citation (year, court, decision number).")

    if not neutral and not reported:
        raise ValueError("Provide either a neutral citation (year, court, decision number) or a reported citation.")

    pin = _pinpoint_suffix(d.pinpoint_type, d.pinpoint)

    pieces_txt: List[str] = []
    pieces_html: List[str] = []

    # AGLC: case name italicised, citations not italicised
    if neutral and reported:
        if d.neutral_citation_first:
            pieces_txt = [f"{case_name_txt} {neutral}", reported]
            pieces_html = [f"{case_name_html} {_plain(' ' + neutral).strip()}", _plain(reported)]
        else:
            pieces_txt = [f"{case_name_txt} {reported}", neutral]
            pieces_html = [f"{case_name_html} {_plain(' ' + reported).strip()}", _plain(neutral)]
    elif neutral:
        pieces_txt = [f"{case_name_txt} {neutral}"]
        pieces_html = [f"{case_name_html} {_plain(' ' + neutral).strip()}"]
    else:
        pieces_txt = [f"{case_name_txt} {reported}"]
        pieces_html = [f"{case_name_html} {_plain(' ' + reported).strip()}"]

    out_txt = ", ".join(pieces_txt) + pin
    out_html = ", ".join(pieces_html) + _plain(pin)

    return CitationResult(SourceType.CASE, mode, _end_full_stop(out_txt), _end_full_stop(out_html))


def _format_legislation(d: LegislationData, mode: Mode) -> CitationResult:
    title = d.title.strip()
    year = d.year.strip()
    jur = d.jurisdiction.strip()
    bill = " Bill" if d.is_bill else " Act"
    # Users often enter "Fair Work Act" already. Do not double "Act".
    if re.search(r"\b(Act|Bill|Regulations?)\b", title):
        base = f"{title} {year} ({jur})"
    else:
        base = f"{title}{bill} {year} ({jur})"

    suffix = ""
    if _clean(d.pinpoint_unit) and _clean(d.pinpoint_number):
        suffix = f" {_s_or_ss(d.pinpoint_unit)} {d.pinpoint_number.strip()}"

    txt = _end_full_stop(base + suffix)
    html = _end_full_stop(_plain(base + suffix))
    return CitationResult(SourceType.LEGISLATION, mode, txt, html)


def _format_journal(d: JournalArticleData, mode: Mode) -> CitationResult:
    authors_txt = _join_authors_bibliography(d.authors) if mode == "bibliography" else _join_authors_footnote(d.authors)
    if not authors_txt:
        raise ValueError("Journal article requires at least one author.")

    year_part = _maybe_brackets_year(d.year, d.year_in_square_brackets)

    vol_issue = ""
    if _clean(d.volume) and _clean(d.issue):
        vol_issue = f"{d.volume.strip()}({d.issue.strip()})"
    elif _clean(d.volume):
        vol_issue = d.volume.strip()

    journal_txt = d.journal_title.strip()
    journal_html = _italics_html(journal_txt)

    base_txt = f"{authors_txt}, '{d.article_title.strip()}' {year_part}"
    base_html = f"{_plain(authors_txt)}, '{_plain(d.article_title.strip())}' {_plain(year_part)}"

    mid_txt = f"{vol_issue} {journal_txt} {d.starting_page.strip()}".strip()
    mid_html = f"{_plain(vol_issue + ' ').strip()}{journal_html} {_plain(' ' + d.starting_page.strip()).strip()}".strip()

    pin = f" {_clean(d.pinpoint)}" if _clean(d.pinpoint) else ""

    txt = _end_full_stop(f"{base_txt} {mid_txt}{pin}".strip())
    html = _end_full_stop(f"{base_html} {mid_html}{_plain(pin)}".strip())
    return CitationResult(SourceType.JOURNAL_ARTICLE, mode, txt, html)


def _format_book(d: BookData, mode: Mode) -> CitationResult:
    authors_txt = _join_authors_bibliography(d.authors) if mode == "bibliography" else _join_authors_footnote(d.authors)
    if not authors_txt:
        raise ValueError("Book requires at least one author.")

    title_txt = d.title.strip()
    title_html = _italics_html(title_txt)

    inside_bits = [d.publisher.strip()]
    if _clean(d.edition):
        inside_bits.append(d.edition.strip())
    inside_bits.append(d.year.strip())
    inside = ", ".join(inside_bits)

    pin = f" {_clean(d.pinpoint)}" if _clean(d.pinpoint) else ""

    txt = _end_full_stop(f"{authors_txt}, {title_txt} ({inside}){pin}")
    html = _end_full_stop(f"{_plain(authors_txt)}, {title_html} ({_plain(inside)}){_plain(pin)}")
    return CitationResult(SourceType.BOOK, mode, txt, html)


def _format_book_chapter(d: BookChapterData, mode: Mode) -> CitationResult:
    ca_txt = _join_authors_bibliography(d.chapter_authors) if mode == "bibliography" else _join_authors_footnote(d.chapter_authors)
    if not ca_txt:
        raise ValueError("Book chapter requires at least one chapter author.")

    ed_txt = _join_authors_footnote(d.editors)
    if not ed_txt:
        raise ValueError("Book chapter requires editor(s).")

    book_title_txt = d.book_title.strip()
    book_title_html = _italics_html(book_title_txt)

    inside_bits = [d.publisher.strip()]
    if _clean(d.edition):
        inside_bits.append(d.edition.strip())
    inside_bits.append(d.year.strip())
    inside = ", ".join(inside_bits)

    start = f", {d.starting_page.strip()}" if _clean(d.starting_page) else ""
    pin = f" {_clean(d.pinpoint)}" if _clean(d.pinpoint) else ""

    txt = _end_full_stop(
        f"{ca_txt}, '{d.chapter_title.strip()}' in {ed_txt} (ed{'s' if len(d.editors) != 1 else ''}), "
        f"{book_title_txt} ({inside}){start}{pin}"
    )
    html = _end_full_stop(
        f"{_plain(ca_txt)}, '{_plain(d.chapter_title.strip())}' in {_plain(ed_txt)} (ed{'s' if len(d.editors) != 1 else ''}), "
        f"{book_title_html} ({_plain(inside)}){_plain(start)}{_plain(pin)}"
    )
    return CitationResult(SourceType.BOOK_CHAPTER, mode, txt, html)


def _format_media(d: MediaArticleData, mode: Mode) -> CitationResult:
    authors_txt = _join_authors_bibliography(d.authors) if mode == "bibliography" else _join_authors_footnote(d.authors)
    org = _clean(d.org_as_author)

    author_part = authors_txt if authors_txt else org
    if not author_part:
        raise ValueError("Media article requires author(s) or an organisation author.")

    paper_txt = d.newspaper_title.strip()
    paper_html = _italics_html(paper_txt)

    city = f", {d.city.strip()}" if _clean(d.city) else ""
    page = f", {d.page.strip()}" if _clean(d.page) else ""

    txt = _end_full_stop(f"{author_part}, '{d.article_title.strip()}', {paper_txt}{city}, {d.date.strip()}{page}")
    html = _end_full_stop(f"{_plain(author_part)}, '{_plain(d.article_title.strip())}', {paper_html}{_plain(city)}, {_plain(d.date.strip())}{_plain(page)}")
    return CitationResult(SourceType.MEDIA_ARTICLE, mode, txt, html)


def _format_report(d: ReportData, mode: Mode) -> CitationResult:
    author = d.author_or_org.strip()
    title_txt = d.title.strip()
    title_html = _italics_html(title_txt)

    inside_bits = []
    if _clean(d.report_number_or_series):
        inside_bits.append(d.report_number_or_series.strip())
    if _clean(d.publisher):
        inside_bits.append(d.publisher.strip())
    if _clean(d.place):
        inside_bits.append(d.place.strip())
    if _clean(d.date):
        inside_bits.append(d.date.strip())
    inside = ", ".join(inside_bits) if inside_bits else ""

    pin = f" {_clean(d.pinpoint)}" if _clean(d.pinpoint) else ""

    if inside:
        txt = _end_full_stop(f"{author}, {title_txt} ({inside}){pin}")
        html = _end_full_stop(f"{_plain(author)}, {title_html} ({_plain(inside)}){_plain(pin)}")
    else:
        txt = _end_full_stop(f"{author}, {title_txt}{pin}")
        html = _end_full_stop(f"{_plain(author)}, {title_html}{_plain(pin)}")

    return CitationResult(SourceType.REPORT, mode, txt, html)


def _format_website(d: WebsiteData, mode: Mode) -> CitationResult:
    author = _clean(d.author_or_org)
    if not author:
        # Organisation can be the site name, but keep distinction in case user supplied org in site_name
        author = d.site_name.strip()

    title = d.page_title.strip()
    site = d.site_name.strip()
    date = _clean(d.date)
    url = d.url.strip()
    access = _clean(d.access_date)

    if date:
        base_txt = f"{author}, '{title}' ({site}, {date}) <{url}>"
        base_html = f"{_plain(author)}, '{_plain(title)}' ({_plain(site)}, {_plain(date)}) &lt;{_plain(url)}&gt;"
    else:
        base_txt = f"{author}, '{title}' ({site}) <{url}>"
        base_html = f"{_plain(author)}, '{_plain(title)}' ({_plain(site)}) &lt;{_plain(url)}&gt;"

    if access:
        base_txt += f" accessed {access}"
        base_html += f" {_plain(' accessed ' + access)}"

    return CitationResult(SourceType.WEBSITE, mode, _end_full_stop(base_txt), _end_full_stop(base_html))


# -------------------------
# Public API
# -------------------------

def format_citation(source_type: Union[SourceType, str], data: dict, mode: Mode = "footnote") -> CitationResult:
    st = SourceType(source_type) if isinstance(source_type, str) else source_type
    if mode not in ("footnote", "bibliography"):
        raise ValueError("mode must be footnote or bibliography")

    d = _validate_data(st, data)

    if st == SourceType.CASE:
        return _format_case(d, mode)  # type: ignore[arg-type]
    if st == SourceType.LEGISLATION:
        return _format_legislation(d, mode)  # type: ignore[arg-type]
    if st == SourceType.JOURNAL_ARTICLE:
        return _format_journal(d, mode)  # type: ignore[arg-type]
    if st == SourceType.BOOK:
        return _format_book(d, mode)  # type: ignore[arg-type]
    if st == SourceType.BOOK_CHAPTER:
        return _format_book_chapter(d, mode)  # type: ignore[arg-type]
    if st == SourceType.MEDIA_ARTICLE:
        return _format_media(d, mode)  # type: ignore[arg-type]
    if st == SourceType.REPORT:
        return _format_report(d, mode)  # type: ignore[arg-type]
    if st == SourceType.WEBSITE:
        return _format_website(d, mode)  # type: ignore[arg-type]

    raise ValueError("unsupported source_type")
