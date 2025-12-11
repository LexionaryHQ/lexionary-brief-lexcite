from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ValidationError, field_validator


class SourceType(str, Enum):
    CASE = "case"
    LEGISLATION = "legislation"
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    MEDIA_ARTICLE = "media_article"
    REPORT = "report"
    WEBSITE = "website"


@dataclass
class CitationResult:
    source_type: SourceType
    mode: str  # "footnote" or "bibliography"
    text: str
    html: str


# ---------- CASE ----------


class CaseCitation(BaseModel):
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
    pinpoint_type: Optional[str] = None  # "page" | "paragraph" | None
    pinpoint: Optional[str] = None

    @field_validator("pinpoint_type")
    @classmethod
    def validate_pinpoint_type(cls, v):
        if v is None or v == "":
            return None
        if v not in {"page", "paragraph"}:
            raise ValueError("pinpoint_type must be 'page', 'paragraph', or null")
        return v


def format_case(data: dict, mode: str) -> CitationResult:
    c = CaseCitation(**data)

    # Italicised case name
    italic_name = f"<i>{c.case_name}</i>"

    # Neutral citation if we have court + decision
    neutral_part = None
    if c.court and c.decision_number:
        neutral_part = f"[{c.year}] {c.court} {c.decision_number}"

    # Reported citation if we have volume + reporter + first page
    report_part = None
    if c.volume and c.reporter and c.first_page:
        if c.reporter_series_by_year:
            # Year-based series: [1998] 194 CLR 1
            report_part = f"[{c.year}] {c.volume} {c.reporter} {c.first_page}"
        else:
            # Standard: (1992) 175 CLR 1
            report_part = f"({c.year}) {c.volume} {c.reporter} {c.first_page}"

    segments_text: List[str] = [c.case_name]
    segments_html: List[str] = [italic_name]

    if c.unreported:
        # Unreported: neutral only
        if neutral_part:
            segments_text.append(neutral_part)
            segments_html.append(neutral_part)
        else:
            # Fallback: at least put year
            segments_text.append(f"[{c.year}]")
            segments_html.append(f"[{c.year}]")
    else:
        # Reported / mixed scenarios
        if neutral_part and report_part:
            # Both neutral and report
            if c.neutral_citation_first:
                segments_text.append(neutral_part)
                segments_text.append(report_part)
                segments_html.append(neutral_part)
                segments_html.append(report_part)
            else:
                segments_text.append(report_part)
                segments_text.append(neutral_part)
                segments_html.append(report_part)
                segments_html.append(neutral_part)
        elif report_part:
            # Report only
            segments_text.append(report_part)
            segments_html.append(report_part)
        elif neutral_part:
            # Neutral only
            segments_text.append(neutral_part)
            segments_html.append(neutral_part)
        else:
            # Very minimal: name + year
            segments_text.append(f"({c.year})")
            segments_html.append(f"({c.year})")

    # Pinpoint logic
    pin_str_text = ""
    if c.pinpoint and c.pinpoint.strip():
        if c.pinpoint_type == "paragraph":
            # AGLC: [150]
            pin_str_text = f"[{c.pinpoint.strip()}]"
        else:
            # Page or generic: 42
            pin_str_text = c.pinpoint.strip()

    text = " ".join(seg for seg in segments_text if seg)
    html = " ".join(seg for seg in segments_html if seg)

    if pin_str_text:
        text = f"{text}, {pin_str_text}"
        html = f"{html}, {pin_str_text}"

    return CitationResult(source_type=SourceType.CASE, mode=mode, text=text, html=html)


# ---------- LEGISLATION ----------


class LegislationCitation(BaseModel):
    title: str
    year: str
    jurisdiction: str
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None  # s, ss, pt, sch etc
    pinpoint_number: Optional[str] = None


def format_legislation(data: dict, mode: str) -> CitationResult:
    l = LegislationCitation(**data)

    full_title = f"{l.title} {l.year}"
    italic_title = f"<i>{l.title} {l.year}</i>"

    base_text = f"{full_title} ({l.jurisdiction})"
    base_html = f"{italic_title} ({l.jurisdiction})"

    if l.is_bill:
        full_title_bill = f"{l.title} Bill {l.year}"
        base_text = f"{full_title_bill} ({l.jurisdiction})"
        base_html = f"<i>{full_title_bill}</i> ({l.jurisdiction})"

    pin = ""
    if l.pinpoint_unit and l.pinpoint_number:
        unit = l.pinpoint_unit.strip()
        num = l.pinpoint_number.strip()
        pin = f"{unit} {num}"

    if pin:
        base_text = f"{base_text} {pin}"
        base_html = f"{base_html} {pin}"

    return CitationResult(source_type=SourceType.LEGISLATION, mode=mode, text=base_text, html=base_html)


# ---------- JOURNAL ARTICLE ----------


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


def format_authors(authors: List[str]) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al"


def format_journal_article(data: dict, mode: str) -> CitationResult:
    j = JournalArticleCitation(**data)
    author_str = format_authors(j.authors)

    year_part = f"[{j.year}]" if j.year_in_square_brackets else f"({j.year})"

    vol_issue = ""
    if j.volume and j.issue:
        vol_issue = f"{j.volume}({j.issue})"
    elif j.volume:
        vol_issue = j.volume
    elif j.issue:
        vol_issue = j.issue

    parts_text: List[str] = []
    parts_html: List[str] = []

    if author_str:
        parts_text.append(f"{author_str},")
        parts_html.append(f"{author_str},")
    parts_text.append(f"'{j.article_title}'")
    parts_html.append(f"'{j.article_title}'")

    core = year_part
    if vol_issue:
        core = f"{year_part} {vol_issue}"

    parts_text.append(core)
    parts_text.append(j.journal_title)
    parts_html.append(core)
    parts_html.append(f"<i>{j.journal_title}</i>")

    parts_text.append(j.starting_page)
    parts_html.append(j.starting_page)

    if j.pinpoint:
        parts_text[-1] = f"{parts_text[-1]}, {j.pinpoint}"
        parts_html[-1] = f"{parts_html[-1]}, {j.pinpoint}"

    text = " ".join(p for p in parts_text if p)
    html = " ".join(p for p in parts_html if p)

    if j.is_online and j.url:
        tail = f"<{j.url}>"
        if j.access_date:
            tail = f"{tail} accessed {j.access_date}"
        text = f"{text} {tail}"
        html = f"{html} {tail}"

    return CitationResult(source_type=SourceType.JOURNAL_ARTICLE, mode=mode, text=text, html=html)


# ---------- BOOK ----------


class BookCitation(BaseModel):
    authors: List[str]
    title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    pinpoint: Optional[str] = None


def format_book(data: dict, mode: str) -> CitationResult:
    b = BookCitation(**data)
    author_str = format_authors(b.authors)

    pub_parts = [b.publisher]
    if b.edition:
        pub_parts.append(b.edition)
    pub_parts.append(b.year)
    pub_segment = ", ".join(pub_parts)

    text = f"{author_str}, {b.title} ({pub_segment})"
    html = f"{author_str}, <i>{b.title}</i> ({pub_segment})"

    if b.pinpoint:
        text = f"{text} {b.pinpoint}"
        html = f"{html} {b.pinpoint}"

    return CitationResult(source_type=SourceType.BOOK, mode=mode, text=text, html=html)


# ---------- BOOK CHAPTER ----------


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


def format_book_chapter(data: dict, mode: str) -> CitationResult:
    c = BookChapterCitation(**data)
    chapter_authors = format_authors(c.chapter_authors)
    editors = format_authors(c.editors)

    text_parts: List[str] = []
    html_parts: List[str] = []

    if chapter_authors:
        text_parts.append(f"{chapter_authors},")
        html_parts.append(f"{chapter_authors},")
    text_parts.append(f"'{c.chapter_title}',")
    html_parts.append(f"'{c.chapter_title}',")

    if editors:
        ed_suffix = "ed" if len(c.editors) == 1 else "eds"
        text_parts.append(f"in {editors} ({ed_suffix}),")
        html_parts.append(f"in {editors} ({ed_suffix}),")

    pub_parts = [c.publisher]
    if c.edition:
        pub_parts.append(c.edition)
    pub_parts.append(c.year)
    pub_segment = ", ".join(pub_parts)

    text_parts.append(f"{c.book_title} ({pub_segment})")
    html_parts.append(f"<i>{c.book_title}</i> ({pub_segment})")

    if c.starting_page:
        text_parts[-1] = f"{text_parts[-1]} {c.starting_page}"
        html_parts[-1] = f"{html_parts[-1]} {c.starting_page}"

    if c.pinpoint:
        text_parts[-1] = f"{text_parts[-1]}, {c.pinpoint}"
        html_parts[-1] = f"{html_parts[-1]}, {c.pinpoint}"

    text = " ".join(text_parts)
    html = " ".join(html_parts)
    return CitationResult(source_type=SourceType.BOOK_CHAPTER, mode=mode, text=text, html=html)


# ---------- MEDIA ARTICLE ----------


class MediaArticleCitation(BaseModel):
    authors: List[str] = []
    org_as_author: Optional[str] = None
    article_title: str
    newspaper_title: str
    city: Optional[str] = None
    date: str
    page: Optional[str] = None
    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None


def format_media_article(data: dict, mode: str) -> CitationResult:
    m = MediaArticleCitation(**data)

    author_str = ""
    if m.authors:
        author_str = format_authors(m.authors)
    elif m.org_as_author:
        author_str = m.org_as_author

    parts_text: List[str] = []
    parts_html: List[str] = []

    if author_str:
        parts_text.append(f"{author_str},")
        parts_html.append(f"{author_str},")
    parts_text.append(f"'{m.article_title}',")
    parts_html.append(f"'{m.article_title}',")

    if m.city:
        loc_text = f"{m.newspaper_title} ({m.city}, {m.date})"
        loc_html = f"<i>{m.newspaper_title}</i> ({m.city}, {m.date})"
    else:
        loc_text = f"{m.newspaper_title} ({m.date})"
        loc_html = f"<i>{m.newspaper_title}</i> ({m.date})"

    parts_text.append(loc_text)
    parts_html.append(loc_html)

    if m.page:
        parts_text[-1] = f"{parts_text[-1]} {m.page}"
        parts_html[-1] = f"{parts_html[-1]} {m.page}"

    text = " ".join(parts_text)
    html = " ".join(parts_html)

    if m.is_online and m.url:
        tail = f"<{m.url}>"
        if m.access_date:
            tail = f"{tail} accessed {m.access_date}"
        text = f"{text} {tail}"
        html = f"{html} {tail}"

    return CitationResult(source_type=SourceType.MEDIA_ARTICLE, mode=mode, text=text, html=html)


# ---------- REPORT ----------


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


def format_report(data: dict, mode: str) -> CitationResult:
    r = ReportCitation(**data)

    parts_text: List[str] = [f"{r.author_or_org},"]



::contentReference[oaicite:0]{index=0}
