# aglc_engine.py

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from pydantic import BaseModel, validator


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
    mode: str          # "footnote" or "bibliography"
    text: str          # plain text (no italics)
    html: str          # with <i> for italics


def join_authors(authors: List[str]) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al"


def italic_html(text: str) -> str:
    return f"<i>{text}</i>" if text else ""


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

    pinpoint_type: Optional[str] = None   # "page" or "paragraph"
    pinpoint: Optional[str] = None        # "7" or "45"

    @validator("case_name")
    def strip_case_name(cls, v: str) -> str:
        return v.strip()

    @validator("pinpoint_type")
    def validate_pinpoint_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"page", "paragraph"}:
            raise ValueError("pinpoint_type must be 'page' or 'paragraph'")
        return v

    def year_with_brackets(self) -> str:
        if self.reporter_series_by_year or self.unreported:
            return f"[{self.year}]"
        return f"({self.year})"

    def neutral_year(self) -> str:
        return f"[{self.year}]"


class LegislationCitation(BaseModel):
    title: str
    year: str
    jurisdiction: str
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None   # s, ss, pt, ch, div, sch
    pinpoint_number: Optional[str] = None

    @validator("title", "jurisdiction")
    def strip_fields(cls, v: str) -> str:
        return v.strip()


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

    @validator("authors", "article_title", "journal_title")
    def strip_strings(cls, v):
        if isinstance(v, list):
            return [s.strip() for s in v]
        return v.strip()


class BookCitation(BaseModel):
    authors: List[str]
    title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @validator("authors", "title", "publisher")
    def strip_strings(cls, v):
        if isinstance(v, list):
            return [s.strip() for s in v]
        return v.strip()


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

    @validator("chapter_authors", "editors", "chapter_title", "book_title", "publisher")
    def strip_strings(cls, v):
        if isinstance(v, list):
            return [s.strip() for s in v]
        return v.strip()


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

    @validator("article_title", "newspaper_title")
    def strip_strings(cls, v: str) -> str:
        return v.strip()


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

    @validator("author_or_org", "title")
    def strip_strings(cls, v: str) -> str:
        return v.strip()


class WebsiteCitation(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str
    site_name: str
    date: Optional[str] = None
    url: str
    access_date: str

    @validator("page_title", "site_name", "url", "access_date")
    def strip_strings(cls, v: str) -> str:
        return v.strip()


def _format_case(case: CaseCitation, mode: str) -> CitationResult:
    text_case = case.case_name
    html_case = italic_html(case.case_name)

    pinpoint_str = ""
    if case.pinpoint:
        if case.pinpoint_type == "paragraph":
            pinpoint_str = f" [para {case.pinpoint}]"
        elif case.pinpoint_type == "page":
            pinpoint_str = f" {case.pinpoint}"

    parts_text: List[str] = [text_case]
    parts_html: List[str] = [html_case]

    neutral_str_text = None
    neutral_str_html = None
    report_str_text = None
    report_str_html = None

    if case.court and case.decision_number:
        neutral_year = case.neutral_year()
        neutral_str_text = f"{neutral_year} {case.court} {case.decision_number}"
        neutral_str_html = neutral_str_text

    if case.reporter and case.first_page:
        year_part = case.year_with_brackets()
        vol = f" {case.volume}" if case.volume else ""
        report_str_text = f"{year_part}{vol} {case.reporter} {case.first_page}"
        report_str_html = report_str_text

    if case.unreported and neutral_str_text:
        parts_text.append(neutral_str_text)
        parts_html.append(neutral_str_html)
    elif neutral_str_text and report_str_text:
        if case.neutral_citation_first:
            parts_text.append(neutral_str_text)
            parts_text.append(report_str_text)
            parts_html.append(neutral_str_html)
            parts_html.append(report_str_html)
        else:
            parts_text.append(report_str_text)
            parts_text.append(neutral_str_text)
            parts_html.append(report_str_html)
            parts_html.append(neutral_str_html)
    elif report_str_text:
        parts_text.append(report_str_text)
        parts_html.append(report_str_html)

    base_text = " ".join(parts_text).strip()
    base_html = " ".join(parts_html).strip()

    if pinpoint_str:
        base_text = f"{base_text},{pinpoint_str}"
        base_html = f"{base_html},{pinpoint_str}"

    return CitationResult(
        source_type=SourceType.CASE,
        mode=mode,
        text=base_text,
        html=base_html,
    )


def _format_legislation(leg: LegislationCitation, mode: str) -> CitationResult:
    title_year = f"{leg.title} {leg.year}"
    if leg.is_bill:
        text_main = title_year
        html_main = title_year
    else:
        text_main = title_year
        html_main = italic_html(title_year)

    main_text = f"{text_main} ({leg.jurisdiction})"
    main_html = f"{html_main} ({leg.jurisdiction})"

    if leg.pinpoint_unit and leg.pinpoint_number:
        pin = f"{leg.pinpoint_unit} {leg.pinpoint_number}"
        main_text = f"{main_text} {pin}"
        main_html = f"{main_html} {pin}"

    return CitationResult(
        source_type=SourceType.LEGISLATION,
        mode=mode,
        text=main_text,
        html=main_html,
    )


def _format_journal_article(art: JournalArticleCitation, mode: str) -> CitationResult:
    authors_str = join_authors(art.authors)
    year_brackets = f"[{art.year}]" if art.year_in_square_brackets else f"({art.year})"
    vol_issue = ""
    if art.volume:
        vol_issue = art.volume
        if art.issue:
            vol_issue += f"({art.issue})"

    base_text = f"{authors_str}, '{art.article_title}', {year_brackets} {vol_issue} {art.journal_title} {art.starting_page}"
    base_html = f"{authors_str}, '{art.article_title}', {year_brackets} {vol_issue} {italic_html(art.journal_title)} {art.starting_page}"

    if art.pinpoint:
        base_text += f", {art.pinpoint}"
        base_html += f", {art.pinpoint}"

    if art.is_online and art.url and art.access_date:
        base_text += f", {art.url}, {art.access_date}"
        base_html += f", {art.url}, {art.access_date}"

    return CitationResult(
        source_type=SourceType.JOURNAL_ARTICLE,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def _format_book(book: BookCitation, mode: str) -> CitationResult:
    authors_str = join_authors(book.authors)

    if book.edition:
        pub_block = f"{book.publisher}, {book.edition}, {book.year}"
    else:
        pub_block = f"{book.publisher}, {book.year}"

    base_text = f"{authors_str}, {book.title} ({pub_block})"
    base_html = f"{authors_str}, {italic_html(book.title)} ({pub_block})"

    if book.pinpoint:
        base_text += f" {book.pinpoint}"
        base_html += f" {book.pinpoint}"

    return CitationResult(
        source_type=SourceType.BOOK,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def _format_book_chapter(ch: BookChapterCitation, mode: str) -> CitationResult:
    chapter_authors_str = join_authors(ch.chapter_authors)
    editors_str = join_authors(ch.editors)
    ed_label = "ed" if len(ch.editors) == 1 else "eds"

    if ch.edition:
        pub_block = f"{ch.publisher}, {ch.edition}, {ch.year}"
    else:
        pub_block = f"{ch.publisher}, {ch.year}"

    base_text = (
        f"{chapter_authors_str}, '{ch.chapter_title}', in "
        f"{editors_str} ({ed_label}), {ch.book_title} ({pub_block})"
    )
    base_html = (
        f"{chapter_authors_str}, '{ch.chapter_title}', in "
        f"{editors_str} ({ed_label}), {italic_html(ch.book_title)} ({pub_block})"
    )

    if ch.starting_page:
        base_text += f" {ch.starting_page}"
        base_html += f" {ch.starting_page}"

    if ch.pinpoint:
        base_text += f", {ch.pinpoint}"
        base_html += f", {ch.pinpoint}"

    return CitationResult(
        source_type=SourceType.BOOK_CHAPTER,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def _format_media_article(m: MediaArticleCitation, mode: str) -> CitationResult:
    if m.authors:
        author_str = join_authors(m.authors)
    elif m.org_as_author:
        author_str = m.org_as_author
    else:
        author_str = ""

    parts_text = []
    parts_html = []

    if author_str:
        parts_text.append(author_str)
        parts_html.append(author_str)

    parts_text.append(f"'{m.article_title}'")
    parts_html.append(f"'{m.article_title}'")

    loc = m.newspaper_title
    if m.city:
        loc += f" ({m.city})"

    loc_with_date = f"{loc}, {m.date}"

    parts_text.append(loc_with_date)
    parts_html.append(italic_html(m.newspaper_title) + (f" ({m.city})" if m.city else "") + f", {m.date}")

    if m.page:
        parts_text.append(m.page)
        parts_html.append(m.page)

    base_text = ", ".join(parts_text)
    base_html = ", ".join(parts_html)

    if m.is_online and m.url and m.access_date:
        base_text += f", {m.url}, {m.access_date}"
        base_html += f", {m.url}, {m.access_date}"

    return CitationResult(
        source_type=SourceType.MEDIA_ARTICLE,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def _format_report(r: ReportCitation, mode: str) -> CitationResult:
    parts_text = [r.author_or_org, r.title]
    parts_html = [r.author_or_org, italic_html(r.title)]

    if r.report_number_or_series:
        parts_text.append(r.report_number_or_series)
        parts_html.append(r.report_number_or_series)

    pub_bits = []
    if r.publisher:
        pub_bits.append(r.publisher)
    if r.place:
        pub_bits.append(r.place)
    if r.date:
        pub_bits.append(r.date)

    if pub_bits:
        pub_block = ", ".join(pub_bits)
        parts_text.append(f"({pub_block})")
        parts_html.append(f"({pub_block})")

    if r.pinpoint:
        parts_text.append(r.pinpoint)
        parts_html.append(r.pinpoint)

    base_text = ", ".join(parts_text)
    base_html = ", ".join(parts_html)

    if r.is_online and r.url and r.access_date:
        base_text += f", {r.url}, {r.access_date}"
        base_html += f", {r.url}, {r.access_date}"

    return CitationResult(
        source_type=SourceType.REPORT,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def _format_website(w: WebsiteCitation, mode: str) -> CitationResult:
    parts_text = []
    parts_html = []

    if w.author_or_org:
        parts_text.append(w.author_or_org)
        parts_html.append(w.author_or_org)

    parts_text.append(f"'{w.page_title}'")
    parts_html.append(f"'{w.page_title}'")

    site_block = w.site_name
    parts_text.append(site_block)
    parts_html.append(italic_html(w.site_name))

    if w.date:
        parts_text.append(w.date)
        parts_html.append(w.date)

    base_text = ", ".join(parts_text) + f", {w.url}, {w.access_date}"
    base_html = ", ".join(parts_html) + f", {w.url}, {w.access_date}"

    return CitationResult(
        source_type=SourceType.WEBSITE,
        mode=mode,
        text=base_text.strip(),
        html=base_html.strip(),
    )


def format_citation(
    source_type: Union[SourceType, str],
    data: Dict[str, Any],
    mode: str = "footnote",
) -> CitationResult:
    if isinstance(source_type, str):
        source_type = SourceType(source_type)

    if mode not in {"footnote", "bibliography"}:
        raise ValueError("mode must be 'footnote' or 'bibliography'")

    if source_type == SourceType.CASE:
        model = CaseCitation(**data)
        return _format_case(model, mode)
    if source_type == SourceType.LEGISLATION:
        model = LegislationCitation(**data)
        return _format_legislation(model, mode)
    if source_type == SourceType.JOURNAL_ARTICLE:
        model = JournalArticleCitation(**data)
        return _format_journal_article(model, mode)
    if source_type == SourceType.BOOK:
        model = BookCitation(**data)
        return _format_book(model, mode)
    if source_type == SourceType.BOOK_CHAPTER:
        model = BookChapterCitation(**data)
        return _format_book_chapter(model, mode)
    if source_type == SourceType.MEDIA_ARTICLE:
        model = MediaArticleCitation(**data)
        return _format_media_article(model, mode)
    if source_type == SourceType.REPORT:
        model = ReportCitation(**data)
        return _format_report(model, mode)
    if source_type == SourceType.WEBSITE:
        model = WebsiteCitation(**data)
        return _format_website(model, mode)

    raise ValueError(f"Unsupported source_type: {source_type}")
