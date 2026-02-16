from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape as _html_escape
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, field_validator, model_validator


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


# -----------------------------
# Helpers: safety + typography
# -----------------------------

def html_escape(s: Optional[str]) -> str:
    return _html_escape(s or "", quote=True)

def curly_quotes(s: str) -> str:
    """
    Turn straight single quotes into curly single quotes for titles.
    We do not attempt full typographic transformation across the entire string.
    """
    # This is intentionally conservative.
    return s.replace("'", "’")

def quote_title(title: str) -> str:
    # AGLC titles commonly use single quotes. We output curly ‘ ’.
    t = title.strip()
    return f"‘{t}’"

def normalise_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in {"footnote", "bibliography"}:
        return "footnote"
    return m

def join_people(names: List[str]) -> str:
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def join_people_html(names: List[str]) -> str:
    # Escape each name safely.
    safe = [html_escape(n.strip()) for n in (names or []) if n and n.strip()]
    if not safe:
        return ""
    if len(safe) == 1:
        return safe[0]
    if len(safe) == 2:
        return f"{safe[0]} and {safe[1]}"
    return ", ".join(safe[:-1]) + f", and {safe[-1]}"

def require(condition: bool, msg: str):
    if not condition:
        raise ValueError(msg)

def clean_pinpoint_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    vv = v.strip()
    return vv or None


# ----------
# CASE
# ----------

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

    @model_validator(mode="after")
    def validate_case(self):
        self.case_name = self.case_name.strip()
        self.year = self.year.strip()

        require(self.case_name != "", "Case name is required.")
        require(self.year.isdigit() and len(self.year) == 4, "Year must be a 4-digit year.")

        has_neutral = bool((self.court or "").strip()) and bool((self.decision_number or "").strip())
        has_report = bool((self.volume or "").strip()) and bool((self.reporter or "").strip()) and bool((self.first_page or "").strip())

        if self.unreported:
            require(has_neutral, "Unreported case requires court and decision number for a neutral citation.")
        else:
            require(has_neutral or has_report, "Provide either a neutral citation (court + decision number) or a reported citation (volume + reporter + first page).")

        self.pinpoint = clean_pinpoint_value(self.pinpoint)
        if self.pinpoint:
            # If pinpoint provided but no type, default to page
            if not self.pinpoint_type:
                self.pinpoint_type = "page"
        return self


def format_case(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    c = CaseCitation(**data)

    # Safe HTML
    italic_name_html = f"<i>{html_escape(c.case_name)}</i>"

    neutral_part = None
    if c.court and c.decision_number:
        neutral_part = f"[{c.year}] {c.court.strip()} {c.decision_number.strip()}"

    report_part = None
    if c.volume and c.reporter and c.first_page:
        vol = c.volume.strip()
        rep = c.reporter.strip()
        fp = c.first_page.strip()
        if c.reporter_series_by_year:
            report_part = f"[{c.year}] {vol} {rep} {fp}"
        else:
            report_part = f"({c.year}) {vol} {rep} {fp}"

    segments_text: List[str] = [c.case_name]
    segments_html: List[str] = [italic_name_html]

    if c.unreported:
        # neutral only
        segments_text.append(neutral_part or "")
        segments_html.append(html_escape(neutral_part or ""))
    else:
        if neutral_part and report_part:
            if c.neutral_citation_first:
                segments_text.extend([neutral_part, report_part])
                segments_html.extend([html_escape(neutral_part), html_escape(report_part)])
            else:
                segments_text.extend([report_part, neutral_part])
                segments_html.extend([html_escape(report_part), html_escape(neutral_part)])
        elif report_part:
            segments_text.append(report_part)
            segments_html.append(html_escape(report_part))
        elif neutral_part:
            segments_text.append(neutral_part)
            segments_html.append(html_escape(neutral_part))

    base_text = " ".join(s for s in segments_text if s).strip()
    base_html = " ".join(s for s in segments_html if s).strip()

    # Pinpoints: use "at"
    if c.pinpoint:
        pin = c.pinpoint.strip()
        if c.pinpoint_type == "paragraph":
            pin_str = f"at [{pin}]"
        else:
            pin_str = f"at {pin}"
        base_text = f"{base_text} {pin_str}"
        base_html = f"{base_html} {html_escape(pin_str)}"

    return CitationResult(source_type=SourceType.CASE, mode=mode, text=base_text, html=base_html)


# ----------
# LEGISLATION
# ----------

class LegislationCitation(BaseModel):
    title: str
    year: str
    jurisdiction: str
    is_bill: bool = False
    pinpoint_unit: Optional[str] = None  # s, ss, pt, sch etc
    pinpoint_number: Optional[str] = None

    @model_validator(mode="after")
    def validate_leg(self):
        self.title = self.title.strip()
        self.year = self.year.strip()
        self.jurisdiction = self.jurisdiction.strip()

        require(self.title != "", "Legislation title is required.")
        require(self.year.isdigit() and len(self.year) == 4, "Year must be a 4-digit year.")
        require(self.jurisdiction != "", "Jurisdiction is required (eg Cth, NSW).")

        if (self.pinpoint_unit and not self.pinpoint_number) or (self.pinpoint_number and not self.pinpoint_unit):
            raise ValueError("Provide both pinpoint unit and pinpoint number, or leave both blank.")
        return self


def format_legislation(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    l = LegislationCitation(**data)

    if l.is_bill:
        base = f"{l.title} Bill {l.year} ({l.jurisdiction})"
    else:
        base = f"{l.title} {l.year} ({l.jurisdiction})"

    if l.pinpoint_unit and l.pinpoint_number:
        base = f"{base} {l.pinpoint_unit.strip()} {l.pinpoint_number.strip()}"

    # Trust choice: do not italicise legislation titles.
    text = base
    html = html_escape(base)

    return CitationResult(source_type=SourceType.LEGISLATION, mode=mode, text=text, html=html)


# ----------
# JOURNAL ARTICLE
# ----------

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

    @model_validator(mode="after")
    def validate_ja(self):
        self.article_title = self.article_title.strip()
        self.journal_title = self.journal_title.strip()
        self.year = self.year.strip()
        self.starting_page = self.starting_page.strip()
        self.pinpoint = clean_pinpoint_value(self.pinpoint)

        require(len([a for a in self.authors if a and a.strip()]) > 0, "At least one author is required.")
        require(self.article_title != "", "Article title is required.")
        require(self.journal_title != "", "Journal title is required.")
        require(self.year.isdigit() and len(self.year) == 4, "Year must be a 4-digit year.")
        require(self.starting_page.isdigit(), "Starting page must be numeric.")

        if self.is_online:
            require(bool((self.url or "").strip()), "Online journal articles require a URL.")
            require(bool((self.access_date or "").strip()), "Online journal articles require an access date.")
        return self


def format_journal_article(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    j = JournalArticleCitation(**data)

    author_str = join_people(j.authors)
    author_html = join_people_html(j.authors)

    year_part = f"[{j.year}]" if j.year_in_square_brackets else f"({j.year})"

    vol_issue = ""
    if j.volume and j.issue:
        vol_issue = f"{j.volume.strip()}({j.issue.strip()})"
    elif j.volume:
        vol_issue = j.volume.strip()
    elif j.issue:
        vol_issue = j.issue.strip()

    title_q = quote_title(j.article_title)
    title_q_html = html_escape(title_q)

    parts_text: List[str] = []
    parts_html: List[str] = []

    parts_text.append(f"{author_str},")
    parts_html.append(f"{author_html},")

    parts_text.append(title_q)
    parts_html.append(title_q_html)

    core = year_part + (f" {vol_issue}" if vol_issue else "")
    parts_text.append(core)
    parts_html.append(html_escape(core))

    parts_text.append(j.journal_title)
    parts_html.append(f"<i>{html_escape(j.journal_title)}</i>")

    # Starting page and optional pinpoint
    tail = j.starting_page
    if j.pinpoint:
        tail = f"{tail}, {j.pinpoint.strip()}"
    parts_text.append(tail)
    parts_html.append(html_escape(tail))

    text = " ".join(parts_text).strip()
    html = " ".join(parts_html).strip()

    if j.is_online:
        url = (j.url or "").strip()
        access = (j.access_date or "").strip()
        online_tail = f"<{url}> accessed {access}"
        text = f"{text} {online_tail}"
        html = f"{html} {html_escape(online_tail)}"

    return CitationResult(source_type=SourceType.JOURNAL_ARTICLE, mode=mode, text=text, html=html)


# ----------
# BOOK
# ----------

class BookCitation(BaseModel):
    authors: List[str]
    title: str
    publisher: str
    year: str
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @model_validator(mode="after")
    def validate_book(self):
        self.title = self.title.strip()
        self.publisher = self.publisher.strip()
        self.year = self.year.strip()
        self.edition = clean_pinpoint_value(self.edition)
        self.pinpoint = clean_pinpoint_value(self.pinpoint)

        require(len([a for a in self.authors if a and a.strip()]) > 0, "At least one author or editor is required.")
        require(self.title != "", "Book title is required.")
        require(self.publisher != "", "Publisher is required.")
        require(self.year.isdigit() and len(self.year) == 4, "Year must be a 4-digit year.")
        return self


def format_book(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    b = BookCitation(**data)

    author_str = join_people(b.authors)
    author_html = join_people_html(b.authors)

    pub_parts = [b.publisher]
    if b.edition:
        pub_parts.append(b.edition)
    pub_parts.append(b.year)
    pub_segment = ", ".join(pub_parts)

    text = f"{author_str}, {b.title} ({pub_segment})"
    html = f"{author_html}, <i>{html_escape(b.title)}</i> ({html_escape(pub_segment)})"

    if b.pinpoint:
        text = f"{text} {b.pinpoint}"
        html = f"{html} {html_escape(' ' + b.pinpoint)}"

    return CitationResult(source_type=SourceType.BOOK, mode=mode, text=text, html=html)


# ----------
# BOOK CHAPTER
# ----------

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

    @model_validator(mode="after")
    def validate_chapter(self):
        self.chapter_title = self.chapter_title.strip()
        self.book_title = self.book_title.strip()
        self.publisher = self.publisher.strip()
        self.year = self.year.strip()
        self.edition = clean_pinpoint_value(self.edition)
        self.starting_page = clean_pinpoint_value(self.starting_page)
        self.pinpoint = clean_pinpoint_value(self.pinpoint)

        require(len([a for a in self.chapter_authors if a and a.strip()]) > 0, "Chapter author is required.")
        require(self.chapter_title != "", "Chapter title is required.")
        require(self.book_title != "", "Book title is required.")
        require(self.publisher != "", "Publisher is required.")
        require(self.year.isdigit() and len(self.year) == 4, "Year must be a 4-digit year.")
        return self


def format_book_chapter(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    c = BookChapterCitation(**data)

    chapter_authors = join_people(c.chapter_authors)
    chapter_authors_html = join_people_html(c.chapter_authors)

    editors = join_people(c.editors)
    editors_html = join_people_html(c.editors)

    title_q = quote_title(c.chapter_title)

    text_parts: List[str] = [f"{chapter_authors}," , f"{title_q},"]
    html_parts: List[str] = [f"{chapter_authors_html},", f"{html_escape(title_q)},"]

    if editors:
        # AGLC often uses ed/eds
        ed_suffix = "ed" if len([e for e in c.editors if e and e.strip()]) == 1 else "eds"
        text_parts.append(f"in {editors} ({ed_suffix}),")
        html_parts.append(f"in {editors_html} ({html_escape(ed_suffix)}),")

    pub_parts = [c.publisher]
    if c.edition:
        pub_parts.append(c.edition)
    pub_parts.append(c.year)
    pub_segment = ", ".join(pub_parts)

    book_seg_text = f"{c.book_title} ({pub_segment})"
    book_seg_html = f"<i>{html_escape(c.book_title)}</i> ({html_escape(pub_segment)})"

    if c.starting_page:
        book_seg_text = f"{book_seg_text} {c.starting_page}"
        book_seg_html = f"{book_seg_html} {html_escape(' ' + c.starting_page)}"

    if c.pinpoint:
        book_seg_text = f"{book_seg_text}, {c.pinpoint}"
        book_seg_html = f"{book_seg_html}{html_escape(', ' + c.pinpoint)}"

    text_parts.append(book_seg_text)
    html_parts.append(book_seg_html)

    return CitationResult(
        source_type=SourceType.BOOK_CHAPTER,
        mode=mode,
        text=" ".join(text_parts).strip(),
        html=" ".join(html_parts).strip(),
    )


# ----------
# MEDIA ARTICLE
# ----------

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

    @model_validator(mode="after")
    def validate_media(self):
        self.article_title = self.article_title.strip()
        self.newspaper_title = self.newspaper_title.strip()
        self.date = self.date.strip()
        self.city = clean_pinpoint_value(self.city)
        self.page = clean_pinpoint_value(self.page)

        require(self.article_title != "", "Article title is required.")
        require(self.newspaper_title != "", "Newspaper title is required.")
        require(self.date != "", "Date is required (eg 1 January 2025).")

        # Require at least one of authors or org
        has_author = len([a for a in self.authors if a and a.strip()]) > 0
        has_org = bool((self.org_as_author or "").strip())
        require(has_author or has_org, "Provide either author(s) or an organisation as author.")

        if self.is_online:
            require(bool((self.url or "").strip()), "Online media articles require a URL.")
            require(bool((self.access_date or "").strip()), "Online media articles require an access date.")
        return self


def format_media_article(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    m = MediaArticleCitation(**data)

    if m.authors:
        author_str = join_people(m.authors)
        author_html = join_people_html(m.authors)
    else:
        author_str = (m.org_as_author or "").strip()
        author_html = html_escape(author_str)

    title_q = quote_title(m.article_title)

    if m.city:
        loc_text = f"{m.newspaper_title} ({m.city}, {m.date})"
        loc_html = f"<i>{html_escape(m.newspaper_title)}</i> ({html_escape(m.city)}, {html_escape(m.date)})"
    else:
        loc_text = f"{m.newspaper_title} ({m.date})"
        loc_html = f"<i>{html_escape(m.newspaper_title)}</i> ({html_escape(m.date)})"

    if m.page:
        loc_text = f"{loc_text} {m.page}"
        loc_html = f"{loc_html}{html_escape(' ' + m.page)}"

    text = f"{author_str}, {title_q}, {loc_text}"
    html = f"{author_html}, {html_escape(title_q)}, {loc_html}"

    if m.is_online:
        online_tail = f"<{(m.url or '').strip()}> accessed {(m.access_date or '').strip()}"
        text = f"{text} {online_tail}"
        html = f"{html} {html_escape(' ' + online_tail)}"

    return CitationResult(source_type=SourceType.MEDIA_ARTICLE, mode=mode, text=text, html=html)


# ----------
# REPORT
# ----------

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

    @model_validator(mode="after")
    def validate_report(self):
        self.author_or_org = self.author_or_org.strip()
        self.title = self.title.strip()
        self.report_number_or_series = clean_pinpoint_value(self.report_number_or_series)
        self.publisher = clean_pinpoint_value(self.publisher)
        self.place = clean_pinpoint_value(self.place)
        self.date = clean_pinpoint_value(self.date)
        self.pinpoint = clean_pinpoint_value(self.pinpoint)

        require(self.author_or_org != "", "Author or organisation is required.")
        require(self.title != "", "Report title is required.")

        if self.is_online:
            require(bool((self.url or "").strip()), "Online reports require a URL.")
            require(bool((self.access_date or "").strip()), "Online reports require an access date.")
        return self


def format_report(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    r = ReportCitation(**data)

    title_ital = r.title  # many reports are italicised; safest: italicise the report title
    text_parts: List[str] = [f"{r.author_or_org},", title_ital]
    html_parts: List[str] = [f"{html_escape(r.author_or_org)},", f"<i>{html_escape(title_ital)}</i>"]

    pub_bits: List[str] = []
    if r.report_number_or_series:
        pub_bits.append(r.report_number_or_series)
    if r.publisher:
        pub_bits.append(r.publisher)
    if r.place:
        pub_bits.append(r.place)
    if r.date:
        pub_bits.append(r.date)

    if pub_bits:
        seg = f"({', '.join(pub_bits)})"
        text_parts.append(seg)
        html_parts.append(html_escape(seg))

    if r.pinpoint:
        text_parts.append(r.pinpoint)
        html_parts.append(html_escape(r.pinpoint))

    text = " ".join(text_parts).strip()
    html = " ".join(html_parts).strip()

    if r.is_online:
        online_tail = f"<{(r.url or '').strip()}> accessed {(r.access_date or '').strip()}"
        text = f"{text} {online_tail}"
        html = f"{html} {html_escape(' ' + online_tail)}"

    return CitationResult(source_type=SourceType.REPORT, mode=mode, text=text, html=html)


# ----------
# WEBSITE
# ----------

class WebsiteCitation(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str
    site_name: str
    date: Optional[str] = None
    url: str
    access_date: str

    @model_validator(mode="after")
    def validate_web(self):
        self.author_or_org = (self.author_or_org or "").strip() or None
        self.page_title = self.page_title.strip()
        self.site_name = self.site_name.strip()
        self.date = clean_pinpoint_value(self.date)
        self.url = self.url.strip()
        self.access_date = self.access_date.strip()

        require(self.page_title != "", "Page title is required.")
        require(self.site_name != "", "Site name is required.")
        require(self.url != "", "URL is required.")
        require(self.access_date != "", "Access date is required.")
        return self


def format_website(data: dict, mode: str) -> CitationResult:
    mode = normalise_mode(mode)
    w = WebsiteCitation(**data)

    title_q = quote_title(w.page_title)

    author_prefix = f"{w.author_or_org}, " if w.author_or_org else ""
    author_prefix_html = f"{html_escape(w.author_or_org)}, " if w.author_or_org else ""

    if w.date:
        core = f"({w.site_name}, {w.date})"
    else:
        core = f"({w.site_name})"

    tail = f"<{w.url}> accessed {w.access_date}"

    text = f"{author_prefix}{title_q} {core} {tail}"
    html = f"{author_prefix_html}{html_escape(title_q)} {html_escape(core)} {html_escape(tail)}"

    return CitationResult(source_type=SourceType.WEBSITE, mode=mode, text=text, html=html)


# ----------
# Dispatcher
# ----------

def format_citation(source_type: SourceType | str, data: Dict[str, Any], mode: str = "footnote") -> CitationResult:
    mode = normalise_mode(mode)

    st = source_type
    if isinstance(st, str):
        st = st.strip().lower()
        try:
            st = SourceType(st)
        except Exception:
            raise ValueError(f"Unsupported source_type: {source_type}")

    if st == SourceType.CASE:
        return format_case(data, mode)
    if st == SourceType.LEGISLATION:
        return format_legislation(data, mode)
    if st == SourceType.JOURNAL_ARTICLE:
        return format_journal_article(data, mode)
    if st == SourceType.BOOK:
        return format_book(data, mode)
    if st == SourceType.BOOK_CHAPTER:
        return format_book_chapter(data, mode)
    if st == SourceType.MEDIA_ARTICLE:
        return format_media_article(data, mode)
    if st == SourceType.REPORT:
        return format_report(data, mode)
    if st == SourceType.WEBSITE:
        return format_website(data, mode)

    raise ValueError(f"Unsupported source_type: {st}")
