# aglc_engine.py
# Lexcite AGLC4 formatting engine (deterministic, HTML italics support)
# Version: 2.1.0 (whitespace + punctuation normalisation, safer joins)
#
# Key upgrade: a single, central cleaner that removes accidental extra spaces
# everywhere (text + html), including edge cases like:
#  - double spaces introduced by optional fields
#  - spaces before punctuation (",", ".", ")", "]")
#  - spaces after opening brackets ("(", "[")
#  - " ;" / " ," style artifacts from join logic

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator


# -----------------------------
# Public API
# -----------------------------

class SourceType(str, Enum):
    case = "case"
    legislation = "legislation"
    journal_article = "journal_article"
    book = "book"
    book_chapter = "book_chapter"
    media_article = "media_article"
    report = "report"
    website = "website"
    other = "other"


@dataclass
class CitationResult:
    source_type: SourceType
    mode: str  # "footnote" | "bibliography"
    text: str
    html: str


def format_citation(
    source_type: Union[SourceType, str],
    data: Dict[str, Any],
    mode: str = "footnote",
) -> CitationResult:
    """
    Deterministic formatter. Returns both plain text and HTML (with controlled <i> tags).
    Always runs the final output through a strict cleaner that normalises whitespace and punctuation.
    """
    st = SourceType(str(source_type))
    mode = (mode or "footnote").strip().lower()
    if mode not in ("footnote", "bibliography"):
        mode = "footnote"

    if st == SourceType.case:
        model = CaseInput.model_validate(data)
        text, html = format_case(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.legislation:
        model = LegislationInput.model_validate(data)
        text, html = format_legislation(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.journal_article:
        model = JournalArticleInput.model_validate(data)
        text, html = format_journal_article(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.book:
        model = BookInput.model_validate(data)
        text, html = format_book(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.book_chapter:
        model = BookChapterInput.model_validate(data)
        text, html = format_book_chapter(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.media_article:
        model = MediaArticleInput.model_validate(data)
        text, html = format_media_article(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.report:
        model = ReportInput.model_validate(data)
        text, html = format_report(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    if st == SourceType.website:
        model = WebsiteInput.model_validate(data)
        text, html = format_website(model)
        return CitationResult(source_type=st, mode=mode, text=text, html=html)

    # Fallback
    raw = str(data.get("raw") or "").strip()
    raw = clean_text_output(raw)
    return CitationResult(source_type=SourceType.other, mode=mode, text=raw, html=escape_text(raw))


# -----------------------------
# Paste mode helpers (freeform)
# -----------------------------

class PasteEntry(BaseModel):
    raw: str
    source_type: str
    text: str
    html: str
    validated: bool
    validation_errors: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


def format_freeform_line(raw: str) -> PasteEntry:
    s = (raw or "").strip()
    if not s:
        return PasteEntry(raw=raw, source_type="OTHER", text="", html="", validated=False, validation_errors=["Empty line."])

    st = detect_source_type_freeform(s)

    if st == SourceType.case:
        return format_case_freeform(s)

    if st == SourceType.legislation:
        return format_legislation_freeform(s)

    if st == SourceType.journal_article:
        return format_journal_freeform(s)

    if st == SourceType.book:
        return format_book_freeform(s)

    if st == SourceType.website:
        return format_website_freeform(s)

    # Unknown
    txt = clean_text_output(s)
    return PasteEntry(
        raw=s,
        source_type="OTHER",
        text=txt,
        html=escape_text(txt),
        validated=False,
        validation_errors=["Unsupported or unrecognised source pattern in paste mode. Use Build mode for reliability."],
        meta={},
    )


def detect_source_type_freeform(text: str) -> SourceType:
    t = text.strip()

    # Website
    if re.search(r"https?://", t) or re.search(r"<https?://", t):
        return SourceType.website

    # Legislation
    if re.search(r"\bAct\s+\d{4}\b", t) or re.search(r"\bRegulations?\b", t):
        return SourceType.legislation

    # Case: " v " plus either neutral or reported
    if re.search(r"\bv\b", t) and re.search(r"\s+v\s+", t, re.I):
        return SourceType.case

    # Journal: has quoted title + year + journal + page
    if re.search(r"'.+?'\s*\(\d{4}\)", t) and re.search(r"\b\d+\s*$", t):
        return SourceType.journal_article

    # Book: (Publisher, ed, year) or similar
    if re.search(r"\([^,]+,\s*\d+(st|nd|rd|th)\s+ed,\s*\d{4}\)", t, re.I):
        return SourceType.book

    return SourceType.other


# -----------------------------
# Models
# -----------------------------

class CaseInput(BaseModel):
    case_name: str = Field(..., min_length=2)
    year: str = Field(..., pattern=r"^\d{4}$")

    # Reported
    reporter_series_by_year: bool = False
    volume: Optional[str] = None
    reporter: Optional[str] = None
    first_page: Optional[str] = None

    # Neutral
    court: Optional[str] = None
    decision_number: Optional[str] = None

    neutral_citation_first: bool = True
    unreported: bool = False  # if True, neutral is required

    # Pinpoint
    pinpoint_type: Optional[str] = None  # "page" | "paragraph"
    pinpoint: Optional[str] = None

    @model_validator(mode="after")
    def _check_case(self):
        self.case_name = normalise_case_name(self.case_name)

        if self.reporter:
            self.reporter = normalise_reporter(self.reporter)

        if self.court:
            self.court = normalise_court(self.court)

        has_neutral = bool(self.court and self.decision_number)
        has_reported = bool(self.volume and self.reporter and self.first_page)

        if self.unreported and not has_neutral:
            raise ValueError("Unreported cases require a neutral citation (court and decision number).")

        if not has_neutral and not has_reported:
            raise ValueError(
                "Provide either a neutral citation (court + decision number) or a reported citation (volume + reporter + first page)."
            )

        if self.pinpoint_type:
            pt = self.pinpoint_type.strip().lower()
            if pt not in ("page", "paragraph"):
                raise ValueError("Pinpoint type must be 'page' or 'paragraph'.")
            if not self.pinpoint:
                raise ValueError("Pinpoint value is required when a pinpoint type is selected.")

        return self


class LegislationInput(BaseModel):
    title: str = Field(..., min_length=2)
    year: str = Field(..., pattern=r"^\d{4}$")
    jurisdiction: str = Field(..., min_length=2)

    is_bill: bool = False
    pinpoint_unit: Optional[str] = None  # s | ss | reg | regs etc
    pinpoint_number: Optional[str] = None

    @model_validator(mode="after")
    def _check_leg(self):
        self.title = normalise_titlecase(self.title)
        self.jurisdiction = normalise_jurisdiction(self.jurisdiction)
        if self.pinpoint_unit:
            self.pinpoint_unit = self.pinpoint_unit.strip()
        if self.pinpoint_number:
            self.pinpoint_number = self.pinpoint_number.strip()
        return self


class JournalArticleInput(BaseModel):
    authors: List[str] = Field(..., min_length=1)
    article_title: str = Field(..., min_length=2)
    year: str = Field(..., pattern=r"^\d{4}$")
    year_in_square_brackets: bool = False

    volume: Optional[str] = None
    issue: Optional[str] = None
    journal_title: str = Field(..., min_length=2)
    starting_page: str = Field(..., min_length=1)
    pinpoint: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None

    @model_validator(mode="after")
    def _check_ja(self):
        self.authors = [normalise_person_name(a) for a in self.authors if a and a.strip()]
        self.article_title = normalise_titlecase(self.article_title)
        self.journal_title = normalise_titlecase(self.journal_title)

        if self.is_online:
            if not self.url:
                raise ValueError("Online journal articles require a URL.")
            if not self.access_date:
                raise ValueError("Online journal articles require an access date.")
        return self


class BookInput(BaseModel):
    authors: List[str] = Field(..., min_length=1)
    title: str = Field(..., min_length=2)
    publisher: str = Field(..., min_length=1)
    year: str = Field(..., pattern=r"^\d{4}$")
    edition: Optional[str] = None
    pinpoint: Optional[str] = None

    @model_validator(mode="after")
    def _check_book(self):
        self.authors = [normalise_person_name(a) for a in self.authors if a and a.strip()]
        self.title = normalise_titlecase(self.title)
        self.publisher = self.publisher.strip()
        if self.edition:
            self.edition = self.edition.strip()
        if self.pinpoint:
            self.pinpoint = self.pinpoint.strip()
        return self


class BookChapterInput(BaseModel):
    chapter_authors: List[str] = Field(..., min_length=1)
    chapter_title: str = Field(..., min_length=2)
    editors: List[str] = Field(..., min_length=1)
    book_title: str = Field(..., min_length=2)
    publisher: str = Field(..., min_length=1)
    year: str = Field(..., pattern=r"^\d{4}$")

    edition: Optional[str] = None
    starting_page: Optional[str] = None
    pinpoint: Optional[str] = None

    @model_validator(mode="after")
    def _check_ch(self):
        self.chapter_authors = [normalise_person_name(a) for a in self.chapter_authors if a and a.strip()]
        self.editors = [normalise_person_name(a) for a in self.editors if a and a.strip()]
        self.chapter_title = normalise_titlecase(self.chapter_title)
        self.book_title = normalise_titlecase(self.book_title)
        self.publisher = self.publisher.strip()
        if self.edition:
            self.edition = self.edition.strip()
        return self


class MediaArticleInput(BaseModel):
    authors: List[str] = Field(default_factory=list)
    org_as_author: Optional[str] = None
    article_title: str = Field(..., min_length=2)
    newspaper_title: str = Field(..., min_length=2)
    city: Optional[str] = None
    date: str = Field(..., min_length=4)
    page: Optional[str] = None

    is_online: bool = False
    url: Optional[str] = None
    access_date: Optional[str] = None

    @model_validator(mode="after")
    def _check_ma(self):
        self.authors = [normalise_person_name(a) for a in self.authors if a and a.strip()]
        if self.org_as_author:
            self.org_as_author = normalise_titlecase(self.org_as_author)
        self.article_title = normalise_titlecase(self.article_title)
        self.newspaper_title = normalise_titlecase(self.newspaper_title)
        if self.city:
            self.city = normalise_titlecase(self.city)
        if self.is_online:
            if not self.url:
                raise ValueError("Online media articles require a URL.")
            if not self.access_date:
                raise ValueError("Online media articles require an access date.")
        return self


class ReportInput(BaseModel):
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

    @model_validator(mode="after")
    def _check_rep(self):
        self.author_or_org = normalise_titlecase(self.author_or_org)
        self.title = normalise_titlecase(self.title)
        if self.publisher:
            self.publisher = normalise_titlecase(self.publisher)
        if self.place:
            self.place = normalise_titlecase(self.place)
        if self.is_online:
            if not self.url:
                raise ValueError("Online reports require a URL.")
            if not self.access_date:
                raise ValueError("Online reports require an access date.")
        return self


class WebsiteInput(BaseModel):
    author_or_org: Optional[str] = None
    page_title: str = Field(..., min_length=2)
    site_name: str = Field(..., min_length=2)
    date: Optional[str] = None
    url: str = Field(..., min_length=8)
    access_date: str = Field(..., min_length=4)

    @model_validator(mode="after")
    def _check_web(self):
        if self.author_or_org:
            self.author_or_org = normalise_titlecase(self.author_or_org)
        self.page_title = normalise_titlecase(self.page_title)
        self.site_name = normalise_titlecase(self.site_name)
        self.url = self.url.strip()
        self.access_date = self.access_date.strip()
        if self.date:
            self.date = self.date.strip()
        return self


# -----------------------------
# Formatting rules (AGLC-ish)
# -----------------------------

def format_case(m: CaseInput) -> Tuple[str, str]:
    name_html = f"<i>{escape_text(m.case_name)}</i>"
    name_text = m.case_name

    has_neutral = bool(m.court and m.decision_number)
    has_reported = bool(m.volume and m.reporter and m.first_page)

    pin_text = ""
    pin_html = ""
    if m.pinpoint_type and m.pinpoint:
        pt = m.pinpoint_type.strip().lower()
        val = normalise_pinpoint_value(m.pinpoint)
        if pt == "paragraph":
            pin_text = f"[{val}]"
            pin_html = f"[{escape_text(val)}]"
        else:
            pin_text = f"{val}"
            pin_html = escape_text(val)

    neutral_text = ""
    neutral_html = ""
    if has_neutral:
        neutral_text = f"[{m.year}] {m.court} {m.decision_number}"
        neutral_html = f"[{escape_text(m.year)}] {escape_text(m.court)} {escape_text(m.decision_number)}"

    reported_text = ""
    reported_html = ""
    if has_reported:
        year_brackets = f"[{m.year}]" if m.reporter_series_by_year else f"({m.year})"
        reported_text = f"{year_brackets} {m.volume} {m.reporter} {m.first_page}"
        reported_html = f"{escape_text(year_brackets)} {escape_text(m.volume)} {escape_text(m.reporter)} {escape_text(m.first_page)}"

    core_text_parts: List[str] = [name_text]
    core_html_parts: List[str] = [name_html]

    if has_neutral and has_reported:
        if m.neutral_citation_first:
            core_text_parts.append(neutral_text + ";")
            core_text_parts.append(reported_text)
            core_html_parts.append(neutral_html + ";")
            core_html_parts.append(reported_html)
        else:
            core_text_parts.append(reported_text + ";")
            core_text_parts.append(neutral_text)
            core_html_parts.append(reported_html + ";")
            core_html_parts.append(neutral_html)
    elif has_neutral:
        core_text_parts.append(neutral_text)
        core_html_parts.append(neutral_html)
    elif has_reported:
        core_text_parts.append(reported_text)
        core_html_parts.append(reported_html)

    if pin_text:
        core_text_parts[-1] = core_text_parts[-1].rstrip(".")
        core_html_parts[-1] = core_html_parts[-1].rstrip(".")
        core_text_parts.append(f", {pin_text}")
        core_html_parts.append(f", {pin_html}")

    text = " ".join(core_text_parts).strip() + "."
    html = " ".join(core_html_parts).strip() + "."

    text = normalise_v(text)
    html = normalise_v(html)

    return clean_text_output(text), clean_html_output(html)


def format_legislation(m: LegislationInput) -> Tuple[str, str]:
    title = f"{m.title} {m.year}"
    jur = f"({m.jurisdiction})"
    title_text = title
    title_html = f"<i>{escape_text(title)}</i>"

    if m.is_bill:
        title_text += " (Bill)"
        title_html = f"<i>{escape_text(title)}</i> (Bill)"

    provision = ""
    if m.pinpoint_unit and m.pinpoint_number:
        provision = f" {m.pinpoint_unit} {m.pinpoint_number}"

    text = f"{title_text} {jur}{provision}."
    html = f"{title_html} {escape_text(jur)}{escape_text(provision)}."
    return clean_text_output(text), clean_html_output(html)


def format_journal_article(m: JournalArticleInput) -> Tuple[str, str]:
    authors = join_authors(m.authors)
    title = f"'{m.article_title}'"
    year = f"[{m.year}]" if m.year_in_square_brackets else f"({m.year})"

    vol_issue = ""
    if m.volume and m.issue:
        vol_issue = f"{m.volume}({m.issue})"
    elif m.volume:
        vol_issue = f"{m.volume}"
    elif m.issue:
        vol_issue = f"({m.issue})"

    journal_html = f"<i>{escape_text(m.journal_title)}</i>"
    journal_text = m.journal_title

    base_text = f"{authors}, {title} {year} {vol_issue} {journal_text} {m.starting_page}".replace("  ", " ").strip()
    base_html = f"{escape_text(authors)}, {escape_text(title)} {escape_text(year)} {escape_text(vol_issue)} {journal_html} {escape_text(m.starting_page)}".replace("  ", " ").strip()

    if m.pinpoint:
        p = normalise_pinpoint_value(m.pinpoint)
        base_text += f", {p}"
        base_html += f", {escape_text(p)}"

    if m.is_online:
        base_text += f" <{m.url}> accessed {m.access_date}"
        base_html += f" &lt;{escape_text(m.url)}&gt; accessed {escape_text(m.access_date)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


def format_book(m: BookInput) -> Tuple[str, str]:
    authors = join_authors(m.authors)
    title_html = f"<i>{escape_text(m.title)}</i>"
    title_text = m.title

    inner = [m.publisher]
    if m.edition:
        inner.append(m.edition)
    inner.append(m.year)
    paren = f"({', '.join(inner)})"

    base_text = f"{authors}, {title_text} {paren}"
    base_html = f"{escape_text(authors)}, {title_html} {escape_text(paren)}"

    if m.pinpoint:
        p = normalise_pinpoint_value(m.pinpoint)
        base_text += f", {p}"
        base_html += f", {escape_text(p)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


def format_book_chapter(m: BookChapterInput) -> Tuple[str, str]:
    chap_auth = join_authors(m.chapter_authors)
    editors = join_editors(m.editors)
    chapter_title = f"'{m.chapter_title}'"
    book_title_html = f"<i>{escape_text(m.book_title)}</i>"
    book_title_text = m.book_title

    inner = [m.publisher]
    if m.edition:
        inner.append(m.edition)
    inner.append(m.year)
    paren = f"({', '.join(inner)})"

    base_text = f"{chap_auth}, {chapter_title} in {editors} (ed), {book_title_text} {paren}".strip()
    base_html = f"{escape_text(chap_auth)}, {escape_text(chapter_title)} in {escape_text(editors)} (ed), {book_title_html} {escape_text(paren)}".strip()

    if m.starting_page:
        base_text += f" {m.starting_page}"
        base_html += f" {escape_text(m.starting_page)}"
    if m.pinpoint:
        p = normalise_pinpoint_value(m.pinpoint)
        base_text += f", {p}"
        base_html += f", {escape_text(p)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


def format_media_article(m: MediaArticleInput) -> Tuple[str, str]:
    author_part = ""
    if m.org_as_author:
        author_part = m.org_as_author
    elif m.authors:
        author_part = join_authors(m.authors)

    title = f"'{m.article_title}'"
    paper_html = f"<i>{escape_text(m.newspaper_title)}</i>"
    paper_text = m.newspaper_title

    parts_text: List[str] = []
    parts_html: List[str] = []

    if author_part:
        parts_text.append(author_part + ",")
        parts_html.append(escape_text(author_part) + ",")

    parts_text.append(title)
    parts_html.append(escape_text(title))

    inner = [paper_text]
    inner_html = [paper_html]
    if m.city:
        inner.append(m.city)
        inner_html.append(escape_text(m.city))
    inner.append(m.date)
    inner_html.append(escape_text(m.date))
    if m.page:
        inner.append(m.page)
        inner_html.append(escape_text(m.page))

    paren_text = f"({', '.join(inner)})"
    paren_html = "(" + ", ".join(inner_html) + ")"

    parts_text.append(paren_text)
    parts_html.append(paren_html)

    base_text = " ".join(parts_text)
    base_html = " ".join(parts_html)

    if m.is_online:
        base_text += f" <{m.url}> accessed {m.access_date}"
        base_html += f" &lt;{escape_text(m.url)}&gt; accessed {escape_text(m.access_date)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


def format_report(m: ReportInput) -> Tuple[str, str]:
    author = m.author_or_org
    title_html = f"<i>{escape_text(m.title)}</i>"
    title_text = m.title

    inner: List[str] = []
    if m.report_number_or_series:
        inner.append(m.report_number_or_series)
    if m.publisher:
        inner.append(m.publisher)
    if m.place:
        inner.append(m.place)
    if m.date:
        inner.append(m.date)

    paren = f"({', '.join(inner)})" if inner else ""

    base_text = f"{author}, {title_text} {paren}".strip()
    base_html = f"{escape_text(author)}, {title_html} {escape_text(paren)}".strip()

    if m.pinpoint:
        p = normalise_pinpoint_value(m.pinpoint)
        base_text += f", {p}"
        base_html += f", {escape_text(p)}"

    if m.is_online:
        base_text += f" <{m.url}> accessed {m.access_date}"
        base_html += f" &lt;{escape_text(m.url)}&gt; accessed {escape_text(m.access_date)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


def format_website(m: WebsiteInput) -> Tuple[str, str]:
    author = (m.author_or_org or "").strip()
    title = f"'{m.page_title}'"
    site = m.site_name

    if m.date:
        paren = f"({site}, {m.date})"
    else:
        paren = f"({site})"

    if author:
        base_text = f"{author}, {title} {paren} <{m.url}>"
        base_html = f"{escape_text(author)}, {escape_text(title)} {escape_text(paren)} &lt;{escape_text(m.url)}&gt;"
    else:
        base_text = f"{title} {paren} <{m.url}>"
        base_html = f"{escape_text(title)} {escape_text(paren)} &lt;{escape_text(m.url)}&gt;"

    if not m.date:
        base_text += f" accessed {m.access_date}"
        base_html += f" accessed {escape_text(m.access_date)}"

    return clean_text_output(base_text + "."), clean_html_output(base_html + ".")


# -----------------------------
# Freeform formatters (paste)
# -----------------------------

_NEUTRAL_RE = re.compile(r"(?P<year>\d{4})\]?\s+(?P<court>[A-Za-z]{2,8})\s+(?P<num>\d{1,4})")
_REPORTED_RE = re.compile(r"\((?P<year>\d{4})\)\s+(?P<vol>\d+)\s+(?P<rep>[A-Za-z\.]+)\s+(?P<page>\d+)", re.I)
_PINPARA_RE = re.compile(r"\[(?P<p>\d{1,5})\]\s*$")
_PINPAGE_RE = re.compile(r"\b(?P<p>\d{1,5})\s*$")


def format_case_freeform(raw: str) -> PasteEntry:
    s = raw.strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    split = re.split(r"\s+(?=\[\d{4}\]|\(\d{4}\))", s, maxsplit=1)
    case_name = split[0].strip()
    rest = split[1].strip() if len(split) > 1 else ""

    if not case_name or " v " not in case_name.lower():
        errs.append("Could not confidently identify a case name. Use Build mode for best results.")
        case_name = case_name or s

    case_name = normalise_case_name(case_name)

    pin_text = ""
    pin_html = ""
    mpara = _PINPARA_RE.search(s)
    if mpara:
        val = mpara.group("p")
        pin_text = f"[{val}]"
        pin_html = f"[{escape_text(val)}]"
        s_wo = s[: mpara.start()].strip()
    else:
        s_wo = s

    neutral = ""
    m_neu = re.search(r"\[(\d{4})\]\s+([A-Za-z]{2,8})\s+(\d{1,4})", s_wo)
    if m_neu:
        y = m_neu.group(1)
        court = normalise_court(m_neu.group(2))
        num = m_neu.group(3)
        neutral = f"[{y}] {court} {num}"
        meta["neutral"] = neutral

    reported = ""
    m_rep = _REPORTED_RE.search(s_wo)
    if m_rep:
        y = m_rep.group("year")
        vol = m_rep.group("vol")
        rep = normalise_reporter(m_rep.group("rep"))
        page = m_rep.group("page")
        reported = f"({y}) {vol} {rep} {page}"
        meta["reported"] = reported

    if not neutral and not reported:
        errs.append("Missing a recognised neutral citation or reported citation. Use Build mode for this case.")
        out_text = f"{case_name}."
        out_html = f"<i>{escape_text(case_name)}</i>."
        return PasteEntry(raw=raw, source_type="CASE", text=clean_text_output(out_text), html=clean_html_output(out_html), validated=False, validation_errors=errs, meta=meta)

    name_html = f"<i>{escape_text(case_name)}</i>"
    parts_text = [case_name]
    parts_html = [name_html]

    if neutral and reported:
        parts_text.append(neutral + ";")
        parts_text.append(reported)
        parts_html.append(escape_text(neutral) + ";")
        parts_html.append(escape_text(reported))
    elif neutral:
        parts_text.append(neutral)
        parts_html.append(escape_text(neutral))
    else:
        parts_text.append(reported)
        parts_html.append(escape_text(reported))

    if pin_text:
        parts_text.append(f", {pin_text}")
        parts_html.append(f", {pin_html}")

    out_text = normalise_v(" ".join(parts_text).strip()) + "."
    out_html = normalise_v(" ".join(parts_html).strip()) + "."

    validated = len(errs) == 0
    return PasteEntry(
        raw=raw,
        source_type="CASE",
        text=clean_text_output(out_text),
        html=clean_html_output(out_html),
        validated=validated,
        validation_errors=errs,
        meta=meta,
    )


def format_legislation_freeform(raw: str) -> PasteEntry:
    s = raw.strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    m = re.search(
        r"^(?P<title>.+?)\s+(?P<year>\d{4})\s*\(\s*(?P<jur>[A-Za-z]{2,6})\s*\)\s*(?P<prov>.*)?$",
        s,
    )
    if not m:
        m = re.search(
            r"^(?P<title>.+?)\s+(?P<year>\d{4})\s+(?P<jur>[A-Za-z]{2,6})\s*(?P<prov>.*)?$",
            s,
        )

    if not m:
        errs.append("Could not parse legislation reliably. Use Build mode.")
        return PasteEntry(raw=raw, source_type="LEGISLATION", text=clean_text_output(s), html=clean_html_output(escape_text(s)), validated=False, validation_errors=errs, meta=meta)

    title = normalise_titlecase(m.group("title").strip())
    year = m.group("year").strip()
    jur = normalise_jurisdiction(m.group("jur").strip())
    prov = (m.group("prov") or "").strip()
    prov = normalise_leg_provision(prov)

    core_text = f"{title} {year} ({jur})".strip()
    core_html = f"<i>{escape_text(f'{title} {year}')}</i> {escape_text(f'({jur})')}"

    if prov:
        core_text += f" {prov}"
        core_html += f" {escape_text(prov)}"

    out_text = core_text + "."
    out_html = core_html + "."
    meta.update({"title": title, "year": year, "jurisdiction": jur, "provision": prov})

    return PasteEntry(raw=raw, source_type="LEGISLATION", text=clean_text_output(out_text), html=clean_html_output(out_html), validated=True, validation_errors=[], meta=meta)


def format_journal_freeform(raw: str) -> PasteEntry:
    s = raw.strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    m = re.search(
        r"^(?P<author>[^,]+(?:, [^,]+)*),\s*'(?P<title>[^']+)'\s*\((?P<year>\d{4})\)\s*(?P<vol>\d+)\((?P<issue>\d+)\)\s+(?P<journal>.+?)\s+(?P<page>\d+)(?:,\s*(?P<pin>\d+))?$",
        s,
    )
    if not m:
        errs.append("Could not parse journal article reliably. Use Build mode.")
        return PasteEntry(raw=raw, source_type="JOURNAL", text=clean_text_output(s), html=clean_html_output(escape_text(s)), validated=False, validation_errors=errs, meta=meta)

    author = normalise_person_name(m.group("author").strip())
    title = normalise_titlecase(m.group("title").strip())
    year = m.group("year")
    vol = m.group("vol")
    issue = m.group("issue")
    journal = normalise_titlecase(m.group("journal").strip())
    page = m.group("page")
    pin = m.group("pin")

    out_text = f"{author}, '{title}' ({year}) {vol}({issue}) {journal} {page}"
    out_html = f"{escape_text(author)}, '{escape_text(title)}' ({escape_text(year)}) {escape_text(vol)}({escape_text(issue)}) <i>{escape_text(journal)}</i> {escape_text(page)}"

    if pin:
        out_text += f", {pin}"
        out_html += f", {escape_text(pin)}"

    return PasteEntry(raw=raw, source_type="JOURNAL", text=clean_text_output(out_text + "."), html=clean_html_output(out_html + "."), validated=True, validation_errors=[], meta=meta)


def format_book_freeform(raw: str) -> PasteEntry:
    s = raw.strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    m = re.search(
        r"^(?P<author>.+?),\s*(?P<title>.+?)\s*\((?P<publisher>[^,]+),\s*(?P<edition>[^,]+),\s*(?P<year>\d{4})\)(?:,\s*(?P<pin>\d+))?$",
        s,
    )
    if not m:
        errs.append("Could not parse book reliably. Use Build mode.")
        return PasteEntry(raw=raw, source_type="BOOK", text=clean_text_output(s), html=clean_html_output(escape_text(s)), validated=False, validation_errors=errs, meta=meta)

    author = normalise_person_name(m.group("author").strip())
    title = normalise_titlecase(m.group("title").strip())
    publisher = m.group("publisher").strip()
    edition = m.group("edition").strip()
    year = m.group("year").strip()
    pin = m.group("pin")

    paren = f"({publisher}, {edition}, {year})"
    out_text = f"{author}, {title} {paren}"
    out_html = f"{escape_text(author)}, <i>{escape_text(title)}</i> {escape_text(paren)}"

    if pin:
        out_text += f", {pin}"
        out_html += f", {escape_text(pin)}"

    return PasteEntry(raw=raw, source_type="BOOK", text=clean_text_output(out_text + "."), html=clean_html_output(out_html + "."), validated=True, validation_errors=[], meta=meta)


def format_website_freeform(raw: str) -> PasteEntry:
    s = raw.strip()
    errs: List[str] = []
    meta: Dict[str, Any] = {}

    url = extract_url(s)
    if not url:
        errs.append("Website line missing URL. Include https://... or <https://...>.")
        return PasteEntry(raw=raw, source_type="WEBSITE", text=clean_text_output(s), html=clean_html_output(escape_text(s)), validated=False, validation_errors=errs, meta=meta)

    has_quotes = bool(re.search(r"'[^']+'", s))
    has_paren = "(" in s and ")" in s

    if not (has_quotes and has_paren):
        errs.append("Paste mode cannot reliably build website metadata. Use Build mode for websites.")
        out = ensure_angle_brackets(s, url)
        out = out.rstrip(".") + "."
        out = clean_text_output(out)
        return PasteEntry(raw=raw, source_type="WEBSITE", text=out, html=escape_text(out), validated=False, validation_errors=errs, meta={"url": url})

    out = ensure_angle_brackets(s, url).rstrip(".") + "."
    out_text = clean_text_output(strip_tags(out))
    out_html = clean_html_output(escape_text(out))
    return PasteEntry(raw=raw, source_type="WEBSITE", text=out_text, html=out_html, validated=True, validation_errors=[], meta={"url": url})


# -----------------------------
# Normalisation helpers
# -----------------------------

def normalise_v(s: str) -> str:
    return re.sub(r"\sv\.\s", " v ", s, flags=re.I)


def normalise_reporter(rep: str) -> str:
    rep = (rep or "").strip().upper().replace(".", "")
    return rep


def normalise_court(court: str) -> str:
    return (court or "").strip().upper()


def normalise_jurisdiction(j: str) -> str:
    j0 = (j or "").strip()
    if not j0:
        return j0
    mapping = {
        "CTH": "Cth",
        "NSW": "NSW",
        "VIC": "Vic",
        "QLD": "Qld",
        "SA": "SA",
        "WA": "WA",
        "TAS": "Tas",
        "ACT": "ACT",
        "NT": "NT",
    }
    up = j0.upper()
    return mapping.get(up, j0)


def normalise_person_name(name: str) -> str:
    s = " ".join((name or "").split())
    if not s:
        return s
    if looks_mostly_lower(s):
        return s.title()
    return s


def normalise_titlecase(text: str) -> str:
    s = " ".join((text or "").split())
    if not s:
        return s
    if looks_mostly_lower(s):
        return s.title()
    return s


def normalise_case_name(case_name: str) -> str:
    s = " ".join((case_name or "").split())
    if not s:
        return s

    s = re.sub(r"\sv\.\s", " v ", s, flags=re.I)

    if looks_mostly_lower(s):
        s = s.title()

    s = re.sub(r"\sV\s", " v ", s)

    return s


def looks_mostly_lower(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    lower = sum(1 for c in letters if c.islower())
    return (lower / max(1, len(letters))) > 0.75


def normalise_pinpoint_value(p: str) -> str:
    return re.sub(r"[^\d]", "", (p or "").strip()) or (p or "").strip()


def normalise_leg_provision(prov: str) -> str:
    if not prov:
        return ""
    s = " ".join(prov.split())
    s = re.sub(r"\bsec\b", "s", s, flags=re.I)
    s = re.sub(r"\bsecs\b", "ss", s, flags=re.I)
    s = re.sub(r"\bsection\b", "s", s, flags=re.I)
    s = re.sub(r"\bsections\b", "ss", s, flags=re.I)
    return s


def join_authors(authors: List[str]) -> str:
    a = [x.strip() for x in authors if x and x.strip()]
    if not a:
        return ""
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return f"{a[0]} and {a[1]}"
    return ", ".join(a[:-1]) + f", and {a[-1]}"


def join_editors(editors: List[str]) -> str:
    e = [x.strip() for x in editors if x and x.strip()]
    if not e:
        return ""
    if len(e) == 1:
        return e[0]
    if len(e) == 2:
        return f"{e[0]} and {e[1]}"
    return ", ".join(e[:-1]) + f", and {e[-1]}"


def extract_url(text: str) -> Optional[str]:
    m = re.search(r"<(https?://[^>]+)>", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(https?://\S+)", text)
    if m2:
        return m2.group(1).rstrip(").,;")
    return None


def ensure_angle_brackets(text: str, url: str) -> str:
    if f"<{url}>" in text:
        return text
    return re.sub(re.escape(url), f"<{url}>", text)


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def escape_text(s: str) -> str:
    s = s or ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s


# -----------------------------
# Output cleaners (new, central)
# -----------------------------

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:\)\]\}])")
_SPACE_AFTER_OPEN = re.compile(r"([\(\[\{])\s+")
_WEIRD_SEMI = re.compile(r"\s*;\s*")
_WEIRD_COMMA = re.compile(r"\s*,\s*")
_TRAILING_SPACE_BEFORE_PERIOD = re.compile(r"\s+\.")


def clean_text_output(s: str) -> str:
    """
    Clean final plain-text citation output.
    Removes extra spaces and fixes punctuation spacing deterministically.
    """
    if not s:
        return ""

    out = str(s)

    # Collapse whitespace first
    out = _WS.sub(" ", out).strip()

    # Standardise semicolons in case citations: " ; " -> "; "
    out = _WEIRD_SEMI.sub("; ", out)

    # Remove spaces before punctuation: " ,", " .", " )"
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)

    # Remove spaces after opening brackets: "( 2009)" -> "(2009)"
    out = _SPACE_AFTER_OPEN.sub(r"\1", out)

    # Fix comma spacing to single: "a , b" -> "a, b"
    out = _WEIRD_COMMA.sub(", ", out)

    # Remove " ." patterns
    out = _TRAILING_SPACE_BEFORE_PERIOD.sub(".", out)

    # Final collapse
    out = _WS.sub(" ", out).strip()

    return out


def clean_html_output(s: str) -> str:
    """
    Clean final HTML citation output without breaking controlled <i> tags.
    This does NOT attempt to parse HTML. It just normalises whitespace outside
    of tag syntax and fixes punctuation spacing similarly to text.

    Assumption: only <i> tags are introduced by this engine (controlled).
    """
    if not s:
        return ""

    out = str(s)

    # Collapse whitespace runs (safe because we don't rely on formatting whitespace)
    out = _WS.sub(" ", out).strip()

    # Standardise semicolons
    out = _WEIRD_SEMI.sub("; ", out)

    # Remove spaces before punctuation
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)

    # Remove spaces after opening brackets
    out = _SPACE_AFTER_OPEN.sub(r"\1", out)

    # Fix comma spacing
    out = _WEIRD_COMMA.sub(", ", out)

    # Remove " ." patterns
    out = _TRAILING_SPACE_BEFORE_PERIOD.sub(".", out)

    # Final collapse
    out = _WS.sub(" ", out).strip()

    return out
