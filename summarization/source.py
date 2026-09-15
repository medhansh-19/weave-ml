"""Build bounded, high-signal model input without mutating canonical Markdown."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any


_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_SETEXT = re.compile(r"^\s*(?:=+|-+)\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+] |\d+[.)]\s+)")
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_COMMAND = re.compile(
    r"(?:\b(?:npm|pnpm|yarn|pipx?|uv|cargo|go)\s+(?:install|add|run|get)\b"
    r"|\b(?:brew|apt(?:-get)?|dnf|yum|pacman|choco|winget)\s+install\b"
    r"|\bgit\s+clone\b|\b(?:curl|wget)\s+https?://|(?:^|\s)\$\s*\w)",
    re.IGNORECASE,
)
_BLOCKED_HEADING = re.compile(
    r"\b(?:install(?:ation)?|setup|getting started|quick ?start|api\s+(?:reference|docs?)|"
    r"reference|changelog|release notes?|licen[cs]e|contributors?|contributing|"
    r"development|build from source|table of contents|contents)\b",
    re.IGNORECASE,
)
_HIGH_SIGNAL_HEADING = re.compile(
    r"\b(?:overview|about|introduction|what is|why|features?|highlights?|"
    r"use cases?|who is|audience|benefits?|key concepts?|motivation)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SummarySource:
    prompt_input: str
    comparison_texts: tuple[str, ...]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = (_clean_line(str(item)) for item in value)
    return [item for item in cleaned if item and not _COMMAND.search(item)]


def _source_variants(source: Any, repo: Mapping[str, Any]) -> tuple[str, ...]:
    variants: list[str] = []
    raw = repo.get("readme")
    if isinstance(raw, str) and raw.strip():
        variants.append(raw)
    for key in ("clean_text", "readme_md"):
        value = repo.get(key)
        if isinstance(value, str) and value.strip():
            variants.append(value)

    readme = getattr(source, "readme", None)
    for key in ("raw_markdown", "clean_text", "readme_md"):
        value = getattr(readme, key, None)
        if isinstance(value, str) and value.strip():
            variants.append(value)

    distinct: list[str] = []
    seen: set[str] = set()
    for value in variants:
        fingerprint = value.strip()
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            distinct.append(value)
    return tuple(distinct)


def _clean_line(line: str) -> str:
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _HTML.sub(" ", line)
    line = _URL.sub(" ", line)
    line = _LIST_MARKER.sub("", line)
    line = line.replace("`", " ")
    return re.sub(r"\s+", " ", line).strip()


def _readme_sections(markdown: str) -> tuple[list[str], list[str]]:
    """Return high-signal and neutral lines from a Markdown copy."""

    high: list[str] = []
    neutral: list[str] = []
    current = neutral
    blocked = False
    in_fence = False
    previous_was_heading = False
    for raw_line in markdown.splitlines():
        if _FENCE.match(raw_line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading_match = _HEADING.match(raw_line)
        if heading_match:
            heading = heading_match.group(1)
            blocked = bool(_BLOCKED_HEADING.search(heading))
            current = high if _HIGH_SIGNAL_HEADING.search(heading) else neutral
            previous_was_heading = True
            continue
        if previous_was_heading and _SETEXT.match(raw_line):
            previous_was_heading = False
            continue
        previous_was_heading = False
        if blocked or _TABLE_RULE.match(raw_line) or raw_line.count("|") >= 3:
            continue
        cleaned = _clean_line(raw_line)
        if (
            len(cleaned) < 24
            or _COMMAND.search(cleaned)
            or _BLOCKED_HEADING.fullmatch(cleaned.rstrip(":"))
        ):
            continue
        current.append(cleaned)
    return high, neutral


def build_summary_source(
    source: Any,
    repo: Mapping[str, Any],
    *,
    max_chars: int,
) -> SummarySource:
    """Build a metadata-led prompt input capped before provider transport."""

    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")
    variants = _source_variants(source, repo)
    raw_markdown = variants[0] if variants else ""
    high, neutral = _readme_sections(raw_markdown)

    full_name = _clean_line(str(repo.get("full_name") or ""))
    metadata: list[str] = [f"Repository: {full_name}"]
    description = _clean_line(str(repo.get("description") or ""))
    if description and not _COMMAND.search(description) and not _BLOCKED_HEADING.search(description):
        metadata.append(f"Description: {description}")
    topics = _string_list(repo.get("topics"))
    languages = _string_list(repo.get("languages"))
    if topics:
        metadata.append(f"Topics: {', '.join(topics[:20])}")
    if languages:
        metadata.append(f"Languages: {', '.join(languages[:12])}")

    chunks = metadata + ["README high-signal excerpts:"] + high + neutral
    selected: list[str] = []
    used = 0
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if not normalized:
            continue
        remaining = max_chars - used - (2 if selected else 0)
        if remaining <= 0:
            break
        if len(normalized) > remaining:
            normalized = normalized[:remaining].rsplit(" ", 1)[0].rstrip(" ,;:-")
        if normalized:
            selected.append(normalized)
            used += len(normalized) + (2 if len(selected) > 1 else 0)
        if used >= max_chars:
            break

    prompt_input = "\n\n".join(selected)
    comparisons = tuple(dict.fromkeys((*variants, prompt_input)))
    return SummarySource(prompt_input=prompt_input, comparison_texts=comparisons)
