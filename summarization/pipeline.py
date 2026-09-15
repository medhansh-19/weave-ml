"""Generate, repair, validate, and safely fall back for card summaries."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .contracts import (
    CARD_SUMMARY_FALLBACK_MODEL_VERSION,
    CARD_SUMMARY_FORMAT_VERSION,
    CARD_SUMMARY_MAX_CHARS,
    CARD_SUMMARY_PROMPT_VERSION,
    CardSummaryArtifact,
)
from .provider import SummaryProvider, SummaryProviderError
from .settings import SummarySettings
from .source import build_summary_source
from .validation import (
    SummaryValidationError,
    normalize_summary,
    parse_summary_response,
    validate_generated_summary,
)


_UNSAFE_SENTENCE = re.compile(
    r"(?:```|`|\b(?:npm|pnpm|yarn|pipx?|uv|cargo|go)\s+"
    r"(?:install|add|run|get)\b|\b(?:brew|apt(?:-get)?|dnf|yum|pacman|choco|"
    r"winget)\s+install\b|\bgit\s+clone\b|\b(?:curl|wget)\s+https?://|"
    r"\b(?:installation|installing|setup|getting started|quick ?start|changelog|"
    r"licen[cs]e|contributors?|contributing|api\s+(?:reference|docs?)|reference|"
    r"table of contents|badges?)\b)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r".+?[.!?](?=\s|$)|.+$", re.DOTALL)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MARKDOWN_STRUCTURE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)|```|~~~|`"
)
_URL = re.compile(r"https?://\S+")


def _bounded(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    candidate = value[: maximum + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not candidate:
        candidate = value[:maximum].rstrip(" ,;:-")
    if candidate and candidate[-1] not in ".!?":
        candidate = candidate[: maximum - 1].rstrip(" ,;:-") + "."
    return candidate[:maximum]


def _plain_description(value: Any) -> str:
    """Reduce a repository description to plain prose for the metadata fallback."""

    text = str(value or "")
    text = _MARKDOWN_IMAGE.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _MARKDOWN_STRUCTURE.sub("", text)
    return normalize_summary(text)


def description_fallback_artifact(
    repo: Mapping[str, Any],
    *,
    prompt_version: str = CARD_SUMMARY_PROMPT_VERSION,
    format_version: str = CARD_SUMMARY_FORMAT_VERSION,
) -> CardSummaryArtifact:
    """Build a bounded metadata fallback that never reads README content."""

    raw_description = _plain_description(repo.get("description"))
    safe_sentences = [
        normalize_summary(match.group(0))
        for match in _SENTENCE.finditer(raw_description)
        if match.group(0).strip() and not _UNSAFE_SENTENCE.search(match.group(0))
    ]
    if len(safe_sentences) >= 2:
        first = _bounded(safe_sentences[0], 175)
        second = _bounded(
            safe_sentences[1],
            CARD_SUMMARY_MAX_CHARS - len(first) - 1,
        )
        summary = f"{first} {second}".strip()
    else:
        summary = " ".join(safe_sentences).strip()
    if not summary:
        full_name = _plain_description(repo.get("full_name") or "This repository")
        language = _plain_description(repo.get("primary_language"))
        topics = repo.get("topics") if isinstance(repo.get("topics"), list) else []
        topic_text = ", ".join(
            item for item in (_plain_description(value) for value in topics[:3]) if item
        )
        summary = f"{full_name} is a GitHub repository."
        if language and language.casefold() != "unknown":
            summary += f" Its primary language is {language}."
        elif topic_text:
            summary += f" Its published topics include {topic_text}."
        else:
            summary += " Its description does not yet provide more project details."
    elif len(safe_sentences) == 1:
        language = _plain_description(repo.get("primary_language"))
        topics = repo.get("topics") if isinstance(repo.get("topics"), list) else []
        topic_text = ", ".join(
            item for item in (_plain_description(value) for value in topics[:3]) if item
        )
        full_name = _plain_description(repo.get("full_name") or "this repository")
        if language and language.casefold() != "unknown":
            supplement = f"Its primary language is {language}."
        elif topic_text:
            supplement = f"Its published topics include {topic_text}."
        else:
            supplement = f"It is published as {full_name}."
        summary = _bounded(
            summary,
            CARD_SUMMARY_MAX_CHARS - len(supplement) - 1,
        )
        if summary and summary[-1] not in ".!?":
            summary = f"{summary}."
        summary = f"{summary} {supplement}"
    summary = _bounded(summary, CARD_SUMMARY_MAX_CHARS)
    return CardSummaryArtifact(
        summary=summary,
        highlights=[],
        prompt_version=prompt_version,
        model_version=CARD_SUMMARY_FALLBACK_MODEL_VERSION,
        format_version=format_version,
        source="description_fallback",
    )


class CardSummaryPipeline:
    """One generation attempt, at most one repair, then safe fallback."""

    def __init__(
        self,
        *,
        settings: SummarySettings | None = None,
        provider: SummaryProvider | None = None,
    ) -> None:
        self.settings = settings or SummarySettings()
        self.provider = provider

    def is_current(self, artifact: CardSummaryArtifact) -> bool:
        if (
            artifact.prompt_version != self.settings.prompt_version
            or artifact.format_version != self.settings.format_version
        ):
            return False
        if artifact.source == "generated":
            return artifact.model_version == self.settings.model_id
        return (
            self.provider is None
            and artifact.model_version == CARD_SUMMARY_FALLBACK_MODEL_VERSION
        )

    def summarize(self, source: Any, repo: Mapping[str, Any]) -> CardSummaryArtifact:
        material = build_summary_source(
            source,
            repo,
            max_chars=self.settings.input_max_chars,
        )
        if self.provider is not None and material.prompt_input:
            repair_feedback: str | None = None
            for _attempt in range(2):
                try:
                    raw = self.provider.generate(
                        material.prompt_input,
                        repair_feedback=repair_feedback,
                    )
                    content = validate_generated_summary(
                        parse_summary_response(raw),
                        comparison_texts=material.comparison_texts,
                    )
                    return CardSummaryArtifact(
                        **content.model_dump(),
                        prompt_version=self.settings.prompt_version,
                        model_version=self.settings.model_id,
                        format_version=self.settings.format_version,
                        source="generated",
                    )
                except SummaryValidationError as exc:
                    repair_feedback = str(exc)
                except (SummaryProviderError, ValueError, TypeError):
                    break
        return description_fallback_artifact(
            repo,
            prompt_version=self.settings.prompt_version,
            format_version=self.settings.format_version,
        )
