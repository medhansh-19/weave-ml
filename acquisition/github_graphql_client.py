"""GitHub GraphQL API client for repository discovery and enrichment."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from typing import Any
import logging
from pathlib import PurePosixPath
from urllib.parse import quote

import requests

from .graphql_queries import (
    GET_README_QUERY,
    GET_REPOSITORY_QUERY,
    README_CANDIDATES,
    build_batch_metadata_query,
)

logger = logging.getLogger(__name__)


class GitHubGraphQLClientError(RuntimeError):
    """Raised when the GitHub GraphQL API returns an unrecoverable error."""

# Backwards-compatible alias so existing callers don't break
GitHubClientError = GitHubGraphQLClientError


class FetchedReadme(str):
    """Canonical README Markdown with the metadata needed to resolve media.

    This remains a ``str`` subclass so existing acquisition callers that only
    consume README text continue to work while V2 ingestion can retain the
    source metadata as a separate artifact.
    """

    raw_markdown: str
    source_path: str | None
    default_branch: str | None
    base_url: str | None

    def __new__(
        cls,
        raw_markdown: str = "",
        *,
        source_path: str | None = None,
        default_branch: str | None = None,
        base_url: str | None = None,
    ) -> "FetchedReadme":
        value = str(raw_markdown or "")
        instance = super().__new__(cls, value)
        instance.raw_markdown = value
        instance.source_path = source_path
        instance.default_branch = default_branch
        instance.base_url = base_url
        return instance


def build_readme_base_url(
    owner: str,
    name: str,
    default_branch: str | None,
    source_path: str | None,
) -> str | None:
    """Build an HTTPS raw-content directory URL for relative README media."""
    if not source_path:
        return None
    ref = (default_branch or "HEAD").strip()
    if not ref:
        ref = "HEAD"
    owner_part = quote(owner.strip(), safe="")
    name_part = quote(name.strip(), safe="")
    ref_part = "/".join(quote(part, safe="") for part in ref.split("/"))
    parent = PurePosixPath(source_path).parent
    parent_part = "" if str(parent) == "." else "/".join(
        quote(part, safe="") for part in parent.parts if part not in {"", "."}
    )
    suffix = f"{parent_part}/" if parent_part else ""
    if default_branch:
        return (
            f"https://raw.githubusercontent.com/{owner_part}/{name_part}/"
            f"refs/heads/{ref_part}/{suffix}"
        )
    return (
        f"https://raw.githubusercontent.com/{owner_part}/{name_part}/"
        f"{ref_part}/{suffix}"
    )


class GitHubGraphQLClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com/graphql",
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        sleep_on_rate_limit: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.session = session or requests.Session()
        token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.session.headers.update(
            {
                "User-Agent": "osiris-repository-ingestion-pipeline",
                "Content-Type": "application/json",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def get_repository(self, owner: str, name: str) -> dict[str, Any] | None:
        """Fetches a single repository using GraphQL."""
        variables = {"owner": owner, "name": name}
        response = self.execute(GET_REPOSITORY_QUERY, variables)
        if not response:
            return None
        data = response.get("data", {})
        return data.get("repository")

    def get_readme(self, owner: str, name: str) -> FetchedReadme:
        """Fetch canonical README Markdown and its source metadata.

        ``FetchedReadme`` is string-compatible for older text-only callers.
        """
        try:
            response = self.execute(GET_README_QUERY, {"owner": owner, "name": name})
            if not isinstance(response, dict):
                raise GitHubClientError("README query returned no response envelope")
            gql_errors = response.get("errors")
            data = response.get("data")
            # GitHub can return a partial response: usable repository data (and
            # valid README aliases) alongside a field-level error for a *different*
            # alias.  Only treat errors as fatal when there is no accompanying
            # data envelope — otherwise log them as warnings and continue so the
            # pipeline can still use whichever README aliases resolved correctly.
            if gql_errors:
                if not isinstance(data, dict) or "repository" not in data:
                    raise GitHubClientError(
                        f"README query returned GraphQL errors: {gql_errors}"
                    )
                logger.warning(
                    "README query for %s/%s returned field-level errors alongside "
                    "partial data — proceeding with available aliases. Errors: %s",
                    owner,
                    name,
                    gql_errors,
                )
            if not isinstance(data, dict) or "repository" not in data:
                raise GitHubClientError("README query response is missing repository data")
            repo = data["repository"]
            if not isinstance(repo, dict):
                raise GitHubClientError("README query could not resolve the repository")
            if "defaultBranchRef" not in repo:
                raise GitHubClientError(
                    "README query response is missing default-branch metadata"
                )

            default_branch = (repo.get("defaultBranchRef") or {}).get("name")
            for key, source_path in README_CANDIDATES:
                blob = repo.get(key)
                if blob is None:
                    continue
                if not isinstance(blob, dict) or "text" not in blob:
                    raise GitHubClientError(
                        f"README query returned an incomplete {key} object"
                    )
                text = blob["text"]
                if not isinstance(text, str):
                    raise GitHubClientError(
                        f"README query returned invalid text for {key}"
                    )
                if text:
                    return FetchedReadme(
                        text,
                        source_path=source_path,
                        default_branch=default_branch,
                        base_url=build_readme_base_url(
                            owner, name, default_branch, source_path
                        ),
                    )
            logger.info(
                "README not found for %s/%s: every candidate resolved empty or null.",
                owner,
                name,
            )
        except Exception as exc:
            logger.warning(f"README fetch failed for {owner}/{name}: {exc}", exc_info=True)
            # Missing READMEs are represented by an empty FetchedReadme above.
            # Transport/API failures must propagate so acquisition marks the
            # repository for retry instead of overwriting canonical content
            # with an accidental empty README.
            raise
        return FetchedReadme()

    def get_repositories_batch(self, repos: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        """Fetches multiple repositories using a lean metadata-only batch query."""
        if not repos:
            return {}

        query = build_batch_metadata_query(repos)
        response = self.execute(query)
        if not response:
            return {}
        if response.get("errors") and not response.get("data"):
            raise GitHubClientError(
                f"batch metadata query returned GraphQL errors: {response['errors']}"
            )

        data = response.get("data", {})
        if not isinstance(data, dict):
            raise GitHubClientError("batch metadata query returned invalid data")
        results = {}
        for i, (owner, name) in enumerate(repos):
            alias = f"repo_{i}"
            if alias in data and data[alias]:
                results[f"{owner}/{name}"] = data[alias]
        return results

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Executes a GraphQL query with retries and rate limit handling."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub GraphQL request failed: {exc}") from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 403 and self._is_rate_limited(response):
                if attempt >= self.max_retries or not self.sleep_on_rate_limit:
                    raise GitHubClientError("GitHub GraphQL rate limit exceeded")
                self._sleep_until_reset(response)
                continue

            if response.status_code in {500, 502, 503, 504}:
                if attempt >= self.max_retries:
                    raise GitHubClientError(f"GitHub GraphQL transient failure {response.status_code}: {response.text[:300]}")
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise GitHubClientError(f"GitHub GraphQL error {response.status_code}: {response.text[:300]}")

            result = response.json()
            
            # Rate limit tracking from GraphQL payload
            data_field = result.get("data")
            if isinstance(data_field, dict) and "rateLimit" in data_field:
                rl = data_field["rateLimit"]
                logger.debug("GraphQL rate limit: %s remaining", rl.get("remaining"))
            
            if "errors" in result:
                if result.get("data"):
                    logger.warning(
                        "GitHub GraphQL returned partial errors: %s",
                        result["errors"],
                    )
                    return result
                if any(
                    "Could not resolve to a Repository" in error.get("message", "")
                    for error in result["errors"]
                ):
                    logger.warning(
                        "GitHub GraphQL could not resolve repository: %s",
                        result["errors"],
                    )
                    return None
                raise GitHubClientError(
                    f"GitHub GraphQL returned errors: {result['errors']}"
                )

            return result

        raise GitHubClientError("GitHub GraphQL request exhausted retries")

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        return remaining == "0" or "rate limit" in response.text.lower()

    def _sleep_until_reset(self, response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            sleep_for = max(int(reset) - int(time.time()) + 2, 1)
        else:
            retry_after = response.headers.get("Retry-After")
            sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 60
        time.sleep(min(sleep_for, 300))

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min((2**attempt) + random.random(), 30.0))
