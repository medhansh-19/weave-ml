"""Repository embed response, Qdrant replay, and summary-backfill contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http import models

from api.contracts import RepositoryJob
from api.v2 import _embed_repository_job_locked
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.qdrant_store import QdrantRepositoryStore
from summarization.contracts import (
    CARD_SUMMARY_FALLBACK_MODEL_VERSION,
    CARD_SUMMARY_PAYLOAD_FIELDS,
    CardSummaryArtifact,
    card_summary_from_payload,
    card_summary_payload,
)
from summarization.pipeline import CardSummaryPipeline


REPO_ID = "00000000-0000-4000-8000-000000000221"


class FakeEmbedder:
    def embed_texts(self, texts, *, normalize=True):
        return [self._vector() for _ in texts]

    def embed_text(self, text, *, normalize=True):
        return self._vector()

    @staticmethod
    def _vector():
        return [1.0] + [0.0] * 383


class OwnedLock:
    def assert_owned(self) -> None:
        return None


def _job(*, job_id: str | None = None, content_hash: str = "content-hash") -> RepositoryJob:
    return RepositoryJob.model_validate(
        {
            "schema_version": 2,
            "job_id": job_id or str(uuid.uuid4()),
            "repo_id": REPO_ID,
            "content_version": 7,
            "repository": {
                "repo_id": REPO_ID,
                "github_id": "123456",
                "full_name": "weave/summary-contract",
                "description": "A review workspace for teams maintaining shared developer tools.",
                "readme": "# Summary Contract\n\nA review workspace for shared developer tools.",
                "primary_language": "Python",
                "languages": ["Python"],
                "topics": ["developer-tools"],
                "content_hash": content_hash,
            },
        }
    )


def _artifact(summary: str = "A concise discovery summary explains the repository to developer-tool maintainers. Its focused review workflow helps teams coordinate changes before they affect shared users.") -> CardSummaryArtifact:
    return CardSummaryArtifact(
        summary=summary,
        highlights=["Review workflow"],
        prompt_version="repo-card-summary-v1",
        model_version="meta-llama/llama-3.3-70b-instruct",
        format_version="repo-card-summary-json-v1",
        source="generated",
    )


def _point(job: RepositoryJob, *, artifact: CardSummaryArtifact | None, stored_job: str) -> SimpleNamespace:
    payload = {
        "content_version": job.content_version,
        "content_job_id": stored_job,
        "content_hash": job.repository.content_hash,
        "embedding_version": "repo-embedding-v2",
    }
    if artifact is not None:
        payload.update(card_summary_payload(artifact))
    return SimpleNamespace(id=REPO_ID, payload=payload)


@pytest.mark.parametrize("same_job", [True, False], ids=["duplicate", "current"])
def test_duplicate_and_current_jobs_replay_the_durable_artifact(same_job: bool) -> None:
    request = _job()
    stored_job = str(request.job_id) if same_job else str(uuid.uuid4())
    point = _point(request, artifact=_artifact(), stored_job=stored_job)
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=store):
        response = _embed_repository_job_locked(request, OwnedLock())

    assert response["schema_version"] == 2
    assert response["job_id"] == str(request.job_id)
    assert response["status"] == ("duplicate" if same_job else "current")
    assert response["card_summary"] == _artifact().model_dump(mode="json")
    store.compare_and_set_card_summary.assert_not_called()


def test_same_content_missing_summary_is_backfilled_without_replacing_content() -> None:
    request = _job()
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())

    class Store:
        def compare_and_set_card_summary(self, *, expected_point, artifact):
            payload = dict(expected_point.payload)
            payload.update(card_summary_payload(artifact))
            return SimpleNamespace(id=expected_point.id, payload=payload)

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=Store()):
        response = _embed_repository_job_locked(request, OwnedLock())

    assert response["status"] == "current"
    assert response["card_summary"]["source"] == "description_fallback"
    assert response["card_summary"]["summary"].startswith(
        "A review workspace for teams"
    )
    assert (
        response["card_summary"]["model_version"]
        == CARD_SUMMARY_FALLBACK_MODEL_VERSION
    )


def test_same_content_fallback_is_upgraded_after_provider_recovery() -> None:
    request = _job()
    fallback = CardSummaryPipeline().summarize(
        request.repository.model_dump(mode="json"),
        request.repository.model_dump(mode="json"),
    )
    point = _point(request, artifact=fallback, stored_job=str(uuid.uuid4()))
    generated_summary = (
        "A focused repository workspace helps maintainers review shared "
        "developer-tool changes with clear ownership, context, and durable "
        "decision history. Its coordinated workflow helps distributed teams "
        "catch risky updates before they disrupt downstream users or integrations."
    )

    class RecoveredProvider:
        def generate(self, source, *, repair_feedback=None, deadline=None):
            return json.dumps({"summary": generated_summary})

    class Store:
        def compare_and_set_card_summary(self, *, expected_point, artifact):
            payload = dict(expected_point.payload)
            payload.update(card_summary_payload(artifact))
            return SimpleNamespace(id=expected_point.id, payload=payload)

    pipeline = RepositoryEmbeddingPipeline(
        embedder=FakeEmbedder(),
        card_summarizer=CardSummaryPipeline(provider=RecoveredProvider()),
    )
    assert pipeline.card_summarizer.is_current(fallback) is False

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=Store()):
        response = _embed_repository_job_locked(request, OwnedLock())

    assert response["status"] == "current"
    assert response["card_summary"]["source"] == "generated"
    assert response["card_summary"]["summary"] == generated_summary
    assert (
        response["card_summary"]["model_version"]
        == pipeline.card_summarizer.settings.model_id
    )


def test_same_version_different_content_cannot_attach_a_summary() -> None:
    request = _job(content_hash="requested")
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    point.payload["content_hash"] = "stored"
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline",
        return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder()),
    ), patch("api.v2.repository_store", return_value=store):
        with pytest.raises(HTTPException) as exc_info:
            _embed_repository_job_locked(request, OwnedLock())

    assert exc_info.value.status_code == 409
    store.compare_and_set_card_summary.assert_not_called()


def test_same_version_missing_stored_content_hash_cannot_attach_a_summary() -> None:
    request = _job()
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    point.payload.pop("content_hash")
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline",
        return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder()),
    ), patch("api.v2.repository_store", return_value=store):
        with pytest.raises(HTTPException) as exc_info:
            _embed_repository_job_locked(request, OwnedLock())

    assert exc_info.value.status_code == 409
    store.compare_and_set_card_summary.assert_not_called()


def test_artifact_hash_rejects_highlight_only_payload_tampering() -> None:
    payload = card_summary_payload(_artifact())
    payload["card_summary_highlights"] = ["Tampered highlight"]

    assert card_summary_from_payload(payload) is None


def test_applied_response_is_replayed_after_backend_crash() -> None:
    request = _job()
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    state: dict[str, SimpleNamespace | None] = {"point": None}
    embed_calls = 0
    original_embed = pipeline.embed_repository

    def counted_embed(source, deadline=None):
        nonlocal embed_calls
        embed_calls += 1
        return original_embed(source, deadline=deadline)

    pipeline.embed_repository = counted_embed  # type: ignore[method-assign]

    class Store:
        def compare_and_set_content(self, result, *, expected_point):
            state["point"] = SimpleNamespace(id=REPO_ID, payload=dict(result.payload))
            return state["point"]

    with patch(
        "api.v2._repository_points",
        side_effect=lambda _repo_id: [] if state["point"] is None else [state["point"]],
    ), patch("api.v2.repository_embedding_pipeline", return_value=pipeline), patch(
        "api.v2.repository_store", return_value=Store()
    ):
        first = _embed_repository_job_locked(request, OwnedLock())
        retry = _embed_repository_job_locked(request, OwnedLock())

    assert first["status"] == "applied"
    assert retry["status"] == "duplicate"
    assert retry["card_summary"] == first["card_summary"]
    assert embed_calls == 1


def _qdrant_store() -> tuple[QdrantClient, QdrantRepositoryStore]:
    client = QdrantClient(":memory:")
    store = QdrantRepositoryStore(client=client)
    client.create_collection(
        store.collection_name,
        vectors_config={
            store.vector_name: models.VectorParams(
                size=store.vector_size,
                distance=models.Distance.COSINE,
            )
        },
    )
    return client, store


def test_qdrant_summary_cas_backfills_missing_artifact_and_fences_new_content() -> None:
    client, store = _qdrant_store()
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        {
            "repo_id": REPO_ID,
            "full_name": "weave/qdrant-summary",
            "description": "A repository used to verify atomic summary updates.",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["qdrant"],
            "content_version": 1,
        }
    )
    result.payload["content_job_id"] = str(uuid.uuid4())
    store.upsert([result])
    client.delete_payload(
        store.collection_name,
        keys=list(CARD_SUMMARY_PAYLOAD_FIELDS),
        points=[REPO_ID],
        wait=True,
    )
    point = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]

    updated = store.compare_and_set_card_summary(
        expected_point=point,
        artifact=_artifact(),
    )
    assert card_summary_from_payload(updated.payload) == _artifact()

    stale = updated
    client.set_payload(
        store.collection_name,
        payload={"content_version": 2, "content_job_id": str(uuid.uuid4())},
        points=[REPO_ID],
        wait=True,
    )
    stale_write = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=_artifact("A different summary remains fenced from the newer content revision. Its text must not replace the artifact selected for the winning repository state."),
    )
    assert stale_write.payload["content_version"] == 2
    assert card_summary_from_payload(stale_write.payload) == _artifact()


def test_qdrant_summary_cas_fences_a_concurrent_highlight_update() -> None:
    client, store = _qdrant_store()
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        {
            "repo_id": REPO_ID,
            "full_name": "weave/qdrant-summary-race",
            "description": "A repository used to verify atomic summary artifact updates.",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["qdrant"],
            "content_version": 1,
        }
    )
    result.payload["content_job_id"] = str(uuid.uuid4())
    store.upsert([result])
    stale = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]

    winning = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=_artifact(),
    )
    losing = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=CardSummaryArtifact(
            **{
                **_artifact().model_dump(),
                "highlights": ["Stale highlight"],
            }
        ),
    )

    assert card_summary_from_payload(winning.payload) == _artifact()
    assert card_summary_from_payload(losing.payload) == _artifact()
