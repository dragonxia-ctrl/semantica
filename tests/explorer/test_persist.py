"""Tests for persisting Explorer edits back to the source file (#1817).

Node merges and annotations only ever lived in the in-memory ``GraphSession``,
so a restart silently discarded them. ``POST /api/persist`` writes them back.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from semantica.explorer.app import create_app  # noqa: E402
from semantica.explorer.session import GraphSession  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402


def _write_source(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "A", "type": "entity", "content": "stale apple"},
                    {"id": "A2", "type": "entity", "content": "apple duplicate"},
                    {"id": "K", "type": "entity", "content": "kept kiwi"},
                ],
                "edges": [
                    {"source_id": "A", "target_id": "K", "type": "related_to"},
                    {"source_id": "A2", "target_id": "K", "type": "related_to"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def source(tmp_path):
    return _write_source(tmp_path / "graph.json")


@pytest.fixture()
def session(source):
    return GraphSession.from_file(str(source))


@pytest.fixture()
def client(session):
    with TestClient(create_app(session=session)) as test_client:
        yield test_client


def _restart(source: Path) -> GraphSession:
    """What a server restart does: load the file again."""
    return GraphSession.from_file(str(source))


class TestSourcePath:
    def test_from_file_remembers_the_file(self, source, session):
        assert session.source_path == str(source.resolve())

    def test_in_memory_session_has_no_source(self):
        from semantica.context.context_graph import ContextGraph

        assert GraphSession(ContextGraph(advanced_analytics=False)).source_path is None

    def test_persist_without_a_source_is_a_conflict(self):
        from semantica.context.context_graph import ContextGraph

        session = GraphSession(ContextGraph(advanced_analytics=False))
        with TestClient(create_app(session=session)) as client:
            response = client.post("/api/persist")
        assert response.status_code == 409
        assert "--graph" in response.json()["detail"]


class TestPersist:
    def test_merge_and_annotation_survive_a_restart(self, client, source):
        merge = client.post(
            "/api/enrich/merge", json={"primary_id": "A", "duplicate_ids": ["A2"]}
        )
        assert merge.status_code == 200
        created = client.post(
            "/api/annotations", json={"node_id": "K", "content": "checked by hand"}
        )
        assert created.status_code == 201
        annotation_id = created.json()["annotation_id"]

        response = client.post("/api/persist")
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "status": "success",
            "file": "graph.json",  # basename only, never the server's path
            "nodes": 2,
            # A2 -> K was rewired to A -> K, which is the edge A already had:
            # edge ids are content-derived, so the duplicate collapses.
            "edges": 1,
            "annotations": 1,
        }

        restarted = _restart(source)
        assert set(restarted.graph.nodes) == {"A", "K"}  # A2 stays merged away
        restored = restarted.get_annotation(annotation_id)
        assert restored is not None
        assert restored["content"] == "checked by hand"
        assert restored["node_id"] == "K"

    def test_without_persist_a_restart_still_loses_edits(self, client, source):
        # Documents why the explicit route exists: nothing is written implicitly.
        client.post(
            "/api/enrich/merge", json={"primary_id": "A", "duplicate_ids": ["A2"]}
        )
        assert set(_restart(source).graph.nodes) == {"A", "A2", "K"}

    def test_replace_import_then_persist_matches_the_source_after_restart(
        self, client, source
    ):
        updated = json.dumps(
            {
                "nodes": [{"id": "K", "type": "entity"}, {"id": "B", "type": "entity"}],
                "edges": [{"source": "K", "target": "B", "type": "related_to"}],
            }
        )
        assert (
            client.post(
                "/api/import",
                params={"mode": "replace"},
                files={"file": ("graph.json", updated, "application/json")},
            ).status_code
            == 200
        )
        assert client.post("/api/persist").status_code == 200
        assert set(_restart(source).graph.nodes) == {"K", "B"}

    def test_deleting_the_last_annotation_does_not_resurrect_it(self, client, source):
        annotation_id = client.post(
            "/api/annotations", json={"node_id": "K", "content": "temporary"}
        ).json()["annotation_id"]
        client.post("/api/persist")
        sidecar = Path(str(source) + ".annotations.json")
        assert sidecar.exists()

        assert client.delete(f"/api/annotations/{annotation_id}").status_code == 204
        client.post("/api/persist")

        assert not sidecar.exists()
        assert _restart(source).annotations == {}

    def test_no_annotations_means_no_sidecar(self, client, source):
        client.post("/api/persist")
        assert not Path(str(source) + ".annotations.json").exists()

    def test_persisted_file_is_loadable_by_the_graph_loader(self, client, source):
        client.post("/api/persist")
        data = json.loads(source.read_text(encoding="utf-8"))
        assert {node["id"] for node in data["nodes"]} == {"A", "A2", "K"}
        assert len(data["edges"]) == 2


class TestPersistFailureModes:
    def test_unreadable_sidecar_does_not_block_startup(self, source):
        Path(str(source) + ".annotations.json").write_text(
            "{not json", encoding="utf-8"
        )
        session = _restart(source)
        assert set(session.graph.nodes) == {"A", "A2", "K"}
        assert session.annotations == {}

    def test_sidecar_with_unexpected_layout_is_ignored(self, source):
        Path(str(source) + ".annotations.json").write_text('["nope"]', encoding="utf-8")
        assert _restart(source).annotations == {}

    def test_failed_write_leaves_the_original_file_intact(
        self, client, session, source, monkeypatch
    ):
        before = source.read_text(encoding="utf-8")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(session.graph, "save_to_file", boom)
        response = client.post("/api/persist")

        assert response.status_code == 500
        assert "disk full" not in response.text  # no internals in the response
        assert source.read_text(encoding="utf-8") == before


class TestPersistAuth:
    def test_persist_requires_the_api_key(self, client, monkeypatch):
        monkeypatch.delenv("SEMANTICA_ALLOW_ANONYMOUS", raising=False)
        monkeypatch.setenv("SEMANTICA_API_KEY", "secret")
        assert client.post("/api/persist").status_code == 401
        assert (
            client.post("/api/persist", headers={"X-API-Key": "secret"}).status_code
            == 200
        )
