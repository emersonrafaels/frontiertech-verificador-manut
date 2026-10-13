import json

import pytest

from atest_ai.services.batch import run_batch
from tests.conftest import FakeBedrockClient


def _write_manifest(tmp_path, image, scope, images):
    """Grava imagens fictícias e um manifesto com um chamado no diretório temporário."""
    (tmp_path / "images").mkdir()
    for item in images:
        if not item.get("missing"):
            (tmp_path / item["file"]).write_bytes(image)
    manifest = {"requests": [{
        "request_id": "CH-1",
        "scope": scope.model_dump(),
        "images": [{k: v for k, v in item.items() if k != "missing"} for item in images],
    }]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_batch_writes_one_json_per_request_and_summary(tmp_path, cfg, image, scope, fake_client):
    """O lote grava <request_id>.json e summary.json com a concordância do rótulo."""
    manifest = _write_manifest(tmp_path, image, scope, [
        {"file": "images/a.jpg", "expected_compliant": True},
        {"file": "images/b.jpg", "expected_compliant": False},
    ])

    summary = run_batch(manifest, tmp_path / "out", cfg, client=fake_client)

    run_dir = tmp_path / "out" / summary["run_id"]
    request = json.loads((run_dir / "CH-1.json").read_text(encoding="utf-8"))
    assert [i["status"] for i in request["images"]] == ["visually_compliant", "visually_compliant"]
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["images_total"] == 2
    assert (summary["labeled_total"], summary["labeled_matches"]) == (2, 1)
    assert summary["images_failed"] == 0


def test_batch_isolates_failures_per_image(tmp_path, cfg, image, scope, make_reading):
    """Falha em uma foto (arquivo ausente ou recusa) não interrompe as demais."""
    manifest = _write_manifest(tmp_path, image, scope, [
        {"file": "images/missing.jpg", "missing": True},
        {"file": "images/ok.jpg"},
    ])
    client = FakeBedrockClient(make_reading(["met"] * 4).model_dump())

    summary = run_batch(manifest, tmp_path / "out", cfg, client=client)

    statuses = {row["file"]: row["status"] for row in summary["images"]}
    assert statuses == {"images/missing.jpg": "analysis_not_completed", "images/ok.jpg": "visually_compliant"}
    assert summary["images_failed"] == 1


def test_example_manifest_is_valid():
    """O manifesto de teste em data/input é válido e aponta para fotos existentes."""
    from atest_ai.config import PROJECT_ROOT
    from atest_ai.services.batch import load_manifest

    path = PROJECT_ROOT / "data" / "input" / "manifest.json"
    manifest = load_manifest(path)
    files = [path.parent / i.file for r in manifest.requests for i in r.images]
    if not all(f.exists() for f in files):
        pytest.skip("Fotos de teste não versionadas (data/input/images).")
    assert files


def test_batch_logs_progress_and_failures_with_loguru(tmp_path, cfg, image, scope, fake_client):
    """O lote registra progresso e falhas via loguru, sem a mensagem da exceção."""
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda m: messages.append((m.record["level"].name, m.record["message"])), level="INFO")
    try:
        manifest = _write_manifest(tmp_path, image, scope, [
            {"file": "images/missing.jpg", "missing": True},
            {"file": "images/ok.jpg"},
        ])
        run_batch(manifest, tmp_path / "out", cfg, client=fake_client)
    finally:
        logger.remove(sink_id)

    assert ("INFO", "Analisando CH-1 / images/ok.jpg") in messages
    assert ("WARNING", "Falha em CH-1 / images/missing.jpg: FileNotFoundError") in messages
