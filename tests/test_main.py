import json

import pytest

import main
from atest_ai.services import analysis


def test_single_mode_reads_image_and_scope_files(tmp_path, cfg, image, scope, fake_client, monkeypatch, capsys):
    """Modo avulso lê imagem e escopo de arquivos e imprime o resultado em JSON."""
    image_path = tmp_path / "foto.jpg"
    image_path.write_bytes(image)
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(scope.model_dump_json(), encoding="utf-8")
    real_analyze = analysis.analyze
    monkeypatch.setattr(main, "analyze", lambda *a, **kw: real_analyze(*a, client=fake_client, **kw))

    result = main.run(cfg, image_path, scope_path)

    assert result["status"] == "visually_compliant"
    assert json.loads(capsys.readouterr().out)["image"]["file_name"] == "foto.jpg"


def test_run_requires_manifest_without_single_inputs(cfg):
    """Sem imagem/escopo avulsos e sem manifesto configurado, falha antes de chamar o modelo."""
    with pytest.raises(ValueError):
        main.run(cfg)


def test_single_mode_requires_both_inputs(cfg, tmp_path):
    """O modo avulso exige imagem e escopo juntos."""
    with pytest.raises(ValueError):
        main.run(cfg, image_path=tmp_path / "foto.jpg")


def test_main_sanitizes_errors(tmp_path, capsys):
    """Erros saem como JSON sanitizado em stderr, com exit code 1 e sem score."""
    with pytest.raises(SystemExit) as exc:
        main.main(tmp_path / "missing.jpg", tmp_path / "missing.json")
    assert exc.value.code == 1
    error = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert (error["status"], error["error_type"]) == ("analysis_not_completed", "FileNotFoundError")
    assert "visual_compliance_score" not in error
