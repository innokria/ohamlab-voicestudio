from pathlib import Path

def test_training_backend_files_present():
    base = Path(__file__).resolve().parents[1] / "src" / "ohamlab_voicestudio" / "training" / "backend"
    assert (base / "train.py").is_file()
    assert (base / "prepare.py").is_file()
    assert (base / "export.py").is_file()

def test_kaggle_uses_safe_worker_count():
    import json
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "OhamLab_Kaggle_Trainer.ipynb"
    nb = json.loads(notebook.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert '"--workers","0"' in code
    assert 'GITHUB_BRANCH = "main"' in code
