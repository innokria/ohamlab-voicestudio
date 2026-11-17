# Releasing OhamLab VoiceStudio to PyPI

Current candidate version: **0.1.0b3**. Do not reuse a version number already published on PyPI.

## Prerequisites

- Confirm all source changes and license notices are committed to GitHub `main`.
- Confirm Kaggle training works with `--workers 0`; separately verify export and inference if claiming those capabilities.
- Configure PyPI Trusted Publishing (recommended) or use a scoped PyPI API token. Never commit tokens.

## Local build and checks

```bash
python -m pip install --upgrade build twine pytest
python -m pytest -q
python -m build
python -m twine check dist/*
```

Inspect the generated wheel and sdist to confirm `ohamlab_voicestudio/training/backend` and `LICENSES` are included.

## Upload

TestPyPI first if desired:

```bash
python -m twine upload --repository testpypi dist/*
```

Publish to PyPI only when ready:

```bash
python -m twine upload dist/*
```

Use `__token__` as username and a PyPI API token as password if prompted. A successful upload cannot be undone; published versions are immutable.

## Verify in a clean environment

```bash
python -m venv /tmp/ohamlab-release-check
/tmp/ohamlab-release-check/bin/python -m pip install 'ohamlab-voicestudio[train]==0.1.0b3'
/tmp/ohamlab-release-check/bin/python -c 'import ohamlab_voicestudio.training.backend.train; print("OK")'
```
