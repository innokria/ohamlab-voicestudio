"""Explicit opt-in publishing to a Hugging Face model repository."""
import json
from pathlib import Path

def publish_model(folder, repo_id, token=None, private=True, base_model="rahul7star/tts-nano-o-hi",
                  dataset_id=None, allow_public=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    if not private and not allow_public:
        raise ValueError("Public uploads require --allow-public and manual rights verification")
    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        raise ValueError("Export directory is empty")
    # Do not generate a model card that falsely claims ownership of upstream weights.
    card = folder / "README.md"
    if not card.exists():
        card.write_text(
            "---\nlanguage:\n- hi\npipeline_tag: text-to-speech\n"
            "license: other\n---\n"
            "# OhamLab fine-tuning output\n\n"
            f"Base checkpoint: `{base_model}`\n\n"
            f"Training dataset: `{dataset_id or 'user-provided; see provenance'}`\n\n"
            "## Rights and limitations\n"
            "This repository may contain derivative model artifacts. "
            "Verify upstream model and dataset rights before public redistribution. "
            "Voice cloning requires the speaker's consent.\n",
            encoding="utf-8")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                      commit_message="Upload OhamLab fine-tuning output")
    return f"https://huggingface.co/{repo_id}"
