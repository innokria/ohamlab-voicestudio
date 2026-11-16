import csv
import json
import shutil
import wave
from pathlib import Path
import pytest
from ohamlab_voicestudio.training.prepare import prepare_own

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_prepare_own(tmp_path):
    folder = tmp_path / "my"
    folder.mkdir()
    with wave.open(str(folder / "clip.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(b"\0\0" * 24000)
    with (folder / "transcripts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "text"])
        w.writerow(["clip.wav", "नमस्ते"])
    report = prepare_own(folder, tmp_path / "out")
    assert report["accepted"] == 1
    assert (tmp_path / "out" / "manifest.jsonl").is_file()

def test_no_public_upload_without_confirmation(tmp_path):
    from ohamlab_voicestudio.training.publish import publish_model
    (tmp_path / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="Public uploads"):
        publish_model(tmp_path, "test/repo", private=False)
