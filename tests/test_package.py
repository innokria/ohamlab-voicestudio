from ohamlab_voicestudio import VoiceStudio
from ohamlab_voicestudio.training.dataset import validate_manifest
import json

def test_lazy_init():
    studio = VoiceStudio()
    assert studio._tts is None
    assert studio.model == "rahul7star/tts-nano-o-hi"

def test_manifest(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"fake")
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"audio": "a.wav", "text": "नमस्ते", "speaker": "a"}))
    assert validate_manifest(path)["valid_rows"] == 1
