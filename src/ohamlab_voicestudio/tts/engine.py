"""Adapter for a compatible Hugging Face Hindi TTS checkpoint.

The checkpoint is fetched at runtime; it is not redistributed with this package.
"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "rahul7star/tts-nano-o-hi"

class HindiTTS:
    def __init__(self, model_id=DEFAULT_MODEL, device="cpu", token=None):
        self.model_id = model_id
        self.path = Path(snapshot_download(repo_id=model_id, token=token))
        # The compatible checkpoint includes its inference implementation.
        if not (self.path / "hima_tts").is_dir():
            raise RuntimeError("Model repo has no compatible hima_tts/ inference source. Check checkpoint layout.")
        sys.path.insert(0, str(self.path))
        try:
            from hima_tts import TTS, SAMPLE_RATE
        except ImportError as exc:
            raise RuntimeError("Missing model runtime dependencies. Install the [tts] extra and review the model's requirements.") from exc
        self.sample_rate = SAMPLE_RATE
        self.engine = TTS(str(self.path), device=device, use_voices=True)

    @property
    def voices(self):
        return list(self.engine.voices)

    def speak(self, text, voice=None, seed=0):
        if not str(text).strip():
            raise ValueError("Text must not be empty")
        if voice is None:
            choices = self.voices
            if not choices:
                raise ValueError("No preset voices found; supply a reference voice")
            voice = choices[0]
        return self.sample_rate, self.engine.say(text, voice=voice, seed=seed)

    def clone(self, text, reference_audio, seed=0):
        path = Path(reference_audio)
        if not path.is_file():
            raise FileNotFoundError(path)
        return self.speak(text, voice=str(path), seed=seed)
