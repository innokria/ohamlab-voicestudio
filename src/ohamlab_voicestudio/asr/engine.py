class SpeechRecognizer:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install pip install "ohamlab-voicestudio[asr]"') from exc
        self.engine = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language=None):
        segments, info = self.engine.transcribe(str(audio_path), language=language)
        return {"text": " ".join(s.text.strip() for s in segments).strip(), "language": info.language}
