class VoiceStudio:
    """Simple facade; models load only when requested."""
    def __init__(self, model="rahul7star/tts-nano-o-hi", device="cpu", token=None):
        self.model, self.device, self.token = model, device, token
        self._tts = None
        self._asr = None

    def _speech(self):
        if self._tts is None:
            from .tts import HindiTTS
            self._tts = HindiTTS(self.model, self.device, self.token)
        return self._tts

    def speak(self, text, voice=None, seed=0):
        return self._speech().speak(text, voice, seed)

    def clone(self, text, reference_audio, seed=0):
        return self._speech().clone(text, reference_audio, seed)

    def transcribe(self, audio_path, language=None):
        if self._asr is None:
            from .asr import SpeechRecognizer
            self._asr = SpeechRecognizer(device=self.device)
        return self._asr.transcribe(audio_path, language)

    def voices(self):
        return self._speech().voices
