"""Provider-neutral turn-based voice chat."""
class VoiceChat:
    def __init__(self, recognizer, synthesizer, reply_fn):
        if not callable(reply_fn):
            raise TypeError("reply_fn must be callable: (text, history) -> str")
        self.asr = recognizer
        self.tts = synthesizer
        self.reply_fn = reply_fn
        self.history = []

    def respond(self, audio_path, voice=None, language=None):
        transcription = self.asr.transcribe(audio_path, language=language)
        user_text = transcription["text"]
        if not user_text:
            raise ValueError("No speech detected")
        response = str(self.reply_fn(user_text, list(self.history))).strip()
        if not response:
            raise ValueError("LLM returned an empty response")
        result = self.tts.speak(response, voice=voice)
        self.history.extend([{"role": "user", "content": user_text},
                             {"role": "assistant", "content": response}])
        return {"transcript": user_text, "response": response, "audio": result}
