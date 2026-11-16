def launch(model="rahul7star/tts-nano-o-hi", device="cpu"):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Install pip install "ohamlab-voicestudio[studio,tts]"') from exc
    from ..studio import VoiceStudio
    studio = VoiceStudio(model=model, device=device)

    def speak(text, voice):
        return studio.speak(text, voice=voice or None)

    def clone(text, audio, consent):
        if not consent:
            raise gr.Error("Confirm you own or have permission to use this voice.")
        if not audio:
            raise gr.Error("Upload or record a voice reference.")
        return studio.clone(text, audio)

    with gr.Blocks(title="OhamLab VoiceStudio") as app:
        gr.Markdown("# OhamLab VoiceStudio")
        with gr.Tab("Text to Speech"):
            text = gr.Textbox(label="Hindi text", value="नमस्ते, आप कैसे हैं?")
            voice = gr.Textbox(label="Preset voice (leave blank for default)")
            out = gr.Audio(label="Generated speech")
            gr.Button("Generate").click(speak, [text, voice], out)
        with gr.Tab("Clone My Voice"):
            clone_text = gr.Textbox(label="Hindi text", value="नमस्ते, यह मेरी आवाज़ है।")
            ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Voice reference")
            consent = gr.Checkbox(label="I have permission to use this voice")
            cloned = gr.Audio(label="Cloned speech")
            gr.Button("Clone voice").click(clone, [clone_text, ref, consent], cloned)
        with gr.Tab("Speech to Text"):
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath")
            transcript = gr.Textbox(label="Transcription")
            gr.Button("Transcribe").click(lambda path: studio.transcribe(path)["text"], audio, transcript)
    app.queue().launch()
