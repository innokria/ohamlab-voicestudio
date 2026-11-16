---
title: OhamLab VoiceStudio
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# OhamLab VoiceStudio

Docker-based Gradio Space using `rahul7star/tts-nano-o-hi`.

## Deploy
Create a Hugging Face Space with Docker SDK and upload the contents of this repository. Set `HF_TOKEN` in Space Settings > Variables and secrets if your checkpoint requires access. For persistence, attach storage mounted at `/data`. Use a GPU Space for GPU training; this Dockerfile defaults to CPU inference.

## Caveats
The current SDK loads the inference implementation from the model repository and its full dependency compatibility has not been verified in a Hugging Face Space. Voice cloning and model training are not end-to-end validated. The Gradio UI currently has TTS, cloning and transcription tabs; the dataset preparation and publishing features are CLI-only. The `train` extra does not yet implement a complete model-specific training loop.
