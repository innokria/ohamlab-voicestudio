# OhamLab VoiceStudio

Modular Python SDK for Hindi speech generation, voice cloning, transcription and turn-based voice chat.

## Install

```pypy
pip install ohamlab-voicestudio
```
```bash
pip install ohamlab-voicestudio
pip install "ohamlab-voicestudio[tts]"
pip install "ohamlab-voicestudio[asr]"
pip install "ohamlab-voicestudio[chat]"
pip install "ohamlab-voicestudio[studio,tts,asr]"
pip install "ohamlab-voicestudio[train]"
```

These commands work **after publishing to PyPI**. For local development: `pip install -e ".[tts,studio,asr]"`.

## Launch

```bash
ohamlab-voice studio --model rahul7star/tts-nano-o-hi --device cpu
```

## Python

```python
from ohamlab_voicestudio import VoiceStudio
studio = VoiceStudio(model="rahul7star/tts-nano-o-hi", device="cpu")
rate, audio = studio.speak("नमस्ते")
rate, cloned = studio.clone("नमस्ते", "my_voice.wav")
print(studio.transcribe("question.wav"))
```

Voice chat is provider-neutral: supply a callable to `VoiceChat(recognizer, synthesizer, reply_fn)`.

## Training

`ohamlab-voice check-data manifest.jsonl` validates training data. The `training.lora` module supplies generic PEFT adapter setup. **A complete checkpoint-specific fine-tuning loop and model export are not yet implemented.** Do not treat this release as a replacement for a proven fine-tuning workflow.

## Model dependencies and rights

The default model is hosted separately at `rahul7star/tts-nano-o-hi`. This package does not bundle its weights or its `hima_tts` inference implementation. The checkpoint must contain compatible inference code and its dependencies; additional requirements may be necessary. Verify the model's terms before redistribution. If third-party code is incorporated later, retain required notices.

Voice cloning requires the speaker's consent. Uploaded recordings should be deleted according to your hosting privacy policy.

## Publishing

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Publish using your PyPI account / trusted publisher only after verifying the package name and metadata.

## Easy dataset preparation (two sources)

### Own voice

Provide a folder containing audio and a `transcripts.csv` with `file_name,text`
columns (optional `speaker`). Supported audio formats depend on ffmpeg.

```bash
ohamlab-voice dataset own --data ./my-voice --output ./prepared
```

### Hugging Face dataset

Inspect the dataset license, speaker consent and columns first. For a multi-speaker
dataset, select a single speaker explicitly.

```bash
ohamlab-voice dataset hf --dataset OWNER/DATASET --output ./prepared \
  --audio-column audio --text-column sentence --max-samples 1000
```

The output includes `audio/`, `metadata.csv`, `manifest.jsonl`, `report.json`.
Audio is mono 24 kHz WAV. The command rejects clips shorter than 0.7s or longer
than 30s. Automatic ASR transcript generation and manual review UI are future work.

## Upload a trained export to Hugging Face

This command uploads an **existing** exported folder; it does not train a model.
By default it creates a private model repository.

```bash
export HF_TOKEN=your_token  # Or run: hf auth login
ohamlab-voice publish --folder ./trained-model --repo USER/ohamlab-hindi-v1
```

Public uploads require `--public --allow-public`, and manual verification of
all base model, dataset, and voice redistribution rights. Do not use this
flag as a substitute for permission. Upload requires a Hugging Face account.

## Training status

**Not production-complete:** dataset preparation and publishing are implemented;
checkpoint-specific TTS LoRA training, checkpoint resume, and compatible model
export still require a verified implementation and GPU integration tests.
The generic `add_lora` helper is not a runnable TTS trainer. We deliberately do
not advertise a `train` CLI command that would silently fail.

## Hugging Face Docker Space

Copy `README_SPACE.md` to `README.md` in the Space repository, and include `Dockerfile`, `pyproject.toml`, and `src/`. Set Space SDK to Docker, port 7860. See `README_SPACE.md` for deployment notes.

## Experimental Kaggle training backend

The `ohamlab_voicestudio.training.backend` modules incorporate MIT-licensed
training code adapted from the uploaded fine-tuning project. The original MIT
copyright notice and third-party notices are retained under `LICENSES/`.
This code is maintained inside OhamLab and **does not install the upstream
fine-tuning Git repository**. It still requires `hima_tts` inference source
distributed with the selected base model. Check model license separately.

Run the notebook `notebooks/OhamLab_Kaggle_Trainer.ipynb` on a Kaggle GPU.
**Not GPU-tested** against `rahul7star/tts-nano-o-hi` yet. The example dataset
is extremely small and intended only for a pipeline smoke test.
