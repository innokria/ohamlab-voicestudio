"""Two-source dataset preparation: own voice or Hugging Face.

Uses soundfile for PCM audio; ffmpeg is required for compressed input.
Every exported clip is 24 kHz mono WAV with metadata.csv and manifest.jsonl.
"""
import csv
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

RATE = 24000
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}

def _safe_id(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))[:80] or "speaker"

def _convert(source, target, rate=RATE):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Colab: !apt-get install -y ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i",
           str(source), "-ac", "1", "-ar", str(rate), "-sample_fmt", "s16", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Audio conversion failed for {source}: {proc.stderr[-400:]}")

def _duration(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def _write_rows(rows, out, min_seconds=0.7, max_seconds=30):
    out = Path(out)
    clips = out / "audio"
    clips.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    for i, row in enumerate(rows):
        source, transcript, speaker = row
        transcript = str(transcript or "").strip()
        if not transcript:
            rejected.append({"source": str(source), "reason": "missing transcript"})
            continue
        dest = clips / f"{i:06d}_{_safe_id(speaker)}.wav"
        try:
            _convert(source, dest)
            duration = _duration(dest)
            if not min_seconds <= duration <= max_seconds:
                dest.unlink(missing_ok=True)
                rejected.append({"source": str(source), "reason": f"duration {duration:.2f}s outside limits"})
                continue
            accepted.append({"audio": f"audio/{dest.name}", "text": transcript,
                             "speaker": str(speaker), "duration": round(duration, 3)})
        except Exception as exc:
            dest.unlink(missing_ok=True)
            rejected.append({"source": str(source), "reason": str(exc)})
    with (out / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "text", "speaker"])
        w.writerows((x["audio"], x["text"], x["speaker"]) for x in accepted)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for item in accepted:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"accepted": len(accepted), "rejected": len(rejected),
              "minutes": round(sum(x["duration"] for x in accepted) / 60, 2),
              "speakers": sorted({x["speaker"] for x in accepted}),
              "rejections": rejected[:100]}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not accepted:
        raise ValueError(f"No valid recordings; see {out / 'report.json'}")
    return report

def prepare_own(folder, output, transcripts="transcripts.csv", speaker="my_voice"):
    folder = Path(folder)
    transcript_file = folder / transcripts
    if not transcript_file.exists():
        raise FileNotFoundError(f"Expected {transcript_file} with columns file_name,text[,speaker]")
    with transcript_file.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"file_name", "text"} <= set(reader.fieldnames or []):
            raise ValueError("transcripts.csv must have file_name and text columns")
        rows = []
        for item in reader:
            audio = (folder / item["file_name"]).resolve()
            if not audio.is_relative_to(folder.resolve()):
                raise ValueError("Audio path must be inside input folder")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            rows.append((audio, item["text"], item.get("speaker") or speaker))
    return _write_rows(rows, output)

def prepare_hf(dataset_id, output, split="train", audio_column="audio",
               text_column="sentence", speaker_column=None, speaker=None,
               max_samples=1000, token=None, trust_remote_code=False):
    try:
        from datasets import load_dataset, Audio
    except ImportError as exc:
        raise RuntimeError('Install "ohamlab-voicestudio[train]" to import datasets') from exc
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    # User is responsible for checking the dataset card/license before downloading.
    dataset = load_dataset(dataset_id, split=split, token=token,
                           trust_remote_code=trust_remote_code)
    required = {audio_column, text_column} | ({speaker_column} if speaker_column else set())
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}. Available: {dataset.column_names}")
    if speaker_column and speaker is None:
        raise ValueError("Specify --speaker to select one speaker from a multi-speaker dataset")
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        rows = []
        for i, item in enumerate(dataset):
            if len(rows) >= max_samples:
                break
            who = str(item[speaker_column]) if speaker_column else "dataset_voice"
            if speaker is not None and who != str(speaker):
                continue
            audio = item[audio_column]
            if isinstance(audio, dict) and audio.get("array") is not None:
                # Hugging Face datasets returns decoded waveform and sampling rate.
                path = temp / f"{i:06d}.wav"
                sf.write(str(path), audio["array"], audio["sampling_rate"])
            elif isinstance(audio, dict) and audio.get("path"):
                path = Path(audio["path"])
            elif isinstance(audio, str):
                path = Path(audio)
            else:
                raise ValueError("Unsupported audio format from dataset")
            rows.append((path, item[text_column], who))
        return _write_rows(rows, output)
