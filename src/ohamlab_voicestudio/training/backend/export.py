"""Package a trained adapter as built-in voices, exactly the way the released model ships its own:
OUT/voices/adapter.safetensors  the adapter (W' = W + B A)
OUT/voices/voices.safetensors   per voice: vec_<Name> (its average voice vector) and ref_<Name> (~5 s of its own audio)
OUT/voices/voices.json          names (and genders, if you give them)
OUT also gets the base model's files, so OUT is a complete model folder:  TTS(OUT).say("...", voice="YourName").

    hima-finetune export --base hima-tts-nano-te --work work --out my-model [--genders Meera=female,Ravi=male]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="default: WORK/run/adapter.safetensors")
    ap.add_argument("--genders", default="", help="Name=female,Name2=male (optional)")
    ap.add_argument("--samples", default="", help="a line to render in every voice (default: one per language)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.base))
    import torch
    import safetensors.torch as st
    import yaml

    lang = yaml.safe_load(open(os.path.join(a.base, "config.yaml")))["language"]
    ref_frames = yaml.safe_load(open(os.path.join(a.base, "config.yaml")))["model"].get("ref_frames", 62)
    enc = os.path.join(a.work, "enc")
    rows = [json.loads(l) for l in open(os.path.join(enc, "index.jsonl"), encoding="utf-8")]
    z = np.load(os.path.join(enc, "vec.npz")); vec_of = {str(i): v.astype(np.float32) for i, v in zip(z["ids"], z["vec"])}
    lat = np.load(os.path.join(enc, "lat.lat.npy"), mmap_mode="r"); off = np.load(os.path.join(enc, "lat.off.npy"))
    genders = dict(kv.split("=", 1) for kv in a.genders.split(",") if "=" in kv)

    table, voices = {}, []
    for v in sorted({r["voice"] for r in rows}):
        tr = [r for r in rows if r["voice"] == v and r["split"] == "train"] or [r for r in rows if r["voice"] == v]
        vec = np.mean([vec_of[r["id"]] for r in tr], 0); vec /= np.linalg.norm(vec)
        # the reference: a complete 4-6 s sentence of this voice (ends on a full stop / question mark), first by id for repeatability
        cand = sorted([r for r in tr if 4.0 <= r["dur"] <= 6.0 and r["text_raw"].rstrip()[-1:] in ".?!।"], key=lambda r: r["id"]) \
            or sorted(tr, key=lambda r: abs(r["dur"] - 5))
        c = cand[0]; ref = np.asarray(lat[off[c["i"]]: off[c["i"] + 1]], dtype=np.float32)[:ref_frames]
        table[f"vec_{v}"] = torch.from_numpy(vec.astype(np.float32)); table[f"ref_{v}"] = torch.from_numpy(np.ascontiguousarray(ref))
        voices.append({"name": v, **({"gender": genders[v]} if v in genders else {})})

    os.makedirs(os.path.join(a.out, "voices"), exist_ok=True)
    for f in os.listdir(a.base):  # a complete model folder: the base files, then our voices
        s, d = os.path.join(a.base, f), os.path.join(a.out, f)
        if f in ("voices", ".git", ".cache") or os.path.exists(d):
            continue
        (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
    ad = a.adapter or os.path.join(a.work, "run", "adapter.safetensors")
    shutil.copy2(ad, os.path.join(a.out, "voices", "adapter.safetensors"))
    st.save_file(table, os.path.join(a.out, "voices", "voices.safetensors"))
    json.dump({"language": lang, "voices": voices}, open(os.path.join(a.out, "voices", "voices.json"), "w"), ensure_ascii=False, indent=1)
    print(f"model folder ready: {a.out}  voices: {[v['name'] for v in voices]}", flush=True)

    from .compat import patch_torchaudio
    patch_torchaudio()                                                          # hima_tts's own cloning path imports speechbrain too
    from hima_tts import TTS
    tts = TTS(a.out)
    line = a.samples or {"te": "నమస్కారం! ఇది నా కొత్త వాయిస్, చిన్న మోడల్ అయినా చక్కగా మాట్లాడుతుంది.",
                         "hi": "नमस्ते! यह मेरी नई आवाज़ है, छोटा मॉडल है पर बहुत साफ़ बोलता है।"}.get(lang, "Hello! This is my new voice.")
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    for v in voices:
        tts.save(tts.say(line, voice=v["name"], seed=2), os.path.join(a.out, "samples", f"{v['name']}.wav"))
    print("EXPORT_DONE samples in", os.path.join(a.out, "samples"), flush=True)


if __name__ == "__main__":
    main()
