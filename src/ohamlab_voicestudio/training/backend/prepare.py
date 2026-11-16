"""Turn a folder of clips + their text into training data for a voice adapter.

Input: DATA_DIR/metadata.csv, one clip per line:   path/to/clip.wav|the exact words spoken|VoiceName
       (paths relative to DATA_DIR; any format soundfile reads: wav, flac, ogg, mp3; any sample rate; a header line is fine)
Output (WORK_DIR): the clips encoded by the model's own codec, each clip's voice vector, and two packs (train / valid) the
trainer memory-maps. About 3% of every voice's clips are held out for validation.

    hima-finetune prepare --base hima-tts-nano-te --data my_voices --work work
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import time

import numpy as np


def read_metadata(data_dir):
    path = next((os.path.join(data_dir, f) for f in ("metadata.csv", "metadata.tsv", "metadata.txt") if os.path.exists(os.path.join(data_dir, f))), None)
    if path is None:
        sys.exit(f"no metadata.csv in {data_dir}: one line per clip, 'path|text|voice'")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, parts in enumerate(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)):
            if not parts or not "".join(parts).strip():
                continue
            if len(parts) != 3:
                sys.exit(f"{path} line {n + 1}: expected 3 fields 'path|text|voice', got {len(parts)}: {parts}")
            p, text, voice = (x.strip() for x in parts)
            if n == 0 and p.lower() in ("path", "file", "audio") and voice.lower() in ("voice", "speaker", "name"):
                continue  # header
            rows.append(dict(path=p, text=text, voice=voice))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the downloaded hima-tts-nano-<lang> folder")
    ap.add_argument("--data", required=True, help="folder with metadata.csv and the clips")
    ap.add_argument("--work", required=True, help="output folder")
    ap.add_argument("--holdout", type=float, default=0.03, help="share of each voice's clips kept for validation")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.base))
    import torch
    import soundfile as sf
    import sentencepiece as spm
    import yaml
    from scipy.signal import resample_poly
    from hima_tts.tts import _codec
    from hima_tts.text.normalize import normalize

    lang = yaml.safe_load(open(os.path.join(a.base, "config.yaml")))["language"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    rows = read_metadata(a.data)
    print(f"{len(rows)} clips in metadata, {len(set(r['voice'] for r in rows))} voices, language {lang}, device {dev}", flush=True)
    codec = _codec(os.path.join(a.base, "codec.safetensors"), dev)
    from .compat import voice_print_model
    ecapa = voice_print_model(os.path.join(a.work, "ecapa"), dev)
    enc_dir = os.path.join(a.work, "enc"); os.makedirs(enc_dir, exist_ok=True)

    def load(r):
        try:
            x, sr = sf.read(os.path.join(a.data, r["path"]), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            return r, None, f"unreadable ({type(e).__name__})"
        x = x.mean(1)
        if sr != 24000:
            x = resample_poly(x, 24000, sr).astype(np.float32)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        if peak > 1.0:
            x = x / peak
        return r, x[: len(x) // 1920 * 1920], None

    lats, vecs, out, drop = [], [], [], {}
    by_voice = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for k, (r, x, err) in enumerate(ex.map(load, rows)):
            why = err
            if x is not None and why is None:
                if len(x) < 1920 * 6:
                    why = "shorter than 0.5 s"
                elif len(x) > 24000 * 29:
                    why = "longer than 29 s (split it)"
            if why is None:
                try:
                    text = " ".join(normalize(r["text"], language=lang).split())
                except Exception:  # noqa: BLE001
                    why = "text could not be normalised"
                else:
                    if not text:
                        why = "empty text"
            if why:
                drop[why] = drop.get(why, 0) + 1
                continue
            with torch.no_grad():
                lat = codec.encode_to_latent(torch.from_numpy(x)[None, None].to(dev)).float()[0].cpu().numpy()
                e = ecapa.encode_batch(torch.from_numpy(resample_poly(x, 2, 3).astype(np.float32))[None].to(dev)).squeeze().float().cpu().numpy()
            if not np.isfinite(lat).all():
                drop["bad audio (nan)"] = drop.get("bad audio (nan)", 0) + 1
                continue
            n = by_voice.setdefault(r["voice"], 0); by_voice[r["voice"]] = n + 1
            split = "train"                                                               # set below, once every voice's count is known
            rid = f"{r['voice']}:{k:06d}"
            out.append(dict(id=rid, voice=r["voice"], lang=lang, text=text, text_raw=r["text"], path=r["path"], split=split, n_frames=int(len(lat)),
                            dur=round(len(x) / 24000, 3), shard=os.path.join(enc_dir, "lat.npz"), i=len(lats)))
            lats.append(lat.astype(np.float16)); vecs.append((e / max(np.linalg.norm(e), 1e-6)).astype(np.float16))
            if len(lats) % 250 == 0:
                print(f"  encoded {len(lats)} clips, {time.time() - t0:.0f} s", flush=True)
    if not lats:
        sys.exit(f"no usable clips: {drop}")
    for v, n in by_voice.items():  # hold out ~holdout of each voice, at least one clip, spread evenly over its clips
        mine = [r for r in out if r["voice"] == v]
        k = max(1, round(n * a.holdout)) if a.holdout > 0 and n >= 4 else 0
        for j in range(k):
            mine[int((j + 0.5) * n / k)]["split"] = "holdout"
    np.save(os.path.join(enc_dir, "lat.lat.npy"), np.concatenate(lats, 0))
    np.save(os.path.join(enc_dir, "lat.off.npy"), np.cumsum([0] + [len(l) for l in lats]).astype(np.int64))
    np.savez(os.path.join(enc_dir, "vec.npz"), ids=np.array([r["id"] for r in out]), vec=np.stack(vecs))
    with open(os.path.join(enc_dir, "index.jsonl"), "w", encoding="utf-8") as o:
        for r in out:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    sp = spm.SentencePieceProcessor(model_file=os.path.join(a.base, "tokenizer.model"))
    allvec = np.stack(vecs).astype(np.float32)
    for split, name in (("train", "pack_train"), ("holdout", "pack_valid")):
        pack(out, split, sp, allvec, os.path.join(a.work, name))
    print("\nvoice              clips   minutes (train)")
    for v in sorted(by_voice):
        tr = [r for r in out if r["voice"] == v and r["split"] == "train"]
        mins = sum(r["dur"] for r in tr) / 60
        flag = "   <- under 20 min: expect a weaker voice" if mins < 20 else ""
        print(f"{v:18s} {len(tr):6d}   {mins:7.1f}{flag}")
    if drop:
        print("skipped:", ", ".join(f"{n} {why}" for why, n in sorted(drop.items(), key=lambda kv: -kv[1])))
    print(f"PREPARE_DONE {len(out)} clips, {sum(r['dur'] for r in out) / 3600:.2f} h, {time.time() - t0:.0f} s", flush=True)


def pack(rows, split, sp, allvec, out_dir):
    """Flat arrays the loader memory-maps: frames index, text pieces, and every clip's voice vector grouped by voice."""
    os.makedirs(out_dir, exist_ok=True)
    rows_all = rows
    sel = [r for r in rows if r["split"] == split]
    toks = [np.asarray(sp.encode(r["text"]), dtype=np.int32) for r in sel]
    voices = sorted({r["voice"] for r in rows_all}); vid = {v: i for i, v in enumerate(voices)}
    pos = {r["id"]: i for i, r in enumerate(rows_all)}
    vec_row = np.arange(len(sel), dtype=np.int32)
    vecs = allvec[[pos[r["id"]] for r in sel]] if sel else np.zeros((0, allvec.shape[1]), np.float32)
    vecs = (vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6)).astype(np.float16)
    spk = np.array([vid[r["voice"]] for r in sel], dtype=np.int32)
    order = np.argsort(spk, kind="stable")
    cnt = np.bincount(spk, minlength=len(voices))
    np.save(f"{out_dir}/tokens.npy", np.concatenate(toks).astype(np.int32) if toks else np.zeros(0, np.int32))
    np.save(f"{out_dir}/vecs.npy", vecs)
    np.savez(f"{out_dir}/meta.npz", n_frames=np.array([r["n_frames"] for r in sel], dtype=np.int32), shard_id=np.zeros(len(sel), dtype=np.int32),
             clip_i=np.array([r["i"] for r in sel], dtype=np.int32), spk=spk, lang=np.zeros(len(sel), dtype=np.int8),
             tok_off=np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64), vec_row=vec_row,
             spk_off=np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64), spk_rows=order.astype(np.int64),
             parent=np.arange(len(sel), dtype=np.int64))
    json.dump([rows_all[0]["shard"][: -len(".npz")]], open(f"{out_dir}/shards.json", "w"))
    json.dump([rows_all[0]["lang"]], open(f"{out_dir}/langs.json", "w"))
    json.dump(voices, open(f"{out_dir}/voices.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
