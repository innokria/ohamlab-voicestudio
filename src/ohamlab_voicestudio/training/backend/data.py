"""Loader for the packed training data written by prepare.py (memory-mapped).

Batches: shuffle, length-sorted pools, padded size <= max_frames frames. Per row: the clip's frames, its text pieces, and the voice
vector + reference frames of ANOTHER clip of the same voice, so the model learns to copy a voice, not memorise a clip."""

import json
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class Pack:
    def __init__(self, pack_dir: str, null_id: int, max_clip_frames: int = 375, max_text: int = 400):
        m = np.load(f"{pack_dir}/meta.npz")
        self.n_frames, self.shard_id, self.clip_i = m["n_frames"], m["shard_id"], m["clip_i"]
        self.spk, self.lang, self.tok_off, self.vec_row = m["spk"], m["lang"], m["tok_off"], m["vec_row"]
        self.spk_off, self.spk_rows = m["spk_off"], m["spk_rows"]
        self.parent = m["parent"] if "parent" in m.files else None  # split pieces of one segment share a parent
        self.tokens = np.load(f"{pack_dir}/tokens.npy", mmap_mode="r")
        self.vecs = np.load(f"{pack_dir}/vecs.npy", mmap_mode="r")
        self.shards = json.load(open(f"{pack_dir}/shards.json"))
        self.langs = json.load(open(f"{pack_dir}/langs.json"))
        self.null_id = null_id
        tl = np.diff(self.tok_off)
        self.ok = np.nonzero((self.n_frames >= 1) & (self.n_frames <= max_clip_frames) & (tl >= 1) & (tl <= max_text))[0]
        self._open = {}

    def frames(self, k: int) -> np.ndarray:
        s = int(self.shard_id[k])
        if s not in self._open:
            stem = self.shards[s]
            lat = np.load(f"{stem}.lat.npy", mmap_mode="r"); off = np.load(f"{stem}.off.npy")
            if off[-1] != len(lat):
                off = np.append(off, len(lat))
            self._open[s] = (lat, off)
        lat, off = self._open[s]
        i = int(self.clip_i[k])
        x = lat[off[i]: off[i + 1]]
        assert len(x) == self.n_frames[k], (self.shards[s], i, len(x), self.n_frames[k])
        return x

    def other_clip(self, k: int, rng: random.Random):
        """Index of ANOTHER clip of the same speaker that has a vector, or None."""
        sp = int(self.spk[k]); a, b = int(self.spk_off[sp]), int(self.spk_off[sp + 1])
        n = b - a
        own = self.vec_row[k] >= 0
        if n - (1 if own else 0) < 1:
            return None
        for _ in range(16):  # never this clip, never a piece of the same parent segment (its vector holds this clip's audio)
            j = int(self.spk_rows[a + rng.randrange(n)])
            if j != k and (self.parent is None or self.parent[j] != self.parent[k]):
                return j
        return None

    def other_vec(self, k: int, rng: random.Random):
        j = self.other_clip(k, rng)
        return None if j is None else self.vecs[self.vec_row[j]]

    def ref_window(self, j: int, rng: random.Random, max_frames: int) -> np.ndarray:
        """Up to max_frames contiguous frames of clip j (a random window when the clip is longer)."""
        x = self.frames(j)
        if len(x) <= max_frames:
            return x
        s = rng.randrange(len(x) - max_frames + 1)
        return x[s: s + max_frames]

    def collate(self, ks, rng: random.Random):
        T = int(max(self.n_frames[k] for k in ks))
        toks = [self.tokens[self.tok_off[k]: self.tok_off[k + 1]] for k in ks]
        L = max(len(t) for t in toks)
        lat = np.zeros((len(ks), T, 32), dtype=np.float32)
        text = np.full((len(ks), L), self.null_id, dtype=np.int64)
        spk = np.zeros((len(ks), self.vecs.shape[1] if len(self.vecs) else 512), dtype=np.float32)  # 192 = ECAPA
        has = np.zeros(len(ks), dtype=bool)
        RF = getattr(self, "ref_frames", 0)  # >0: also return up to RF frames of the SAME other clip the vector comes from
        refs = [None] * len(ks)
        for b, k in enumerate(ks):
            x = self.frames(k); lat[b, :len(x)] = x
            text[b, :len(toks[b])] = toks[b]
            j = self.other_clip(k, rng)
            if j is not None:
                spk[b] = self.vecs[self.vec_row[j]]; has[b] = True
                if RF > 0:
                    r = self.ref_window(j, rng, RF)
                    # longer reference (E2 TTS / VoiceStar: speaker similarity rises with prompt length): join more clips of
                    # the speaker, never the target clip or a piece of its segment, until RF frames
                    parts, used = [r], {j}
                    for _ in range(getattr(self, "ref_join", 0)):
                        if sum(len(p) for p in parts) >= RF:
                            break
                        j2 = self.other_clip(k, rng)
                        if j2 is None or j2 in used:
                            continue
                        used.add(j2); parts.append(self.frames(j2))
                    refs[b] = np.concatenate(parts, 0)[:RF] if len(parts) > 1 else r
        out = {"latents": torch.from_numpy(lat), "lens": torch.from_numpy(self.n_frames[ks].astype(np.int64)),
               "text_ids": torch.from_numpy(text), "text_len": torch.tensor([len(t) for t in toks]), "spk": torch.from_numpy(spk),
               "has_spk": torch.from_numpy(has), "voice_key": torch.from_numpy(self.spk[ks].astype(np.int64)),
               "lang": torch.from_numpy(self.lang[ks].astype(np.int64))}
        if RF > 0:
            R = max([len(r) for r in refs if r is not None] or [1])
            ref = np.zeros((len(ks), R, 32), dtype=np.float32); rl = np.zeros(len(ks), dtype=np.int64)
            for b, r in enumerate(refs):
                if r is not None:
                    ref[b, :len(r)] = r; rl[b] = len(r)
            out.update(ref=torch.from_numpy(ref), ref_len=torch.from_numpy(rl))
        return out

    def plan(self, ids, max_frames: int, rng: random.Random, pool: int):
        ids = list(ids); rng.shuffle(ids)
        out = []
        for i in range(0, len(ids), pool):
            part = sorted(ids[i:i + pool], key=lambda k: self.n_frames[k])
            cur = []
            for k in part:
                if cur and (len(cur) + 1) * self.n_frames[k] > max_frames:
                    out.append(cur); cur = []
                cur.append(k)
            if cur:
                out.append(cur)
        rng.shuffle(out)
        return out


class TrainStream(IterableDataset):
    def __init__(self, pack: Pack, max_frames: int, pool: int, rank: int, world: int, seed: int):
        self.p, self.mf, self.pool, self.rank, self.world, self.seed = pack, max_frames, pool, rank, world, seed

    def __iter__(self):
        wi = get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        epoch = 0
        while True:
            plan = self.p.plan(self.p.ok, self.mf, random.Random(self.seed * 1_000_003 + epoch), self.pool)
            n = len(plan) // self.world * self.world
            mine = plan[:n][self.rank::self.world][wid::nw]
            rng = random.Random((self.seed * 7919 + epoch) * 131 + self.rank * 17 + wid)
            for ks in mine:
                yield self.p.collate(ks, rng)
            epoch += 1


def train_loader(pack, max_frames, pool, rank, world, seed, workers=6):
    return DataLoader(TrainStream(pack, max_frames, pool, rank, world, seed), batch_size=None, num_workers=workers,
                      pin_memory=True, prefetch_factor=4 if workers else None, persistent_workers=workers > 0)


def valid_batches(pack: Pack, per_lang: int, max_frames: int):
    """Fixed validation groups: up to per_lang clips per language spread over the file, each with a fixed vector."""
    out = {}
    for li, lang in enumerate(pack.langs):
        ks = [int(k) for k in pack.ok if pack.lang[k] == li]
        step = max(1, len(ks) // per_lang)
        sel = ks[::step][:per_lang]
        sel.sort(key=lambda k: pack.n_frames[k])
        bs, cur = [], []
        for k in sel:
            if cur and (len(cur) + 1) * pack.n_frames[k] > max_frames:
                bs.append(cur); cur = []
            cur.append(k)
        if cur:
            bs.append(cur)
        out[lang] = [pack.collate(b, random.Random(1000 * li + i)) for i, b in enumerate(bs)]
    return out
