"""Train a voice adapter (LoRA) on the FROZEN base model, on one GPU. A free T4 (Kaggle / Colab) is enough.

The base never changes: the adapter adds y = W x + B (A x) to every linear layer of the backbone, the flow head and the stop head
(B starts at zero, so step 0 IS the base model). Only A, B and the loss-weighting net train, about 5M numbers.
Precision: bf16 on GPUs that have it (A100, L4, H100...), fp16 with a gradient scaler on T4 / P100 / V100.

    hima-finetune train --base hima-tts-nano-te --work work            (--steps N to override, --resume to continue)
Writes WORK/run/adapter_<step>.safetensors (+ adapter.safetensors = the last one) and WORK/run/progress.jsonl.
"""
import argparse
import json
import math
import os
import random
import sys
import time

# batches change shape every step; without this, a 16 GB T4 runs out of memory from fragmentation, not from real use
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ (self.B.t().to(x.dtype) * self.scale)


TARGETS = ("backbone.", "flow_net.", "out_eos")


def add_lora(model, r, alpha):
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith(TARGETS)]
    for n in names:
        parent, _, leaf = n.rpartition(".")
        pm = model.get_submodule(parent) if parent else model
        setattr(pm, leaf, LoRALinear(getattr(pm, leaf), r, alpha))
    return names


def lora_state(model):
    """The adapter as the released loader reads it: <layer>.A and <layer>.B, scale folded into B (so W' = W + B A)."""
    out = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{n}.A"] = m.A.detach().float().cpu().contiguous()
            out[f"{n}.B"] = (m.B.detach().float() * m.scale).cpu().contiguous()
    return out


def lr_at(step, a):
    if step < a.wst_warmup:
        return 0.0
    s, total = step - a.wst_warmup, a.steps - a.wst_warmup
    lr = a.lr * min(1.0, (s + 1) / a.warmup)
    start = int(total * 0.7)
    if s >= start:
        p = (s - start) / max(1, total - start)
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
    return lr


@torch.no_grad()
def validate(model, vb, device, dtype):
    model.eval()
    tot, n = {"loss": 0.0, "flow_loss": 0.0, "eos_loss": 0.0}, 0
    for b in vb:
        bb = {k: (v if k in ("text_len", "ref_len") else v.to(device)) for k, v in b.items() if k != "lang"}
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(0)
            with torch.autocast("cuda", dtype=dtype):
                _, m = model(**bb)
        for k in tot:
            tot[k] += float(m[k])
        n += 1
    model.train()
    return {k: round(v / max(n, 1), 5) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--steps", default="auto", help="auto = about 60 passes over your data (what the released voices got)"); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wst_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--wst_warmup", type=int, default=300)
    ap.add_argument("--batch_frames", type=int, default=3500, help="frames per batch (12.5 per second of audio); lower it if you run out of memory")
    ap.add_argument("--save_every", type=int, default=500); ap.add_argument("--valid_every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--resume", action="store_true", help="continue from WORK/run/state.pt (saved with every adapter), e.g. after a disconnect")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.base))
    import safetensors.torch as st
    import yaml
    from .data import Pack, train_loader, valid_batches
    from .model import ModelConfig, TrainModel

    assert torch.cuda.is_available(), "training needs a GPU (Kaggle: Settings > Accelerator > GPU T4)"
    device = torch.device("cuda")
    bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    torch.manual_seed(a.seed); random.seed(a.seed)
    run = os.path.join(a.work, "run"); os.makedirs(run, exist_ok=True)
    logf = open(os.path.join(run, "progress.jsonl"), "a")
    log = lambda **kv: (logf.write(json.dumps({"time": time.strftime("%H:%M:%S"), **kv}) + "\n"), logf.flush())

    mc = yaml.safe_load(open(os.path.join(a.base, "config.yaml")))["model"]
    model = TrainModel(ModelConfig(**{k: v for k, v in mc.items() if k in ModelConfig.__dataclass_fields__}))
    missing, unexpected = model.load_state_dict(st.load_file(os.path.join(a.base, "model.safetensors")), strict=False)
    assert not unexpected and all(k.startswith("flow.") for k in missing), ("base weights do not match", unexpected[:5], missing[:5])
    for p in model.parameters():
        p.requires_grad_(False)
    names = add_lora(model, a.rank, a.rank)
    for n, p in model.named_parameters():
        if n.startswith("flow.") or n.endswith((".A", ".B")):
            p.requires_grad_(True)
    model.to(device)
    lora_p = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    wst_p = [p for n, p in model.named_parameters() if n.startswith("flow.")]
    opt = torch.optim.AdamW([{"params": lora_p}, {"params": wst_p}], lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    gpu = torch.cuda.get_device_name()
    print(f"GPU {gpu}, {'bf16' if bf16 else 'fp16 + gradient scaler'}; adapter on {len(names)} layers, "
          f"{sum(p.numel() for p in lora_p) / 1e6:.1f}M trainable numbers; base frozen", flush=True)
    log(type="run_start", gpu=gpu, precision="bf16" if bf16 else "fp16", lora_modules=len(names), lora_params=sum(p.numel() for p in lora_p), args=vars(a))

    train = Pack(os.path.join(a.work, "pack_train"), model.cfg.vocab_size); valid = Pack(os.path.join(a.work, "pack_valid"), model.cfg.vocab_size)
    for pk in (train, valid):
        pk.ref_frames = model.cfg.ref_frames; pk.ref_join = 0
    print(f"train clips {len(train.ok)}, valid clips {len(valid.ok)}", flush=True)
    if str(a.steps) == "auto":  # the released voices saw their data ~67 times; aim for ~60 passes, within sensible bounds
        frames = int(train.n_frames[train.ok].sum())
        a.steps = int(min(12000, max(600, round(60 * frames / a.batch_frames / 100) * 100)))
        print(f"steps: auto -> {a.steps} (about 60 passes over {frames / 12.5 / 3600:.1f} h of audio)", flush=True)
    a.steps = int(a.steps)
    gen = torch.Generator().manual_seed(a.seed * 31)
    start = 0
    state_path = os.path.join(run, "state.pt")
    if a.resume:
        assert os.path.exists(state_path), f"--resume: no {state_path} yet (it is written with every saved adapter)"
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        assert ck["steps"] == a.steps, f"--resume: the saved run was for {ck['steps']} steps, this one is {a.steps}; pass --steps {ck['steps']}"
        own = dict(model.named_parameters())
        for n, t in ck["trainable"].items():
            own[n].data.copy_(t.to(device))
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["gen"])
        start = ck["step"]
        print(f"resumed at step {start}/{a.steps}", flush=True); log(type="resume", step=start)
    loader = iter(train_loader(train, a.batch_frames, 2048, 0, 1, a.seed + start, a.workers))
    vb = [b for bs in valid_batches(valid, 256, a.batch_frames).values() for b in bs]
    if start == 0:
        v0 = validate(model, vb, device, dtype); log(type="valid", step=0, **v0)
        print(f"step 0  valid loss {v0['loss']:.4f}  (the base model on your voices)", flush=True)
    acc, nlog, t0, t_start = {}, 0, time.time(), time.time()
    model.train()
    for step in range(start, a.steps):
        lr = lr_at(step, a)
        opt.param_groups[0]["lr"] = lr; opt.param_groups[1]["lr"] = a.wst_lr
        b = next(loader)
        bb = {k: (v if k in ("text_len", "ref_len") else v.to(device, non_blocking=True)) for k, v in b.items() if k != "lang"}
        bb["voice_key"] = b["voice_key"]
        with torch.autocast("cuda", dtype=dtype):
            loss, met = model(**bb, spoil_frac=0.3, gen=gen, update_stats=False)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(lora_p, 1.0); gw = torch.nn.utils.clip_grad_norm_(wst_p, 1.0)
        scaler.step(opt); scaler.update()
        for k, v in met.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        nlog += 1; done = step + 1
        if done % 50 == 0:
            sp = nlog / (time.time() - t0); left = (a.steps - done) / max(sp, 1e-6)
            means = {k: round(v / nlog, 5) for k, v in acc.items()}
            log(type="train", step=done, **means, lr=lr, grad_norm=float(gn), steps_per_s=round(sp, 3),
                mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), scale=float(scaler.get_scale()) if not bf16 else 1.0)
            print(f"step {done}/{a.steps}  loss {means['loss']:.4f}  {sp:.2f} steps/s  ~{left / 60:.0f} min left  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            acc, nlog, t0 = {}, 0, time.time()
        if done % a.valid_every == 0 or done == a.steps:
            v = validate(model, vb, device, dtype); log(type="valid", step=done, **v)
            print(f"  valid loss {v['loss']:.4f}", flush=True)
        if done % a.save_every == 0 or done == a.steps:
            p = os.path.join(run, f"adapter_{done:06d}.safetensors")
            st.save_file(lora_state(model), p)
            tmp = state_path + ".tmp"                     # written whole, then swapped in: a disconnect mid-save cannot corrupt it
            torch.save({"step": done, "steps": a.steps, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "gen": gen.get_state(),
                        "trainable": {n: q.detach().cpu() for n, q in model.named_parameters() if q.requires_grad}}, tmp)
            os.replace(tmp, state_path)
    st.save_file(lora_state(model), os.path.join(run, "adapter.safetensors"))
    log(type="done", step=a.steps, minutes=round((time.time() - t_start) / 60, 1))
    print(f"TRAIN_DONE {a.steps} steps in {(time.time() - t_start) / 60:.1f} min -> {run}/adapter.safetensors", flush=True)


if __name__ == "__main__":
    main()
