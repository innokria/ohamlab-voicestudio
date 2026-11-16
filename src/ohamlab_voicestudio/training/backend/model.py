"""The model, in training form: text + voice vector (+ ~5 s of reference audio) -> 32-number codec frames at 12.5 Hz.

Row per example: [bos][text pieces][voice row][reference frames][bos_frame][frame_0] ... [frame_{T-1}]
Same layers and parameter names as hima_tts.model.SpeechModel, so an adapter trained here loads straight into the released model.
forward() returns the training loss: flow loss on every frame (LSD) + a stop loss."""
import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from hima_tts.flow_head import SimpleMLPAdaLN
from .lsd import LSD

SPK_DIM = 512


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    text_dim: int = 256
    d_model: int = 512
    n_heads: int = 8
    ff: int = 2048
    n_begin: int = 2
    n_middle: int = 4
    n_end: int = 2
    n_loops: int = 2
    n_streams: int = 4          # hyper-connection streams (1 = off)
    latent_dim: int = 32
    flow_dim: int = 512
    flow_depth: int = 6
    flow_batch_multiplier: int = 4
    eos_loss_weight: float = 0.3
    text_dropout: float = 0.1
    voice_dropout: float = 0.1
    stats_ema_decay: float = 0.999
    rope_theta: float = 10000.0
    spoil_max: float = 0.30      # max share of history frames spoiled , reached after spoil_ramp steps
    spoil_ramp: int = 10000
    spoil_min_win: int = 12      # 1 s
    spoil_max_win: int = 50      # 4 s
    hist_noise_max: float = 0.0  # Gaussian noise on the INPUT history (normalised units): per row sigma ~ U(0, max); 0 = off
    hist_noise_prob: float = 1.0 # share of rows that get history noise
    ref_frames: int = 0          # reference audio in context: up to this many frames (62 = 5 s) of ANOTHER clip of the speaker; 0 = off
    ref_dropout: float = 0.2     # share of training rows without reference audio , independent of the vector
    null_both: float = 0.0       # extra share of rows with NO voice at all (vector and reference both dropped): trains the no-speaker
                                 # (random voice) mode; independent drops alone give only ~2%
    spk_dim: int = 512           # voice vector width: 192 = ECAPA (speechbrain spkrec-ecapa-voxceleb)
    max_len: int = 1024          # positions (text + voice + frames) for the KV cache


def rope_cache(n: int, hd: int, theta: float, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(n, device=device).float(), inv)
    return torch.cos(f), torch.sin(f)  # [n, hd/2]


def apply_rope(x, cos, sin):  # x [B, H, T, hd]; cos/sin [T, hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w.to(x.dtype)


class Layer(nn.Module):
    """Pre-LN transformer layer: attention (q/k RMSNorm, RoPE) + GELU feed-forward, LayerScale on both branches."""

    def __init__(self, d, h, ff, scale):
        super().__init__()
        self.h, self.hd = h, d // h
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.ff1, self.ff2 = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
        self.ls1 = nn.Parameter(torch.full((d,), float(scale)))
        self.ls2 = nn.Parameter(torch.full((d,), float(scale)))

    def attn(self, x, cos, sin, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T]), apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # streaming: append to this pass's cache, attend to everything so far
            kc, vc = cache
            kc[:, :, pos:pos + T] = k; vc[:, :, pos:pos + T] = v
            K, V = kc[:, :, :pos + T], vc[:, :, :pos + T]
            if T == 1:
                y = F.scaled_dot_product_attention(q, K, V)
            else:
                m = torch.ones(T, pos + T, dtype=torch.bool, device=x.device).tril(diagonal=pos)
                y = F.scaled_dot_product_attention(q, K, V, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

    def forward(self, x, cos, sin, cache=None, pos=0):
        x = x + self.ls1 * self.attn(self.norm1(x), cos, sin, cache, pos)
        return x + self.ls2 * self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))


class LoopHC(nn.Module):
    """Loop-level hyper-connection (Hyperloop sec. 3): read weights H_pre (n), write weights H_post (n), diagonal H_res (n),
    all input-dependent through RMSNorm(flatten(Y)); a loop position embedding e. Near-identity at init:
    H_pre = 1/n, H_post = 1, H_res ~ 1."""

    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.norm = RMSNorm(n * d)
        self.w = nn.Linear(n * d, 3 * n, bias=False)
        nn.init.normal_(self.w.weight, std=0.02)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        self.b_pre = nn.Parameter(torch.full((n,), -math.log(n - 1) if n > 1 else 10.0))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.full((n,), 4.0))
        self.e = nn.Parameter(torch.zeros(d))

    def weights(self, Y):  # Y [B, T, n, d]
        z = self.w(self.norm(Y.flatten(-2))).float()
        p, q, r = z.split(self.n, dim=-1)
        pre = torch.sigmoid(self.alpha[0] * p + self.b_pre)
        post = 2 * torch.sigmoid(self.alpha[1] * q + self.b_post)
        res = torch.sigmoid(self.alpha[2] * r + self.b_res)
        return pre, post, res


class Backbone(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        d, h, ff = c.d_model, c.n_heads, c.ff
        self.begin = nn.ModuleList([Layer(d, h, ff, 1.0) for _ in range(c.n_begin)])
        self.middle = nn.ModuleList([Layer(d, h, ff, 1.0 / c.n_loops) for _ in range(c.n_middle)])
        self.end = nn.ModuleList([Layer(d, h, ff, 1.0) for _ in range(c.n_end)])
        self.hc = nn.ModuleList([LoopHC(d, c.n_streams) for _ in range(c.n_loops)]) if c.n_streams > 1 else None
        self.out_norm = nn.LayerNorm(d, eps=1e-5)
        cos, sin = rope_cache(c.max_len, d // h, c.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_passes(self) -> int:
        c = self.c
        return c.n_begin + c.n_middle * c.n_loops + c.n_end

    def forward(self, x, caches=None, pos=0):
        """x [B, T, d] -> plan [B, T, d]. caches: list of n_passes (k, v) buffers for streaming, or None (training)."""
        cs, sn = self.cos, self.sin
        it = iter(caches) if caches is not None else None
        nxt = (lambda: next(it)) if it is not None else (lambda: None)
        for L in self.begin:
            x = L(x, cs, sn, nxt(), pos)
        if self.hc is None:
            for _ in range(self.c.n_loops):
                for L in self.middle:
                    x = L(x, cs, sn, nxt(), pos)
        else:
            Y = x.unsqueeze(2).expand(-1, -1, self.c.n_streams, -1)
            for hc in self.hc:
                pre, post, res = hc.weights(Y)
                h = (pre.unsqueeze(-1).to(Y.dtype) * Y).sum(2)
                for L in self.middle:
                    h = L(h, cs, sn, nxt(), pos)
                Y = res.unsqueeze(-1).to(Y.dtype) * Y + post.unsqueeze(-1).to(Y.dtype) * (h + hc.e).unsqueeze(2)
            x = Y.mean(2)
        for L in self.end:
            x = L(x, cs, sn, nxt(), pos)
        return self.out_norm(x)

    def new_caches(self, B, n, device, dtype):
        hd = self.c.d_model // self.c.n_heads
        return [(torch.zeros(B, self.c.n_heads, n, hd, device=device, dtype=dtype),
                 torch.zeros(B, self.c.n_heads, n, hd, device=device, dtype=dtype)) for _ in range(self.n_passes)]


class TrainModel(nn.Module):
    emb_mean: torch.Tensor
    emb_std: torch.Tensor

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, ld = cfg.d_model, cfg.latent_dim
        self.text_embed = nn.Embedding(cfg.vocab_size + 1, cfg.text_dim)  # last row = null / padding
        self.text_up = nn.Linear(cfg.text_dim, d, bias=False)
        self.spk_in = nn.Linear(cfg.spk_dim, d)
        self.no_voice = nn.Parameter(torch.zeros(d))
        self.head_voice = nn.Linear(cfg.spk_dim, d, bias=False)
        self.head_no_voice = nn.Parameter(torch.zeros(d))
        self.bos_text = nn.Parameter(torch.randn(d) * 0.02)
        self.bos_frame = nn.Parameter(torch.randn(ld))
        self.frame_in = nn.Linear(ld, d, bias=False)
        self.backbone = Backbone(cfg)
        self.out_eos = nn.Linear(d, 1)
        self.flow_net = SimpleMLPAdaLN(ld, cfg.flow_dim, ld, d, cfg.flow_depth, num_time_conds=2)
        self.flow = LSD()
        self.register_buffer("emb_mean", torch.zeros(ld))
        self.register_buffer("emb_std", torch.ones(ld))
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head_voice:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.text_embed.weight, std=0.02)
        nn.init.zeros_(self.head_voice.weight)  # the head starts as v1's head: plan only
        if cfg.ref_frames > 0:  # reference frames -> rows after the voice row; zero map: a trained model starts unchanged
            self.ref_in = nn.Linear(ld, d, bias=False)
            nn.init.zeros_(self.ref_in.weight)
            self.ref_tag = nn.Parameter(torch.randn(d) * 0.02)
        for hc in (self.backbone.hc or []):
            nn.init.normal_(hc.w.weight, std=0.02)

    @property
    def null_id(self) -> int:
        return self.cfg.vocab_size

    def normalise(self, raw):
        return (raw - self.emb_mean) / self.emb_std

    def update_stats(self, latents, mask):
        with torch.no_grad():
            sel = latents[mask].float()
            n = torch.tensor(float(sel.shape[0]), device=sel.device)
            s1, s2 = sel.sum(0), sel.square().sum(0)
            if torch.distributed.is_initialized():
                for t in (n, s1, s2):
                    torch.distributed.all_reduce(t)
            mean = s1 / n
            std = (s2 / n - mean.square()).clamp(min=1e-8).sqrt()
            g = self.cfg.stats_ema_decay
            self.emb_mean.mul_(g).add_(mean, alpha=1 - g)
            self.emb_std.mul_(g).add_(std, alpha=1 - g)

    def voice_rows(self, spk, has):
        """spk [B, 512] (unit length), has [B] bool -> (backbone voice row [B, d], head voice add [B, d])."""
        row = torch.where(has[:, None], self.spk_in(spk), self.no_voice.to(spk.dtype))
        head = torch.where(has[:, None], self.head_voice(spk), self.head_no_voice.to(spk.dtype))
        return row, head

    def spoil_history(self, x, lens, voice_key, frac, gen):
        """Span swap on the INPUT history only: windows of 1-4 s replaced by the same positions of another row with a
        different speaker; about frac of each row's frames. Targets stay clean."""
        if frac <= 0 or x.shape[0] < 2:
            return x
        B, T, _ = x.shape
        out = x.clone()
        cpu_lens = lens.tolist()
        keys = voice_key.tolist()
        for b in range(B):
            n = cpu_lens[b]
            need = int(frac * n)
            donors = [j for j in range(B) if keys[j] != keys[b] and cpu_lens[j] >= 12]
            if need < self.cfg.spoil_min_win or not donors:
                continue
            got = 0
            while got < need:
                w = int(torch.randint(self.cfg.spoil_min_win, self.cfg.spoil_max_win + 1, (1,), generator=gen))
                w = min(w, n, max(self.cfg.spoil_min_win, need - got))  # do not overshoot the share by a whole window
                s = int(torch.randint(0, n - w + 1, (1,), generator=gen))
                j = donors[int(torch.randint(0, len(donors), (1,), generator=gen))]
                src = torch.arange(s, s + w) % cpu_lens[j]
                out[b, s:s + w] = x[j, src.to(x.device)]
                got += w
        return out

    def forward(self, latents, lens, text_ids, text_len, spk, has_spk, voice_key=None, spoil_frac=0.0, gen=None, update_stats=False,
                ref=None, ref_len=None, return_plans=False):
        """latents [B, T, 32] raw; lens [B]; text_ids [B, L] (null-padded); text_len [B] CPU; spk [B, 512]; has_spk [B] bool;
        ref [B, R, 32] raw frames of another clip of the speaker, ref_len [B] (0 = none), used only when cfg.ref_frames > 0."""
        cfg = self.cfg
        B, T, _ = latents.shape
        dev = latents.device
        text_len = text_len.cpu()  # shapes the row buffers on the CPU (no device sync)
        voice_key = voice_key.cpu() if voice_key is not None else None
        pos = torch.arange(T + 1, device=dev)[None]
        mask = pos[:, :T] < lens[:, None]
        if update_stats and self.training:
            self.update_stats(latents, mask)
        x = self.normalise(latents) * mask[..., None]
        if self.training:
            drop = torch.rand(B) < cfg.text_dropout
            text_len = torch.where(drop, torch.ones_like(text_len), text_len)
            text_ids = text_ids.masked_fill(drop.to(dev)[:, None], self.null_id)
            has_spk = has_spk & (torch.rand(B, device=dev) >= cfg.voice_dropout)
            null_both = torch.rand(B) < cfg.null_both  # CPU: also zeroes the reference length below
            has_spk = has_spk & ~null_both.to(dev)
        hist = x if (not self.training or voice_key is None) else self.spoil_history(x, lens, voice_key, spoil_frac, gen)
        if self.training and cfg.hist_noise_max > 0:  # make the history look like the model's own imperfect samples; targets stay clean
            on = (torch.rand(B, device=dev) < cfg.hist_noise_prob).to(hist.dtype)
            sig = torch.rand(B, device=dev, dtype=hist.dtype) * cfg.hist_noise_max * on
            hist = hist + sig[:, None, None] * torch.randn_like(hist) * mask[..., None]
        audio_in = torch.cat([self.bos_frame.expand(B, 1, -1).to(x.dtype), hist], dim=1)  # [B, T+1, 32]
        vrow, vhead = self.voice_rows(spk, has_spk)
        # reference audio rows (after the voice row); dropped per row independently of the vector
        R = 0
        if self.training and cfg.ref_frames > 0 and ref is None:
            raise RuntimeError("ref_frames > 0 but the batch carries no reference frames: the loader is not sending them "
                               "(Pack.ref_frames unset?)")
        if cfg.ref_frames > 0 and ref is not None:
            ref_len = ref_len.cpu().clamp(max=cfg.ref_frames)
            if self.training:
                ref_len = torch.where((torch.rand(B) < cfg.ref_dropout) | null_both, torch.zeros_like(ref_len), ref_len)
            R = int(ref_len.max())
        rlen = ref_len if R > 0 else torch.zeros_like(text_len)
        # per-row layout, exactly what streaming sees: [bos][its own pieces][voice][ref frames][bos_frame][frames]; padding only at the END
        L = int(text_len.max())
        temb = self.text_up(self.text_embed(text_ids[:, :L]))
        pre = 2 + text_len + rlen  # CPU: bos + pieces + voice + reference frames
        N = int((pre + T + 1).max())
        xs = torch.zeros(B, N, cfg.d_model, device=dev, dtype=temb.dtype)
        xs[:, 0] = self.bos_text.to(temb.dtype)
        rows = torch.arange(B, device=dev)
        tl = text_len.to(dev)
        tmask = torch.arange(L, device=dev)[None] < tl[:, None]
        tpos = 1 + torch.arange(L, device=dev)[None].expand(B, -1)
        xs[rows[:, None].expand(-1, L)[tmask], tpos[tmask]] = temb[tmask]
        xs[rows, 1 + tl] = vrow.to(xs.dtype)
        if R > 0:
            remb = self.ref_in(self.normalise(ref[:, :R].to(dev)).to(xs.dtype)) + self.ref_tag.to(xs.dtype)  # [B, R, d]
            rmask = torch.arange(R, device=dev)[None] < rlen.to(dev)[:, None]
            rpos = (2 + tl)[:, None] + torch.arange(R, device=dev)[None]
            xs[rows[:, None].expand(-1, R)[rmask], rpos[rmask]] = remb[rmask]
        apos = pre.to(dev)[:, None] + torch.arange(T + 1, device=dev)[None]  # [B, T+1]
        xs[rows[:, None].expand(-1, T + 1).reshape(-1), apos.reshape(-1)] = self.frame_in(audio_in.to(xs.dtype)).reshape(-1, cfg.d_model)
        out = self.backbone(xs)
        z = out.gather(1, apos[:, :, None].expand(-1, -1, out.shape[-1]))  # [B, T+1, d]
        if return_plans:  # tests: the per-frame backbone states, to compare with streaming generation
            return z
        is_eos = pos == lens[:, None]
        eos_valid = pos <= lens[:, None]
        logits = self.out_eos(z).squeeze(-1).float()
        eos = is_eos * F.softplus(-logits) + ~is_eos * F.softplus(logits)
        eos_loss = (eos * eos_valid).sum() / eos_valid.sum().clamp(min=1)
        c = z[:, :T] + vhead[:, None].to(z.dtype)
        fbm = cfg.flow_batch_multiplier
        sel_c = c[mask].repeat(fbm, 1)
        target = x[mask].repeat(fbm, 1)
        noise = torch.randn_like(target)
        flow_loss, m, _ = self.flow.loss(partial(self.flow_net, sel_c), noise, target)
        flow_loss = flow_loss.mean()
        loss = flow_loss + cfg.eos_loss_weight * eos_loss
        if cfg.ref_frames > 0 and R == 0:
            # A batch whose rows ALL lost their reference frames (dropout / none available): an exactly-zero term keeps ref_in and
            # ref_tag in the graph for DDP. It must never stand in for a loader that sends NO
            # reference at all: that is refused above.
            loss = loss + 0.0 * (self.ref_in.weight.sum() + self.ref_tag.sum())
        return loss, {"flow_diag": m["flow_diag"].detach(), "flow_loss": flow_loss.detach(), "eos_loss": eos_loss.detach(), "loss": loss.detach()}

