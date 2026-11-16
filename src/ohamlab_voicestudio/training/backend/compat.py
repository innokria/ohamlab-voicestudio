"""speechbrain 1.0.x calls torchaudio.list_audio_backends(), which newer torchaudio (2.9+) removed: importing speechbrain then fails
with AttributeError. We only use speechbrain's voice-print model on arrays we load ourselves, so a stub is enough."""


def patch_torchaudio():
    try:
        import torchaudio
    except ImportError:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"


def voice_print_model(savedir, device):
    patch_torchaudio()
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir, run_opts={"device": "cuda:0" if str(device) == "cuda" else str(device)})
