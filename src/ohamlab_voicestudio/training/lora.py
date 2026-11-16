"""Generic LoRA adapter insertion for torch Linear modules.

This is a building block, not a complete Hima-compatible training loop.
"""
def add_lora(model, target_modules, rank=8, alpha=16):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError('Install pip install "ohamlab-voicestudio[train]"') from exc
    if not target_modules:
        raise ValueError("Explicit target_modules are required")
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(target_modules),
                     lora_dropout=0.05, bias="none")
    return get_peft_model(model, cfg)
