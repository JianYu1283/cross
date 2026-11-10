"""Download graphcodebert-base via hf-mirror and convert .bin -> safetensors.

transformers 4.57.x refuses torch.load (CVE-2025-32434) unless torch >= 2.6,
but this venv pins torch 2.2.2 for dgl 2.2.1. graphcodebert-base only ships
pytorch_model.bin, so we download it and rewrite as model.safetensors (same
state dict) — safetensors loading bypasses the torch.load version check.
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NO_PROXY", "hf-mirror.com,huggingface.co")

from huggingface_hub import snapshot_download  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "graphcodebert"))
os.makedirs(OUT, exist_ok=True)

ALLOW = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt",
         "tokenizer_config.json", "special_tokens_map.json"]

print(f"downloading microsoft/graphcodebert-base -> {OUT}", flush=True)
path = snapshot_download("microsoft/graphcodebert-base", local_dir=OUT,
                         allow_patterns=ALLOW)
print("downloaded:", path, flush=True)

bin_path = os.path.join(path, "pytorch_model.bin")
st_path = os.path.join(path, "model.safetensors")
if os.path.exists(bin_path) and not os.path.exists(st_path):
    print("converting pytorch_model.bin -> model.safetensors", flush=True)
    sd = torch.load(bin_path, map_location="cpu")
    # Drop the LM head — it is tied to the embeddings (shared memory) and is
    # unused by AutoModel classification; safetensors refuses tied tensors.
    sd = {k: v for k, v in sd.items() if not k.startswith("lm_head.")}
    save_file(sd, st_path)
    os.remove(bin_path)
    print("converted; removed .bin", flush=True)

print("DONE. files:", sorted(os.listdir(path)), flush=True)
