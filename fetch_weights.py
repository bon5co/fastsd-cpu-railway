"""Download the SDXS-512-0.9 OpenVINO weights into /models at image build, pinned to a Hugging Face commit SHA.

A download at boot would sit inside Railway's healthcheck window, so the weights are part of the image.
rupeshs/sdxs-512-0.9-openvino is the OpenVINO conversion of IDKiro/sdxs-512-0.9 that FastSD CPU itself loads for
its SDXS-512 mode (src/frontend/webui/realtime_ui.py). Licence: openrail++ (read 2026-11-24), ungated.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
REV = "f6cacf7a521e8801904f821bafa9dec7e4d10594"  # rupeshs/sdxs-512-0.9-openvino, 2024-06-10, license: openrail++

snapshot_download("rupeshs/sdxs-512-0.9-openvino", revision=REV, local_dir=ROOT / "sdxs-512-0.9-openvino",
                  ignore_patterns=["out_image.png", ".gitattributes"])

# FastSD CPU's tiny autoencoder for OpenVINO (its SDXS-512 mode uses it): MIT, ungated, 2.4 MB per half.
TAESD_REV = "035ad7cc7e7258b69378894821a3e5c52cd4b547"  # rupeshs/taesd-ov, license: mit
snapshot_download("rupeshs/taesd-ov", revision=TAESD_REV, local_dir=ROOT / "taesd-ov", ignore_patterns=[".gitattributes"])

for p in sorted(ROOT.rglob("*")):
    if p.is_file() and not p.is_symlink():
        print(f"{p.stat().st_size:>12} {p}")
