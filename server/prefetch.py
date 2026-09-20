"""Download every checkpoint at image build time.

Without this the first connection after a cold start blocks for roughly two
minutes: 142 MB DiT, 28 MB CAMPPlus, 82 MB HiFT vocoder, and a 1.27 GB
wav2vec2-xls-r-300m content encoder.

Runs on CPU -- no GPU needed at build time.
"""
import os
import sys

sys.path.insert(0, "/opt/seedvc")
os.chdir("/opt/seedvc")

from hf_utils import load_custom_model_from_hf  # noqa: E402

print("==> DiT checkpoint + config")
ckpt, cfg = load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_uvit_tat_xlsr_ema.pth",
    "config_dit_mel_seed_uvit_xlsr_tiny.yml",
)
print("   ", ckpt)

print("==> CAMPPlus speaker encoder")
load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin",
                          config_filename=None)

print("==> HiFT vocoder")
load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)

print("==> wav2vec2-xls-r-300m content encoder (~1.3 GB)")
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # noqa: E402
Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")

print("\nAll checkpoints cached in the image.")
