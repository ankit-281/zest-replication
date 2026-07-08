"""
ZEST Feature Adapter Training — Kaggle Notebook

This script trains the FeatureAdapter MLP that maps ZEST features
(HuBERT codes, wav2vec emotion, EASE speaker, F0 contour) into the
latent space expected by the frozen FACodec decoder.

=====================================================================
PIPELINE OVERVIEW
=====================================================================
1. Load the ZEST dataset (same CodeDataset as HiFi-GAN training)
2. Load the frozen FACodec encoder + decoder
3. For each audio sample:
   a. Extract ZEST features via dataset (HuBERT, wav2vec, EASE, F0)
   b. Encode the ground-truth audio through FACodec encoder → get target latents
   c. Pass ZEST features through FeatureAdapter → get predicted latents
   d. Compute L1 loss between predicted and target latents
   e. Backprop through adapter only (FACodec is frozen)

=====================================================================
KAGGLE SETUP
=====================================================================
1. Upload the following as a Kaggle dataset:
   - code/HiFi-GAN/feature_adapter.py
   - code/HiFi-GAN/facodec_wrapper.py
   - code/HiFi-GAN/setup_facodec.py
   - code/HiFi-GAN/dataset.py
   - code/HiFi-GAN/utils.py
   - code/HiFi-GAN/hubert_alladv.json
   - code/train_esd.txt, val_esd.txt
   - code/esd_f0_stats.pth
   - data/ directory (audio files)
   - F0_predictor/wav2vec_feats/
   - F0_predictor/f0_contours/
   - EASE/EASE_embeddings/

2. Enable GPU accelerator in Kaggle notebook settings

3. Run all cells in order

=====================================================================
ESTIMATED TIME
=====================================================================
- ~2-4 hours on Kaggle T4 GPU for 50 epochs
- Can be done on CPU but would take ~10-20x longer (not recommended)
"""

# =====================================================================
# Cell 1: Setup & Install Dependencies
# =====================================================================
import subprocess
import sys
import os

# Install required packages
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "einops", "amfm_decompy",
                "librosa", "soundfile"], check=True)

print("Dependencies installed successfully!")

# =====================================================================
# Cell 2: Set paths (MODIFY THESE for your Kaggle dataset layout)
# =====================================================================

# === MODIFY THESE PATHS ===
# Point these to your uploaded Kaggle dataset directories
KAGGLE_INPUT = "/kaggle/input"

# Where you uploaded the ZEST code files
CODE_DIR = os.path.join(KAGGLE_INPUT, "zest-code")  # Adjust this name

# Where you uploaded the audio data
DATA_DIR = os.path.join(KAGGLE_INPUT, "zest-data")   # Adjust this name

# Working directory for outputs
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

# === Auto-configured paths ===
HIFIGAN_DIR = os.path.join(CODE_DIR, "HiFi-GAN")
TRAIN_MANIFEST = os.path.join(CODE_DIR, "train_esd.txt")
VAL_MANIFEST = os.path.join(CODE_DIR, "val_esd.txt")
CONFIG_FILE = os.path.join(HIFIGAN_DIR, "hubert_alladv.json")

PITCH_FOLDER = os.path.join(CODE_DIR, "F0_predictor", "f0_contours")
EMO_FOLDER = os.path.join(CODE_DIR, "F0_predictor", "wav2vec_feats")
EASE_FOLDER = os.path.join(CODE_DIR, "EASE", "EASE_embeddings")

# Add HiFi-GAN dir to path so we can import dataset.py etc.
sys.path.insert(0, HIFIGAN_DIR)

print("Paths configured:")
print(f"  CODE_DIR: {CODE_DIR}")
print(f"  HIFIGAN_DIR: {HIFIGAN_DIR}")
print(f"  TRAIN_MANIFEST: {TRAIN_MANIFEST}")
print(f"  CONFIG_FILE: {CONFIG_FILE}")

# =====================================================================
# Cell 3: Download Amphion FACodec & Pretrained Weights
# =====================================================================
os.chdir(HIFIGAN_DIR)

# Run the setup script to download Amphion codec code + weights
exec(open(os.path.join(HIFIGAN_DIR, "setup_facodec.py")).read())

print("\nFACodec setup complete!")

# =====================================================================
# Cell 4: Import modules
# =====================================================================
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import DataLoader

# Import from ZEST codebase
from dataset import CodeDataset, parse_manifest
from utils import AttrDict

# Import new modules
from feature_adapter import FeatureAdapter, save_adapter
from facodec_wrapper import FACodecWrapper

print("All modules imported successfully!")

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================
# Cell 5: Load Config & Dataset
# =====================================================================
with open(CONFIG_FILE) as f:
    json_config = json.loads(f.read())

# Override paths in config for Kaggle
json_config["f0_stats"] = os.path.join(CODE_DIR, "esd_f0_stats.pth")
json_config["input_training_file"] = TRAIN_MANIFEST
json_config["input_validation_file"] = VAL_MANIFEST
h = AttrDict(json_config)

# Load training data
train_files = parse_manifest(TRAIN_MANIFEST)
train_dataset = CodeDataset(
    train_files, h.segment_size, h.code_hop_size,
    h.n_fft, h.num_mels, h.hop_size, h.win_size,
    h.sampling_rate, h.fmin, h.fmax,
    n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
    f0=h.get("f0", None), multispkr=h.get("multispkr", None),
    f0_stats=h.get("f0_stats", None),
    f0_normalize=h.get("f0_normalize", False),
    f0_feats=h.get("f0_feats", False),
    f0_median=h.get("f0_median", False),
    f0_interp=h.get("f0_interp", False),
    vqvae=h.get("code_vq_params", False),
    pitch_folder=PITCH_FOLDER,
    emo_folder=EMO_FOLDER,
)

# Load validation data
val_files = parse_manifest(VAL_MANIFEST)
val_dataset = CodeDataset(
    val_files, -1, h.code_hop_size,
    h.n_fft, h.num_mels, h.hop_size, h.win_size,
    h.sampling_rate, h.fmin, h.fmax,
    split=False,
    n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
    f0=h.get("f0", None), multispkr=h.get("multispkr", None),
    f0_stats=h.get("f0_stats", None),
    f0_normalize=h.get("f0_normalize", False),
    f0_feats=h.get("f0_feats", False),
    f0_median=h.get("f0_median", False),
    f0_interp=h.get("f0_interp", False),
    vqvae=h.get("code_vq_params", False),
    pitch_folder=PITCH_FOLDER,
    emo_folder=EMO_FOLDER,
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False,
    num_workers=2, pin_memory=True, drop_last=True,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# =====================================================================
# Cell 6: Initialize Models
# =====================================================================
# --- Frozen FACodec (encoder + decoder) ---
facodec = FACodecWrapper(
    weights_dir=os.path.join(HIFIGAN_DIR, "facodec_weights"),
    device=str(device),
    load_encoder=True,  # Need encoder for ground-truth latent extraction
)
facodec.eval()

# --- HuBERT code embedding ---
# The original CodeGenerator has nn.Embedding(100, 128) for HuBERT codes.
# We create a trainable embedding layer.
code_embedding = nn.Embedding(
    h.num_embeddings, h.embedding_dim
).to(device)

# --- Feature Adapter ---
adapter = FeatureAdapter(
    content_dim=h.embedding_dim,   # 128 (HuBERT embedding)
    emotion_dim=128,               # wav2vec emotion dim
    speaker_dim=128,               # EASE speaker embedding dim
    f0_dim=1,                      # F0 is 1-dimensional
    facodec_latent_dim=256,        # FACodec vq_post_emb channels
    facodec_spk_dim=256,           # FACodec speaker embedding dim
    hidden_dim=512,
).to(device)

print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
print(f"Code embedding parameters: {sum(p.numel() for p in code_embedding.parameters()):,}")

# =====================================================================
# Cell 7: Training Setup
# =====================================================================
# Only train adapter + code_embedding — FACodec is frozen
trainable_params = list(adapter.parameters()) + list(code_embedding.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

NUM_EPOCHS = 50
CHECKPOINT_EVERY = 10
BEST_VAL_LOSS = float("inf")

print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
print(f"Training for {NUM_EPOCHS} epochs")

# =====================================================================
# Cell 8: Training Loop
# =====================================================================
def extract_facodec_targets(audio_batch, facodec_wrapper):
    """
    Encode ground-truth audio through FACodec to get target latents.

    Args:
        audio_batch: [B, T_audio] — raw audio waveform
        facodec_wrapper: FACodecWrapper with encoder loaded

    Returns:
        vq_post_emb_target: [B, 256, T_codec]
        spk_embs_target: [B, 256]
    """
    # FACodec encoder expects [B, 1, T_audio]
    if audio_batch.dim() == 1:
        audio_batch = audio_batch.unsqueeze(0)
    if audio_batch.dim() == 2:
        audio_batch = audio_batch.unsqueeze(1)

    audio_batch = audio_batch.float()

    with torch.no_grad():
        result = facodec_wrapper.encode(audio_batch)

    return result["vq_post_emb"], result["spk_embs"]


def train_one_epoch(epoch, train_loader, adapter, code_embedding,
                    facodec, optimizer, device):
    adapter.train()
    code_embedding.train()
    total_loss = 0.0
    total_temporal_loss = 0.0
    total_speaker_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        feats, audio, filenames, mel_loss = batch

        # Move to device
        audio = audio.to(device)  # [B, T_audio]

        # Extract ZEST features
        code_indices = feats["code"].to(device)  # [B, T_code]
        f0 = feats.get("f0", torch.zeros(audio.shape[0], 1, 1)).to(device)
        spkr = feats.get("spkr", torch.zeros(audio.shape[0], 128)).to(device)
        emo = feats.get("emo_embed", torch.zeros(audio.shape[0], 128)).to(device)

        # Embed HuBERT codes
        content = code_embedding(code_indices.long())  # [B, T, 128]
        content = content.transpose(1, 2)  # [B, 128, T]

        # Ensure correct shapes
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)  # [B, 1, T]
        if spkr.dim() == 1:
            spkr = spkr.unsqueeze(0)
        if emo.dim() == 1:
            emo = emo.unsqueeze(0)

        # Extract FACodec ground-truth targets
        try:
            vq_target, spk_target = extract_facodec_targets(audio, facodec)
        except Exception as e:
            print(f"  [WARN] Skipping batch {batch_idx}: {e}")
            continue

        # Forward pass through adapter
        vq_pred, spk_pred = adapter(
            content_features=content,
            emotion_features=emo,
            speaker_embedding=spkr,
            f0_features=f0,
        )

        # Align temporal dimensions (adapter output T may differ from FACodec T)
        T_target = vq_target.shape[-1]
        T_pred = vq_pred.shape[-1]
        if T_pred != T_target:
            vq_pred = torchF.interpolate(
                vq_pred, size=T_target, mode="nearest"
            )

        # Compute losses
        temporal_loss = torchF.l1_loss(vq_pred, vq_target)
        speaker_loss = torchF.l1_loss(spk_pred, spk_target)
        loss = temporal_loss + 0.5 * speaker_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_temporal_loss += temporal_loss.item()
        total_speaker_loss += speaker_loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"(temporal: {temporal_loss.item():.4f}, "
                f"speaker: {speaker_loss.item():.4f})"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_temp = total_temporal_loss / max(num_batches, 1)
    avg_spk = total_speaker_loss / max(num_batches, 1)
    return avg_loss, avg_temp, avg_spk


@torch.no_grad()
def validate(val_loader, adapter, code_embedding, facodec, device):
    adapter.eval()
    code_embedding.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        feats, audio, filenames, mel_loss = batch
        audio = audio.to(device)

        code_indices = feats["code"].to(device)
        f0 = feats.get("f0", torch.zeros(audio.shape[0], 1, 1)).to(device)
        spkr = feats.get("spkr", torch.zeros(audio.shape[0], 128)).to(device)
        emo = feats.get("emo_embed", torch.zeros(audio.shape[0], 128)).to(device)

        content = code_embedding(code_indices.long()).transpose(1, 2)

        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        if spkr.dim() == 1:
            spkr = spkr.unsqueeze(0)
        if emo.dim() == 1:
            emo = emo.unsqueeze(0)

        try:
            vq_target, spk_target = extract_facodec_targets(audio, facodec)
        except Exception:
            continue

        vq_pred, spk_pred = adapter(
            content_features=content,
            emotion_features=emo,
            speaker_embedding=spkr,
            f0_features=f0,
        )

        T_target = vq_target.shape[-1]
        if vq_pred.shape[-1] != T_target:
            vq_pred = torchF.interpolate(
                vq_pred, size=T_target, mode="nearest"
            )

        temporal_loss = torchF.l1_loss(vq_pred, vq_target)
        speaker_loss = torchF.l1_loss(spk_pred, spk_target)
        loss = temporal_loss + 0.5 * speaker_loss

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# --- Main training loop ---
print("=" * 60)
print("Starting Adapter Training")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_loss, train_temp, train_spk = train_one_epoch(
        epoch, train_loader, adapter, code_embedding,
        facodec, optimizer, device,
    )
    val_loss = validate(val_loader, adapter, code_embedding, facodec, device)
    scheduler.step()

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} (temp: {train_temp:.4f}, spk: {train_spk:.4f}) | "
        f"Val Loss: {val_loss:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f} | "
        f"Time: {elapsed:.1f}s"
    )

    # Save checkpoints
    if epoch % CHECKPOINT_EVERY == 0:
        ckpt_path = os.path.join(WORK_DIR, f"adapter_epoch_{epoch}.pth")
        save_adapter(adapter, ckpt_path)
        # Also save code embedding
        torch.save(
            code_embedding.state_dict(),
            os.path.join(WORK_DIR, f"code_embedding_epoch_{epoch}.pth"),
        )
        print(f"  Checkpoint saved: {ckpt_path}")

    if val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_loss
        best_path = os.path.join(WORK_DIR, "adapter_best.pth")
        save_adapter(adapter, best_path)
        torch.save(
            code_embedding.state_dict(),
            os.path.join(WORK_DIR, "code_embedding_best.pth"),
        )
        print(f"  ★ New best model saved (val_loss: {val_loss:.4f})")

print("\n" + "=" * 60)
print("Training Complete!")
print(f"Best validation loss: {BEST_VAL_LOSS:.4f}")
print(f"Best model saved to: {os.path.join(WORK_DIR, 'adapter_best.pth')}")
print("=" * 60)

# =====================================================================
# Cell 9: Download trained weights
# =====================================================================
# After training completes, download these files from /kaggle/working/:
#   - adapter_best.pth          → Place in code/HiFi-GAN/
#   - code_embedding_best.pth   → Place in code/HiFi-GAN/
#
# Then run local inference:
#   python decoder_inference.py \
#       --adapter_checkpoint adapter_best.pth \
#       --pitch_folder ../F0_predictor/f0_contours \
#       --emo_folder ../F0_predictor/wav2vec_feats \
#       --convert --debug

print("\nFiles to download from /kaggle/working/:")
for f in os.listdir(WORK_DIR):
    if f.endswith(".pth"):
        fpath = os.path.join(WORK_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
