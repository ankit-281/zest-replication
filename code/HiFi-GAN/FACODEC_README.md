# ZEST × FACodec Integration — Complete Knowledge Transfer Guide

> **Purpose**: This document provides everything needed to understand, maintain,
> and extend the FACodec decoder integration into the ZEST emotion transfer pipeline.
> It is designed for knowledge transfer in case the original implementer is unavailable.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Map](#file-map)
3. [Feature Dimensions Reference](#feature-dimensions-reference)
4. [Setup Instructions](#setup-instructions)
5. [Training the Adapter](#training-the-adapter)
6. [Running Inference](#running-inference)
7. [How the Pipeline Works (Detailed)](#how-the-pipeline-works-detailed)
8. [Key Design Decisions](#key-design-decisions)
9. [Known Incompatibilities](#known-incompatibilities)
10. [Future Upgrade Paths](#future-upgrade-paths)
11. [Troubleshooting](#troubleshooting)
12. [Reference: Original HiFi-GAN Pipeline](#reference-original-hifi-gan-pipeline)

---

## Architecture Overview

### Original Pipeline (HiFi-GAN)
```
Speech → EASE Speaker Embeddings → HuBERT Features → wav2vec Features
→ F0 Predictor → Converted F0 → HiFi-GAN CodeGenerator (trained) → Waveform
```

### New Pipeline (FACodec)
```
Speech → EASE Speaker Embeddings → HuBERT Features → wav2vec Features
→ F0 Predictor → Converted F0 → FeatureAdapter (trained MLP)
→ Frozen FACodec Decoder → Waveform
```

**What changed**: Only the waveform reconstruction stage. Everything upstream
(EASE, HuBERT, wav2vec, F0 predictor, pitch conversion) is identical.

**What was removed**: HiFi-GAN Generator, Discriminators, GAN losses, GAN training loop.
Files are preserved but no longer used in the new pipeline.

---

## File Map

### New Files (FACodec Pipeline)

| File | Purpose |
|------|---------|
| `code/HiFi-GAN/setup_facodec.py` | Downloads Amphion FACodec code + pretrained weights |
| `code/HiFi-GAN/facodec_wrapper.py` | Frozen FACodec decoder/encoder wrapper |
| `code/HiFi-GAN/feature_adapter.py` | Trainable MLP: ZEST features → FACodec latents |
| `code/HiFi-GAN/decoder_inference.py` | Standalone inference script (FACodec path) |
| `code/kaggle-notebooks/train_facodec_adapter.py` | Kaggle notebook for adapter training |
| `code/HiFi-GAN/FACODEC_README.md` | This file |

### Modified Files

| File | Change |
|------|--------|
| `code/HiFi-GAN/inference.py` | Added `--use_facodec` flag for backward-compatible switching |
| `code/requirements.txt` | Added `huggingface_hub` dependency |

### Auto-Downloaded Files (by setup_facodec.py)

| Path | Source |
|------|--------|
| `code/HiFi-GAN/amphion_codec/ns3_codec/` | Amphion GitHub (sparse clone) |
| `code/HiFi-GAN/facodec_weights/ns3_facodec_encoder.bin` | HuggingFace |
| `code/HiFi-GAN/facodec_weights/ns3_facodec_decoder.bin` | HuggingFace |

### Unchanged Files (DO NOT MODIFY)

Everything in `code/EASE/` and `code/F0_predictor/` remains untouched.
`code/HiFi-GAN/dataset.py`, `utils.py`, `modules/` are also unchanged.

### Superseded Files (Kept for Reference)

| File | Status |
|------|--------|
| `code/HiFi-GAN/train.py` | GAN training loop — no longer used |
| `code/HiFi-GAN/models.py` | Generator/Discriminator — not imported by new code |

---

## Feature Dimensions Reference

This table documents every feature's shape as it flows through the pipeline.
**This is the most important reference** for debugging dimension mismatches.

### ZEST Features (Inputs to FeatureAdapter)

| Feature | Source | Shape | How Loaded |
|---------|--------|-------|------------|
| HuBERT codes | `CodeDataset` → `code` key | `[B, T_code]` (LongTensor) | Embedded to `[B, 128, T_code]` via `nn.Embedding(100, 128)` |
| F0 contour | `CodeDataset` → `f0` key | `[B, 1, T_f0]` | From `f0_contours/*.npy` or `pred_DSDT_f0/` |
| EASE speaker | `CodeDataset` → `spkr` key | `[B, 128]` (float) | From `EASE_embeddings/*.npy` (originally 192-dim x-vector → 128 after EASE linear) |
| wav2vec emotion | `CodeDataset` → `emo_embed` key | `[B, 128]` (float) | From `wav2vec_feats/*.npy` |

### FACodec Expected Inputs (Outputs of FeatureAdapter)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `vq_post_emb` | `[B, 256, T_codec]` | Post-VQ embedding (sum of all codebook outputs) |
| `spk_embs` | `[B, 256]` | Global timbre/speaker embedding |

### Key Relationships

- ZEST sample rate: **16kHz**
- FACodec sample rate: **16kHz** ✓ (compatible)
- ZEST `code_hop_size`: **320** samples
- FACodec `hop_size`: **200** samples
- This means T_code ≠ T_codec for the same audio duration
- The adapter uses `F.interpolate(mode='nearest')` to align

---

## Setup Instructions

### Prerequisites
- Python 3.8+ with the existing ZEST virtual environment
- Git (for sparse-cloning Amphion)
- Internet access (for downloading weights)

### Step-by-step

```powershell
cd d:\ZEST\ZEST\code\HiFi-GAN

# 1. Install new dependency
..\venv\Scripts\pip.exe install huggingface_hub

# 2. Run setup (downloads ~200MB of model code + weights)
..\venv\Scripts\python.exe setup_facodec.py
```

After setup, you should see:
```
code/HiFi-GAN/
├── amphion_codec/
│   └── ns3_codec/          ← Amphion model definitions
│       ├── facodec.py
│       ├── quantize/
│       ├── alias_free_torch/
│       ├── transformer.py
│       ├── gradient_reversal.py
│       └── melspec.py
├── facodec_weights/
│   ├── ns3_facodec_encoder.bin  ← Pretrained encoder (~100MB)
│   └── ns3_facodec_decoder.bin  ← Pretrained decoder (~100MB)
├── facodec_wrapper.py      ← Wrapper module
├── feature_adapter.py      ← Adapter module
├── decoder_inference.py    ← New inference script
└── setup_facodec.py        ← This setup script
```

---

## Training the Adapter

The FeatureAdapter MLP must be trained before it can produce meaningful audio.
Training requires GPU — use the provided Kaggle notebook.

### What Gets Trained
- `FeatureAdapter` (MLP): ~400K parameters
- `nn.Embedding(100, 128)` (HuBERT code embedding): ~12K parameters
- **Total**: ~412K trainable parameters

### What Stays Frozen
- FACodec Encoder: used only to extract ground-truth targets
- FACodec Decoder: never trained, only used at inference

### Training Process

1. **Upload to Kaggle** as a dataset:
   - `code/HiFi-GAN/feature_adapter.py`
   - `code/HiFi-GAN/facodec_wrapper.py`
   - `code/HiFi-GAN/setup_facodec.py`
   - `code/HiFi-GAN/dataset.py`
   - `code/HiFi-GAN/utils.py`
   - `code/HiFi-GAN/hubert_alladv.json`
   - `code/train_esd.txt`, `val_esd.txt`
   - `code/esd_f0_stats.pth`
   - `code/data/` (audio directory)
   - `code/F0_predictor/wav2vec_feats/`
   - `code/F0_predictor/f0_contours/`
   - `code/EASE/EASE_embeddings/`

2. **Create Kaggle notebook** from `code/kaggle-notebooks/train_facodec_adapter.py`

3. **Enable GPU** in notebook settings

4. **Adjust paths** in Cell 2 to match your Kaggle dataset names

5. **Run all cells** — training takes ~2-4 hours on T4 GPU

6. **Download outputs** from `/kaggle/working/`:
   - `adapter_best.pth` → place in `code/HiFi-GAN/`
   - `code_embedding_best.pth` → place in `code/HiFi-GAN/`

### Training Loss Explained

The training minimizes:
```
Loss = L1(predicted_vq_post_emb, target_vq_post_emb) 
     + 0.5 * L1(predicted_spk_embs, target_spk_embs)
```

Where targets are extracted by encoding ground-truth audio through FACodec's encoder.

---

## Running Inference

### Prerequisites
- Trained adapter weights (`adapter_best.pth`)
- FACodec pretrained weights (from `setup_facodec.py`)
- Pre-extracted ZEST features (wav2vec feats, predicted F0, EASE embeddings)

### Commands

```powershell
cd d:\ZEST\ZEST\code\HiFi-GAN

# Option 1: Standalone script (recommended)
..\venv\Scripts\python.exe decoder_inference.py ^
    --adapter_checkpoint adapter_best.pth ^
    --pitch_folder ../F0_predictor/f0_contours ^
    --emo_folder ../F0_predictor/wav2vec_feats ^
    --output_dir DSDT_facodec ^
    --convert --debug

# Option 2: Via original inference.py with --use_facodec flag
..\venv\Scripts\python.exe inference.py ^
    --checkpoint_file checkpoints/ESD/g_00004200 ^
    --adapter_checkpoint adapter_best.pth ^
    --pitch_folder ../F0_predictor/f0_contours ^
    --emo_folder ../F0_predictor/wav2vec_feats ^
    --use_facodec --convert --debug

# Option 3: Original HiFi-GAN (still works, no changes needed)
..\venv\Scripts\python.exe inference.py ^
    --checkpoint_file checkpoints/ESD/g_00004200 ^
    --pitch_folder ../F0_predictor/f0_contours ^
    --emo_folder ../F0_predictor/wav2vec_feats ^
    --convert --debug
```

Output WAV files are written to `--output_dir` (default: `DSDT_facodec/`).

---

## How the Pipeline Works (Detailed)

### Step 1: Dataset Loading (unchanged)
`CodeDataset.__getitem__()` loads:
- HuBERT code indices from the manifest file
- F0 contour from `pitch_folder/*.npy`
- EASE speaker embedding from `EASE_embeddings/*.npy`
- wav2vec emotion embedding from `emo_folder/*.npy`

### Step 2: Feature Adaptation (NEW)
`FeatureAdapter.forward()` does:
1. Embeds HuBERT code indices → `[B, 128, T]` via `nn.Embedding`
2. Broadcasts global features (emotion, speaker) to temporal dimension
3. Concatenates: `[B, 128+128+128+1, T]` = `[B, 385, T]`
4. Projects via Conv1d MLP: `[B, 385, T]` → `[B, 512, T]` → `[B, 256, T]`
5. Projects speaker via Linear MLP: `[B, 128]` → `[B, 512]` → `[B, 256]`

### Step 3: Waveform Synthesis (NEW)
`FACodecWrapper.decode()` takes:
- `vq_post_emb [B, 256, T]` → treated as if it came from VQ codebooks
- `spk_embs [B, 256]` → treated as timbre embedding
- Produces waveform `[B, 1, T_audio]` where `T_audio ≈ T × 200`

---

## Key Design Decisions

1. **MLP adapter, not attention**: Keeps it simple, trainable with small data.
   Can be swapped for attention-based fusion later without changing anything else.

2. **Conv1d instead of Linear for temporal projection**: Preserves temporal
   structure. Each time step is projected independently (kernel_size=1).

3. **Separate speaker projection**: FACodec expects a global speaker embedding
   separate from temporal features. We project EASE→FACodec speaker space directly.

4. **Nearest-neighbor interpolation for temporal alignment**: ZEST uses
   hop_size=320, FACodec uses hop_size=200. We interpolate rather than
   resample to avoid information loss.

5. **CPU-only compatible**: All code runs on CPU. Training uses Kaggle GPU.

6. **Vendored Amphion code**: Downloaded via sparse git clone rather than
   pip install (Amphion isn't pip-installable). Stored in `amphion_codec/`.

---

## Known Incompatibilities

### 1. Latent Space Mismatch (Critical)
FACodec expects VQ codebook latents from its own encoder. ZEST produces
HuBERT codes + wav2vec embeddings + EASE embeddings + F0 contours. These
are fundamentally different representations. The adapter MLP must be
**trained** to bridge this gap. Without training, output is noise.

### 2. Temporal Resolution Difference
- ZEST code_hop_size: 320 samples → 50 frames/sec at 16kHz
- FACodec hop_size: 200 samples → 80 frames/sec at 16kHz
- Solved via interpolation in the adapter.

### 3. Speaker Embedding Provenance
- ZEST uses EASE (128-dim, emotion-disentangled x-vector derivative)
- FACodec uses its own timbre extractor (256-dim, from CNNLSTM)
- The adapter's speaker MLP learns to map between these spaces.

---

## Future Upgrade Paths

The modular design supports these swaps without touching the decoder:

| Swap | What to Change | What Stays |
|------|---------------|------------|
| HuBERT → WavLM | Change feature extractor, update `content_dim` in adapter | Adapter architecture, FACodec decoder |
| wav2vec → emotion2vec | Change emotion extractor, update `emotion_dim` | Everything else |
| EASE → CAM++ | Change speaker extractor, update `speaker_dim` | Everything else |
| MLP adapter → Attention | Replace `FeatureAdapter` class, same interface | FACodec decoder, dataset, inference |

The adapter interface always receives:
```python
adapter(
    content_features,    # [B, C_content, T]
    emotion_features,    # [B, C_emotion] or [B, C_emotion, T]
    speaker_embedding,   # [B, C_speaker]
    f0_features,         # [B, C_f0, T]
)
```

---

## Troubleshooting

### "FACodec model code not found"
```
Run: python setup_facodec.py
```

### "Decoder weights not found"
```
Run: python setup_facodec.py
# Or manually download from: https://huggingface.co/amphion/naturalspeech3_facodec
```

### Shape mismatch errors
Check the [Feature Dimensions Reference](#feature-dimensions-reference) table.
Most issues are caused by:
- Missing batch dimension (use `.unsqueeze(0)`)
- Missing channel dimension for F0 (should be `[B, 1, T]`)
- Speaker embedding shape (should be `[B, 128]`, not `[B, 1, 128]`)

### Output is noise/static
The adapter has not been trained. See [Training the Adapter](#training-the-adapter).

### Import errors
Ensure the setup script ran successfully and that `amphion_codec/ns3_codec/`
exists with all submodules (alias_free_torch, quantize, etc.).

### Memory errors on CPU
FACodec decoder is ~100M parameters. For long audio files, process in chunks
or reduce batch size to 1.

---

## Reference: Original HiFi-GAN Pipeline

The original pipeline (still functional) uses:
- `CodeGenerator` from `models.py` — generator with HuBERT embedding, F0 encoder, unit encoder
- `MultiPeriodDiscriminator` + `MultiScaleDiscriminator` — adversarial training
- `train.py` — training loop with mel-spectrogram loss + GAN loss
- `inference.py` — synthesis with `--checkpoint_file` pointing to `g_XXXXXXXX`

To use the original pipeline, simply omit the `--use_facodec` flag:
```powershell
python inference.py --checkpoint_file checkpoints/ESD/g_00004200 ...
```

---

## Contact / Source

- FACodec Paper: *NaturalSpeech 3* (arXiv:2403.03100)
- FACodec Weights: https://huggingface.co/amphion/naturalspeech3_facodec
- Amphion GitHub: https://github.com/open-mmlab/Amphion
- ZEST Pipeline: This repository
