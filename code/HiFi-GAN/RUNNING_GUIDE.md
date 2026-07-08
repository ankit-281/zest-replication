# ZEST × FACodec Quick Run Guide

This guide describes how to run and test the new ZEST waveform reconstruction pipeline using a frozen pre-trained FACodec decoder and the custom `FeatureAdapter`.

---

## Phase 1: Local Setup

1. **Install requirements**:
   Ensure you have installed `huggingface_hub` and `einops` in your local virtual environment:
   ```powershell
   cd d:\ZEST\ZEST\code
   venv\Scripts\pip.exe install huggingface_hub einops
   ```

2. **Download FACodec source and weights**:
   Run the setup script which clones the necessary Amphion modules and fetches pretrained bins from Hugging Face:
   ```powershell
   cd d:\ZEST\ZEST\code\HiFi-GAN
   ..\venv\Scripts\python.exe setup_facodec.py
   ```
   This will auto-generate:
   - `code/HiFi-GAN/amphion_codec/ns3_codec/` (sparse-cloned Amphion model files)
   - `code/HiFi-GAN/facodec_weights/ns3_facodec_encoder.bin`
   - `code/HiFi-GAN/facodec_weights/ns3_facodec_decoder.bin`

---

## Phase 2: Kaggle Training Setup (Heavy Compute)

Since training is heavy, do not run it locally on CPU. Use the Kaggle notebook script provided at [train_facodec_adapter.py](file:///d:/ZEST/ZEST/code/kaggle-notebooks/train_facodec_adapter.py).

### 1. Create a Kaggle Dataset containing:
- The updated code directory (make sure it includes the files under `code/HiFi-GAN/` and `code/kaggle-notebooks/`)
- Pre-extracted features:
  - `code/F0_predictor/wav2vec_feats/`
  - `code/F0_predictor/f0_contours/`
  - `code/EASE/EASE_embeddings/`
- Audio database directory:
  - `code/data/` (or wherever your `test`, `train`, `val` audios are stored)
- Manifest files:
  - `code/train_esd.txt`, `code/val_esd.txt`, `code/test_esd.txt`
  - `code/esd_f0_stats.pth`

### 2. Run the Notebook:
1. Create a new notebook on Kaggle.
2. Add your newly created dataset as input.
3. Copy-paste the content of `train_facodec_adapter.py` into cells.
4. **Important**: Enable **GPU T4** Accelerator in notebook settings.
5. In Cell 2, verify and adjust the target directories of `KAGGLE_INPUT` to match the exact dataset paths.
6. Run all cells.
7. Download the following output files from `/kaggle/working/`:
   - `adapter_best.pth`
   - `code_embedding_best.pth`
8. Place both `.pth` files in your local `code/HiFi-GAN/` directory.

---

## Phase 3: Local Speech Synthesis (Inference on CPU)

Once the best adapter weights are downloaded, execute synthesis on your CPU:

### Option A: Using the standalone script (Recommended)
```powershell
cd d:\ZEST\ZEST\code\HiFi-GAN
..\venv\Scripts\python.exe decoder_inference.py `
    --adapter_checkpoint adapter_best.pth `
    --pitch_folder ../F0_predictor/f0_contours `
    --emo_folder ../F0_predictor/wav2vec_feats `
    --output_dir DSDT_facodec `
    --convert --debug
```

### Option B: Using the backward-compatible CLI (`inference.py`)
```powershell
cd d:\ZEST\ZEST\code\HiFi-GAN
..\venv\Scripts\python.exe inference.py `
    --use_facodec `
    --adapter_checkpoint adapter_best.pth `
    --pitch_folder ../F0_predictor/f0_contours `
    --emo_folder ../F0_predictor/wav2vec_feats `
    --output_dir DSDT_facodec `
    --convert --debug
```

*All synthesized waveforms will be written to `code/HiFi-GAN/DSDT_facodec/` at 16kHz.*
