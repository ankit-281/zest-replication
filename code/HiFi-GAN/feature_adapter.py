"""
Feature Adapter Module for ZEST → FACodec Pipeline.

Maps the existing ZEST feature representations (HuBERT, wav2vec, EASE, F0)
into the latent representation expected by the frozen FACodec decoder.

This is the ONLY trainable component in the new pipeline.

Architecture:
    Content (HuBERT) ─┐
    Emotion (wav2vec) ─┤
    F0 (converted) ────┤─→ Concat → Linear → ReLU → Linear → [B, 256, T]  (vq_post_emb)
                       │
    Speaker (EASE) ────────→ Linear → ReLU → Linear → [B, 256]            (spk_embs)

Design:
    - Modular input interface: named inputs, not positional
    - Dynamic dimension detection: reads input shapes at first forward pass
    - Separate temporal and speaker projections
    - Future-proof: swap HuBERT→WavLM, wav2vec→emotion2vec, EASE→CAM++
      without touching this module (as long as you update input dims)

Usage:
    adapter = FeatureAdapter(
        content_dim=128,    # HuBERT embedding dim
        emotion_dim=128,    # wav2vec emotion dim
        speaker_dim=128,    # EASE speaker embedding dim
        f0_dim=1,           # F0 is 1-dimensional
    )
    vq_post_emb, spk_embs = adapter(
        content_features=hubert_feats,   # [B, 128, T]
        emotion_features=wav2vec_feats,  # [B, 128] (global) or [B, 128, T]
        speaker_embedding=ease_emb,      # [B, 128]
        f0_features=f0,                  # [B, 1, T]
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """
    Lightweight MLP adapter: ZEST features → FACodec decoder latent space.

    Trainable. The FACodec decoder is frozen; only this adapter learns.

    Args:
        content_dim: Channel dimension of content features (HuBERT). Default: 128
        emotion_dim: Channel dimension of emotion features (wav2vec). Default: 128
        speaker_dim: Dimension of speaker embedding (EASE). Default: 128
        f0_dim: Channel dimension of F0 features. Default: 1
        facodec_latent_dim: Target temporal latent dim for FACodec. Default: 256
        facodec_spk_dim: Target speaker embedding dim for FACodec. Default: 256
        hidden_dim: Hidden dimension in the MLP. Default: 512
    """

    def __init__(
        self,
        content_dim=128,
        emotion_dim=128,
        speaker_dim=128,
        f0_dim=1,
        facodec_latent_dim=256,
        facodec_spk_dim=256,
        hidden_dim=512,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.emotion_dim = emotion_dim
        self.speaker_dim = speaker_dim
        self.f0_dim = f0_dim
        self.facodec_latent_dim = facodec_latent_dim
        self.facodec_spk_dim = facodec_spk_dim

        # Total input dim for temporal projection:
        # content + emotion + f0 + speaker (broadcast to temporal)
        temporal_input_dim = content_dim + emotion_dim + f0_dim + speaker_dim

        # --- Temporal MLP ---
        # Maps concatenated ZEST features → FACodec vq_post_emb dimension
        # Architecture: Linear → ReLU → Linear
        self.temporal_mlp = nn.Sequential(
            nn.Conv1d(temporal_input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, facodec_latent_dim, kernel_size=1),
        )

        # --- Speaker MLP ---
        # Maps EASE speaker embedding → FACodec spk_embs dimension
        # Architecture: Linear → ReLU → Linear
        self.speaker_mlp = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, facodec_spk_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _align_temporal(feature, target_length):
        """
        Align a feature tensor to a target temporal length via interpolation.

        Args:
            feature: [B, C, T_src] or [B, C]
            target_length: int — target T dimension

        Returns:
            [B, C, target_length]
        """
        if feature.dim() == 2:
            # Global feature [B, C] → broadcast to [B, C, T]
            return feature.unsqueeze(-1).expand(-1, -1, target_length)
        elif feature.shape[-1] == target_length:
            return feature
        else:
            # Interpolate to match target length
            return F.interpolate(
                feature.float(),
                size=target_length,
                mode="nearest",
            )

    def forward(
        self,
        content_features,
        emotion_features,
        speaker_embedding,
        f0_features,
    ):
        """
        Project ZEST features into FACodec latent space.

        Args:
            content_features: [B, content_dim, T] — HuBERT features
                              (after embedding lookup + UnitEncoder)
            emotion_features: [B, emotion_dim] — wav2vec emotion embedding (global)
                              OR [B, emotion_dim, T] (frame-level)
            speaker_embedding: [B, speaker_dim] — EASE speaker embedding
            f0_features: [B, f0_dim, T] — F0 contour
                         OR [B, 1, T] (raw F0, 1-dim)

        Returns:
            vq_post_emb: [B, facodec_latent_dim, T] — temporal latent for decoder
            spk_embs: [B, facodec_spk_dim] — speaker embedding for decoder
        """
        # Determine target temporal length from content features
        if content_features.dim() == 3:
            target_length = content_features.shape[-1]
        elif f0_features.dim() == 3:
            target_length = f0_features.shape[-1]
        else:
            raise ValueError(
                "At least one of content_features or f0_features must be "
                "3-dimensional [B, C, T] to determine temporal length."
            )

        # Ensure all temporal features are [B, C, T] with the same T
        content = self._align_temporal(content_features, target_length)
        emotion = self._align_temporal(emotion_features, target_length)
        f0 = self._align_temporal(f0_features, target_length)
        spk_temporal = self._align_temporal(speaker_embedding, target_length)

        # Concatenate along channel dimension: [B, total_dim, T]
        concat = torch.cat([content, emotion, f0, spk_temporal], dim=1)

        # Temporal MLP: [B, total_dim, T] → [B, facodec_latent_dim, T]
        vq_post_emb = self.temporal_mlp(concat)

        # Speaker MLP: [B, speaker_dim] → [B, facodec_spk_dim]
        if speaker_embedding.dim() == 3:
            # If speaker embedding is temporal, take the mean
            spk_global = speaker_embedding.mean(dim=-1)
        else:
            spk_global = speaker_embedding
        spk_embs = self.speaker_mlp(spk_global)

        return vq_post_emb, spk_embs

    def get_config(self):
        """Return adapter configuration for serialization."""
        return {
            "content_dim": self.content_dim,
            "emotion_dim": self.emotion_dim,
            "speaker_dim": self.speaker_dim,
            "f0_dim": self.f0_dim,
            "facodec_latent_dim": self.facodec_latent_dim,
            "facodec_spk_dim": self.facodec_spk_dim,
        }

    @classmethod
    def from_config(cls, config):
        """Create adapter from a config dict."""
        return cls(**config)


def save_adapter(adapter, filepath):
    """Save adapter weights and config."""
    torch.save(
        {
            "config": adapter.get_config(),
            "state_dict": adapter.state_dict(),
        },
        filepath,
    )
    print(f"[FeatureAdapter] Saved to {filepath}")


def load_adapter(filepath, device="cpu"):
    """Load adapter weights and config."""
    checkpoint = torch.load(filepath, map_location=device)
    adapter = FeatureAdapter.from_config(checkpoint["config"])
    adapter.load_state_dict(checkpoint["state_dict"])
    print(f"[FeatureAdapter] Loaded from {filepath}")
    return adapter
