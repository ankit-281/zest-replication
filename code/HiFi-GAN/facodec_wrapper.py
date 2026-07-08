"""
FACodec Wrapper Module for ZEST Pipeline.

Wraps the Amphion FACodecDecoder (and optionally FACodecEncoder) with a clean
interface suitable for the ZEST feature adapter pipeline.

The decoder is loaded with pretrained weights and frozen — no gradient flows
through any FACodec parameter.

Usage:
    wrapper = FACodecWrapper(device="cpu")
    waveform = wrapper.decode(vq_post_emb, spk_embs)

Architecture reference:
    FACodec factorizes speech into:
    - Content (2 VQ codebooks)
    - Prosody (1 VQ codebook)
    - Timbre/Speaker (global embedding)
    - Acoustic details / Residual (3 VQ codebooks)
    Total = 6 VQ codebooks, each with dim=256

    The decoder's inference() method expects:
    - vq_post_emb: [B, 256, T] — the sum of all VQ post-embeddings
    - spk_embs: [B, 256] — the global timbre/speaker embedding

    The decoder produces waveform at 16kHz with hop_size=200.
"""

import os
import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup: ensure the vendored Amphion codec code is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AMPHION_DIR = os.path.join(_SCRIPT_DIR, "amphion_codec")
_NS3_CODEC_DIR = os.path.join(_AMPHION_DIR, "ns3_codec")

if not os.path.isdir(_NS3_CODEC_DIR):
    raise RuntimeError(
        f"FACodec model code not found at {_NS3_CODEC_DIR}. "
        f"Run 'python setup_facodec.py' first to download it."
    )

if _AMPHION_DIR not in sys.path:
    sys.path.insert(0, _AMPHION_DIR)

from ns3_codec.facodec import FACodecEncoder, FACodecDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Default architecture parameters (matching the pretrained checkpoint)
# ---------------------------------------------------------------------------
FACODEC_ENCODER_PARAMS = dict(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

FACODEC_DECODER_PARAMS = dict(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

# FACodec operates at 16kHz with hop_size=200
FACODEC_SAMPLE_RATE = 16000
FACODEC_HOP_SIZE = 200
FACODEC_LATENT_DIM = 256  # Channel dim of vq_post_emb
FACODEC_SPK_DIM = 256  # Speaker/timbre embedding dim


class FACodecWrapper(nn.Module):
    """
    Frozen wrapper around the Amphion FACodec decoder.

    This module:
    1. Loads the pretrained FACodecDecoder (and optionally FACodecEncoder)
    2. Freezes all parameters
    3. Provides a clean decode() interface

    The wrapper does NOT train any FACodec parameters.

    Attributes:
        decoder (FACodecDecoder): The frozen pretrained decoder
        encoder (FACodecEncoder or None): Optional encoder for extracting
            ground-truth latents (useful for adapter training)
        sample_rate (int): Output sample rate (16000)
        hop_size (int): FACodec hop size (200)
        latent_dim (int): Expected channel dim of vq_post_emb (256)
        spk_dim (int): Expected speaker embedding dim (256)
    """

    def __init__(
        self,
        weights_dir=None,
        device="cpu",
        load_encoder=False,
    ):
        """
        Args:
            weights_dir: Directory containing ns3_facodec_decoder.bin
                         (and optionally ns3_facodec_encoder.bin).
                         Defaults to <script_dir>/facodec_weights/
            device: Device to load the model on ("cpu" or "cuda:0")
            load_encoder: Whether to also load the encoder (needed for
                          extracting ground-truth latents for adapter training)
        """
        super().__init__()

        if weights_dir is None:
            weights_dir = os.path.join(_SCRIPT_DIR, "facodec_weights")

        self.sample_rate = FACODEC_SAMPLE_RATE
        self.hop_size = FACODEC_HOP_SIZE
        self.latent_dim = FACODEC_LATENT_DIM
        self.spk_dim = FACODEC_SPK_DIM

        # --- Load decoder ---
        self.decoder = FACodecDecoder(**FACODEC_DECODER_PARAMS)
        decoder_ckpt = os.path.join(weights_dir, "ns3_facodec_decoder.bin")
        if not os.path.isfile(decoder_ckpt):
            raise FileNotFoundError(
                f"Decoder weights not found at {decoder_ckpt}. "
                f"Run 'python setup_facodec.py' first."
            )
        self.decoder.load_state_dict(
            torch.load(decoder_ckpt, map_location=device)
        )
        self.decoder.eval()
        self._freeze(self.decoder)
        print(f"[FACodecWrapper] Loaded decoder from {decoder_ckpt}")

        # --- Optionally load encoder ---
        self.encoder = None
        if load_encoder:
            self.encoder = FACodecEncoder(**FACODEC_ENCODER_PARAMS)
            encoder_ckpt = os.path.join(weights_dir, "ns3_facodec_encoder.bin")
            if not os.path.isfile(encoder_ckpt):
                raise FileNotFoundError(
                    f"Encoder weights not found at {encoder_ckpt}. "
                    f"Run 'python setup_facodec.py' first."
                )
            self.encoder.load_state_dict(
                torch.load(encoder_ckpt, map_location=device)
            )
            self.encoder.eval()
            self._freeze(self.encoder)
            print(f"[FACodecWrapper] Loaded encoder from {encoder_ckpt}")

        self.to(device)

    @staticmethod
    def _freeze(module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def decode(self, vq_post_emb, spk_embs):
        """
        Decode latent representations to a waveform.

        Args:
            vq_post_emb: Tensor [B, 256, T] — post-VQ embedding
                         (what the adapter produces)
            spk_embs: Tensor [B, 256] — global speaker/timbre embedding
                      (what the adapter produces from EASE features)

        Returns:
            waveform: Tensor [B, 1, T_audio] where T_audio ≈ T * hop_size
        """
        self.decoder.eval()
        waveform = self.decoder.inference(vq_post_emb, spk_embs)
        return waveform

    @torch.no_grad()
    def encode(self, audio):
        """
        Encode audio to FACodec latent representations.

        This is used to extract ground-truth latent targets for training
        the FeatureAdapter. Not used during inference.

        Args:
            audio: Tensor [B, 1, T_audio] — raw waveform at 16kHz

        Returns:
            dict with keys:
                - vq_post_emb: [B, 256, T] — target for adapter temporal output
                - spk_embs: [B, 256] — target for adapter speaker output
                - vq_id: list of VQ indices (for analysis)
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder not loaded. Initialize with load_encoder=True"
            )

        self.encoder.eval()
        self.decoder.eval()

        enc_out = self.encoder(audio)
        # The decoder's forward pass with vq=True returns quantized outputs
        vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(
            enc_out, eval_vq=False, vq=True
        )

        return {
            "vq_post_emb": vq_post_emb,
            "spk_embs": spk_embs,
            "vq_id": vq_id,
        }

    def get_expected_dims(self):
        """Return the expected input dimensions for the decoder."""
        return {
            "latent_dim": self.latent_dim,
            "spk_dim": self.spk_dim,
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
        }
