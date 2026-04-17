"""
TTS Service using a custom Kinyarwanda XTTS-based health model.

Model:
  - DigitalUmuganda/xtts_based_male_female_health_model  (Kinyarwanda)

Environment variables:
  HF_TOKEN   - Hugging Face token (required for the private Kinyarwanda model)
  MODEL_DIR  - Directory to store the downloaded Kinyarwanda model (default: ./kinyarwanda_health_model)
"""

import glob
import os

import torch
import torchaudio
from huggingface_hub import login, snapshot_download


class TTSService:
    def __init__(self, model_dir: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Using device: {self.device}")

        self.model_dir = model_dir or os.environ.get(
            "MODEL_DIR", "./kinyarwanda_health_model"
        )

        self._authenticate()
        self._load_kinyarwanda_model()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        else:
            print(
                "[TTS] Warning: HF_TOKEN not set — private model download may fail."
            )

    def _load_kinyarwanda_model(self) -> None:
        if not os.path.isdir(self.model_dir):
            print(f"[TTS] Downloading Kinyarwanda model to {self.model_dir} …")
            os.makedirs(self.model_dir, exist_ok=True)
            snapshot_download(
                repo_id="DigitalUmuganda/xtts_based_male_female_health_model",
                local_dir=self.model_dir,
            )

        from TTS.api import TTS

        config_path = os.path.join(self.model_dir, "config.json")
        print("[TTS] Loading Kinyarwanda model …")
        kw_tts = TTS(model_path=self.model_dir, config_path=config_path).to(
            self.device
        )
        self._kinyarwanda_model = kw_tts.synthesizer.tts_model

        wav_files = glob.glob(f"{self.model_dir}/**/*.wav", recursive=True)
        self.speaker_wav: str | None = wav_files[0] if wav_files else None
        print("[TTS] Kinyarwanda model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str = "output.wav") -> str:
        """
        Generate Kinyarwanda speech and write it to *output_path*.
        """
        if self.speaker_wav is None:
            raise RuntimeError(
                "No reference speaker WAV found in the Kinyarwanda model directory."
            )

        gpt_cond_latent, speaker_embedding = (
            self._kinyarwanda_model.get_conditioning_latents(
                audio_path=self.speaker_wav
            )
        )
        out = self._kinyarwanda_model.inference(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,
            repetition_penalty=5.0,
        )

        wav = torch.tensor(out["wav"]).unsqueeze(0)
        torchaudio.save(output_path, wav, 24000)
        return output_path
