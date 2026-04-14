"""
TTS Service using Coqui XTTS v2 and a custom Kinyarwanda health model.

Models:
  - tts_models/multilingual/multi-dataset/xtts_v2  (multilingual, default)
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
        self._load_xtts_v2()
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

    def _load_xtts_v2(self) -> None:
        from TTS.api import TTS

        print("[TTS] Loading XTTS v2 …")
        self.xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            self.device
        )
        print("[TTS] XTTS v2 ready.")

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

    def synthesize(
        self,
        text: str,
        output_path: str = "output.wav",
        language: str = "en",
        use_kinyarwanda: bool = False,
        speaker_wav: str | None = None,
    ) -> str:
        """
        Generate speech and write it to *output_path*.

        Parameters
        ----------
        text            : Input text to synthesise.
        output_path     : Destination WAV file.
        language        : BCP-47 language code for XTTS v2 (default "en").
        use_kinyarwanda : Use the custom Kinyarwanda health model instead.
        speaker_wav     : Optional path to a reference speaker WAV for voice
                          cloning (XTTS v2 only; falls back to the model
                          sample when omitted).

        Returns
        -------
        output_path after successful synthesis.
        """
        if use_kinyarwanda:
            return self._synthesize_kinyarwanda(text, output_path)
        return self._synthesize_xtts(text, output_path, language, speaker_wav)

    # ------------------------------------------------------------------
    # Internal synthesis methods
    # ------------------------------------------------------------------

    def _synthesize_xtts(
        self,
        text: str,
        output_path: str,
        language: str,
        speaker_wav: str | None,
    ) -> str:
        ref = speaker_wav or self.speaker_wav
        self.xtts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=ref,
            language=language,
        )
        return output_path

    def _synthesize_kinyarwanda(self, text: str, output_path: str) -> str:
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
